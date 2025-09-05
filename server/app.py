from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from pathlib import Path
import logging
import asyncio
import gpxpy
import simplekml
from io import StringIO, BytesIO
from typing import Optional
from uuid import uuid4
import os
import json
import time
from typing import Dict, List

# Database (PostgreSQL via SQLAlchemy async)
from sqlalchemy import String, Float, Boolean, BigInteger, Index, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


app = FastAPI(title="FlightTracePro – GPX→KML + Live", version="1.1.0")

# Logger
logger = logging.getLogger("flighttracepro")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger.setLevel(logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local static assets (e.g., 3D models) to avoid CORS issues
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")


# Simple in-memory KML store to provide a link for Google Earth
KML_STORE: dict[str, bytes] = {}

# Live map data structures
class LiveSample(BaseModel):
    lat: float
    lon: float
    alt_m: float | None = None
    spd_kt: float | None = None
    vsi_ms: float | None = None
    hdg_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    ts: float | None = None  # epoch seconds
    callsign: str | None = None
    aircraft: str | None = None

class ChannelState:
    def __init__(self):
        self.connections: set[WebSocket] = set()
        self.viewers: set[WebSocket] = set()
        self.feeders: set[WebSocket] = set()
        self.last_samples: dict[str, tuple[LiveSample, float]] = {}
        self.lock = asyncio.Lock()
        self.active_callsigns: set[str] = set()
        self.history: dict[str, list[dict]] = {}
        self.history_max = 5000
        # per-callsign filter state
        self.filters: dict[str, dict] = {}
        # persistence state
        self._last_persist_ts: float = 0.0
        self._dirty: bool = False

def _haversine_m(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def normalize_sample(st: ChannelState, callsign: str, s: LiveSample, now_ts: float) -> tuple[LiveSample, bool]:
    """Clean obvious bad data, reset paths on teleports, be gap-aware. Returns (sample, break_path)."""
    fs = st.filters.get(callsign) or { 'ema_gs': None, 'ema_vsi': None, 'ema_alt': None, 'last_ts': None, 'last_lat': None, 'last_lon': None }
    # ensure ts
    ts = float(s.ts or now_ts)
    last_ts = fs['last_ts']
    dt = (ts - last_ts) if last_ts else None
    gap_reset = (dt is None) or (dt < 0) or (dt > 10.0)
    # compute implied speed
    spd_implied_kt = None
    break_path = False
    if fs['last_lat'] is not None and fs['last_lon'] is not None and dt and dt > 0:
        dist = _haversine_m(fs['last_lat'], fs['last_lon'], s.lat, s.lon)
        spd_implied_kt = (dist/dt) * 1.94384
        if dist > 20000:  # >20km jump
            break_path = True
            gap_reset = True
    # EMAs
    def ema(prev, x, alpha=0.2):
        return x if prev is None else (prev*(1-alpha) + x*alpha)
    # Update EMAs cautiously
    if gap_reset:
        fs['ema_gs'] = float(s.spd_kt) if (s.spd_kt is not None) else (spd_implied_kt or fs['ema_gs'])
        fs['ema_vsi'] = float(s.vsi_ms) if (s.vsi_ms is not None) else fs['ema_vsi']
        fs['ema_alt'] = float(s.alt_m) if (s.alt_m is not None) else fs['ema_alt']
    else:
        if spd_implied_kt is not None:
            fs['ema_gs'] = ema(fs['ema_gs'], spd_implied_kt)
        elif s.spd_kt is not None:
            fs['ema_gs'] = ema(fs['ema_gs'], float(s.spd_kt))
        if s.vsi_ms is not None:
            fs['ema_vsi'] = ema(fs['ema_vsi'], float(s.vsi_ms))
        if s.alt_m is not None:
            fs['ema_alt'] = ema(fs['ema_alt'], float(s.alt_m), alpha=0.1)
    # Repair obvious bad alt: zero/null spikes while moving
    if (s.alt_m is None or (s.alt_m == 0 and (fs['ema_alt'] or 0) > 50)) and (fs['ema_alt'] is not None):
        s.alt_m = fs['ema_alt']
    # Clamp insane VSI
    if s.vsi_ms is not None and fs['ema_vsi'] is not None:
        if abs(s.vsi_ms) > max(30.0, abs(fs['ema_vsi'])*3 + 5):
            s.vsi_ms = fs['ema_vsi']
    # Accept new sample; update last
    fs['last_ts'] = ts
    fs['last_lat'] = s.lat
    fs['last_lon'] = s.lon
    st.filters[callsign] = fs
    return s, break_path

CHANNELS: dict[str, ChannelState] = {}

# Persistence helpers (per-channel JSON files)
DATA_DIR = Path(os.environ.get("FTPRO_DATA_DIR", "server/data"))
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

def _sanitize_channel(name: str) -> str:
    s = ''.join(ch for ch in name if (ch.isalnum() or ch in ('_', '-')))
    return s or "default"

def _channel_history_path(channel: str) -> Path:
    return DATA_DIR / f"history_{_sanitize_channel(channel)}.json"

def _load_history(channel: str, st: ChannelState) -> None:
    p = _channel_history_path(channel)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        hist = data.get("history") if isinstance(data, dict) else None
        if isinstance(hist, dict):
            # ensure lists
            for k, v in list(hist.items()):
                if not isinstance(v, list):
                    hist.pop(k, None)
            st.history = hist
    except Exception as e:
        try:
            logger.warning("failed to load history for %s: %s", channel, e)
        except Exception:
            pass

def _save_history(channel: str, st: ChannelState) -> None:
    # Write atomically via temp file
    p = _channel_history_path(channel)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        payload = {"channel": channel, "saved_at": datetime.now(timezone.utc).isoformat(), "history": st.history}
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        tmp.replace(p)
        st._dirty = False
        st._last_persist_ts = time.monotonic()
    except Exception as e:
        try:
            logger.warning("failed to save history for %s: %s", channel, e)
        except Exception:
            pass

def _maybe_persist_locked(channel: str, st: ChannelState, interval: float = 2.0) -> None:
    if USE_DB:
        return  # DB-backed; skip JSON persistence
    st._dirty = True
    now = time.monotonic()
    last = st._last_persist_ts or 0.0
    if (now - last) >= interval:
        _save_history(channel, st)

def get_channel(name: str) -> ChannelState:
    st = CHANNELS.get(name)
    if not st:
        st = ChannelState()
        CHANNELS[name] = st
        # Load persisted history only when not using DB
        if not USE_DB:
            _load_history(name, st)
    return st

LIVE_POST_KEY = os.environ.get("LIVE_POST_KEY", "").strip()
LIVE_TTL_SEC = int(os.environ.get("LIVE_TTL_SEC", "60") or "60")
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_DB = bool(DATABASE_URL)

# SQLAlchemy models and engine
class Base(DeclarativeBase):
    pass

class TrackPoint(Base):
    __tablename__ = "track_points"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    channel: Mapped[str] = mapped_column(String(64), index=True)
    callsign: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    alt_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    spd_kt: Mapped[float | None] = mapped_column(Float, nullable=True)
    vsi_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    hdg_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    roll_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    break_path: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

Index("idx_track_channel_callsign_ts", TrackPoint.channel, TrackPoint.callsign, TrackPoint.ts)

engine = create_async_engine(DATABASE_URL) if USE_DB else None
async_session = async_sessionmaker(engine, expire_on_commit=False) if USE_DB else None


async def _broadcast_viewers(st: ChannelState, data: dict):
    dead = []
    for ws in list(st.viewers):
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        st.viewers.discard(ws)
        st.connections.discard(ws)


def _prune_stale_locked(st: ChannelState, now_ts: float) -> list[str]:
    removed = []
    stale = [cs for cs, (_, t) in st.last_samples.items() if now_ts - t > LIVE_TTL_SEC]
    for cs in stale:
        st.last_samples.pop(cs, None)
        if cs in st.active_callsigns:
            st.active_callsigns.discard(cs)
            removed.append(cs)
    return removed


def convert_gpx_to_kml(
    gpx_text: str,
    line_name: str = "Flight Trail 3D",
    line_color: str = simplekml.Color.red,
    line_width: int = 3,
    altitude_mode: str = "absolute",
    extrude: int = 0,
    include_waypoints: int = 0,
    color_by: str = "solid",  # solid | speed
    include_tour: int = 0,
) -> bytes:
    try:
        gpx = gpxpy.parse(StringIO(gpx_text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid GPX: {e}")

    # Collect points
    track_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                track_points.append(point)

    if not track_points:
        # Also try routes and waypoints if no tracks
        for route in gpx.routes:
            for point in route.points:
                track_points.append(point)
        if not track_points:
            for wpt in gpx.waypoints:
                track_points.append(wpt)

    if not track_points:
        raise HTTPException(status_code=400, detail="No coordinates found in GPX")

    kml = simplekml.Kml()

    # Altitude mode mapping
    mode_map = {
        "absolute": simplekml.AltitudeMode.absolute,
        "relativeToGround": simplekml.AltitudeMode.relativetoground,
        "clampToGround": simplekml.AltitudeMode.clamptoground,
    }

    def pt_to_coord(p):
        # Some GPX points may lack elevation
        ele = getattr(p, 'elevation', None)
        return (p.longitude, p.latitude, ele if ele is not None else 0)

    # Helper to create a styled LineString
    def add_linestring(coords, name=None, color=None):
        if len(coords) < 2:
            return None
        ls = kml.newlinestring(name=name or line_name)
        ls.coords = coords
        ls.altitudemode = mode_map.get(altitude_mode, simplekml.AltitudeMode.absolute)
        ls.extrude = extrude
        ls.style.linestyle.width = line_width
        ls.style.linestyle.color = color or line_color
        return ls

    # When color_by == speed, split into short segments with varying color
    if color_by == "speed":
        # Compute speeds m/s between consecutive points
        from math import radians, sin, cos, sqrt, atan2
        def haversine_m(p1, p2):
            R = 6371000.0
            lat1, lon1 = radians(p1.latitude), radians(p1.longitude)
            lat2, lon2 = radians(p2.latitude), radians(p2.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c

        samples = []  # (coord1, coord2, speed_mps)
        for i in range(len(track_points) - 1):
            p1, p2 = track_points[i], track_points[i+1]
            t1, t2 = getattr(p1, 'time', None), getattr(p2, 'time', None)
            if t1 and t2 and t2 > t1:
                dt = (t2 - t1).total_seconds()
                if dt > 0:
                    dist = haversine_m(p1, p2)
                    spd = dist / dt
                else:
                    continue
            else:
                # If no time, skip speed coloring for this segment
                continue
            samples.append((pt_to_coord(p1), pt_to_coord(p2), spd))

        if samples:
            spds = [s[2] for s in samples]
            smin, smax = min(spds), max(spds)
            if smax == smin:
                smax = smin + 1e-6

            def lerp(a, b, t):
                return a + (b - a) * t

            # Blue (slow) -> Red (fast): in web hex then convert to simplekml aabbggrr
            def speed_to_simplekml_color(spd):
                t = (spd - smin) / (smax - smin)
                # RGB from blue(0,114,255) to red(255,0,0)
                r = int(lerp(0, 255, t))
                g = int(lerp(114, 0, t))
                b = int(lerp(255, 0, t))
                return f"ff{b:02x}{g:02x}{r:02x}"

            for (c1, c2, spd) in samples:
                add_linestring([c1, c2], name=None, color=speed_to_simplekml_color(spd))
        else:
            # Fallback: single colored line
            coords = [pt_to_coord(p) for p in track_points]
            add_linestring(coords)
    else:
        coords = [pt_to_coord(p) for p in track_points]
        add_linestring(coords)

    # Add waypoints as placemarks
    if include_waypoints:
        for w in gpx.waypoints:
            pnt = kml.newpoint(name=w.name or "Waypoint", coords=[(w.longitude, w.latitude, (w.elevation or 0))])
            pnt.altitudemode = mode_map.get(altitude_mode, simplekml.AltitudeMode.absolute)
            pnt.extrude = 0
            desc = []
            if w.description:
                desc.append(w.description)
            if w.time:
                desc.append(f"Time: {w.time}")
            if w.elevation is not None:
                desc.append(f"Elevation: {w.elevation} m")
            if desc:
                pnt.description = "\n".join(desc)

    # Optional: simple gx:Tour flythrough (best-effort)
    if include_tour:
        try:
            # Build a minimal tour with a few key frames
            tour = kml.newgxtour(name=f"Flythrough: {line_name}")
            playlist = tour.newgxplaylist()
            # Use every Nth point to limit size
            step = max(1, len(track_points) // 200)
            for i in range(0, len(track_points), step):
                p = track_points[i]
                # Create a smooth fly-to towards a LookAt slightly above the path
                gxflyto = playlist.newgxflyto(gxduration=1.0)
                lookat = simplekml.LookAt(
                    longitude=p.longitude,
                    latitude=p.latitude,
                    altitude=(getattr(p, 'elevation', 0) or 0) + 120,
                    range=250,
                    tilt=70,
                    heading=0,
                    altitudemode=mode_map.get(altitude_mode, simplekml.AltitudeMode.absolute),
                )
                gxflyto.lookat = lookat
        except Exception:
            # Ignore tour creation errors; still return KML
            pass

    kml_bytes = kml.kml().encode("utf-8")
    return kml_bytes


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    # Resolve Cesium Ion token from env or mounted secret file
    token = os.environ.get("CESIUM_ION_TOKEN", "").strip()
    if not token:
        # Support conventional *_FILE pattern and Docker secrets
        token_file = os.environ.get("CESIUM_ION_TOKEN_FILE")
        candidate_files = [
            token_file,
            "/run/secrets/cesium_ion_token",
            "/run/secrets/CESIUM_ION_TOKEN",
        ]
        for path in candidate_files:
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        token = fh.read().strip()
                        break
                except Exception:
                    pass
    return INDEX_HTML.replace("__CESIUM_ION_TOKEN__", token)


@app.get("/api/live/{channel}/recent")
async def live_recent(channel: str):
    st = get_channel(channel)
    now = datetime.now(timezone.utc).timestamp()
    out = []
    async with st.lock:
        _prune_stale_locked(st, now)
        for _, (s, t) in st.last_samples.items():
            if now - t <= LIVE_TTL_SEC:
                out.append(s.dict())
    return {"channel": channel, "samples": out, "ts": now}


@app.get("/api/live/{channel}/info")
async def live_info(channel: str):
    st = get_channel(channel)
    now = datetime.now(timezone.utc).timestamp()
    async with st.lock:
        # prune stale players
        stale = [cs for cs, (_, t) in st.last_samples.items() if now - t > LIVE_TTL_SEC]
        for cs in stale:
            st.last_samples.pop(cs, None)
        players = []
        for cs, (s, t) in st.last_samples.items():
            d = s.dict()
            d.update({"callsign": cs, "ts": t, "age_s": max(0.0, now - t)})
            players.append(d)
        return {
            "channel": channel,
            "ts": now,
            "counts": {
                "connections": len(st.connections),
                "viewers": len(st.viewers),
                "feeders": len(st.feeders),
                "players": len(players),
            },
            "players": players,
        }


@app.get("/api/live/{channel}/history")
async def live_history(channel: str):
    st = get_channel(channel)
    # Prefer DB if available
    if USE_DB:
        # Build tracks grouped by callsign, limited per callsign to history_max
        result: Dict[str, List[dict]] = {}
        try:
            async with async_session() as session:
                # Find callsigns present in channel
                q_cs = select(TrackPoint.callsign).where(TrackPoint.channel == channel).distinct()
                callsigns = [row[0] for row in (await session.execute(q_cs)).all()]
                maxn = st.history_max
                for cs in callsigns:
                    # last N points per callsign ordered ascending by ts
                    q = (
                        select(TrackPoint)
                        .where((TrackPoint.channel == channel) & (TrackPoint.callsign == cs))
                        .order_by(TrackPoint.ts.desc())
                        .limit(maxn)
                    )
                    rows = (await session.execute(q)).scalars().all()
                    rows.sort(key=lambda r: r.ts)
                    result[cs] = [
                        {
                            "lat": r.lat,
                            "lon": r.lon,
                            "alt_m": r.alt_m,
                            "spd_kt": r.spd_kt,
                            "vsi_ms": r.vsi_ms,
                            "hdg_deg": r.hdg_deg,
                            "pitch_deg": r.pitch_deg,
                            "roll_deg": r.roll_deg,
                            "ts": r.ts,
                            "callsign": r.callsign,
                            "break_path": r.break_path,
                        }
                        for r in rows
                    ]
        except Exception as e:
            try:
                logger.warning("history db error for %s: %s", channel, e)
            except Exception:
                pass
            # fall back to in-memory
            async with st.lock:
                result = { cs: (st.history.get(cs) or []) for cs in st.history.keys() }
        return {"channel": channel, "tracks": result}
    else:
        async with st.lock:
            out = { cs: (st.history.get(cs) or []) for cs in st.history.keys() }
        return {"channel": channel, "tracks": out}


@app.post("/api/live/{channel}")
async def live_post(channel: str, sample: LiveSample, key: str | None = None):
    if LIVE_POST_KEY and (not key or key != LIVE_POST_KEY):
        raise HTTPException(status_code=403, detail="forbidden")
    st = get_channel(channel)
    now = datetime.now(timezone.utc).timestamp()
    if not sample.ts:
        sample.ts = now
    callsign = sample.callsign or "n/a"
    # Normalize/clean
    sample, break_path = normalize_sample(st, callsign, sample, now)
    async with st.lock:
        st.last_samples[callsign] = (sample, now)
        is_new = callsign not in st.active_callsigns
        st.active_callsigns.add(callsign)
        # append to history
        hist = st.history.get(callsign) or []
        item = sample.dict(); item['break_path'] = break_path
        item['ts'] = sample.ts
        hist.append(item)
        if len(hist) > st.history_max:
            del hist[: len(hist) - st.history_max]
        st.history[callsign] = hist
        _maybe_persist_locked(channel, st)
    # Persist to DB (non-blocking for channel lock)
    if USE_DB:
        try:
            async with async_session() as session:
                session: AsyncSession
                tp = TrackPoint(
                    channel=channel,
                    callsign=callsign,
                    ts=float(sample.ts or now),
                    lat=float(sample.lat),
                    lon=float(sample.lon),
                    alt_m=float(sample.alt_m) if sample.alt_m is not None else None,
                    spd_kt=float(sample.spd_kt) if sample.spd_kt is not None else None,
                    vsi_ms=float(sample.vsi_ms) if sample.vsi_ms is not None else None,
                    hdg_deg=float(sample.hdg_deg) if sample.hdg_deg is not None else None,
                    pitch_deg=float(sample.pitch_deg) if sample.pitch_deg is not None else None,
                    roll_deg=float(sample.roll_deg) if sample.roll_deg is not None else None,
                    break_path=bool(break_path),
                )
                session.add(tp)
                await session.commit()
        except Exception as e:
            try:
                logger.warning("db insert error: %s", e)
            except Exception:
                pass
        # broadcast to viewers only
        payload = sample.dict(); payload['break_path'] = break_path
        data = {"type": "state", "payload": payload}
        await _broadcast_viewers(st, data)
        # prune and emit leave events
        removed = _prune_stale_locked(st, now)
    try:
        logger.debug(
            "[live] POST ch=%s cs=%s lat=%.5f lon=%.5f alt=%s viewers=%d",
            channel,
            callsign,
            float(sample.lat),
            float(sample.lon),
            ("%.0f" % float(sample.alt_m)) if sample.alt_m is not None else "n/a",
            int(len(st.viewers)),
        )
    except Exception:
        pass
    # Emit join/leave events outside lock
    try:
        if is_new:
            await _broadcast_viewers(st, {"type":"event","event":"join","callsign":callsign, "ts": now})
        for cs in removed:
            await _broadcast_viewers(st, {"type":"event","event":"leave","callsign":cs, "ts": now})
    except Exception:
        pass
    return {"ok": True}


@app.websocket("/ws/live/{channel}")
async def ws_live(websocket: WebSocket, channel: str):
    # Extract query params from WebSocket URL
    try:
        qp = websocket.query_params or {}
    except Exception:
        qp = {}
    mode = qp.get("mode", "viewer")
    key = qp.get("key")
    is_feeder = (mode == "feeder")
    if is_feeder and LIVE_POST_KEY and (not key or key != LIVE_POST_KEY):
        await websocket.close(code=4403)
        return
    await websocket.accept()
    st = get_channel(channel)
    client_addr = None
    try:
        client_addr = getattr(websocket, "client", None)
    except Exception:
        client_addr = None
    async with st.lock:
        st.connections.add(websocket)
        (st.feeders if is_feeder else st.viewers).add(websocket)
    try:
        logger.info("[ws] connect ch=%s mode=%s from=%s", channel, ("feeder" if is_feeder else "viewer"), str(client_addr))
    except Exception:
        pass
    try:
        if not is_feeder:
            now = datetime.now(timezone.utc).timestamp()
            async with st.lock:
                recent = 0
                for _, (s, t) in st.last_samples.items():
                    if now - t <= LIVE_TTL_SEC:
                        await websocket.send_json({"type": "state", "payload": s.dict()})
                        recent += 1
            try:
                logger.info("[ws] init ch=%s recent=%d", channel, int(recent))
            except Exception:
                pass
        while True:
            msg = await websocket.receive_json()
            if not isinstance(msg, dict) or msg.get("type") != "state":
                continue
            try:
                sample = LiveSample(**msg.get("payload", {}))
            except Exception:
                continue
            now = datetime.now(timezone.utc).timestamp()
            if not sample.ts:
                sample.ts = now
            callsign = sample.callsign or "n/a"
            async with st.lock:
                sample, break_path = normalize_sample(st, callsign, sample, now)
                st.last_samples[callsign] = (sample, now)
                is_new = callsign not in st.active_callsigns
                st.active_callsigns.add(callsign)
                # append to history
                hist = st.history.get(callsign) or []
                item = sample.dict(); item['ts'] = sample.ts; item['break_path'] = break_path
                hist.append(item)
                if len(hist) > st.history_max:
                    del hist[: len(hist) - st.history_max]
                st.history[callsign] = hist
                _maybe_persist_locked(channel, st)
            # Persist to DB outside lock
            if USE_DB:
                try:
                    async with async_session() as session:
                        session: AsyncSession
                        tp = TrackPoint(
                            channel=channel,
                            callsign=callsign,
                            ts=float(sample.ts or now),
                            lat=float(sample.lat),
                            lon=float(sample.lon),
                            alt_m=float(sample.alt_m) if sample.alt_m is not None else None,
                            spd_kt=float(sample.spd_kt) if sample.spd_kt is not None else None,
                            vsi_ms=float(sample.vsi_ms) if sample.vsi_ms is not None else None,
                            hdg_deg=float(sample.hdg_deg) if sample.hdg_deg is not None else None,
                            pitch_deg=float(sample.pitch_deg) if sample.pitch_deg is not None else None,
                            roll_deg=float(sample.roll_deg) if sample.roll_deg is not None else None,
                            break_path=bool(break_path),
                        )
                        session.add(tp)
                        await session.commit()
                except Exception as e:
                    try:
                        logger.warning("db insert error: %s", e)
                    except Exception:
                        pass
                payload = sample.dict(); payload['break_path'] = break_path
                data = {"type": "state", "payload": payload}
                await _broadcast_viewers(st, data)
                removed = _prune_stale_locked(st, now)
            try:
                logger.debug(
                    "[ws] state ch=%s cs=%s lat=%.5f lon=%.5f alt=%s viewers=%d",
                    channel,
                    callsign,
                    float(sample.lat),
                    float(sample.lon),
                    ("%.0f" % float(sample.alt_m)) if sample.alt_m is not None else "n/a",
                    int(len(st.viewers)),
                )
            except Exception:
                pass
            # Emit join/leave events outside lock
            try:
                if is_new:
                    await _broadcast_viewers(st, {"type":"event","event":"join","callsign":callsign, "ts": now})
                for cs in removed:
                    await _broadcast_viewers(st, {"type":"event","event":"leave","callsign":cs, "ts": now})
            except Exception:
                pass
    except WebSocketDisconnect as e:
        try:
            logger.info("[ws] disconnect ch=%s mode=%s from=%s code=%s", channel, ("feeder" if is_feeder else "viewer"), str(client_addr), getattr(e, 'code', ''))
        except Exception:
            pass
    except Exception:
        # Ignore client errors
        pass
    finally:
        async with st.lock:
            st.connections.discard(websocket)
            st.viewers.discard(websocket)
            st.feeders.discard(websocket)


@app.post("/api/convert")
async def api_convert(
    file: UploadFile = File(..., description="GPX file"),
    name: Optional[str] = Form(default="Flight Trail 3D"),
    width: Optional[int] = Form(default=3),
    color: Optional[str] = Form(default="#ff0000"),  # hex in RRGGBB or AABBGGRR? simplekml expects aabbggrr
    altitude_mode: Optional[str] = Form(default="absolute"),
    extrude: Optional[int] = Form(default=0),
    include_waypoints: Optional[int] = Form(default=1),
    color_by: Optional[str] = Form(default="solid"),
    include_tour: Optional[int] = Form(default=0),
):
    if not file.filename or not file.filename.lower().endswith(".gpx"):
        raise HTTPException(status_code=400, detail="Please upload a .gpx file")

    # Convert web hex color to simplekml color (aabbggrr). Use full opacity.
    def web_hex_to_simplekml(hex_color: str) -> str:
        c = hex_color.strip().lstrip('#')
        if len(c) == 3:  # expand shorthand like f00
            c = ''.join([ch * 2 for ch in c])
        if len(c) != 6:
            return simplekml.Color.red
        rr, gg, bb = c[0:2], c[2:4], c[4:6]
        # simplekml uses aabbggrr; use ff for alpha (opaque)
        return f"ff{bb}{gg}{rr}"

    gpx_text = (await file.read()).decode("utf-8", errors="ignore")
    kml_bytes = convert_gpx_to_kml(
        gpx_text,
        line_name=name or "Flight Trail 3D",
        line_color=web_hex_to_simplekml(color or "#ff0000"),
        line_width=int(width) if width else 3,
        altitude_mode=altitude_mode or "absolute",
        extrude=int(extrude) if extrude is not None else 0,
        include_waypoints=int(include_waypoints) if include_waypoints is not None else 1,
        color_by=color_by or "solid",
        include_tour=int(include_tour) if include_tour is not None else 0,
    )

    filename_base = file.filename.rsplit('.', 1)[0]
    out_name = f"{filename_base}.kml"

    return StreamingResponse(
        BytesIO(kml_bytes),
        media_type="application/vnd.google-earth.kml+xml",
        headers={"Content-Disposition": f"attachment; filename={out_name}"},
    )


@app.post("/api/convert_link")
async def api_convert_link(
    file: UploadFile = File(..., description="GPX file"),
    name: Optional[str] = Form(default="Flight Trail 3D"),
    width: Optional[int] = Form(default=3),
    color: Optional[str] = Form(default="#ff0000"),
    altitude_mode: Optional[str] = Form(default="absolute"),
    extrude: Optional[int] = Form(default=0),
    include_waypoints: Optional[int] = Form(default=1),
    color_by: Optional[str] = Form(default="solid"),
    include_tour: Optional[int] = Form(default=0),
):
    if not file.filename or not file.filename.lower().endswith(".gpx"):
        raise HTTPException(status_code=400, detail="Please upload a .gpx file")

    def web_hex_to_simplekml(hex_color: str) -> str:
        c = hex_color.strip().lstrip('#')
        if len(c) == 3:
            c = ''.join([ch * 2 for ch in c])
        if len(c) != 6:
            return simplekml.Color.red
        rr, gg, bb = c[0:2], c[2:4], c[4:6]
        return f"ff{bb}{gg}{rr}"

    gpx_text = (await file.read()).decode("utf-8", errors="ignore")
    kml_bytes = convert_gpx_to_kml(
        gpx_text,
        line_name=name or "Flight Trail 3D",
        line_color=web_hex_to_simplekml(color or "#ff0000"),
        line_width=int(width) if width else 3,
        altitude_mode=altitude_mode or "absolute",
        extrude=int(extrude) if extrude is not None else 0,
        include_waypoints=int(include_waypoints) if include_waypoints is not None else 1,
        color_by=color_by or "solid",
        include_tour=int(include_tour) if include_tour is not None else 0,
    )

    kid = uuid4().hex
    KML_STORE[kid] = kml_bytes
    filename_base = file.filename.rsplit('.', 1)[0]
    out_name = f"{filename_base}.kml"
    return {"url": f"/kml/{kid}.kml", "filename": out_name}


@app.get("/kml/{kid}.kml")
def get_kml(kid: str):
    data = KML_STORE.get(kid)
    if not data:
        raise HTTPException(status_code=404, detail="Not found")
    return StreamingResponse(
        BytesIO(data),
        media_type="application/vnd.google-earth.kml+xml",
        headers={"Content-Disposition": f"inline; filename={kid}.kml"},
    )


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>FlightTracePro – Professional Flight Tracking</title>
    <script>window.CESIUM_ION_TOKEN='__CESIUM_ION_TOKEN__';</script>
    <script>
      // Guard for extensions expecting a geoLocationStorage global
      try { if (typeof window.geoLocationStorage === 'undefined') window.geoLocationStorage = {}; } catch (_) {}
      // Defensive shims for third-party scripts or extensions expecting Node-like globals
      try { if (typeof window.global === 'undefined') window.global = window; } catch (_) {}
      try { if (typeof window.process === 'undefined') window.process = { env: {} }; } catch (_) {}
    </script>
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ctext y='14' font-size='14'%3E✈️%3C/text%3E%3C/svg%3E">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
      /* Material Design 3 Design System */
      :root {
        /* Material Design 3 Color Tokens */
        --md-sys-color-primary: #0066cc;
        --md-sys-color-on-primary: #ffffff;
        --md-sys-color-primary-container: #d4e3ff;
        --md-sys-color-on-primary-container: #001c38;
        
        --md-sys-color-secondary: #565f71;
        --md-sys-color-on-secondary: #ffffff;
        --md-sys-color-secondary-container: #dae2f9;
        --md-sys-color-on-secondary-container: #131c2b;
        
        --md-sys-color-tertiary: #705575;
        --md-sys-color-on-tertiary: #ffffff;
        --md-sys-color-tertiary-container: #fad8fd;
        --md-sys-color-on-tertiary-container: #28132e;
        
        --md-sys-color-surface: #fefbff;
        --md-sys-color-on-surface: #1b1b1f;
        --md-sys-color-surface-variant: #e0e2ec;
        --md-sys-color-on-surface-variant: #44474f;
        --md-sys-color-outline: #74777f;
        --md-sys-color-outline-variant: #c4c6d0;
        
        --md-sys-color-background: #fefbff;
        --md-sys-color-on-background: #1b1b1f;
        --md-sys-color-surface-container: #f0f4f8;
        --md-sys-color-surface-container-high: #e8edf2;
        --md-sys-color-surface-container-highest: #e1e6eb;
        
        --md-sys-color-error: #ba1a1a;
        --md-sys-color-on-error: #ffffff;
        --md-sys-color-error-container: #ffdad6;
        --md-sys-color-on-error-container: #410002;
        
        /* Typography Scale */
        --md-sys-typescale-display-large: 700 57px/64px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-display-medium: 700 45px/52px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-display-small: 700 36px/44px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-headline-large: 700 32px/40px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-headline-medium: 700 28px/36px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-headline-small: 700 24px/32px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-title-large: 500 22px/28px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-title-medium: 500 16px/24px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-title-small: 500 14px/20px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-body-large: 400 16px/24px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-body-medium: 400 14px/20px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-body-small: 400 12px/16px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-label-large: 500 14px/20px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-label-medium: 500 12px/16px 'Inter', system-ui, sans-serif;
        --md-sys-typescale-label-small: 500 11px/16px 'Inter', system-ui, sans-serif;
        
        /* Elevation and Shadows */
        --md-sys-elevation-level0: 0px 0px 0px 0px rgba(0, 0, 0, 0.00), 0px 0px 0px 0px rgba(0, 0, 0, 0.00);
        --md-sys-elevation-level1: 0px 1px 2px 0px rgba(0, 0, 0, 0.30), 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        --md-sys-elevation-level2: 0px 1px 2px 0px rgba(0, 0, 0, 0.30), 0px 2px 6px 2px rgba(0, 0, 0, 0.15);
        --md-sys-elevation-level3: 0px 1px 3px 0px rgba(0, 0, 0, 0.30), 0px 4px 8px 3px rgba(0, 0, 0, 0.15);
        --md-sys-elevation-level4: 0px 2px 3px 0px rgba(0, 0, 0, 0.30), 0px 6px 10px 4px rgba(0, 0, 0, 0.15);
        --md-sys-elevation-level5: 0px 4px 4px 0px rgba(0, 0, 0, 0.30), 0px 8px 12px 6px rgba(0, 0, 0, 0.15);
        
        /* Shape System */
        --md-sys-shape-corner-extra-small: 4px;
        --md-sys-shape-corner-small: 8px;
        --md-sys-shape-corner-medium: 12px;
        --md-sys-shape-corner-large: 16px;
        --md-sys-shape-corner-extra-large: 28px;
        
        /* Glass Morphism Variables */
        --glass-bg: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.12);
        --glass-backdrop-filter: blur(16px) saturate(1.8);
        --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
      }
      
      /* Dark Mode */
      @media (prefers-color-scheme: dark) {
        :root {
          --md-sys-color-primary: #a6c8ff;
          --md-sys-color-on-primary: #003060;
          --md-sys-color-primary-container: #004c88;
          --md-sys-color-on-primary-container: #d4e3ff;
          
          --md-sys-color-secondary: #bec6dc;
          --md-sys-color-on-secondary: #293041;
          --md-sys-color-secondary-container: #3f4759;
          --md-sys-color-on-secondary-container: #dae2f9;
          
          --md-sys-color-tertiary: #ddbce0;
          --md-sys-color-on-tertiary: #3e2844;
          --md-sys-color-tertiary-container: #563e5c;
          --md-sys-color-on-tertiary-container: #fad8fd;
          
          --md-sys-color-surface: #10131a;
          --md-sys-color-on-surface: #e3e2e6;
          --md-sys-color-surface-variant: #44474f;
          --md-sys-color-on-surface-variant: #c4c6d0;
          --md-sys-color-outline: #8e9199;
          --md-sys-color-outline-variant: #44474f;
          
          --md-sys-color-background: #10131a;
          --md-sys-color-on-background: #e3e2e6;
          --md-sys-color-surface-container: #1e2128;
          --md-sys-color-surface-container-high: #282a31;
          --md-sys-color-surface-container-highest: #33353c;
          
          --md-sys-color-error: #ffb4ab;
          --md-sys-color-on-error: #690005;
          --md-sys-color-error-container: #93000a;
          --md-sys-color-on-error-container: #ffdad6;
          
          --glass-bg: rgba(0, 0, 0, 0.15);
          --glass-border: rgba(255, 255, 255, 0.08);
          --glass-backdrop-filter: blur(16px) saturate(1.8);
          --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
      }

      /* Reset and Base Styles */
      * {
        box-sizing: border-box;
      }
      
      body {
        font: var(--md-sys-typescale-body-medium);
        margin: 0;
        padding: 0;
        background-color: var(--md-sys-color-background);
        color: var(--md-sys-color-on-background);
        min-height: 100vh;
        overflow-x: hidden;
      }

      /* Glass Morphism Header */
      header {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-backdrop-filter);
        -webkit-backdrop-filter: var(--glass-backdrop-filter);
        border-bottom: 1px solid var(--glass-border);
        box-shadow: var(--glass-shadow);
        color: var(--md-sys-color-on-surface);
        padding: 16px 24px;
        display: flex;
        align-items: center;
        gap: 16px;
        position: sticky;
        top: 0;
        z-index: 1000;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      header h1 {
        font: var(--md-sys-typescale-headline-small);
        margin: 0;
        background: linear-gradient(135deg, var(--md-sys-color-primary), var(--md-sys-color-tertiary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }

      /* Navigation Pills */
      .topnav {
        display: flex;
        gap: 8px;
        margin-left: auto;
        background: var(--md-sys-color-surface-container);
        border-radius: var(--md-sys-shape-corner-extra-large);
        padding: 4px;
        box-shadow: var(--md-sys-elevation-level1);
      }
      
      .topnav button {
        background: transparent;
        color: var(--md-sys-color-on-surface-variant);
        border: none;
        padding: 10px 20px;
        border-radius: var(--md-sys-shape-corner-large);
        cursor: pointer;
        font: var(--md-sys-typescale-label-large);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
      }
      
      .topnav button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--md-sys-color-primary);
        opacity: 0;
        transition: opacity 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        z-index: -1;
      }
      
      .topnav button:hover::before {
        opacity: 0.08;
      }
      
      .topnav button:focus::before {
        opacity: 0.12;
      }
      
      .topnav button.active {
        background: var(--md-sys-color-primary);
        color: var(--md-sys-color-on-primary);
        box-shadow: var(--md-sys-elevation-level2);
      }
      
      .topnav button.active::before {
        opacity: 0;
      }

      /* Container with improved spacing */
      .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 24px;
        min-height: calc(100vh - 80px);
      }

      /* Glass Morphism Cards */
      .card {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-backdrop-filter);
        -webkit-backdrop-filter: var(--glass-backdrop-filter);
        border: 1px solid var(--glass-border);
        border-radius: var(--md-sys-shape-corner-large);
        padding: 32px;
        box-shadow: var(--glass-shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
      }
      
      .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--md-sys-color-primary), transparent);
        opacity: 0.6;
      }
      
      .card:hover {
        box-shadow: var(--md-sys-elevation-level4);
        transform: translateY(-2px);
        border-color: var(--md-sys-color-outline-variant);
      }

      /* Typography */
      h1, h2 {
        font: var(--md-sys-typescale-headline-medium);
        margin: 0 0 24px 0;
        color: var(--md-sys-color-on-surface);
      }
      
      h1 {
        font: var(--md-sys-typescale-headline-large);
      }
      
      p {
        font: var(--md-sys-typescale-body-large);
        color: var(--md-sys-color-on-surface-variant);
        margin: 0 0 24px 0;
        line-height: 1.6;
      }

      /* Form Elements */
      label {
        display: block;
        font: var(--md-sys-typescale-body-medium);
        color: var(--md-sys-color-on-surface);
        margin: 16px 0 8px;
        font-weight: 500;
      }
      
      .lbl {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .info {
        display: inline-flex;
        width: 18px;
        height: 18px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: var(--md-sys-color-primary-container);
        color: var(--md-sys-color-on-primary-container);
        font: var(--md-sys-typescale-label-small);
        cursor: help;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      .info:hover {
        background: var(--md-sys-color-primary);
        color: var(--md-sys-color-on-primary);
        transform: scale(1.1);
      }

      /* Input Fields */
      input[type="file"], 
      input[type="text"], 
      input[type="number"], 
      input[type="color"],
      select {
        width: 100%;
        padding: 14px 16px;
        border-radius: var(--md-sys-shape-corner-small);
        border: 1px solid var(--md-sys-color-outline-variant);
        background: var(--md-sys-color-surface-container);
        color: var(--md-sys-color-on-surface);
        font: var(--md-sys-typescale-body-large);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      input:focus,
      select:focus {
        outline: none;
        border-color: var(--md-sys-color-primary);
        box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.12);
        background: var(--md-sys-color-surface-container-high);
      }
      
      input:hover:not(:focus),
      select:hover:not(:focus) {
        border-color: var(--md-sys-color-outline);
      }

      /* Layout Grids */
      .row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 24px;
        margin: 24px 0;
      }
      
      .grid-3 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 24px 0;
      }

      /* Material Design Buttons */
      button {
        background: var(--md-sys-color-primary);
        color: var(--md-sys-color-on-primary);
        border: none;
        padding: 12px 24px;
        border-radius: var(--md-sys-shape-corner-large);
        cursor: pointer;
        font: var(--md-sys-typescale-label-large);
        min-height: 48px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
      }
      
      button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: currentColor;
        opacity: 0;
        transition: opacity 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      button:hover::before {
        opacity: 0.08;
      }
      
      button:focus::before {
        opacity: 0.12;
      }
      
      button:active::before {
        opacity: 0.16;
      }
      
      button:disabled {
        background: var(--md-sys-color-surface-variant);
        color: var(--md-sys-color-on-surface-variant);
        cursor: default;
        opacity: 0.38;
      }
      
      button:disabled::before {
        display: none;
      }

      /* Secondary Button Variant */
      button.secondary {
        background: var(--md-sys-color-secondary-container);
        color: var(--md-sys-color-on-secondary-container);
      }
      
      button.outlined {
        background: transparent;
        border: 1px solid var(--md-sys-color-outline);
        color: var(--md-sys-color-primary);
      }

      .actions {
        margin-top: 32px;
        display: flex;
        gap: 16px;
        align-items: center;
        flex-wrap: wrap;
      }

      /* Status and Notes */
      .note {
        font: var(--md-sys-typescale-body-small);
        color: var(--md-sys-color-on-surface-variant);
        line-height: 1.4;
      }
      
      .help {
        font: var(--md-sys-typescale-body-small);
        color: var(--md-sys-color-on-surface-variant);
        margin-top: 8px;
        line-height: 1.4;
      }
      
      .small {
        font: var(--md-sys-typescale-body-small);
      }
      
      .inline {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      /* Map/Globe Preview Container */
      .preview {
        margin-top: 32px;
        height: 70vh;
        min-height: 500px;
        border-radius: var(--md-sys-shape-corner-large);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        position: relative;
        background: var(--md-sys-color-surface-container);
        box-shadow: var(--md-sys-elevation-level2);
      }

      /* Tab Controls */
      .preview-tabs {
        display: flex;
        gap: 8px;
        padding: 16px;
        background: var(--glass-bg);
        backdrop-filter: var(--glass-backdrop-filter);
        -webkit-backdrop-filter: var(--glass-backdrop-filter);
        border-bottom: 1px solid var(--md-sys-color-outline-variant);
        align-items: center;
        flex-wrap: wrap;
        position: relative;
        z-index: 10;
      }
      
      .tabbtn {
        background: var(--md-sys-color-surface-container-high);
        color: var(--md-sys-color-on-surface-variant);
        border: none;
        padding: 8px 16px;
        border-radius: var(--md-sys-shape-corner-medium);
        cursor: pointer;
        font: var(--md-sys-typescale-label-medium);
        min-height: 36px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      .tabbtn:hover {
        background: var(--md-sys-color-secondary-container);
        color: var(--md-sys-color-on-secondary-container);
      }
      
      .tabbtn.active {
        background: var(--md-sys-color-primary);
        color: var(--md-sys-color-on-primary);
        box-shadow: var(--md-sys-elevation-level1);
      }
      
      .tabbtn:disabled {
        opacity: 0.38;
        cursor: default;
      }

      /* Map Display Areas */
      #map, #globe, #livemap, #liveglobe {
        flex: 1 1 auto;
        height: calc(100% - 80px);
        width: 100%;
        display: none;
        position: relative;
      }
      
      #map.active, #globe.active, #livemap.active, #liveglobe.active {
        display: block;
      }

      /* Enhanced Telemetry Display */
      .telemetry {
        position: absolute;
        left: 24px;
        bottom: 24px;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        align-items: flex-end;
        z-index: 500;
        max-width: calc(100% - 48px);
      }
      
      .telemetry .tile {
        background: var(--glass-bg);
        backdrop-filter: var(--glass-backdrop-filter);
        -webkit-backdrop-filter: var(--glass-backdrop-filter);
        color: var(--md-sys-color-on-surface);
        padding: 16px 20px;
        border-radius: var(--md-sys-shape-corner-medium);
        border: 1px solid var(--glass-border);
        box-shadow: var(--glass-shadow);
        min-width: 140px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      .telemetry .tile:hover {
        transform: translateY(-2px);
        box-shadow: var(--md-sys-elevation-level4);
      }
      
      .telemetry .tile .label {
        font: var(--md-sys-typescale-label-small);
        color: var(--md-sys-color-on-surface);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
        opacity: 1;
        font-weight: 500;
      }
      
      .telemetry .tile .value {
        font: var(--md-sys-typescale-title-large);
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
        font-weight: 700;
        color: var(--md-sys-color-on-surface);
        display: flex;
        align-items: baseline;
        gap: 4px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
      }
      
      .telemetry .tile .unit {
        font: var(--md-sys-typescale-label-medium);
        color: var(--md-sys-color-on-surface);
        opacity: 0.9;
        font-weight: 500;
      }
      
      .telemetry .wide {
        min-width: 200px;
      }

      /* HUD Display */
      .hud {
        position: absolute;
        right: 24px;
        bottom: 24px;
        background: var(--glass-bg);
        backdrop-filter: var(--glass-backdrop-filter);
        -webkit-backdrop-filter: var(--glass-backdrop-filter);
        color: var(--md-sys-color-on-surface);
        padding: 12px 16px;
        border-radius: var(--md-sys-shape-corner-medium);
        border: 1px solid var(--glass-border);
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.4;
        box-shadow: var(--glass-shadow);
        z-index: 500;
      }

      /* Enhanced Player Chips */
      .chips {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin: 16px 0;
      }
      
      .chip {
        background: var(--md-sys-color-secondary-container);
        color: var(--md-sys-color-on-secondary-container);
        border-radius: var(--md-sys-shape-corner-extra-large);
        padding: 8px 16px;
        cursor: pointer;
        font: var(--md-sys-typescale-label-medium);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        border: 1px solid transparent;
        user-select: none;
      }
      
      .chip::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: currentColor;
        opacity: 0;
        transition: opacity 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      .chip:hover {
        background: var(--md-sys-color-primary-container);
        color: var(--md-sys-color-on-primary-container);
        transform: translateY(-1px);
        box-shadow: var(--md-sys-elevation-level2);
      }
      
      .chip:hover::before {
        opacity: 0.08;
      }
      
      .chip:active {
        transform: translateY(0);
        box-shadow: var(--md-sys-elevation-level1);
      }
      
      .chip.active {
        background: var(--md-sys-color-primary);
        color: var(--md-sys-color-on-primary);
        box-shadow: var(--md-sys-elevation-level2);
      }

      /* Live Events Panel */
      #live_events {
        font: var(--md-sys-typescale-body-small);
        background: var(--md-sys-color-surface-container);
        border: 1px solid var(--md-sys-color-outline-variant);
        border-radius: var(--md-sys-shape-corner-small);
        padding: 16px;
        max-height: 180px;
        overflow-y: auto;
        margin: 16px 0;
        color: var(--md-sys-color-on-surface);
        line-height: 1.5;
      }
      
      #live_events::-webkit-scrollbar {
        width: 8px;
      }
      
      #live_events::-webkit-scrollbar-track {
        background: var(--md-sys-color-surface-variant);
        border-radius: 4px;
      }
      
      #live_events::-webkit-scrollbar-thumb {
        background: var(--md-sys-color-outline);
        border-radius: 4px;
      }
      
      #live_events::-webkit-scrollbar-thumb:hover {
        background: var(--md-sys-color-outline-variant);
      }

      /* Aircraft Icon Rotation */
      .ac-icon .ac-rot {
        display: inline-block;
        transform-origin: 50% 50%;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }

      /* Aircraft Tooltip Styling - Match Telemetry Cards */
      .aircraft-telemetry-tooltip {
        background: var(--glass-bg) !important;
        backdrop-filter: var(--glass-backdrop-filter);
        -webkit-backdrop-filter: var(--glass-backdrop-filter);
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--md-sys-shape-corner-medium) !important;
        box-shadow: var(--glass-shadow) !important;
        padding: 16px !important;
        min-width: 280px;
        color: var(--md-sys-color-on-surface) !important;
      }
      
      .aircraft-callsign {
        font: var(--md-sys-typescale-title-medium);
        font-weight: 700;
        color: var(--md-sys-color-primary);
        margin-bottom: 12px;
        text-align: center;
      }
      
      .telemetry-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 12px;
      }
      
      .telem-tile {
        background: rgba(var(--md-sys-color-surface-variant-rgb), 0.3);
        border-radius: var(--md-sys-shape-corner-small);
        padding: 8px;
        text-align: center;
        border: 1px solid var(--md-sys-color-outline-variant);
      }
      
      .telem-label {
        font: var(--md-sys-typescale-label-small);
        color: var(--md-sys-color-on-surface);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
        font-weight: 500;
        opacity: 1;
      }
      
      .telem-value {
        font: var(--md-sys-typescale-title-small);
        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
        font-weight: 700;
        color: var(--md-sys-color-on-surface);
        display: flex;
        align-items: baseline;
        justify-content: center;
        gap: 2px;
      }
      
      .telem-unit {
        font: var(--md-sys-typescale-label-small);
        color: var(--md-sys-color-on-surface);
        opacity: 0.9;
        font-weight: 500;
      }
      
      .tooltip-hint {
        font: var(--md-sys-typescale-body-small);
        color: var(--md-sys-color-on-surface-variant);
        text-align: center;
        padding-top: 8px;
        border-top: 1px solid var(--md-sys-color-outline-variant);
        opacity: 0.8;
      }
      
      .leaflet-tooltip.aircraft-tooltip {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
      }
      
      .leaflet-tooltip.aircraft-tooltip::before {
        display: none !important;
      }

      /* Interactive Elements */
      #live_hud {
        pointer-events: none;
      }
      
      #live_hud .tile {
        pointer-events: auto;
      }

      /* Footer */
      footer {
        margin-top: 48px;
        font: var(--md-sys-typescale-body-small);
        color: var(--md-sys-color-on-surface-variant);
        text-align: center;
        padding: 24px 0;
        border-top: 1px solid var(--md-sys-color-outline-variant);
      }

      /* Responsive Design */
      @media (max-width: 768px) {
        .container {
          padding: 16px;
        }
        
        .card {
          padding: 24px;
        }
        
        header {
          padding: 12px 16px;
          gap: 12px;
        }
        
        header h1 {
          font: var(--md-sys-typescale-title-large);
        }
        
        .topnav button {
          padding: 8px 16px;
        }
        
        .row {
          grid-template-columns: 1fr;
          gap: 16px;
        }
        
        .grid-3 {
          grid-template-columns: 1fr;
          gap: 16px;
        }
        
        .actions {
          flex-direction: column;
          align-items: stretch;
        }
        
        .actions button {
          width: 100%;
        }
        
        .telemetry {
          left: 16px;
          bottom: 16px;
          gap: 12px;
          max-width: calc(100% - 32px);
        }
        
        .telemetry .tile {
          min-width: 120px;
          padding: 12px 16px;
        }
        
        .hud {
          right: 16px;
          bottom: 16px;
        }
        
        .preview-tabs {
          padding: 12px;
          gap: 6px;
          flex-wrap: wrap;
        }
        
        .chips {
          gap: 8px;
        }
        
        .chip {
          padding: 6px 12px;
        }
      }
      
      @media (max-width: 480px) {
        .topnav {
          width: 100%;
          justify-content: center;
        }
        
        header {
          flex-direction: column;
          gap: 16px;
          align-items: stretch;
        }
        
        .telemetry {
          flex-direction: column;
          gap: 8px;
        }
        
        .telemetry .tile {
          min-width: auto;
          width: 100%;
        }
      }

      /* Focus and Accessibility */
      :focus-visible {
        outline: 2px solid var(--md-sys-color-primary);
        outline-offset: 2px;
      }
      
      button:focus-visible {
        outline-offset: 4px;
      }
      
      /* Animation Classes */
      .fade-in {
        animation: fadeIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(16px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      .slide-in {
        animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      @keyframes slideIn {
        from {
          opacity: 0;
          transform: translateX(-24px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }
      
      /* Loading States */
      .loading {
        position: relative;
        overflow: hidden;
      }
      
      .loading::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
          90deg,
          transparent,
          rgba(255, 255, 255, 0.1),
          transparent
        );
        animation: shimmer 1.5s infinite;
      }
      
      @keyframes shimmer {
        0% {
          left: -100%;
        }
        100% {
          left: 100%;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>FlightTracePro</h1>
      <div class="topnav">
        <button id="nav-live" class="active">Live Map</button>
        <button id="nav-conv">Converter</button>
      </div>
    </header>
    
    <div class="container">
      <!-- Converter View -->
      <div id="view-conv" style="display:none;" class="fade-in">
        <div class="card">
          <h1>GPX to KML Converter</h1>
          <p>Transform your GPS tracks into Google Earth-compatible KML files with advanced styling and 3D visualization options.</p>
          
          <form id="form" method="post" enctype="multipart/form-data">
            <label for="file">GPX file</label>
            <input id="file" name="file" type="file" accept=".gpx,application/gpx+xml" required />

            <div class="row">
              <div>
                <label for="plan_file" class="lbl">
                  Flight plan (optional) 
                  <span class="info" title="Little Navmap .lnmpln, MSFS/FSX .pln, or GPX route.">i</span>
                </label>
                <input id="plan_file" name="plan_file" type="file" accept=".lnmpln,.pln,.gpx,application/xml,application/gpx+xml" />
              </div>
              <div>
                <label for="airspace_file" class="lbl">
                  Airspaces (optional) 
                  <span class="info" title="OpenAir .txt, KML, or GeoJSON.">i</span>
                </label>
                <input id="airspace_file" name="airspace_file" type="file" accept=".txt,.kml,.json,.geojson,text/plain,application/json,application/vnd.google-earth.kml+xml" />
              </div>
            </div>
            
            <div class="row">
              <div>
                <label for="airports_file" class="lbl">
                  Airports (optional) 
                  <span class="info" title="CSV export from LNM or GPX waypoints.">i</span>
                </label>
                <input id="airports_file" name="airports_file" type="file" accept=".csv,.gpx,text/csv,application/gpx+xml" />
              </div>
              <div>
                <label class="lbl">
                  Layer toggles 
                  <span class="info" title="Control overlay visibility and filters.">i</span>
                </label>
                <div class="inline small" style="gap:16px; flex-wrap:wrap;">
                  <label><input type="checkbox" id="show_plan" checked /> Plan</label>
                  <label><input type="checkbox" id="show_airspaces" /> Airspaces</label>
                  <label><input type="checkbox" id="show_airports" /> Airports</label>
                  <span class="inline small">
                    <label class="note">ASP Alt</label> 
                    <input id="asp_alt" type="number" step="100" value="1500" style="width:90px;" />
                    <span class="note">m MSL</span>
                  </span>
                </div>
              </div>
            </div>

            <div class="row">
              <div>
                <label for="name" class="lbl">
                  Line name 
                  <span class="info" title="Name to show in the exported KML.">i</span>
                </label>
                <input id="name" name="name" type="text" value="Flight Trail 3D" />
              </div>
              <div>
                <label for="color" class="lbl">
                  Line color 
                  <span class="info" title="Pick a color for the path in preview and KML.">i</span>
                </label>
                <input id="color" name="color" type="color" value="#ff0000" />
              </div>
            </div>

            <div class="row">
              <div>
                <label for="width" class="lbl">
                  Line width 
                  <span class="info" title="Path stroke width in pixels.">i</span>
                </label>
                <input id="width" name="width" type="number" min="1" max="10" value="3" />
              </div>
              <div>
                <label for="altitude_mode" class="lbl">
                  Altitude mode 
                  <span class="info" title="absolute: use GPX elevation; relativeToGround: height above terrain; clampToGround: drape on terrain.">i</span>
                </label>
                <select id="altitude_mode" name="altitude_mode">
                  <option value="absolute" selected>absolute</option>
                  <option value="relativeToGround">relativeToGround</option>
                  <option value="clampToGround">clampToGround</option>
                </select>
                <div class="help" style="display:none"></div>
              </div>
            </div>

            <div class="grid-3">
              <div>
                <label for="extrude" class="lbl">
                  Extrude 
                  <span class="info" title="Draw vertical lines from the path to the ground to show height.">i</span>
                </label>
                <select id="extrude" name="extrude">
                  <option value="0" selected>no</option>
                  <option value="1">yes</option>
                </select>
              </div>
              <div>
                <label for="include_waypoints" class="lbl">
                  Waypoints 
                  <span class="info" title="Include GPX waypoints as labeled placemarks.">i</span>
                </label>
                <select id="include_waypoints" name="include_waypoints">
                  <option value="1" selected>include</option>
                  <option value="0">exclude</option>
                </select>
              </div>
              <div>
                <label for="color_by" class="lbl">
                  Coloring 
                  <span class="info" title="Color by a single color or speed (when timestamps are present).">i</span>
                </label>
                <select id="color_by" name="color_by">
                  <option value="solid" selected>solid</option>
                  <option value="speed">speed (blue→red)</option>
                </select>
              </div>
            </div>

            <div class="actions">
              <button id="convert" type="submit">Convert to KML</button>
              <button id="open_ge" type="button" class="outlined">Open in Google Earth</button>
              <input id="include_tour" name="include_tour" type="hidden" value="0" />
              <span class="note" id="status"></span>
            </div>
          </form>

          <div class="preview" id="preview">
            <div class="preview-tabs">
              <button id="tab-2d" class="tabbtn active" type="button">2D Map</button>
              <button id="tab-3d" class="tabbtn" type="button">3D Globe</button>
              <span class="inline small" style="margin-left:16px;">
                <label for="track_select" class="note">Track</label>
                <select id="track_select" class="small"><option value="auto">Auto</option></select>
              </span>
              <button id="play3d" class="tabbtn" type="button" disabled>Play</button>
              <button id="pause3d" class="tabbtn" type="button" disabled>Pause</button>
              <button id="stop3d" class="tabbtn" type="button" disabled>Stop</button>
              <button id="home3d" class="tabbtn" type="button" disabled>Home</button>
              <button id="follow3d" class="tabbtn" type="button" disabled>Follow</button>
              <span class="inline small">
                <label class="note">Presets</label>
                <button id="preset_chase" class="tabbtn" type="button" disabled>Chase</button>
                <button id="preset_front" class="tabbtn" type="button" disabled>Front</button>
                <button id="preset_left" class="tabbtn" type="button" disabled>Left</button>
                <button id="preset_right" class="tabbtn" type="button" disabled>Right</button>
                <button id="preset_top" class="tabbtn" type="button" disabled>Top</button>
              </span>
              <button id="rec3d" class="tabbtn" type="button" disabled>Record</button>
              <button id="stoprec3d" class="tabbtn" type="button" disabled>Stop Rec</button>
              <label class="note">Progress</label>
              <input id="fly_progress" type="range" min="0" max="1" value="0" style="flex:1; height: 4px;" />
              <label class="note">Speed</label>
              <input id="fly_speed" type="number" min="0.1" step="0.1" value="1.5" style="width:72px;" />
              <label class="note">Cam Dist</label>
              <input id="cam_dist" type="number" min="50" step="10" value="800" style="width:80px;" />
              <span class="note" style="margin-left:auto;">Preview updates after choosing a file</span>
            </div>
            <div id="map" class="active"></div>
            <div id="globe"></div>
            <div id="fly_hud" class="hud small" style="display:none">flythrough: idle</div>
            <div id="telemetry" class="telemetry" style="display:none">
              <div class="tile wide"><div class="label">Elapsed</div><div class="value"><span id="tl_elapsed">0.0</span><span class="unit">s</span></div></div>
              <div class="tile"><div class="label">Altitude</div><div class="value"><span id="tl_alt">0</span><span class="unit">m</span></div></div>
              <div class="tile"><div class="label">Ground Speed</div><div class="value"><span id="tl_gs">0</span><span class="unit">km/h</span></div></div>
              <div class="tile"><div class="label">Speed 3D</div><div class="value"><span id="tl_s3d">0</span><span class="unit">km/h</span></div></div>
              <div class="tile"><div class="label">Vertical Speed</div><div class="value"><span id="tl_vs">0.0</span><span class="unit">m/s</span></div></div>
              <div class="tile"><div class="label">Heading</div><div class="value"><span id="tl_hdg">0</span><span class="unit">°</span></div></div>
              <div class="tile"><div class="label">Pitch</div><div class="value"><span id="tl_pitch">0</span><span class="unit">°</span></div></div>
              <div class="tile"><div class="label">Roll</div><div class="value"><span id="tl_roll">0</span><span class="unit">°</span></div></div>
              <div class="tile"><div class="label">FPS</div><div class="value"><span id="tl_fps">0</span></div></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Live Map View -->
      <div id="view-live" style="display:block;" class="slide-in">
        <div class="card">
          <h1>Live Flight Tracking</h1>
          <p>Connect to a shared channel to see real-time aircraft positions from bridge clients. Experience professional flight tracking with modern visualization.</p>
          
          <div style="display:flex; gap:16px; align-items:flex-end; flex-wrap:wrap; margin: 24px 0;">
            <div>
              <label class="lbl" for="live_channel">Channel</label>
              <div style="display:flex; gap:8px;">
                <input id="live_channel" value="default" />
                <button id="channel_manage_btn" class="btn" type="button" style="padding:8px 12px;">⚙️</button>
              </div>
            </div>
            <div>
              <label class="lbl" for="live_callsign">Callsign</label>
              <input id="live_callsign" placeholder="N123AB" />
            </div>
            <div>
              <label class="lbl" for="live_key">Post Key (optional)</label>
              <input id="live_key" placeholder="Security key if required" />
            </div>
            <div>
              <label class="lbl">Connection</label><br/>
              <div style="display: flex; gap: 8px;">
                <button id="live_connect" type="button">Connect</button>
                <button id="live_disconnect" type="button" class="outlined">Disconnect</button>
                <button id="live_center" type="button" class="secondary">Center Map</button>
              </div>
            </div>
          </div>
          
          <div style="display: flex; justify-content: space-between; align-items: center; margin: 16px 0;">
            <div class="note" id="live_meta">Viewers: 0 • Feeders: 0 • Players: 0</div>
            <div class="note" id="live_status" style="font-weight: 500; color: var(--md-sys-color-primary);">disconnected</div>
          </div>
          
          <div class="chips" id="live_players_chips"></div>
          <div class="note small" id="live_players_list" style="margin:8px 0;">Players: -</div>
          
          <div id="live_events">
            <div class="note">Connection events will appear here...</div>
          </div>
          
          <div class="preview" id="live_preview">
            <div class="preview-tabs">
              <button id="live-tab-2d" class="tabbtn active" type="button">2D Map</button>
              <button id="live-tab-3d" class="tabbtn" type="button">3D Globe</button>
              <label class="note" style="margin-left: 16px;">Terrain</label>
              <select id="terrain_mode" class="small" style="width: auto;">
                <option value="flat" selected>Flat</option>
                <option value="real">Real Terrain</option>
              </select>
              <button id="live_follow" class="tabbtn" type="button">Follow</button>
              <button id="live_home3d" class="tabbtn" type="button">Home View</button>
              <button id="live_debug3d" class="tabbtn" type="button">Debug Info</button>
              <span class="note" style="margin-left:auto;">Channel: <span id="live_ch_label" style="font-weight: 500; color: var(--md-sys-color-primary);">default</span></span>
            </div>
            <div id="livemap" class="active"></div>
            <div id="liveglobe"></div>
            <div id="live_hud" class="telemetry" style="position:absolute; left:24px; top:100px; z-index:500; display:flex;">
              <div class="tile"><div class="label">ALT</div><div class="value"><span id="lv_alt">-</span><span class="unit">m</span></div></div>
              <div class="tile"><div class="label">SPD</div><div class="value"><span id="lv_spd">-</span><span class="unit">kt</span></div></div>
              <div class="tile"><div class="label">VSI</div><div class="value"><span id="lv_vsi">-</span><span class="unit">m/s</span></div></div>
              <div class="tile"><div class="label">HDG</div><div class="value"><span id="lv_hdg">-</span><span class="unit">°</span></div></div>
              <div class="tile"><div class="label">PITCH</div><div class="value"><span id="lv_pitch">-</span><span class="unit">°</span></div></div>
              <div class="tile"><div class="label">ROLL</div><div class="value"><span id="lv_roll">-</span><span class="unit">°</span></div></div>
            </div>
          </div>
        </div>
      </div>

      <footer>
      </footer>
    </div>

    <!-- Channel Management Dialog -->
    <div id="channel_dialog" class="modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 2000; backdrop-filter: blur(4px);">
      <div class="modal-content" style="max-width: 500px; margin: 10vh auto; padding: 32px; background: var(--md-sys-color-surface-container); border-radius: 28px; border: 1px solid var(--md-sys-color-outline-variant); box-shadow: 0 24px 32px rgba(0,0,0,0.3);">
        <h2 style="margin: 0 0 24px 0; color: var(--md-sys-color-on-surface);">Channel Management</h2>
        
        <div style="margin-bottom: 24px;">
          <h3 style="margin: 0 0 12px 0; font-size: 16px; font-weight: 500;">Recent Channels</h3>
          <div id="recent_channels" style="display: flex; flex-direction: column; gap: 8px; max-height: 150px; overflow-y: auto;">
            <!-- Recent channels will be populated here -->
          </div>
        </div>
        
        <div style="margin-bottom: 24px;">
          <h3 style="margin: 0 0 12px 0; font-size: 16px; font-weight: 500;">Create New Channel</h3>
          <div style="display: flex; flex-direction: column; gap: 12px;">
            <div>
              <label class="lbl" for="new_channel_name">Channel Name</label>
              <input id="new_channel_name" placeholder="Enter unique channel name" />
            </div>
            <div>
              <label class="lbl" for="new_channel_key">Post Key (optional)</label>
              <input id="new_channel_key" type="password" placeholder="Leave empty for public channel" />
            </div>
            <button id="create_channel_btn" class="btn primary" type="button">Create Channel</button>
          </div>
        </div>
        
        <div style="margin-bottom: 24px; padding: 16px; background: var(--md-sys-color-surface-variant); border-radius: 16px;">
          <h3 style="margin: 0 0 8px 0; font-size: 14px; font-weight: 500; color: var(--md-sys-color-on-surface-variant);">💡 Channel Tips</h3>
          <ul style="margin: 0; padding-left: 16px; font-size: 13px; color: var(--md-sys-color-on-surface-variant);">
            <li>Channels are created automatically when first used</li>
            <li>Post keys are required for flight data uploads only</li>
            <li>Viewers can join any channel without keys</li>
            <li>Use descriptive names like "event-name-2024" or "training-session-1"</li>
          </ul>
        </div>
        
        <div style="display: flex; gap: 12px; justify-content: flex-end;">
          <button id="channel_dialog_close" class="btn" type="button">Close</button>
        </div>
      </div>
    </div>

    <script>
      // URL Hash Navigation System
      function updateURL(view) {
        if (view === 'conv') {
          window.location.hash = '#converter';
        } else if (view === 'live') {
          window.location.hash = '#livemap';
        }
      }
      
      function handleHashChange() {
        const hash = window.location.hash;
        if (hash === '#livemap') {
          setTopTab('live');
        } else if (hash === '#converter' || hash === '') {
          setTopTab('conv');
        }
      }
      
      // Initialize hash navigation
      window.addEventListener('hashchange', handleHashChange);
      document.addEventListener('DOMContentLoaded', handleHashChange);

      // Persist settings in localStorage
      const PERSIST_KEYS = [
        'name','color','width','altitude_mode','extrude','include_waypoints','color_by','include_tour'
      ];
      function saveSettings() {
        PERSIST_KEYS.forEach(k => {
          const el = document.getElementById(k);
          if (el) localStorage.setItem('ftpro_'+k, el.value);
        });
      }
      function loadSettings() {
        PERSIST_KEYS.forEach(k => {
          const v = localStorage.getItem('ftpro_'+k);
          if (v !== null) {
            const el = document.getElementById(k);
            if (el) el.value = v;
          }
        });
      }

      // Enhanced top navigation with URL updates
      const navConv = document.getElementById('nav-conv');
      const navLive = document.getElementById('nav-live');
      const viewConv = document.getElementById('view-conv');
      const viewLive = document.getElementById('view-live');
      
      function setTopTab(tab) {
        const isConv = (tab === 'conv');
        navConv.classList.toggle('active', isConv);
        navLive.classList.toggle('active', !isConv);
        viewConv.style.display = isConv ? 'block' : 'none';
        viewLive.style.display = isConv ? 'none' : 'block';
        
        // Add smooth animations
        if (isConv) {
          viewConv.classList.add('fade-in');
          viewLive.classList.remove('fade-in');
        } else {
          viewLive.classList.add('slide-in');
          viewConv.classList.remove('fade-in');
        }
        
        // Update URL
        updateURL(tab);
        
        // Initialize maps when switching
        if (!isConv) {
          ensureLiveMap().then(() => setTimeout(() => liveMap && liveMap.invalidateSize(), 50));
        }
        if (isConv) {
          ensureMap().then(() => setTimeout(() => map && map.invalidateSize(), 50));
        }
      }
      
      navConv.addEventListener('click', () => setTopTab('conv'));
      navLive.addEventListener('click', () => setTopTab('live'));

      const form = document.getElementById('form');
      const statusEl = document.getElementById('status');
      const btn = document.getElementById('convert');
      form.addEventListener('change', saveSettings);
      document.addEventListener('DOMContentLoaded', loadSettings);
      // Drag & drop support on the card
      const card = document.querySelector('.card');
      const fileInput = document.getElementById('file');
      ;['dragenter','dragover'].forEach(evt => card.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation(); card.style.borderColor = 'var(--md-sys-color-primary)';
      }));
      ;['dragleave','drop'].forEach(evt => card.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation(); card.style.borderColor = 'var(--md-sys-color-outline-variant)';
      }));
      card.addEventListener('drop', e => {
        const dt = e.dataTransfer; if (!dt || !dt.files || !dt.files.length) return;
        const f = dt.files[0];
        if (!f.name.endsWith('.gpx')) { statusEl.textContent = 'Drop a .gpx file'; return; }
        fileInput.files = dt.files;
        previewFile(f);
      });

      // Form submission handler
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        statusEl.textContent = 'Converting...';
        btn.disabled = true;
        
        try {
          const formData = new FormData(form);
          const response = await fetch('/api/convert_link', {
            method: 'POST',
            body: formData
          });
          
          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
          }
          
          const result = await response.json();
          statusEl.textContent = 'Conversion complete!';
          
          // Update "Open in Google Earth" button
          const openBtn = document.getElementById('open_ge');
          if (openBtn && result.url) {
            openBtn.onclick = async () => {
              try {
                // Try to determine if this is accessible externally
                const isLocalhost = window.location.hostname === 'localhost' || 
                                  window.location.hostname === '127.0.0.1' || 
                                  window.location.hostname.startsWith('192.168.') ||
                                  window.location.hostname.startsWith('10.') ||
                                  window.location.hostname.includes('local');
                
                if (!isLocalhost) {
                  // Try Google Earth Web with public URL
                  const kmlUrl = new URL(result.url, window.location.origin).href;
                  const earthUrl = `https://earth.google.com/web/@0,0,0a,0d,0y,0t,0r?url=${encodeURIComponent(kmlUrl)}`;
                  window.open(earthUrl, '_blank');
                  statusEl.textContent = 'Opened in Google Earth Web';
                } else {
                  // Local development - download the file
                  const kmlResponse = await fetch(result.url);
                  const kmlBlob = await kmlResponse.blob();
                  
                  // Create download link
                  const downloadUrl = URL.createObjectURL(kmlBlob);
                  const downloadLink = document.createElement('a');
                  downloadLink.href = downloadUrl;
                  downloadLink.download = result.filename || 'flight.kml';
                  
                  // Show user-friendly message
                  statusEl.textContent = 'KML file downloaded - open with Google Earth Pro or upload to earth.google.com/web';
                  
                  // Auto-download
                  document.body.appendChild(downloadLink);
                  downloadLink.click();
                  document.body.removeChild(downloadLink);
                  URL.revokeObjectURL(downloadUrl);
                }
              } catch (error) {
                statusEl.textContent = `Error: ${error.message}`;
              }
            };
            openBtn.disabled = false;
          }
          
          // Auto-preview the file if one was uploaded
          const fileInput = document.getElementById('file');
          if (fileInput.files && fileInput.files[0]) {
            previewFile(fileInput.files[0]);
          }
          
        } catch (error) {
          statusEl.textContent = `Error: ${error.message}`;
        } finally {
          btn.disabled = false;
        }
      });

      // Convert button click handler (prefer trusted submission)
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        if (typeof form.requestSubmit === 'function') {
          form.requestSubmit();
        } else {
          // Fallback for very old browsers
          form.dispatchEvent(new Event('submit', { cancelable: true }));
        }
      });

      // File input change handler for preview
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          previewFile(e.target.files[0]);
        }
      });

      // Preview file function (multi-track with labeling and auto-selection)
      let __previewBusy = false;
      async function previewFile(file, preserveSelection=false) {
        if (!file || !file.name || !file.name.toLowerCase().endsWith('.gpx')) {
          statusEl.textContent = 'Please select a .gpx file';
          return;
        }
        if (__previewBusy) {
          // Avoid concurrent renders that can race with resize/layout
          return;
        }
        __previewBusy = true;
        statusEl.textContent = 'Loading preview...';
        try {
          const text = await file.text();
          const { tracks, waypoints } = parseGpx(text);
          // Begin loading the globe in background immediately
          ensureGlobe(() => { try { converterGlobe && converterGlobe.resize(); } catch(_) {} });
          lastFileBlob = file;
          // Populate track selector
          const prevSel = preserveSelection ? trackSelect.value : 'auto';
          trackSelect.innerHTML = '<option value="auto">Auto</option>';
          const trackStats = tracks.map((t, idx) => {
            const seg0 = (t.segments && t.segments[0]) ? t.segments[0] : [];
            const p0 = seg0[0] || null;
            const country = p0 ? (countryAt(p0.lat, p0.lon) || null) : null;
            const tstr = p0 ? (fmtTime(p0.time) || null) : null;
            const label = `${country ? country : 'Flight'}${tstr ? ' – '+tstr : ''}`;
            return { idx, points: t.segments.reduce((n,s)=>n+s.length,0), label };
          });
          trackStats.forEach(ts => {
            const opt = document.createElement('option');
            opt.value = String(ts.idx);
            opt.textContent = `${ts.idx+1}: ${ts.label} (${ts.points} pts)`;
            trackSelect.appendChild(opt);
          });
          if (preserveSelection && trackStats.some(ts => String(ts.idx) === prevSel)) {
            trackSelect.value = prevSel;
          }
          const pickTrack = () => {
            const sel = trackSelect.value;
            if (sel === 'auto') {
              return tracks.reduce((a,b) => (a.segments.reduce((n,s)=>n+s.length,0) >= b.segments.reduce((n,s)=>n+s.length,0)) ? a : b, tracks[0] || {segments:[], legs:[]});
            }
            const t = tracks[Number(sel)];
            return t || {segments:[], legs:[]};
          };
          const chosen = pickTrack();
          // Ensure preview visible but keep current tab (2D or 3D) as-is
          document.getElementById('preview').style.display = 'block';
          await ensureMap();
          // Wait until map container has non-zero size to avoid Leaflet path clipping errors
          const mapEl = document.getElementById('map');
          let tries = 0;
          while (tries < 20) {
            try {
              const rect = mapEl.getBoundingClientRect();
              const size = map.getSize();
              if (rect.width > 0 && rect.height > 0 && size.x > 0 && size.y > 0) break;
            } catch(_) {}
            await new Promise(r => setTimeout(r, 16));
            tries++;
          }
          try { map.invalidateSize(true); } catch(_) {}
          // Draw 2D
          const includeWpts = document.getElementById('include_waypoints').value === '1';
          const color = document.getElementById('color').value || '#ff0000';
          const width = parseInt(document.getElementById('width').value || '3', 10);
          const colorBy = document.getElementById('color_by').value || 'solid';
          const group = [];
          const isFiniteNum = (n) => typeof n === 'number' && Number.isFinite(n);
          const isValidLL = (lat, lon) => isFiniteNum(lat) && isFiniteNum(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
          const safeAddPolyline = (latlngs, opts) => {
            try {
              const clean = (latlngs || []).filter(ll => Array.isArray(ll) && isValidLL(ll[0], ll[1]));
              if (clean.length < 2) return null;
              const baseOpts = { interactive: false };
              if (canvasRenderer) baseOpts.renderer = canvasRenderer;
              const pl = L.polyline(clean, Object.assign(baseOpts, opts || {}));
              trackLayer.addLayer(pl);
              return pl;
            } catch (e) {
              console.warn('skip bad polyline', e);
              return null;
            }
          };
          const safeAddMarker = (lat, lon) => {
            try {
              if (!isValidLL(lat, lon)) return null;
              const m = L.marker([lat, lon]);
              wptLayer.addLayer(m);
              return m;
            } catch (e) { console.warn('skip bad marker', e); return null; }
          };
          if (trackLayer) trackLayer.clearLayers();
          if (wptLayer) wptLayer.clearLayers();
          // Precompute bounds from raw coordinates first, then add layers (avoids race during animations)
          const allLatLngs = [];
          if (chosen.segments.length) {
            chosen.segments.forEach(seg => {
              if (seg && seg.length > 0) {
                seg.forEach(p => { if (isValidLL(p.lat, p.lon)) allLatLngs.push([p.lat, p.lon]); });
              }
            });
          }
          try {
            if (allLatLngs.length > 1) {
              const bounds = L.latLngBounds(allLatLngs);
              if (bounds && bounds.isValid && bounds.isValid()) {
                map.fitBounds(bounds, { padding: [20,20], animate: false });
              } else {
                map.setView(allLatLngs[0], 12, { animate: false });
              }
            } else if (allLatLngs.length === 1) {
              map.setView(allLatLngs[0], 12, { animate: false });
            }
          } catch(_) {}

          if (chosen.segments.length) {
            if (colorBy === 'speed' && chosen.legs && chosen.legs.length) {
              const speeds = chosen.legs.map(l => l.speed).filter(s => s !== null);
              const smin = speeds.length ? Math.min(...speeds) : 0;
              const smax = speeds.length ? Math.max(...speeds) : 1;
              const lerp = (a,b,t) => a + (b-a)*t;
              const speedColor = (s) => {
                if (s === null) return color;
                const t = (s - smin) / Math.max(1e-6, (smax - smin));
                const r = Math.round(lerp(0,255,t));
                const g = Math.round(lerp(114,0,t));
                const b = Math.round(lerp(255,0,t));
                return `rgb(${r},${g},${b})`;
              };
              chosen.legs.forEach(({a,b,speed}) => {
                // Validate leg endpoints before using them
                if (a && b && isValidLL(a.lat, a.lon) && isValidLL(b.lat, b.lon)) {
                  const pl = safeAddPolyline([[a.lat,a.lon],[b.lat,b.lon]], { color: speedColor(speed), weight: width });
                  if (pl) group.push(pl);
                }
              });
            } else {
              chosen.segments.forEach(seg => {
                if (seg && seg.length > 1) {
                  const latlngs = seg.filter(p => p && isValidLL(p.lat, p.lon)).map(p => [p.lat, p.lon]);
                  if (latlngs.length > 1) {
                    const pl = safeAddPolyline(latlngs, { color, weight: width });
                    if (pl) group.push(pl);
                  }
                }
              });
            }
          }
          if (includeWpts && waypoints.length) {
            waypoints.forEach(w => {
              const m = safeAddMarker(w.lat, w.lon);
              if (m) { try { m.bindPopup(w.name); } catch(_) {} group.push(m); }
            });
          }
          // No extra fitBounds here; we already fitted using raw points above with animate:false
          // Draw 3D using reusable Globe3D component
          ensureGlobe(async () => {
            try {
              let longestSeg = [];
              
              if (colorBy === 'speed' && chosen.legs && chosen.legs.length) {
                // For speed coloring, we'll use the default color for now
                // TODO: Add multi-color track support to Globe3D
                await converterGlobe.addTrackSegments(chosen.segments || [], color, width);
                longestSeg = (chosen.segments && chosen.segments[0]) ? chosen.segments[0] : [];
              } else {
                await converterGlobe.addTrackSegments(chosen.segments || [], color, width);
                (chosen.segments||[]).forEach(seg => {
                  if (seg.length > longestSeg.length) longestSeg = seg;
                });
              }
              
              // Prepare simple flythrough data
              globePositions = longestSeg.map(p => ({lat:p.lat, lon:p.lon, ele:p.ele||0}));
              setupFlyUI();
              
              // Keep backward compatibility
              globeTracks = converterGlobe.tracks;
            } catch (e) { 
              console.error('3D preview error:', e); 
            }
          });
          statusEl.textContent = `Loaded ${tracks.reduce((n,t)=>n+t.segments.reduce((m,s)=>m+s.length,0),0)} points in ${tracks.length} tracks`;
        } catch (error) {
          statusEl.textContent = `Error reading GPX: ${error.message}`;
          console.error('GPX preview error:', error);
        } finally { __previewBusy = false; }
      }

      // Enhanced map interaction detection for follow mode
      let userInteractionFlag = false;
      let interactionTimeout = null;

      function resetUserInteraction() {
        userInteractionFlag = false;
      }

      function markUserInteraction() {
        userInteractionFlag = true;
        clearTimeout(interactionTimeout);
        interactionTimeout = setTimeout(resetUserInteraction, 3000); // Reset after 3 seconds
      }

      // Quick preview using Leaflet (2D)
      let map, trackLayer, wptLayer;
      // Overlay layers (2D)
      let planLayer, airspaceLayer, airportLayer;
      let globeViewer, globeTracks = [], globeHandler, globeFlyTimer, globePositions = [], gliderEnt = null;
      // Overlay layers (3D)
      let plan3D = [], airspace3D = [], airport3D = [];
      let flyReq = null, flyPlaying = false, flyIndex = 0, flySpeed = 1.0, lastTs = 0, camDistance = 800, camPitchDeg = -35;
      const playBtn = document.getElementById('play3d');
      const pauseBtn = document.getElementById('pause3d');
      const stopBtn = document.getElementById('stop3d');
      const flyRange = document.getElementById('fly_progress');
      const flySpeedInput = document.getElementById('fly_speed');
      const camDistInput = document.getElementById('cam_dist');
      const homeBtn = document.getElementById('home3d');
      const followBtn = document.getElementById('follow3d');
      let followCam = false; // start 3D in non-follow mode
      let followOffsetHeading = 0.0, followOffsetPitch = 0.0, followRangeScale = 1.0;
      function updateFollowBtn(){ if (followBtn) { followBtn.textContent = followCam ? 'Following' : 'Follow'; } }
      document.addEventListener('DOMContentLoaded', updateFollowBtn);
      function releaseCamera(){ try { if (globeViewer) globeViewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY); } catch(_) {} }
      function setFollowCam(v){ followCam = !!v; updateFollowBtn(); if (!followCam) { releaseCamera(); } try { window.__cfgControls && window.__cfgControls(); } catch(_) {} }
      const trackSelect = document.getElementById('track_select');
      if (trackSelect) {
        trackSelect.addEventListener('change', () => {
          if (lastFileBlob) previewFile(lastFileBlob, true);
        });
      }
      const showPlan = document.getElementById('show_plan');
      const showAirspaces = document.getElementById('show_airspaces');
      const showAirports = document.getElementById('show_airports');
      const aspAltInput = document.getElementById('asp_alt');
      const planFileInput = document.getElementById('plan_file');
      const airspaceFileInput = document.getElementById('airspace_file');
      const airportsFileInput = document.getElementById('airports_file');
      let lastFileBlob = null;
      const tab2d = document.getElementById('tab-2d');
      const tab3d = document.getElementById('tab-3d');
      const mapDiv = document.getElementById('map');
      const globeDiv = document.getElementById('globe');
      let leafletReady = null;
      let canvasRenderer = null;
      let cesiumReady = null;
      // Optional country lookup from static JSON (coarse bboxes)
      let countryIndex = null;
      fetch('/static/country-bboxes.json').then(r=>r.ok?r.json():null).then(j=>{ countryIndex=j; }).catch(()=>{});
      function countryAt(lat, lon) {
        if (!countryIndex) return null;
        for (const c of countryIndex) {
          if (lon >= c.minLon && lon <= c.maxLon && lat >= c.minLat && lat <= c.maxLat) return c.name;
        }
        return null;
      }
      function fmtTime(d) {
        if (!(d instanceof Date) || isNaN(d.getTime())) return null;
        try { return d.toLocaleString(); } catch(_) { return d.toISOString(); }
      }

      // Robust GPX parser with multi-track recognition and splitting
      function parseGpx(text) {
        const dom = new DOMParser().parseFromString(text, 'application/xml');
        const toNum = v => Number.parseFloat(v);
        const isValid = (lat, lon) => Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
        const parseTime = (el) => {
          const t = el && el.querySelector ? el.querySelector('time') : null;
          if (!t) return null;
          const d = new Date(t.textContent.trim());
          return isNaN(d.getTime()) ? null : d;
        };
        function splitByMeridian(points) {
          if (!points || points.length < 2) return points && points.length ? [points] : [];
          const segs = [];
          let cur = [points[0]];
          for (let i = 1; i < points.length; i++) {
            const prev = cur[cur.length - 1];
            const p = points[i];
            if (Math.abs(p.lon - prev.lon) > 180) { // lon jump
              if (cur.length > 1) segs.push(cur);
              cur = [p];
            } else {
              cur.push(p);
            }
          }
          if (cur.length > 1) segs.push(cur);
          return segs;
        }
        function smartSplit(points) {
          if (!points || points.length < 2) return [];
          const segs = [];
          const GAP_S = 300;       // 5 minutes gap splits
          const STOP_SPEED = 0.5;  // m/s consider stationary below this
          const STOP_MIN_S = 60;   // 1 minute stationary splits
          const MAX_SPEED = 150;   // m/s (540 km/h) split if exceeded
          const BIG_JUMP_M = 250000; // 250 km jump also splits
          const R = 6371000.0;
          const toRad = (x) => x * Math.PI / 180;
          const haversine = (p1, p2) => {
            const lat1 = toRad(p1.lat), lon1 = toRad(p1.lon);
            const lat2 = toRad(p2.lat), lon2 = toRad(p2.lon);
            const dlat = lat2 - lat1, dlon = lon2 - lon1;
            const a = Math.sin(dlat/2)**2 + Math.cos(lat1)*Math.cos(lat2)*Math.sin(dlon/2)**2;
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
          };
          let cur = [points[0]];
          let stopAccum = 0;
          for (let i=1;i<points.length;i++){
            const a = points[i-1], b = points[i];
            const dt = (a.time && b.time) ? Math.max(0, (b.time - a.time) / 1000) : null;
            const dist = haversine(a, b);
            const spd = (dt && dt > 0) ? (dist / dt) : null;
            const gap = (dt !== null && dt > GAP_S);
            const bigJump = dist > BIG_JUMP_M;
            const tooFast = (spd !== null && spd > MAX_SPEED);
            if (spd !== null && spd < STOP_SPEED) { stopAccum += dt || 0; } else { stopAccum = 0; }
            const stationaryTooLong = stopAccum >= STOP_MIN_S;
            if (gap || bigJump || tooFast || stationaryTooLong) {
              if (cur.length > 1) segs.push(cur);
              cur = [b];
              stopAccum = 0;
            } else {
              cur.push(b);
            }
          }
          if (cur.length > 1) segs.push(cur);
          return segs;
        }
        const tracksOut = [];
        const tracks = Array.from(dom.querySelectorAll('trk'));
        if (tracks.length) {
          tracks.forEach(trk => {
            const _trkNameEl = trk.querySelector('name');
            const trkName = (_trkNameEl && _trkNameEl.textContent ? _trkNameEl.textContent.trim() : 'Track');
            const tsegs = Array.from(trk.querySelectorAll('trkseg'));
            let segs = [];
            if (tsegs.length) {
              tsegs.forEach(ts => {
                const pts = Array.from(ts.querySelectorAll('trkpt')).map(tp => ({
                  lat: toNum(tp.getAttribute('lat')),
                  lon: toNum(tp.getAttribute('lon')),
                  ele: (function(){ const _e = tp.querySelector('ele'); return toNum((_e && _e.textContent) || '0') || 0; })(),
                  time: (function(){ const _t = tp.querySelector('time'); if(!_t) return null; const d=new Date(_t.textContent.trim()); return isNaN(d.getTime())?null:d; })()
                })).filter(p => isValid(p.lat, p.lon));
                splitByMeridian(pts).forEach(s => smartSplit(s).forEach(ss => segs.push(ss)));
              });
            } else {
              const pts = Array.from(trk.querySelectorAll('trkpt')).map(tp => ({
                lat: toNum(tp.getAttribute('lat')),
                lon: toNum(tp.getAttribute('lon')),
                ele: (function(){ const _e = tp.querySelector('ele'); return toNum((_e && _e.textContent) || '0') || 0; })(),
                time: (function(){ const _t = tp.querySelector('time'); if(!_t) return null; const d=new Date(_t.textContent.trim()); return isNaN(d.getTime())?null:d; })()
              })).filter(p => isValid(p.lat, p.lon));
              splitByMeridian(pts).forEach(s => smartSplit(s).forEach(ss => segs.push(ss)));
            }
            const R = 6371000.0; const toRad = (x) => x * Math.PI / 180;
            const haversine = (p1, p2) => {
              const lat1 = toRad(p1.lat), lon1 = toRad(p1.lon);
              const lat2 = toRad(p2.lat), lon2 = toRad(p2.lon);
              const dlat = lat2 - lat1, dlon = lon2 - lon1;
              const a = Math.sin(dlat/2)**2 + Math.cos(lat1)*Math.cos(lat2)*Math.sin(dlon/2)**2;
              const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
              return R * c;
            };
            let flightIdx = 1;
            for (const seg of segs) {
              if (!seg || seg.length < 2) continue;
              const segDuration = (() => { const t0 = seg[0].time, t1 = seg[seg.length-1].time; return (t0 && t1) ? Math.max(0, (t1 - t0) / 1000) : null; })();
              const segDist = (() => { let d=0; for (let i=0;i<seg.length-1;i++) d += haversine(seg[i], seg[i+1]); return d; })();
              const minPts = 8, minDist = 200; // meters
              const avgSpd = (segDuration && segDuration>0) ? (segDist/segDuration) : null;
              const isStationary = (avgSpd !== null && avgSpd < 0.5 && (segDuration||0) >= 60);
              if (seg.length < minPts || segDist < minDist || isStationary) continue;
              const legs = [];
              for (let i=0;i<seg.length-1;i++) {
                const a = seg[i], b = seg[i+1];
                let speed = null;
                if (a.time && b.time) {
                  const dt = (b.time - a.time) / 1000;
                  if (dt > 0) speed = haversine(a, b) / dt;
                }
                legs.push({ a, b, speed });
              }
              const name = `${trkName} – Flight ${flightIdx++}`;
              tracksOut.push({ name, segments: [seg], legs });
            }
          });
        } else {
          const rpts = Array.from(dom.querySelectorAll('rtept')).map(tp => ({
            lat: toNum(tp.getAttribute('lat')),
            lon: toNum(tp.getAttribute('lon')),
            ele: (function(){ const _e = tp.querySelector('ele'); return toNum((_e && _e.textContent) || '0') || 0; })(),
            time: (function(){ const _t = tp.querySelector('time'); if(!_t) return null; const d=new Date(_t.textContent.trim()); return isNaN(d.getTime())?null:d; })()
          })).filter(p => isValid(p.lat, p.lon));
          const segs = [];
          splitByMeridian(rpts).forEach(s => smartSplit(s).forEach(ss => segs.push(ss)));
          let flightIdx = 1;
          for (const seg of segs) {
            if (!seg || seg.length < 2) continue;
            const legs = [];
            for (let i=0;i<seg.length-1;i++) legs.push({ a: seg[i], b: seg[i+1], speed: null });
            const name = `Route – Flight ${flightIdx++}`;
            tracksOut.push({ name, segments: [seg], legs });
          }
        }
        const waypoints = Array.from(dom.querySelectorAll('wpt')).map(w => ({
          lat: toNum(w.getAttribute('lat')),
          lon: toNum(w.getAttribute('lon')),
          name: (function(){ const _n = w.querySelector('name'); return (_n && _n.textContent) ? _n.textContent : 'wpt'; })()
        })).filter(w => isValid(w.lat, w.lon));
        return { tracks: tracksOut, waypoints };
      }

      // --- Simple 3D flythrough implementation for preview ---
      function setupFlyUI() {
        const canFly = globePositions && globePositions.length >= 2 && globeViewer;
        // Enable/disable controls
        [playBtn, pauseBtn, stopBtn, homeBtn, followBtn, flyRange, flySpeedInput, camDistInput].forEach(el => { if (el) el.disabled = !canFly; });
        if (!canFly) return;
        flyRange.min = 0; flyRange.max = Math.max(1, globePositions.length - 1); flyRange.step = 1;
        flyRange.value = String(flyIndex);
        if (!flySpeedInput.value) flySpeedInput.value = '1.5';

        function posAt(i) { 
          i = Math.max(0, Math.min(globePositions.length-1, i)); 
          return globePositions[i]; 
        }

        async function ensureGlider() {
          if (!converterGlobe.entities.has('preview')) {
            const p0 = posAt(0);
            // Convert ele to alt for Globe3D API compatibility
            const position = { lat: p0.lat, lon: p0.lon, alt: p0.ele || 0 };
            await converterGlobe.addAircraft('preview', position, {}, 'Preview');
            gliderEnt = converterGlobe.entities.get('preview');
          }
          return converterGlobe.entities.get('preview');
        }

        async function update3DView() {
          const p = posAt(flyIndex);
          // Convert ele to alt for Globe3D API compatibility
          const position = { lat: p.lat, lon: p.lon, alt: p.ele || 0 };
          await converterGlobe.updateAircraft('preview', position);
          if (followCam) {
            await converterGlobe.trackEntity('preview');
          }
        }

        function step(ts) {
          if (!flyPlaying) return;
          const spd = Number(flySpeedInput.value) || 1.0;
          flyIndex += Math.max(1, Math.floor(spd));
          if (flyIndex >= globePositions.length) { flyIndex = globePositions.length-1; flyPlaying = false; return; }
          flyRange.value = String(flyIndex);
          update3DView();
          flyReq = requestAnimationFrame(step);
        }

        // Wire buttons
        playBtn && playBtn.addEventListener('click', async () => {
          await ensureGlider();
          if (!followCam) { await converterGlobe.trackEntity('preview'); }
          if (!flyPlaying) { flyPlaying = true; cancelAnimationFrame(flyReq); flyReq = requestAnimationFrame(step); }
        });
        pauseBtn && pauseBtn.addEventListener('click', () => { flyPlaying = false; cancelAnimationFrame(flyReq); });
        stopBtn && stopBtn.addEventListener('click', async () => { 
          flyPlaying = false; 
          cancelAnimationFrame(flyReq); 
          flyIndex = 0; 
          flyRange.value = '0'; 
          await update3DView(); 
        });
        homeBtn && homeBtn.addEventListener('click', async () => {
          try { globeViewer.trackedEntity = undefined; } catch(_) {}
          await converterGlobe.flyToTracks();
        });
        followBtn && followBtn.addEventListener('click', () => { setFollowCam(!followCam); update3DView(); });
        flyRange && flyRange.addEventListener('input', async () => { 
          flyIndex = Number(flyRange.value) || 0; 
          flyPlaying = false; 
          cancelAnimationFrame(flyReq); 
          await update3DView(); 
        });
        camDistInput && camDistInput.addEventListener('change', () => { /* trackedEntity handles distance via mouse; keep for future */ });

        // Initial view
        update3DView();
      }
      
      function ensureMap() {
        if (map) return Promise.resolve();
        if (leafletReady) return leafletReady;
        // Load Leaflet lazily and wait until the map is fully ready and sized
        leafletReady = new Promise((resolve, reject) => {
          const css = document.createElement('link');
          css.rel = 'stylesheet';
          css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
          document.head.appendChild(css);
          const script = document.createElement('script');
          script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
          script.crossOrigin = 'anonymous';
          script.onload = () => {
            try {
              map = L.map('map', { preferCanvas: true, zoomAnimation: false, fadeAnimation: false, markerZoomAnimation: false, inertia: false }).setView([46.8, 8.2], 8);
              L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OpenStreetMap' }).addTo(map);
              trackLayer = L.layerGroup().addTo(map);
              wptLayer = L.layerGroup().addTo(map);
              planLayer = L.layerGroup().addTo(map);
              airspaceLayer = L.layerGroup().addTo(map);
              airportLayer = L.layerGroup().addTo(map);
              try { canvasRenderer = L.canvas({ padding: 0.5 }); } catch(_) {}

              // Interaction detection for follow mode
              map.on('movestart', markUserInteraction);
              map.on('zoomstart', markUserInteraction);
              map.on('drag', markUserInteraction);

              // Wait until Leaflet reports ready and container is sized
              map.whenReady(async () => {
                const mapEl = document.getElementById('map');
                let tries = 0;
                while (tries < 30) {
                  try {
                    const rect = mapEl.getBoundingClientRect();
                    const size = map.getSize();
                    if (rect.width > 0 && rect.height > 0 && size.x > 0 && size.y > 0) break;
                  } catch(_) {}
                  await new Promise(r => setTimeout(r, 16));
                  tries++;
                }
                try { map.invalidateSize(true); } catch(_) {}
                resolve();
              });
            } catch (e) {
              reject(e);
            }
          };
          script.onerror = reject;
          document.head.appendChild(script);
        });
        return leafletReady;
      }

      // Reusable 3D Globe Component
      class Globe3D {
        constructor(containerId, options = {}) {
          this.containerId = containerId;
          this.viewer = null;
          this.entities = new Map();
          this.tracks = [];
          this.ready = null;
          this.options = {
            enableFlythrough: false,
            enableLiveTracking: false,
            performanceMode: false,
            ...options
          };
        }

        async initialize(callback) {
          if (this.viewer) { 
            callback && callback(this.viewer); 
            return Promise.resolve(this.viewer); 
          }
          if (this.ready) { 
            const viewer = await this.ready; 
            callback && callback(viewer); 
            return viewer; 
          }
          
          this.ready = new Promise((resolve, reject) => {
            // Load CSS
            if (!document.querySelector('link[href*="cesium"]')) {
              const css = document.createElement('link');
              css.rel = 'stylesheet';
              css.href = 'https://unpkg.com/cesium/Build/Cesium/Widgets/widgets.css';
              document.head.appendChild(css);
            }
            
            // Load Script
            if (window.Cesium) {
              this._createViewer().then(resolve).catch(reject);
            } else {
              const script = document.createElement('script');
              window.CESIUM_BASE_URL = 'https://unpkg.com/cesium/Build/Cesium/';
              script.src = window.CESIUM_BASE_URL + 'Cesium.js';
              script.crossOrigin = 'anonymous';
              script.onload = () => this._createViewer().then(resolve).catch(reject);
              script.onerror = reject;
              document.body.appendChild(script);
            }
          });
          
          const viewer = await this.ready;
          callback && callback(viewer);
          return viewer;
        }

        async _createViewer() {
          try {
            const C = Cesium;
            this.viewer = new C.Viewer(this.containerId, {
              terrainProvider: new C.EllipsoidTerrainProvider(),
              animation: false,
              timeline: false,
              baseLayerPicker: false,
              geocoder: false,
              sceneModePicker: false,
              infoBox: false,
              selectionIndicator: false,
              navigationHelpButton: false,
              fullscreenButton: false,
              shouldAnimate: !this.options.performanceMode
            });

            // Performance optimizations
            this.viewer.scene.requestRenderMode = true;
            this.viewer.scene.maximumRenderTimeChange = this.options.performanceMode ? 1.0 : Infinity;
            
            if (this.options.performanceMode) {
              this.viewer.scene.fog.enabled = false;
              this.viewer.scene.skyAtmosphere.show = false;
              this.viewer.scene.globe.tileCacheSize = 100;
              this.viewer.scene.globe.enableLighting = false;
              this.viewer.scene.postProcessStages.fxaa.enabled = false;
            }
            
            this.viewer.cesiumWidget.creditContainer.style.display = 'none';

            // Setup terrain and imagery
            await this._setupTerrain();
            
            return this.viewer;
          } catch (e) {
            throw e;
          }
        }

        async _setupTerrain() {
          try {
            const C = Cesium;
            const tok = (window.CESIUM_ION_TOKEN && window.CESIUM_ION_TOKEN !== '') ? 
                       window.CESIUM_ION_TOKEN : 
                       (localStorage.getItem('cesium_token') || '');
            
            if (tok) {
              C.Ion.defaultAccessToken = tok;
              this.viewer.terrainProvider = await C.createWorldTerrainAsync();
              const imagery = await C.IonImageryProvider.fromAssetId(3);
              this.viewer.imageryLayers.removeAll();
              this.viewer.imageryLayers.addImageryProvider(imagery);
              try { 
                const tiles = await C.createOsmBuildingsAsync(); 
                this.viewer.scene.primitives.add(tiles); 
              } catch(_) {}
            } else {
              this.viewer.imageryLayers.removeAll();
              this.viewer.imageryLayers.addImageryProvider(
                new C.OpenStreetMapImageryProvider({ url: 'https://tile.openstreetmap.org/' })
              );
              this.viewer.terrainProvider = new C.EllipsoidTerrainProvider();
            }
          } catch(_) {}
        }

        async addTrackSegments(segments, color = '#FF0000', width = 3) {
          await this.initialize();
          const C = Cesium;
          
          // Clear existing tracks
          this.clearTracks();
          
          segments.forEach(segment => {
            if (segment.length > 1) {
              const positions = segment.map(p => C.Cartesian3.fromDegrees(p.lon, p.lat, p.ele || 0));
              const entity = this.viewer.entities.add({
                polyline: {
                  positions,
                  width,
                  material: C.Color.fromCssColorString(color),
                  clampToGround: false
                }
              });
              this.tracks.push(entity);
            }
          });

          if (this.tracks.length) {
            this.viewer.flyTo(this.tracks, { duration: 0.6 });
          }
        }

        async addAircraft(callsign, position, orientation = {}, label = '') {
          await this.initialize();
          const C = Cesium;
          
          const pos = C.Cartesian3.fromDegrees(position.lon, position.lat, position.alt || 0);
          const entity = this.viewer.entities.add({
            id: callsign,
            name: callsign,
            position: pos,
            orientation: C.Transforms.headingPitchRollQuaternion(
              pos,
              C.HeadingPitchRoll.fromDegrees(
                orientation.hdg || 0, 
                orientation.pitch || 0, 
                orientation.roll || 0
              )
            ),
            model: {
              uri: '/static/models/glider/Glider.glb',
              scale: 1.0,
              minimumPixelSize: 64,
              maximumScale: 100,
              runAnimations: false
            },
            label: {
              text: label || callsign,
              font: '12pt sans-serif',
              pixelOffset: new C.Cartesian2(0, -60),
              fillColor: C.Color.YELLOW,
              outlineColor: C.Color.BLACK,
              outlineWidth: 2,
              style: C.LabelStyle.FILL_AND_OUTLINE,
              horizontalOrigin: C.HorizontalOrigin.CENTER,
              verticalOrigin: C.VerticalOrigin.BOTTOM,
              show: true
            }
          });
          
          this.entities.set(callsign, entity);
          
          // Add dynamic flight trail for live tracking
          if (this.options.enableLiveTracking) {
            this._initFlightTrail(callsign, pos);
          }
          
          return entity;
        }

        _initFlightTrail(callsign, initialPos) {
          const C = Cesium;
          
          // Store trail points
          if (!this.trailData) this.trailData = new Map();
          this.trailData.set(callsign, [initialPos]);
          
          // Create trail polyline with dynamic positions
          const trailEntity = this.viewer.entities.add({
            id: callsign + '_trail',
            name: callsign + ' Flight Trail',
            polyline: {
              positions: new C.CallbackProperty(() => {
                const points = this.trailData.get(callsign) || [];
                return points.slice(); // Return copy for performance
              }, false),
              width: 3,
              material: this._getCallsignColor(callsign),
              clampToGround: false,
              followSurface: false,
              arcType: C.ArcType.NONE,
              granularity: C.Math.RADIANS_PER_DEGREE,
              show: true
            }
          });
          
          // Store trail reference
          if (!this.trails) this.trails = new Map();
          this.trails.set(callsign, trailEntity);
        }

        _getCallsignColor(callsign) {
          // Use the same color function as the existing app if available
          if (typeof getCallsignColor !== 'undefined') {
            return Cesium.Color.fromCssColorString(getCallsignColor(callsign));
          }
          // Fallback color scheme
          const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'];
          const hash = callsign.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
          return Cesium.Color.fromCssColorString(colors[hash % colors.length]);
        }

        async updateAircraft(callsign, position, orientation = {}) {
          const entity = this.entities.get(callsign);
          if (!entity) return;
          
          const C = Cesium;
          const pos = C.Cartesian3.fromDegrees(position.lon, position.lat, position.alt || 0);
          entity.position = pos;
          entity.orientation = C.Transforms.headingPitchRollQuaternion(
            pos,
            C.HeadingPitchRoll.fromDegrees(
              orientation.hdg || 0, 
              orientation.pitch || 0, 
              orientation.roll || 0
            )
          );
          
          // Update flight trail for live tracking
          if (this.options.enableLiveTracking && this.trailData) {
            const trail = this.trailData.get(callsign);
            if (trail) {
              trail.push(pos);
              // Keep only last 500 points for performance
              if (trail.length > 500) {
                trail.shift();
              }
            }
          }
          
          this.viewer.scene.requestRender();
        }

        async trackEntity(callsign) {
          const entity = this.entities.get(callsign);
          if (entity && this.viewer) {
            this.viewer.trackedEntity = entity;
          }
        }

        async flyToEntity(callsign, duration = 0.8) {
          const entity = this.entities.get(callsign);
          if (entity && this.viewer) {
            this.viewer.flyTo(entity, { duration });
          }
        }

        async flyToTracks(duration = 0.8) {
          if (this.tracks.length && this.viewer) {
            this.viewer.flyTo(this.tracks, { duration });
          }
        }

        clearTracks() {
          this.tracks.forEach(track => {
            try { this.viewer.entities.remove(track); } catch(_) {}
          });
          this.tracks = [];
        }

        clearAircraft() {
          this.entities.forEach(entity => {
            try { this.viewer.entities.remove(entity); } catch(_) {}
          });
          this.entities.clear();
          
          // Clear trails
          if (this.trails) {
            this.trails.forEach(trail => {
              try { this.viewer.entities.remove(trail); } catch(_) {}
            });
            this.trails.clear();
          }
          
          // Clear trail data
          if (this.trailData) {
            this.trailData.clear();
          }
        }

        clearAll() {
          this.clearTracks();
          this.clearAircraft();
        }

        resize() {
          if (this.viewer) {
            this.viewer.resize();
          }
        }

        destroy() {
          if (this.viewer) {
            this.viewer.destroy();
            this.viewer = null;
          }
          this.entities.clear();
          this.tracks = [];
          this.ready = null;
        }
      }

      // Create global instances
      let converterGlobe = null;
      let liveGlobe = null;

      // Legacy wrapper for converter
      function ensureGlobe(cb) {
        if (!converterGlobe) {
          converterGlobe = new Globe3D('globe', { enableFlythrough: true });
        }
        return converterGlobe.initialize((viewer) => {
          globeViewer = viewer; // Keep legacy reference
          cb && cb();
        });
      }

      // Legacy wrapper for live tracking  
      function ensureLiveGlobe(cb) {
        if (!liveGlobe) {
          liveGlobe = new Globe3D('liveglobe', { 
            enableLiveTracking: true, 
            performanceMode: true 
          });
        }
        return liveGlobe.initialize((viewer) => {
          liveGlobeViewer = viewer; // Keep legacy reference
          cb && cb();
        });
      }

      // Tab switching for converter preview (2D/3D)
      function setTab(which) {
        if (which === '2d') {
          tab2d.classList.add('active'); tab3d.classList.remove('active');
          mapDiv.classList.add('active'); globeDiv.classList.remove('active');
          ensureMap().then(() => setTimeout(() => map && map.invalidateSize(), 50));
        } else {
          tab3d.classList.add('active'); tab2d.classList.remove('active');
          globeDiv.classList.add('active'); mapDiv.classList.remove('active');
          ensureGlobe(() => setTimeout(() => converterGlobe && converterGlobe.resize(), 50));
        }
      }
      tab2d && tab2d.addEventListener('click', () => setTab('2d'));
      tab3d && tab3d.addEventListener('click', () => setTab('3d'));
      window.addEventListener('resize', () => { 
        try { 
          if (map) map.invalidateSize(); 
          if (converterGlobe) converterGlobe.resize(); 
        } catch(_) {} 
      });

      // Live tracking variables
      const liveChInput = document.getElementById('live_channel');
      const liveChLabel = document.getElementById('live_ch_label');
      const liveCallsign = document.getElementById('live_callsign');
      const liveKey = document.getElementById('live_key');
      const liveStatus = document.getElementById('live_status');
      const liveConnectBtn = document.getElementById('live_connect');
      const liveDisconnectBtn = document.getElementById('live_disconnect');
      const liveCenterBtn = document.getElementById('live_center');
      const liveMeta = document.getElementById('live_meta');
      const livePlayersList = document.getElementById('live_players_list');
      const livePlayersChips = document.getElementById('live_players_chips');
      const liveEventsEl = document.getElementById('live_events');
      // Live telemetry elements
      const lvAlt = document.getElementById('lv_alt');
      const lvSpd = document.getElementById('lv_spd');
      const lvVsi = document.getElementById('lv_vsi');
      const lvHdg = document.getElementById('lv_hdg');
      const lvPitch = document.getElementById('lv_pitch');
      const lvRoll = document.getElementById('lv_roll');
      let liveInfoTimer = null;
      let liveKnownPlayers = new Set();
      
      // Enhanced follow state with user interaction detection
      let liveFollow2D = false;
      let liveFollow3D = false;
      let liveFollowCS = null;

      // Helper function for coordinate validation
      const isFiniteNum = (n) => typeof n === 'number' && Number.isFinite(n);

      // Live map variables
      let liveMap = null, liveLeafletReady = null;
      let liveMarkers = new Map(); // callsign -> marker
      let livePaths = new Map();   // callsign -> L.polyline
      let liveHoverPaths = new Map(); // callsign -> hover L.polyline for awesome track tooltips
      let liveTrackData = new Map(); // callsign -> array of {pos, data, timestamp}
      
      function ensureLiveMap() {
        if (liveMap) return Promise.resolve();
        if (liveLeafletReady) return liveLeafletReady;
        liveLeafletReady = new Promise((resolve, reject) => {
          const css = document.createElement('link');
          css.rel = 'stylesheet';
          css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
          document.head.appendChild(css);
          const script = document.createElement('script');
          script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
          script.crossOrigin = 'anonymous';
          script.onload = () => {
            liveMap = L.map('livemap').setView([46.8, 8.2], 8);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
              attribution: '© OpenStreetMap'
            }).addTo(liveMap);
            
            // Add interaction detection for live map follow mode
            let userDragDetected = false;
            
            liveMap.on('dragstart', () => {
              userDragDetected = true;
            });
            
            liveMap.on('movestart', () => {
              // Only disable following if user manually moved (not programmatic)
              if (liveFollow2D && userDragDetected) {
                liveFollow2D = false;
                updateFollowBtn();
                pushEvent('Following disabled - manual map movement');
              }
            });
            
            // Remove zoom disable - zooming should zoom into the followed aircraft
            
            liveMap.on('moveend', () => {
              userDragDetected = false;
            });
            
            resolve();
          };
          script.onerror = reject;
          document.head.appendChild(script);
        });
        return liveLeafletReady;
      }

      // Live 3D Globe
      let liveGlobeViewer = null;
      let live3DEntities = new Map(); // callsign -> entity
      let live3DPaths = new Map();    // callsign -> path entity
      let live3DTrackData = new Map(); // callsign -> array of track points
      let live3DDebugAxes = new Map(); // callsign -> debug axes entities
      let live3DLastLL = new Map();   // callsign -> {lat,lon}
      
      // ensureLiveGlobe is now handled by the Globe3D class wrapper above
      
      // Function to update terrain provider dynamically
      async function updateTerrainProvider() {
        if (!liveGlobeViewer) return;
        
        const C = Cesium;
        const terrainSel = document.getElementById('terrain_mode');
        const mode = terrainSel ? terrainSel.value : 'flat';
        
        console.log('Updating terrain to:', mode);
        
        try {
          let terrainProvider = new C.EllipsoidTerrainProvider();
          
          if (mode === 'real' && C.Ion && C.Ion.defaultAccessToken) {
            try {
              terrainProvider = await C.createWorldTerrainAsync();
              console.log('World terrain created successfully');
            } catch(e) { 
              console.warn('Failed to create world terrain, using ellipsoid:', e);
              terrainProvider = new C.EllipsoidTerrainProvider(); 
            }
          }
          
          liveGlobeViewer.terrainProvider = terrainProvider;
          
          // Force a render to show the change
          liveGlobeViewer.scene.requestRender();
          
          console.log('Terrain successfully changed to:', mode);
        } catch(e) {
          console.error('Error updating terrain:', e);
        }
      }

      // Enhanced player chip functionality
      function updateLiveInfo() {
        const ch = liveChInput.value || 'default';
        fetch(`/api/live/${ch}/info`).then(r=>r.json()).then(j => {
          const c = (j.counts)||{};
          liveMeta.textContent = `Viewers: ${c.viewers||0} • Feeders: ${c.feeders||0} • Players: ${c.players||0}`;
          const players = (j.players||[]).map(p => p.callsign || 'ACFT');
          livePlayersList.textContent = players.length ? `Players: ${players.join(', ')}` : 'Players: -';
          
          // Enhanced chips with active map detection
          if (livePlayersChips){
            livePlayersChips.innerHTML = '';
            players.forEach(cs => {
              const chip = document.createElement('div'); 
              chip.className = 'chip'; 
              chip.textContent = cs;
              chip.onclick = () => {
                liveFollowCS = cs;
                
                // Determine which map is currently active
                const live2DActive = document.getElementById('live-tab-2d').classList.contains('active');
                const live3DActive = document.getElementById('live-tab-3d').classList.contains('active');
                
                if (live2DActive) {
                  // Focus on 2D map
                  liveFollow2D = true;
                  updateFollow2DBtn();
                  const m = liveMarkers.get(cs);
                  if (m) {
                    try {
                      const ll = m.getLatLng();
                      const targetZ = Math.max(12, liveMap.getZoom());
                      liveMap.flyTo(ll, targetZ, { animate: true, duration: 0.6 });
                    } catch(_) {}
                  }
                } else if (live3DActive) {
                  // Focus on 3D globe
                  liveFollow3D = true;
                  updateFollow3DBtn();
                  ensureLiveGlobe(() => {
                    follow3D && follow3D(cs);
                  });
                }
                
                // Add active state to clicked chip
                document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
              };
              livePlayersChips.appendChild(chip);
            });
          }
          
          // eventize presence changes
          const next = new Set(players);
          // joins
          for (const p of next){ if (!liveKnownPlayers.has(p)) pushEvent(`Player ${p} joined`); }
          // leaves
          for (const p of liveKnownPlayers){ if (!next.has(p)) pushEvent(`Player ${p} left`); }
          liveKnownPlayers = next;
        }).catch(()=>{});
      }

      // Enhanced unified follow button update
      function updateFollowBtn() {
        const btn = document.getElementById('live_follow');
        if (!btn) return;
        
        const is2DActive = liveTab2D && liveTab2D.classList.contains('active');
        const is3DActive = liveTab3D && liveTab3D.classList.contains('active');
        
        if (is2DActive && liveFollow2D) {
          btn.textContent = 'Following 2D';
          btn.classList.add('active');
        } else if (is3DActive && liveFollow3D) {
          btn.textContent = 'Following 3D';
          btn.classList.add('active');
        } else {
          btn.textContent = 'Follow';
          btn.classList.remove('active');
        }
      }

      // Back-compat helpers referenced by some UI paths
      function updateFollow2DBtn() { try { updateFollowBtn(); } catch(_) {} }
      function updateFollow3DBtn() { try { updateFollowBtn(); } catch(_) {} }

      // Live tab management
      const liveTab2D = document.getElementById('live-tab-2d');
      const liveTab3D = document.getElementById('live-tab-3d');
      const liveMapDiv = document.getElementById('livemap');
      const liveGlobeDiv = document.getElementById('liveglobe');
      
      function setLiveTab(tab) {
        const is2D = (tab === '2d');
        liveTab2D.classList.toggle('active', is2D);
        liveTab3D.classList.toggle('active', !is2D);
        liveMapDiv.classList.toggle('active', is2D);
        liveGlobeDiv.classList.toggle('active', !is2D);
        
        if (is2D) {
          ensureLiveMap().then(() => setTimeout(() => liveMap && liveMap.invalidateSize(), 50));
        } else {
          ensureLiveGlobe(() => setTimeout(() => liveGlobe && liveGlobe.resize(), 50));
        }
      }
      
      liveTab2D.addEventListener('click', () => setLiveTab('2d'));
      liveTab3D.addEventListener('click', () => setLiveTab('3d'));
      
      // Terrain dropdown event listener
      const terrainSelect = document.getElementById('terrain_mode');
      if (terrainSelect) {
        terrainSelect.addEventListener('change', () => {
          updateTerrainProvider();
        });
      }

      // Enhanced live connection management
      let liveWs = null, liveConnected = false;
      
      // Professional color palette for aircraft tracking
      const AIRCRAFT_COLORS = [
        '#2E8B57', // Sea Green
        '#4682B4', // Steel Blue
        '#CD853F', // Peru
        '#8B008B', // Dark Magenta
        '#FF6347', // Tomato
        '#4169E1', // Royal Blue
        '#32CD32', // Lime Green
        '#FF4500', // Orange Red
        '#9370DB', // Medium Purple
        '#20B2AA', // Light Sea Green
        '#DC143C', // Crimson
        '#00CED1', // Dark Turquoise
        '#FF1493', // Deep Pink
        '#1E90FF', // Dodger Blue
        '#FFD700', // Gold
        '#8A2BE2', // Blue Violet
        '#00FA9A', // Medium Spring Green
        '#FF69B4', // Hot Pink
        '#5F9EA0', // Cadet Blue
        '#D2691E'  // Chocolate
      ];
      
      // Callsign to color mapping with persistence
      const callsignColors = new Map();
      
      // Load persisted color mappings on startup
      function loadCallsignColors() {
        try {
          const stored = localStorage.getItem('ftpro_callsign_colors');
          if (stored) {
            const parsed = JSON.parse(stored);
            Object.entries(parsed).forEach(([callsign, color]) => {
              callsignColors.set(callsign, color);
            });
          }
        } catch (e) {
          console.warn('Failed to load callsign colors:', e);
        }
      }
      
      // Save color mappings to localStorage
      function saveCallsignColors() {
        try {
          const colorMap = {};
          callsignColors.forEach((color, callsign) => {
            colorMap[callsign] = color;
          });
          localStorage.setItem('ftpro_callsign_colors', JSON.stringify(colorMap));
        } catch (e) {
          console.warn('Failed to save callsign colors:', e);
        }
      }
      
      // Generate deterministic color for callsign
      function getCallsignColor(callsign) {
        if (callsignColors.has(callsign)) {
          return callsignColors.get(callsign);
        }
        
        // Simple hash function for deterministic color selection
        let hash = 0;
        for (let i = 0; i < callsign.length; i++) {
          const char = callsign.charCodeAt(i);
          hash = ((hash << 5) - hash) + char;
          hash = hash & hash; // Convert to 32-bit integer
        }
        
        // Select color from palette based on hash
        const colorIndex = Math.abs(hash) % AIRCRAFT_COLORS.length;
        const color = AIRCRAFT_COLORS[colorIndex];
        
        // Store mapping for consistency
        callsignColors.set(callsign, color);
        
        // Persist to localStorage
        saveCallsignColors();
        
        return color;
      }
      
      function pushEvent(msg) {
        if (liveEventsEl) {
          const div = document.createElement('div');
          div.textContent = `${new Date().toLocaleTimeString()}: ${msg}`;
          liveEventsEl.appendChild(div);
          liveEventsEl.scrollTop = liveEventsEl.scrollHeight;
        }
      }

      // WebSocket connection handling
      let currentChannel = null;
      function connectWs() {
        if (liveWs) return;
        
        const channel = (liveChInput.value || 'default').trim();
        liveChLabel.textContent = channel;
        
        // Clear map if switching to a different channel
        if (currentChannel && currentChannel !== channel) {
          clearMapData();
          pushEvent(`Switched from channel '${currentChannel}' to '${channel}'`);
        } else if (!currentChannel) {
          // First connection - clear any stale data
          clearMapData();
        }
        currentChannel = channel;
        
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const url = `${proto}://${location.host}/ws/live/${encodeURIComponent(channel)}?mode=viewer`;
        
        liveStatus.textContent = 'connecting...';
        liveWs = new WebSocket(url);
        
        liveWs.onopen = () => {
          liveConnected = true;
          liveStatus.textContent = 'connected';
          pushEvent('Connected to live channel');
          
          // Load persisted track data first
          loadTrackData(channel);
          
          // Fetch initial data and history
          fetchRecent();
          fetchInfo();
          fetchHistory(channel);
          
          // Start periodic info updates and track saving
          if (liveInfoTimer) clearInterval(liveInfoTimer);
          liveInfoTimer = setInterval(() => {
            fetchInfo();
            pruneStale();
            saveTrackData(channel);  // Save tracks every 3 seconds
          }, 3000);
        };
        
        liveWs.onclose = () => {
          liveConnected = false;
          liveStatus.textContent = 'disconnected';
          liveWs = null;
          currentChannel = null;  // Reset current channel on unexpected disconnect
          
          if (liveInfoTimer) {
            clearInterval(liveInfoTimer);
            liveInfoTimer = null;
          }
          
          // Save track data before closing
          saveTrackData(channel);
          
          pushEvent('Disconnected from live channel');
        };
        
        liveWs.onerror = (err) => {
          liveStatus.textContent = 'error';
          pushEvent('Connection error');
          console.error('WebSocket error:', err);
        };
        
        liveWs.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'state' && msg.payload) {
              updateLivePosition(msg.payload);
            }
          } catch (e) {
            console.error('Message parse error:', e);
          }
        };
      }
      
      // Clear all map data (aircraft, tracks, UI)
      function clearMapData() {
        // Clear 2D map data
        if (liveMap) {
          liveMarkers.forEach(marker => liveMap.removeLayer(marker));
          livePaths.forEach(path => liveMap.removeLayer(path));
        }
        
        // Clear 3D globe data using Globe3D component
        if (liveGlobe) {
          liveGlobe.clearAll();
        }
        
        // Clear data structures
        liveMarkers.clear();
        livePaths.clear();
        liveTrackData.clear();
        if (live3DEntities) live3DEntities.clear();
        if (live3DPaths) live3DPaths.clear();
        if (live3DTrackData) live3DTrackData.clear();
        if (live3DDebugAxes) live3DDebugAxes.clear();
        
        // Clear telemetry display
        const telemetryElements = ['lv_alt', 'lv_spd', 'lv_vsi', 'lv_hdg', 'lv_pitch', 'lv_roll'];
        telemetryElements.forEach(id => {
          const el = document.getElementById(id);
          if (el) el.textContent = '-';
        });
        
        // Clear player chips
        if (livePlayersChips) {
          livePlayersChips.innerHTML = '';
        }
        
        // Reset follow state
        liveFollow2D = false;
        liveFollow3D = false;
        liveFollowCS = null;
        updateFollowBtn();
        
        pushEvent('Map cleared');
      }

      function disconnectWs() {
        if (liveWs) {
          // Save track data before disconnecting
          const channel = (liveChInput.value || 'default').trim();
          saveTrackData(channel);
          
          liveWs.close();
          liveWs = null;
        }
        liveConnected = false;
        currentChannel = null;  // Reset current channel
        
        // Clear all map data when disconnecting
        clearMapData();
      }

      // Track data persistence functions
      function saveTrackData(channel) {
        try {
          const trackDataObj = {};
          liveTrackData.forEach((data, callsign) => {
            trackDataObj[callsign] = data;
          });
          localStorage.setItem(`ftpro_tracks_${channel}`, JSON.stringify(trackDataObj));
        } catch (e) {
          console.error('Failed to save track data:', e);
        }
      }

      function loadTrackData(channel) {
        try {
          const saved = localStorage.getItem(`ftpro_tracks_${channel}`);
          if (saved) {
            const trackDataObj = JSON.parse(saved);
            Object.entries(trackDataObj).forEach(([callsign, data]) => {
              liveTrackData.set(callsign, data);
            });
          }
        } catch (e) {
          console.error('Failed to load track data:', e);
        }
      }

      // Fetch functions
      async function fetchRecent() {
        try {
          const channel = liveChInput.value || 'default';
          const resp = await fetch(`/api/live/${encodeURIComponent(channel)}/recent`);
          const data = await resp.json();
          data.samples.forEach(sample => updateLivePosition(sample));
        } catch (e) {
          console.error('Failed to fetch recent:', e);
        }
      }

      async function fetchHistory() {
        try {
          const channel = liveChInput.value || 'default';
          const resp = await fetch(`/api/live/${encodeURIComponent(channel)}/history`);
          const data = await resp.json();
          
          // Process historical tracks
          Object.entries(data.tracks || {}).forEach(([callsign, trackPoints]) => {
            if (trackPoints && trackPoints.length > 0) {
              // Convert server history to our format with validation
              const formattedTrack = trackPoints
                .filter(point => point && isFiniteNum(point.lat) && isFiniteNum(point.lon) && 
                        Math.abs(point.lat) <= 90 && Math.abs(point.lon) <= 180)
                .map(point => ({
                  pos: [point.lat, point.lon],
                  data: point,
                  timestamp: point.ts ? point.ts * 1000 : Date.now() // Convert to milliseconds
                }));
              
              // Merge with existing track data
              const existing = liveTrackData.get(callsign) || [];
              const combined = [...existing, ...formattedTrack];
              
              // Remove duplicates and sort by timestamp
              const unique = combined.filter((point, index, arr) => 
                index === 0 || arr[index - 1].timestamp !== point.timestamp
              ).sort((a, b) => a.timestamp - b.timestamp);
              
              // Keep only last 500 points
              if (unique.length > 500) {
                unique.splice(0, unique.length - 500);
              }
              
              liveTrackData.set(callsign, unique);
              
              // Create/update track visualization
              if (liveMap && unique.length > 1) {
                const positions = unique
                  .filter(p => p.pos && Array.isArray(p.pos) && p.pos.length === 2 &&
                          isFiniteNum(p.pos[0]) && isFiniteNum(p.pos[1]) &&
                          Math.abs(p.pos[0]) <= 90 && Math.abs(p.pos[1]) <= 180)
                  .map(p => p.pos);
                const color = getCallsignColor(callsign);
                
                // Only create polylines if we have at least 2 valid positions
                if (positions.length < 2) return;
                
                if (!livePaths.has(callsign)) {
                  const pathLine = L.polyline(positions, { 
                    color: color, 
                    weight: 3, 
                    opacity: 0.7,
                    bubblingMouseEvents: false
                  }).addTo(liveMap);
                  
                  // Create invisible wider polyline for better hover detection - AWESOME for historical tracks too!
                  const hoverLine = L.polyline(positions, { 
                    color: 'transparent', 
                    weight: 15, 
                    opacity: 0,
                    interactive: true,
                    bubblingMouseEvents: false
                  }).addTo(liveMap);
                  
                  // Function to find closest track point and show tooltip - SUPER NICE for history!
                  function showHistoricalTooltip(e) {
                    // Find closest track point
                    let closestPoint = unique[0];
                    let minDistance = Infinity;
                    
                    unique.forEach(point => {
                      const distance = liveMap.distance(e.latlng, point.pos);
                      if (distance < minDistance) {
                        minDistance = distance;
                        closestPoint = point;
                      }
                    });
                    
                    const trackTooltipContent = `
                      <div class="aircraft-telemetry-tooltip">
                        <div class="aircraft-callsign">${callsign} - Historical Track</div>
                        <div class="telemetry-grid">
                          <div class="telem-tile">
                            <div class="telem-label">ALT</div>
                            <div class="telem-value">${closestPoint.data.alt_m ? Math.round(closestPoint.data.alt_m) : '-'}<span class="telem-unit">m</span></div>
                          </div>
                          <div class="telem-tile">
                            <div class="telem-label">SPD</div>
                            <div class="telem-value">${closestPoint.data.spd_kt ? Math.round(closestPoint.data.spd_kt) : '-'}<span class="telem-unit">kt</span></div>
                          </div>
                          <div class="telem-tile">
                            <div class="telem-label">HDG</div>
                            <div class="telem-value">${closestPoint.data.hdg_deg ? Math.round(closestPoint.data.hdg_deg) : '-'}<span class="telem-unit">°</span></div>
                          </div>
                          <div class="telem-tile">
                            <div class="telem-label">V/S</div>
                            <div class="telem-value">${closestPoint.data.vsi_ms ? closestPoint.data.vsi_ms.toFixed(1) : '-'}<span class="telem-unit">m/s</span></div>
                          </div>
                        </div>
                        <div class="tooltip-hint">Historical - ${new Date(closestPoint.timestamp).toLocaleTimeString()}</div>
                      </div>
                    `;
                    
                    // Show tooltip at the closest track point position for better snapping
                    hoverLine.bindTooltip(trackTooltipContent, {
                      maxWidth: 320,
                      className: 'aircraft-tooltip',
                      direction: 'top',
                      offset: [0, -10],
                      sticky: false
                    }).openTooltip(closestPoint.pos);
                  }
                  
                  // Add awesome hover functionality to historical track
                  hoverLine.on('mouseover', function(e) {
                    pathLine.setStyle({ weight: 5, opacity: 1 });
                    showHistoricalTooltip(e);
                  });
                  
                  hoverLine.on('mousemove', function(e) {
                    showHistoricalTooltip(e);
                  });
                  
                  hoverLine.on('mouseout', function(e) {
                    pathLine.setStyle({ weight: 3, opacity: 0.7 });
                    this.closeTooltip();
                  });
                  
                  livePaths.set(callsign, pathLine);
                  liveHoverPaths.set(callsign, hoverLine);
                } else {
                  // Update existing path
                  const pathLine = livePaths.get(callsign);
                  const hoverLine = liveHoverPaths.get(callsign);
                  if (pathLine) {
                    pathLine.setLatLngs(positions);
                  }
                  if (hoverLine) {
                    hoverLine.setLatLngs(positions);
                  }
                }
              }

              // Initialize 3D track data for interpolation system (simplified seeding)
              if (typeof Cesium !== 'undefined' && liveGlobeViewer) {
                try {
                  if (!live3DTrackData.has(callsign)) {
                    live3DTrackData.set(callsign, []);
                  }
                } catch (e) {
                  console.warn('Failed to initialize 3D data for', callsign, e);
                }
              }
            }
          });
          
          pushEvent(`Loaded historical tracks for ${Object.keys(data.tracks || {}).length} aircraft`);
        } catch (e) {
          console.error('Failed to fetch history:', e);
        }
      }
      
      async function fetchInfo() {
        try {
          const channel = liveChInput.value || 'default';
          const resp = await fetch(`/api/live/${encodeURIComponent(channel)}/info`);
          const data = await resp.json();
          
          if (liveMeta) {
            liveMeta.textContent = `Viewers: ${data.counts.viewers} • Feeders: ${data.counts.feeders} • Players: ${data.counts.players}`;
          }
          
          updatePlayerChips(data.players.map(p => p.callsign));
        } catch (e) {
          console.error('Failed to fetch info:', e);
        }
      }

      // Update player chips
      function updatePlayerChips(players) {
        if (!livePlayersChips) return;
        
        livePlayersChips.innerHTML = '';
        players.forEach(cs => {
          const chip = document.createElement('div');
          chip.className = 'chip';
          chip.textContent = cs;
          chip.onclick = () => {
            liveFollowCS = cs;
            
            const is2DActive = liveTab2D && liveTab2D.classList.contains('active');
            
            if (is2DActive) {
              liveFollow2D = true;
              liveFollow3D = false;
              const marker = liveMarkers.get(cs);
              if (marker) {
                const pos = marker.getLatLng();
                liveMap.flyTo(pos, Math.max(12, liveMap.getZoom()));
              }
            } else {
              liveFollow2D = false;
              liveFollow3D = true;
              ensureLiveGlobe(() => {
                liveGlobe.trackEntity(cs);
              });
            }
            
            updateFollowBtn();
            pushEvent(`Following ${cs}`);
          };
          livePlayersChips.appendChild(chip);
        });
      }

      // Performance optimization: Cache for icons and throttling
      const iconCache = new Map();
      const tooltipCache = new Map();
      let lastUpdateTime = new Map();
      let last3DUpdateTime = new Map(); // Throttling for 3D updates  
      const UPDATE_THROTTLE_MS = 33; // Direct updates at 30fps (1000/30)
      const UPDATE_3D_THROTTLE_MS = 33; // 3D updates at 30fps

      // Interpolation helpers (safe defaults)
      // 2D interpolation buffer is unused for now but kept to avoid reference errors
      const interpolationBuffer = new Map();
      const interpolation3DBuffer = new Map();
      const INTERPOLATION_DELAY_MS = 0;
      function smoothEasing(t) { return t; }
      function lerpPosition(a, b, t) {
        if (!a || !b) return b || a;
        const ax = Array.isArray(a) ? a[0] : a.lat || 0;
        const ay = Array.isArray(a) ? a[1] : a.lon || 0;
        const bx = Array.isArray(b) ? b[0] : b.lat || 0;
        const by = Array.isArray(b) ? b[1] : b.lon || 0;
        return [ax + (bx - ax) * t, ay + (by - ay) * t];
      }
      function lerpPosition3D(a, b, t) {
        try {
          return new Cesium.Cartesian3(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t
          );
        } catch (_) {
          return b || a;
        }
      }

      // Live position update - direct updates with 30fps throttling
      function updateLivePosition(sample) {
        if (!sample || !sample.lat || !sample.lon) return;
        
        // Validate coordinates are valid finite numbers within proper ranges
        if (!isFiniteNum(sample.lat) || !isFiniteNum(sample.lon) || 
            Math.abs(sample.lat) > 90 || Math.abs(sample.lon) > 180) {
          console.warn('Invalid coordinates in updateLivePosition:', sample.lat, sample.lon);
          return;
        }
        
        const cs = sample.callsign || 'ACFT';
        const now = Date.now();
        const pos = [sample.lat, sample.lon];
        
        // Throttle updates to 30fps per aircraft
        const lastUpdate = lastUpdateTime.get(cs) || 0;
        if (now - lastUpdate < UPDATE_THROTTLE_MS) {
          return;
        }
        lastUpdateTime.set(cs, now);
        
        // Ensure live map is initialized
        if (!liveMap) {
          ensureLiveMap().then(() => updateLivePosition(sample));
          return;
        }
        
        // Update telemetry display
        if (!liveFollowCS || liveFollowCS === cs) {
          if (lvAlt) lvAlt.textContent = sample.alt_m ? Math.round(sample.alt_m) : '-';
          if (lvSpd) lvSpd.textContent = sample.spd_kt ? Math.round(sample.spd_kt) : '-';
          if (lvVsi) lvVsi.textContent = sample.vsi_ms ? sample.vsi_ms.toFixed(1) : '-';
          if (lvHdg) lvHdg.textContent = sample.hdg_deg ? Math.round(sample.hdg_deg) : '-';
          if (lvPitch) lvPitch.textContent = sample.pitch_deg ? Math.round(sample.pitch_deg) : '-';
          if (lvRoll) lvRoll.textContent = sample.roll_deg ? Math.round(sample.roll_deg) : '-';
        }
        
        // Direct position updates without interpolation
        renderActual2DPosition(cs, pos, sample);
        
        // Update 3D if available
        if (liveGlobeViewer) {
          updateLive3DPosition(sample);
        }
        
        // Handle 2D following
        if (liveFollowCS === cs && liveFollow2D) {
          try {
            const targetZoom = Math.max(12, liveMap.getZoom());
            liveMap.flyTo(pos, targetZoom, { animate: true, duration: 0.3 });
          } catch(e) {
            console.error('Follow error:', e);
          }
        }
        
        updateFollowBtn();
      }
      
      // Remove the interpolation system - now using direct updates
      // Wrap the detailed 2D rendering logic in a function that accepts the
      // current callsign, position and optional full sample for rich tooltips.
      function renderActual2DPosition(cs, pos, sample) {
        const now = Date.now();
        const color = getCallsignColor(cs);
        const headingKey = `${cs}-${Math.round((sample.hdg_deg || 0) / 5) * 5}-${color}`;
        let icon = iconCache.get(headingKey);
        
        if (!icon) {
          icon = L.divIcon({
            className: 'ac-icon',
            html: `<div class="ac-rot" style="transform: rotate(${sample.hdg_deg || 0}deg); background: rgba(255,255,255,0.9); border: 2px solid ${color}; border-radius: 50%; padding: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
              <div style="width: 32px; height: 32px; background: ${color}; mask: url('/static/icons/airplane.svg') no-repeat center; -webkit-mask: url('/static/icons/airplane.svg') no-repeat center; mask-size: 24px 24px; -webkit-mask-size: 24px 24px;"></div>
            </div>`,
            iconSize: [40, 40],
            iconAnchor: [20, 20]
          });
          // Keep cache size reasonable
          if (iconCache.size > 100) {
            const firstKey = iconCache.keys().next().value;
            iconCache.delete(firstKey);
          }
          iconCache.set(headingKey, icon);
        }
        
        // Get cached or create new tooltip content - cache by rounded values to reduce regeneration
        const tooltipKey = `${cs}-${Math.round(sample.alt_m || 0)}-${Math.round(sample.spd_kt || 0)}-${Math.round(sample.hdg_deg || 0)}`;
        let tooltipContent = tooltipCache.get(tooltipKey);
        
        if (!tooltipContent) {
          tooltipContent = `
            <div class="aircraft-telemetry-tooltip">
              <div class="aircraft-callsign">${cs}</div>
              <div class="telemetry-grid">
                <div class="telem-tile">
                  <div class="telem-label">ALT</div>
                  <div class="telem-value">${sample.alt_m ? Math.round(sample.alt_m) : '-'}<span class="telem-unit">m</span></div>
                </div>
                <div class="telem-tile">
                  <div class="telem-label">SPD</div>
                  <div class="telem-value">${sample.spd_kt ? Math.round(sample.spd_kt) : '-'}<span class="telem-unit">kt</span></div>
                </div>
                <div class="telem-tile">
                  <div class="telem-label">HDG</div>
                  <div class="telem-value">${sample.hdg_deg ? Math.round(sample.hdg_deg) : '-'}<span class="telem-unit">°</span></div>
                </div>
                <div class="telem-tile">
                  <div class="telem-label">V/S</div>
                  <div class="telem-value">${sample.vsi_ms ? sample.vsi_ms.toFixed(1) : '-'}<span class="telem-unit">m/s</span></div>
                </div>
              </div>
              <div class="tooltip-hint">Click to follow</div>
            </div>
          `;
          // Keep cache size reasonable
          if (tooltipCache.size > 50) {
            const firstKey = tooltipCache.keys().next().value;
            tooltipCache.delete(firstKey);
          }
          tooltipCache.set(tooltipKey, tooltipContent);
        }

        // Update 2D marker - optimized for performance
        if (!liveMarkers.has(cs)) {
          const marker = L.marker(pos, { icon: icon }).addTo(liveMap);
          marker.bindTooltip(tooltipContent, { 
            maxWidth: 320,
            className: 'aircraft-tooltip',
            direction: 'top',
            offset: [0, -20]
          });
          
          // Add click handler for following
          marker.on('click', () => {
            liveFollowCS = cs;
            liveFollow2D = true;
            updateFollowBtn();
            pushEvent(`Following ${cs} from marker`);
          });
          
          liveMarkers.set(cs, marker);
          
          // Create beautiful flight path with hover functionality
          const pathLine = L.polyline([pos], { 
            color: color, 
            weight: 3, 
            opacity: 0.7,
            bubblingMouseEvents: false
          });
          if (pathLine) pathLine.addTo(liveMap);
          
          // Create invisible wider polyline for better hover detection - this is AWESOME!
          const hoverLine = L.polyline([pos], { 
            color: 'transparent', 
            weight: 15, 
            opacity: 0,
            interactive: true,
            bubblingMouseEvents: false
          });
          if (hoverLine) hoverLine.addTo(liveMap);
          
          // Function to find closest track point and show tooltip - SUPER NICE feature!
          function showTrackTooltip(e) {
            const trackData = liveTrackData.get(cs) || [];
            if (trackData.length > 0) {
              let closestPoint = trackData[0];
              let minDistance = Infinity;
              
              trackData.forEach(point => {
                const distance = liveMap.distance(e.latlng, point.pos);
                if (distance < minDistance) {
                  minDistance = distance;
                  closestPoint = point;
                }
              });
              
              // Create track point tooltip
              const trackTooltipContent = `
                <div class="aircraft-telemetry-tooltip">
                  <div class="aircraft-callsign">${cs} - Track Point</div>
                  <div class="telemetry-grid">
                    <div class="telem-tile">
                      <div class="telem-label">ALT</div>
                      <div class="telem-value">${closestPoint.data.alt_m ? Math.round(closestPoint.data.alt_m) : '-'}<span class="telem-unit">m</span></div>
                    </div>
                    <div class="telem-tile">
                      <div class="telem-label">SPD</div>
                      <div class="telem-value">${closestPoint.data.spd_kt ? Math.round(closestPoint.data.spd_kt) : '-'}<span class="telem-unit">kt</span></div>
                    </div>
                    <div class="telem-tile">
                      <div class="telem-label">HDG</div>
                      <div class="telem-value">${closestPoint.data.hdg_deg ? Math.round(closestPoint.data.hdg_deg) : '-'}<span class="telem-unit">°</span></div>
                    </div>
                    <div class="telem-tile">
                      <div class="telem-label">V/S</div>
                      <div class="telem-value">${closestPoint.data.vsi_ms ? closestPoint.data.vsi_ms.toFixed(1) : '-'}<span class="telem-unit">m/s</span></div>
                    </div>
                  </div>
                  <div class="tooltip-hint">Time: ${new Date(closestPoint.timestamp).toLocaleTimeString()}</div>
                </div>
              `;
              
              // Show tooltip at the closest track point position for better snapping
              hoverLine.bindTooltip(trackTooltipContent, {
                maxWidth: 320,
                className: 'aircraft-tooltip',
                direction: 'top',
                offset: [0, -10],
                sticky: false
              }).openTooltip(closestPoint.pos);
            }
          }
          
          // Add awesome hover and click interactions to the invisible hover line
          hoverLine.on('mouseover', function(e) {
            pathLine.setStyle({ weight: 5, opacity: 1 });
            showTrackTooltip(e);
          });
          
          hoverLine.on('mousemove', function(e) {
            showTrackTooltip(e);
          });
          
          hoverLine.on('mouseout', function(e) {
            pathLine.setStyle({ weight: 3, opacity: 0.7 });
            this.closeTooltip();
          });
          
          hoverLine.on('click', function(e) {
            liveFollowCS = cs;
            liveFollow2D = true;
            updateFollowBtn();
            liveMap.flyTo(e.latlng, Math.max(12, liveMap.getZoom()));
            pushEvent(`Following ${cs} from track`);
          });
          
          livePaths.set(cs, pathLine);
          liveHoverPaths.set(cs, hoverLine);
          
          // Initialize track data for this aircraft  
          liveTrackData.set(cs, [{
            pos: pos,
            data: { ...sample },
            timestamp: now
          }]);
        } else {
          // Update existing marker
          const marker = liveMarkers.get(cs);
          marker.setLatLng(pos);
          marker.setIcon(icon);
          marker.setTooltipContent(tooltipContent);
          
          // Update flight path and track data
          const pathLine = livePaths.get(cs);
          const hoverLine = liveHoverPaths.get(cs);
          const trackData = liveTrackData.get(cs) || [];
          
          if (pathLine) {
            const currentPath = pathLine.getLatLngs();
            
            // Reset path if break_path flag is set or if large jump detected
            if (sample.break_path) {
              pathLine.setLatLngs([pos]);
              if (hoverLine) hoverLine.setLatLngs([pos]);
              liveTrackData.set(cs, [{
                pos: pos,
                data: { ...sample },
                timestamp: now
              }]);
            } else {
              // Check for large jumps (teleportation)
              if (currentPath.length > 0) {
                const lastPos = currentPath[currentPath.length - 1];
                const distance = liveMap.distance(lastPos, pos);
                if (distance > 20000) { // 20km jump
                  pathLine.setLatLngs([pos]);
                  if (hoverLine) hoverLine.setLatLngs([pos]);
                  liveTrackData.set(cs, [{
                    pos: pos,
                    data: { ...sample },
                    timestamp: now
                  }]);
                } else {
                  // Add new point to path and track data
                  currentPath.push(pos);
                  if (hoverLine) hoverLine.addLatLng(pos);
                  trackData.push({
                    pos: pos,
                    data: { ...sample },
                    timestamp: now
                  });
                  
                  // Keep only last 500 points for performance
                  if (currentPath.length > 500) currentPath.shift();
                  if (trackData.length > 500) trackData.shift();
                  
                  pathLine.setLatLngs(currentPath);
                  liveTrackData.set(cs, trackData);
                }
              } else {
                pathLine.setLatLngs([pos]);
                trackData.push({
                  pos: pos,
                  data: { ...sample },
                  timestamp: Date.now()
                });
                liveTrackData.set(cs, trackData);
              }
            }
          }
        }
        
        // Active following logic - continuous tracking
        if (liveFollow2D && (liveFollowCS === null || liveFollowCS === cs)) {
          liveFollowCS = cs;
          try {
            // Use setView for immediate positioning, maintaining current zoom
            const currentZoom = liveMap.getZoom();
            liveMap.setView(pos, Math.max(currentZoom, 12), { animate: true, duration: 0.5 });
          } catch(e) {
            console.error('Follow error:', e);
          }
        }
        
        // Update 3D if available
        if (liveGlobeViewer) {
          updateLive3DPosition(sample);
        }
        
        updateFollowBtn();
      }
      
      // Smooth interpolation rendering loop - runs continuously at 60fps
      function renderInterpolatedPositions() {
        const now = Date.now();
        
        // Handle 2D interpolation
        for (let [cs, buffer] of interpolationBuffer) {
          const elapsed = now - buffer.startTime;
          
          if (elapsed < 0) continue; // Not yet time to start interpolation
          
          const t = elapsed / buffer.duration;
          
          let currentPos;
          if (t >= 1) {
            // Animation complete - use final position and stay there
            currentPos = buffer.to;
            // Keep the buffer at the final position to prevent flickering
            buffer.from = buffer.to;
            buffer.startTime = now;
            buffer.duration = 100;
          } else {
            // Interpolate between from and to
            const easedT = smoothEasing(t);
            currentPos = lerpPosition(buffer.from, buffer.to, easedT);
          }
          
          // Now actually update the map with the interpolated position
          renderActual2DPositionSimple(cs, currentPos);
        }
        
        // Handle 3D interpolation
        for (let [cs, buffer] of interpolation3DBuffer) {
          const elapsed = now - buffer.startTime;
          
          if (elapsed < 0) continue; // Not yet time to start interpolation
          
          const t = elapsed / buffer.duration;
          
          let currentPos;
          if (t >= 1) {
            // Animation complete - use final position and stay there
            currentPos = buffer.to;
            // Keep the buffer at the final position to prevent flickering
            buffer.from = buffer.to;
            buffer.startTime = now;
            buffer.duration = 100;
          } else {
            // Interpolate between from and to
            const easedT = smoothEasing(t);
            currentPos = lerpPosition3D(buffer.from, buffer.to, easedT);
          }
          
          // Now actually update the 3D globe with the interpolated position
          renderActual3DPosition(cs, currentPos);
        }
        
        // Schedule next frame
        requestAnimationFrame(renderInterpolatedPositions);
      }
      
      // Actually render position to 2D map (separated from interpolation logic)
      function renderActual2DPositionSimple(cs, pos) {
        // Validate position coordinates
        if (!pos || !Array.isArray(pos) || pos.length !== 2 ||
            !isFiniteNum(pos[0]) || !isFiniteNum(pos[1]) ||
            Math.abs(pos[0]) > 90 || Math.abs(pos[1]) > 180) {
          return; // Skip invalid positions
        }
        
        // Ensure live map is initialized
        if (!liveMap) {
          ensureLiveMap().then(() => renderActual2DPositionSimple(cs, pos));
          return;
        }
        
        // Throttle actual rendering
        const now = Date.now();
        const lastUpdate = lastUpdateTime.get(cs) || 0;
        if (now - lastUpdate < UPDATE_THROTTLE_MS) {
          return;
        }
        lastUpdateTime.set(cs, now);
        
        // Get or create marker
        let marker = liveMarkers.get(cs);
        if (!marker) {
          const icon = L.divIcon({
            className: 'ac-icon',
            html: `<div class="ac-rot" style="background: rgba(255,255,255,0.9); border: 2px solid ${getCallsignColor(cs)}; border-radius: 50%; padding: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
              <div style="width: 32px; height: 32px; background: ${getCallsignColor(cs)}; mask: url('/static/icons/airplane.svg') no-repeat center; -webkit-mask: url('/static/icons/airplane.svg') no-repeat center; mask-size: 24px 24px; -webkit-mask-size: 24px 24px;"></div>
            </div>`,
            iconSize: [40, 40],
            iconAnchor: [20, 20]
          });
          
          marker = L.marker(pos, { icon: icon })
            .addTo(liveMap)
            .bindTooltip(`<div class="aircraft-telemetry-tooltip"><div class="aircraft-callsign">${cs}</div></div>`, {
              permanent: false,
              direction: 'top',
              offset: [0, -10],
              className: 'aircraft-tooltip'
            });
          
          liveMarkers.set(cs, marker);
        } else {
          // Update existing marker position
          marker.setLatLng(pos);
        }
        
        // Update flight path
        let pathLine = livePaths.get(cs);
        if (!pathLine) {
          pathLine = L.polyline([pos], {
            color: getCallsignColor(cs),
            weight: 2,
            opacity: 0.8
          });
          if (pathLine) {
            pathLine.addTo(liveMap);
            livePaths.set(cs, pathLine);
          }
        } else {
          // Add new position to path
          const currentPath = pathLine.getLatLngs();
          currentPath.push(pos);
          
          // Keep only last 500 points for performance
          if (currentPath.length > 500) {
            currentPath.shift();
          }
          
          pathLine.setLatLngs(currentPath);
        }
      }
      
      // Actually render position to 3D globe using Globe3D component
      async function renderActual3DPosition(cs, pos) {
        if (!liveGlobeViewer) return;
        
        // Throttle actual 3D rendering
        const now = Date.now();
        const lastUpdate = last3DUpdateTime.get(cs) || 0;
        if (now - lastUpdate < UPDATE_3D_THROTTLE_MS) {
          return;
        }
        last3DUpdateTime.set(cs, now);
        
        const C = Cesium;
        
        // Convert Cesium position back to lat/lon/alt for Globe3D API
        const cartographic = C.Cartographic.fromCartesian(pos);
        const position = {
          lat: C.Math.toDegrees(cartographic.latitude),
          lon: C.Math.toDegrees(cartographic.longitude),
          alt: cartographic.height
        };
        
        // Check if aircraft exists in Globe3D component
        if (!liveGlobe.entities.has(cs)) {
          // Create new aircraft using Globe3D API
          await liveGlobe.addAircraft(cs, position, {}, cs);
          // Keep backward compatibility references
          live3DEntities.set(cs, liveGlobe.entities.get(cs));
          live3DTrackData.set(cs, [{position: pos, timestamp: now}]);
          
          // Create debug axes if debug mode is active
          if (typeof debugAxes !== 'undefined' && debugAxes && typeof create3DDebugAxes === 'function') {
            const entity = liveGlobe.entities.get(cs);
            try { create3DDebugAxes(cs, entity.position, entity.orientation); } catch(_) {}
          }
        } else {
          // Update existing aircraft using Globe3D API
          await liveGlobe.updateAircraft(cs, position);
          
          // Update flight trail data for backward compatibility
          const trackData = live3DTrackData.get(cs);
          if (trackData) {
            trackData.push({position: pos, timestamp: now});
            // Keep only last 500 points for performance
            if (trackData.length > 500) {
              trackData.shift();
            }
          }
        }
        
        // Request render for 3D (Globe3D handles this internally but keep for compatibility)
        liveGlobeViewer.scene.requestRender();
      }
      
      // Start the interpolation rendering loop
      if (!window.__interpolationStarted) {
        window.__interpolationStarted = true;
        requestAnimationFrame(renderInterpolatedPositions);
      }
      
      // Update 3D position - now with smooth interpolation buffering
      function updateLive3DPosition(sample) {
        if (!liveGlobeViewer || !sample) return;
        
        const cs = sample.callsign || 'ACFT';
        const now = Date.now();
        const C = Cesium;
        const pos = C.Cartesian3.fromDegrees(sample.lon, sample.lat, sample.alt_m || 0);
        
        // Update 3D interpolation buffer instead of direct position
        const currentBuffer = interpolation3DBuffer.get(cs);
        if (!currentBuffer) {
          // First position - start immediately
          interpolation3DBuffer.set(cs, {
            from: pos,
            to: pos, 
            startTime: now,
            duration: 100 // Small duration for first position
          });
        } else {
          // New target position - set up smooth transition
          const timeSinceLastUpdate = now - currentBuffer.startTime;
          const duration = Math.max(100, Math.min(400, timeSinceLastUpdate * 0.9)); // Adaptive duration for 300ms delay
          interpolation3DBuffer.set(cs, {
            from: currentBuffer.to, // Start from where we were going
            to: pos,
            startTime: now + INTERPOLATION_DELAY_MS, // 1-second delay for smooth buffering
            duration: duration
          });
        }
      }

      function pruneStale() {
        // Remove stale markers - implement based on timestamp if needed
      }

      // Event listeners
      liveConnectBtn.addEventListener('click', () => {
        ensureLiveMap().then(() => {
          connectWs();
          setTimeout(() => liveMap && liveMap.invalidateSize(), 100);
        });
      });
      
      liveDisconnectBtn.addEventListener('click', disconnectWs);
      
      liveCenterBtn.addEventListener('click', () => {
        if (liveMarkers.size > 0) {
          const positions = Array.from(liveMarkers.values()).map(m => m.getLatLng());
          liveMap.fitBounds(L.latLngBounds(positions));
        }
      });

      // Enhanced unified follow button
      const liveFollowBtn = document.getElementById('live_follow');
      if (liveFollowBtn) {
        liveFollowBtn.addEventListener('click', () => {
          const is2DActive = liveTab2D && liveTab2D.classList.contains('active');
          
          if (is2DActive) {
            liveFollow2D = !liveFollow2D;
            liveFollow3D = false;
            if (!liveFollowCS && liveMarkers.size > 0) {
              liveFollowCS = Array.from(liveMarkers.keys())[0];
            }
          } else {
            liveFollow3D = !liveFollow3D;
            liveFollow2D = false;
            if (!liveFollowCS && live3DEntities.size > 0) {
              liveFollowCS = Array.from(live3DEntities.keys())[0];
            }
          }
          
          updateFollowBtn();
          const mode = is2DActive ? '2D' : '3D';
          const state = (is2DActive ? liveFollow2D : liveFollow3D) ? 'enabled' : 'disabled';
          pushEvent(`Follow ${mode} ${state}`);
        });
      }

      // Channel management functions
      function getCallsignColor(callsign) {
        if (callsignColors.has(callsign)) {
          return callsignColors.get(callsign);
        }
        
        // Simple hash function for deterministic color selection
        let hash = 0;
        for (let i = 0; i < callsign.length; i++) {
          const char = callsign.charCodeAt(i);
          hash = ((hash << 5) - hash) + char;
          hash = hash & hash; // Convert to 32bit integer
        }
        
        // Map hash to color palette
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD'];
        const colorIndex = Math.abs(hash) % colors.length;
        const color = colors[colorIndex];
        
        callsignColors.set(callsign, color);
        saveCallsignColors();
        return color;
      }

      // Recent channel management
      function getRecentChannels() {
        try {
          const recent = JSON.parse(localStorage.getItem('ftpro_recent_channels') || '[]');
          return recent.slice(0, 10); // Keep only last 10
        } catch (e) {
          return [];
        }
      }
      
      function saveRecentChannel(channelName, key) {
        const recent = getRecentChannels();
        recent.unshift({ name: channelName, key: key, lastUsed: Date.now() });
        localStorage.setItem('ftpro_recent_channels', JSON.stringify(recent.slice(0, 10)));
        updateRecentChannelsUI();
      }
      
      function updateRecentChannelsUI() {
        // Implementation would go here
      }
      
      // Override connectWs to save recent channels
      const originalConnectWs = connectWs;
      connectWs = function() {
        const channel = liveChInput?.value || 'default';
        const key = liveKey?.value || '';
        saveRecentChannel(channel, key);
        return originalConnectWs();
      };

      // Auto-connect on page load
      document.addEventListener('DOMContentLoaded', () => {
        // Load persisted callsign colors
        loadCallsignColors();
        
        setTimeout(() => {
          if (!liveConnected && liveChInput.value !== '') {
            ensureLiveMap().then(() => connectWs());
          }
        }, 1000);
      });
      
    </script>
  </body>
</html>
"""
