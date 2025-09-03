from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import gpxpy
import simplekml
from io import StringIO, BytesIO
from typing import Optional
from uuid import uuid4
import os


app = FastAPI(title="GPX → KML Converter", version="1.0.0")

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
    <title>GPX → KML Converter</title>
    <script>window.CESIUM_ION_TOKEN='__CESIUM_ION_TOKEN__';</script>
    <script>
      // Guard for extensions expecting a geoLocationStorage global
      try { if (typeof window.geoLocationStorage === 'undefined') window.geoLocationStorage = {}; } catch (_) {}
      // Defensive shims for third-party scripts or extensions expecting Node-like globals
      try { if (typeof window.global === 'undefined') window.global = window; } catch (_) {}
      try { if (typeof window.process === 'undefined') window.process = { env: {} }; } catch (_) {}
    </script>
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ctext y='14' font-size='14'%3E%F0%9F%97%BA%EF%B8%8F%3C/text%3E%3C/svg%3E">
    <style>
      :root { color-scheme: light dark; }
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; }
      .container { max-width: 1200px; margin: 24px auto; padding: 24px; }
      .card { border: 1px solid #ddd; border-radius: 12px; padding: 24px; }
      h1 { margin-top: 0; }
      label { display: block; margin: 12px 0 4px; font-weight: 600; }
      .lbl { display: flex; align-items: center; gap: 6px; }
      .info { display:inline-flex; width: 14px; height: 14px; align-items:center; justify-content:center; border-radius:50%; background:#e5e7eb; color:#111; font-size:10px; cursor:help; }
      input[type="file"], input[type="text"], input[type="number"], select {
        width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc;
      }
      .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .actions { margin-top: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      button { background: #2563eb; color: white; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
      button:disabled { opacity: 0.6; cursor: default; }
      .note { font-size: 12px; color: #666; }
      .help { font-size: 12px; color: #666; margin-top: 6px; }
      .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
      .preview { margin-top: 16px; height: 65vh; min-height: 420px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; display:flex; flex-direction:column; position: relative; }
      .preview-tabs { display: flex; gap: 8px; padding: 8px; border-bottom: 1px solid #eee; align-items: center; flex-wrap: wrap; }
      .tabbtn { background: #f3f4f6; color: #111; border: 1px solid #ddd; padding: 4px 8px; border-radius: 6px; cursor: pointer; font-weight: 600; }
      .tabbtn.active { background: #2563eb; color: #fff; border-color: #2563eb; }
      #map, #globe { flex: 1 1 auto; height: calc(100% - 40px); width: 100%; display: none; }
      #map.active, #globe.active { display: block; }
      .hud { position: absolute; right: 8px; bottom: 8px; background: rgba(0,0,0,0.55); color:#fff; padding:6px 8px; border-radius:6px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; line-height: 1.2; }
      /* Telemetry panel (SpaceX-style) */
      .telemetry { position:absolute; left:8px; bottom:8px; right:auto; display:flex; gap:8px; flex-wrap:wrap; align-items:flex-end; }
      .telemetry .tile { background: rgba(0,0,0,0.65); color:#fff; padding:8px 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.12); min-width:120px; }
      .telemetry .tile .label { font-size:11px; opacity:0.8; letter-spacing: .06em; text-transform:uppercase; }
      .telemetry .tile .value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:18px; font-weight:700; }
      .telemetry .tile .unit { font-size:11px; opacity:0.85; margin-left:4px; }
      .telemetry .wide { min-width: 180px; }
      .inline { display:flex; align-items:center; gap:6px; }
      .small { font-size: 12px; }
      footer { margin-top: 24px; font-size: 12px; color: #666; text-align: center; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h1>GPX → KML Converter</h1>
        <p>Upload a <strong>.gpx</strong> file and download a <strong>.kml</strong>.</p>
        <form id="form">
          <label for="file">GPX file</label>
          <input id="file" name="file" type="file" accept=".gpx,application/gpx+xml" required />

          <div class="row">
            <div>
              <label for="plan_file" class="lbl">Flight plan (optional) <span class="info" title="Little Navmap .lnmpln, MSFS/FSX .pln, or GPX route.">i</span></label>
              <input id="plan_file" name="plan_file" type="file" accept=".lnmpln,.pln,.gpx,application/xml,application/gpx+xml" />
            </div>
            <div>
              <label for="airspace_file" class="lbl">Airspaces (optional) <span class="info" title="OpenAir .txt, KML, or GeoJSON.">i</span></label>
              <input id="airspace_file" name="airspace_file" type="file" accept=".txt,.kml,.json,.geojson,text/plain,application/json,application/vnd.google-earth.kml+xml" />
            </div>
          </div>
          <div class="row">
            <div>
              <label for="airports_file" class="lbl">Airports (optional) <span class="info" title="CSV export from LNM or GPX waypoints.">i</span></label>
              <input id="airports_file" name="airports_file" type="file" accept=".csv,.gpx,text/csv,application/gpx+xml" />
            </div>
            <div>
              <label class="lbl">Layer toggles <span class="info" title="Control overlay visibility and filters.">i</span></label>
              <div class="inline small" style="gap:12px; flex-wrap:wrap;">
                <label><input type="checkbox" id="show_plan" checked /> Plan</label>
                <label><input type="checkbox" id="show_airspaces" /> Airspaces</label>
                <label><input type="checkbox" id="show_airports" /> Airports</label>
                <span class="inline small"><label class="note">ASP Alt</label> <input id="asp_alt" type="number" step="100" value="1500" style="width:90px;" /><span class="note">m MSL</span></span>
              </div>
            </div>
          </div>

          <div class="row">
            <div>
              <label for="name" class="lbl">Line name <span class="info" title="Name to show in the exported KML.">i</span></label>
              <input id="name" name="name" type="text" value="Flight Trail 3D" />
            </div>
            <div>
              <label for="color" class="lbl">Line color <span class="info" title="Pick a color for the path in preview and KML.">i</span></label>
              <input id="color" name="color" type="color" value="#ff0000" />
            </div>
          </div>

          <div class="row">
            <div>
              <label for="width" class="lbl">Line width <span class="info" title="Path stroke width in pixels.">i</span></label>
              <input id="width" name="width" type="number" min="1" max="10" value="3" />
            </div>
            <div>
              <label for="altitude_mode" class="lbl">Altitude mode <span class="info" title="absolute: use GPX elevation; relativeToGround: height above terrain; clampToGround: drape on terrain.">i</span></label>
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
              <label for="extrude" class="lbl">Extrude <span class="info" title="Draw vertical lines from the path to the ground to show height.">i</span></label>
              <select id="extrude" name="extrude">
                <option value="0" selected>no</option>
                <option value="1">yes</option>
              </select>
              <div class="help" style="display:none"></div>
            </div>
            <div>
              <label for="include_waypoints" class="lbl">Waypoints <span class="info" title="Include GPX waypoints as labeled placemarks.">i</span></label>
              <select id="include_waypoints" name="include_waypoints">
                <option value="1" selected>include</option>
                <option value="0">exclude</option>
              </select>
              <div class="help" style="display:none"></div>
            </div>
            <div>
              <label for="color_by" class="lbl">Coloring <span class="info" title="Color by a single color or speed (when timestamps are present).">i</span></label>
              <select id="color_by" name="color_by">
                <option value="solid" selected>solid</option>
                <option value="speed">speed (blue→red)</option>
              </select>
              <div class="help" style="display:none"></div>
            </div>
          </div>

          <div class="actions">
            <button id="convert" type="submit">Convert to KML</button>
            <button id="open_ge" type="button">Open in Google Earth</button>
            <input id="include_tour" name="include_tour" type="hidden" value="0" />
            <span class="note" id="status"></span>
          </div>
        </form>

          <div class="preview" id="preview">
            <div class="preview-tabs">
              <button id="tab-2d" class="tabbtn active" type="button">2D Map</button>
              <button id="tab-3d" class="tabbtn" type="button">3D Globe (beta)</button>
              <span class="inline small" style="margin-left:8px;">
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
              <button id="fullbtn" class="tabbtn" type="button">Full</button>
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
      <footer>
        Powered by FastAPI, GPXPy, and simplekml.
      </footer>
    </div>

    <script>
      // Persist settings in localStorage
      const PERSIST_KEYS = [
        'name','color','width','altitude_mode','extrude','include_waypoints','color_by','include_tour'
      ];
      function saveSettings() {
        PERSIST_KEYS.forEach(k => {
          const el = document.getElementById(k);
          if (el) localStorage.setItem('navmap_'+k, el.value);
        });
      }
      function loadSettings() {
        PERSIST_KEYS.forEach(k => {
          const v = localStorage.getItem('navmap_'+k);
          if (v !== null) {
            const el = document.getElementById(k);
            if (el) el.value = v;
          }
        });
      }

      const form = document.getElementById('form');
      const statusEl = document.getElementById('status');
      const btn = document.getElementById('convert');
      form.addEventListener('change', saveSettings);
      document.addEventListener('DOMContentLoaded', loadSettings);

      // Drag & drop support on the card
      const card = document.querySelector('.card');
      const fileInput = document.getElementById('file');
      ;['dragenter','dragover'].forEach(evt => card.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation(); card.style.borderColor = '#2563eb';
      }));
      ;['dragleave','drop'].forEach(evt => card.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation(); card.style.borderColor = '#ddd';
      }));
      card.addEventListener('drop', e => {
        const dt = e.dataTransfer; if (!dt || !dt.files || !dt.files.length) return;
        const f = dt.files[0];
        if (!f.name.endsWith('.gpx')) { statusEl.textContent = 'Drop a .gpx file'; return; }
        fileInput.files = dt.files;
        previewFile(f);
      });

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
      // Optional country lookup from static JSON (coarse bboxes)
      let countryIndex = null;
      fetch('/static/country-bboxes.json').then(r=>r.ok?r.json():null).then(j=>{ countryIndex=j; }).catch(()=>{});
      function ensureMap() {
        if (map) return Promise.resolve();
        if (leafletReady) return leafletReady;
        // Load Leaflet lazily and wait until ready
        leafletReady = new Promise((resolve, reject) => {
          const css = document.createElement('link');
          css.rel = 'stylesheet';
          css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
          document.head.appendChild(css);
          const script = document.createElement('script');
          script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
          script.onload = () => { try { initMap(); resolve(); } catch (e) { reject(e); } };
          script.onerror = reject;
          document.body.appendChild(script);
        });
        return leafletReady;
      }
      function ensureGlobe(cb) {
        if (globeViewer) { cb && cb(); return; }
        const css = document.createElement('link');
        css.rel = 'stylesheet';
        css.href = 'https://unpkg.com/cesium/Build/Cesium/Widgets/widgets.css';
        document.head.appendChild(css);
        const script = document.createElement('script');
        // Ensure Cesium can load its workers/assets from the CDN
        window.CESIUM_BASE_URL = 'https://unpkg.com/cesium/Build/Cesium/';
        script.src = window.CESIUM_BASE_URL + 'Cesium.js';
        script.crossOrigin = 'anonymous';
        script.onload = () => { initGlobe().then(() => cb && cb()); };
        document.body.appendChild(script);
      }
      function initMap() {
        if (map) return;
        map = L.map('map');
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19, attribution: '&copy; OpenStreetMap contrib.'
        }).addTo(map);
        trackLayer = L.layerGroup().addTo(map);
        wptLayer = L.layerGroup().addTo(map);
        planLayer = L.layerGroup().addTo(map);
        airspaceLayer = L.layerGroup().addTo(map);
        airportLayer = L.layerGroup().addTo(map);
      }
      async function initGlobe() {
        if (globeViewer) return;
        globeViewer = new Cesium.Viewer('globe', {
          terrainProvider: new Cesium.EllipsoidTerrainProvider(),
          animation: false, timeline: false, baseLayerPicker: false, geocoder: false, sceneModePicker: false,
          infoBox: false, selectionIndicator: false, navigationHelpButton: false, fullscreenButton: false
        });
        // Performance: only render on demand; we call requestRender from the RAF step
        globeViewer.scene.requestRenderMode = true;
        globeViewer.scene.maximumRenderTimeChange = Infinity;
        // Quick FPS overlay (Cesium built-in)
        globeViewer.scene.debugShowFramesPerSecond = true;
        // Detect user input and orbit around target without disabling follow
        const canvas = globeViewer.scene.canvas;
        let drag = null;
        const ctrl = globeViewer.scene.screenSpaceCameraController;
        function configureControls(){
          if (!ctrl) return;
          if (followCam) {
            // In follow mode, our custom handlers drive the camera
            ctrl.enableRotate = false;
            ctrl.enableTranslate = false;
            ctrl.enableZoom = false;
            ctrl.enableTilt = false;
            ctrl.enableLook = false;
          } else {
            // Free mode: map left-drag to pan/translate, middle-drag to rotate/look, wheel to zoom
            ctrl.enableRotate = true;
            ctrl.enableTranslate = true;
            ctrl.enableZoom = true;
            ctrl.enableTilt = true;
            ctrl.enableLook = true;
            try {
              ctrl.translateEventTypes = [Cesium.CameraEventType.LEFT_DRAG];
              ctrl.lookEventTypes = [Cesium.CameraEventType.MIDDLE_DRAG, Cesium.CameraEventType.PINCH];
              ctrl.zoomEventTypes = [Cesium.CameraEventType.WHEEL, Cesium.CameraEventType.PINCH];
              // Optional: right-drag tilt
              ctrl.tiltEventTypes = [Cesium.CameraEventType.RIGHT_DRAG];
            } catch(_) {}
          }
        }
        try { window.__cfgControls = configureControls; } catch(_) {}
        function disableDefaultControls() { if (!ctrl) return; drag && (drag.prev = { rot: ctrl.enableRotate, zoom: ctrl.enableZoom, tilt: ctrl.enableTilt }); ctrl.enableRotate = false; ctrl.enableZoom = false; ctrl.enableTilt = false; }
        function restoreDefaultControls() { if (!ctrl || !drag || !drag.prev) return; ctrl.enableRotate = drag.prev.rot; ctrl.enableZoom = drag.prev.zoom; ctrl.enableTilt = drag.prev.tilt; }
        canvas.addEventListener('pointerdown', (e) => {
          if (!followCam) return; // allow native Cesium controls when not following
          e.preventDefault();
          drag = { x: e.clientX, y: e.clientY, startH: followOffsetHeading, startP: followOffsetPitch, prev: null };
          disableDefaultControls();
        });
        canvas.addEventListener('pointermove', (e) => {
          if (!followCam) return; // let Cesium handle free camera
          if (e.buttons === 0) { restoreDefaultControls(); drag = null; return; }
          e.preventDefault();
          if (!drag) return;
          const dx = e.clientX - drag.x; const dy = e.clientY - drag.y;
          followOffsetHeading = drag.startH + dx * 0.005; // radians per px
          followOffsetPitch = Cesium.Math.clamp(drag.startP + dy * -0.003, Cesium.Math.toRadians(-80), Cesium.Math.toRadians(30));
        });
        window.addEventListener('pointerup', () => { restoreDefaultControls(); drag = null; });
        canvas.addEventListener('wheel', (e) => {
          if (!followCam) return; // free mode uses native zoom
          e.preventDefault();
          // Scale camera distance input directly so UI reflects changes
          const s = Math.pow(1.0 + (e.deltaY>0 ? 0.06 : -0.06), 1);
          const cur = Math.max(10, Number(camDistInput.value) || 800);
          const next = Cesium.Math.clamp(cur * s, 10, 200000);
          camDistInput.value = String(Math.round(next));
        }, { passive: false });
        window.addEventListener('keydown', (e) => {
          if (e.key === 'f' || e.key === 'F') { setFollowCam(!followCam); }
          if (e.key === 'r' || e.key === 'R') { followOffsetHeading = 0; followOffsetPitch = 0; followRangeScale = 1; }
        });
        configureControls();
        // Configure scene: prefer Cesium Ion assets if token is provided
        async function setupCesiumScene() {
          try {
            const tok = (window.CESIUM_ION_TOKEN && window.CESIUM_ION_TOKEN !== '') ? window.CESIUM_ION_TOKEN : (localStorage.getItem('cesium_token') || '');
            if (tok) Cesium.Ion.defaultAccessToken = tok;
            globeViewer.scene.globe = new Cesium.Globe(Cesium.Ellipsoid.WGS84);
            if (tok) {
              globeViewer.terrainProvider = await Cesium.createWorldTerrainAsync();
              const imagery = await Cesium.IonImageryProvider.fromAssetId(3);
              globeViewer.imageryLayers.removeAll();
              globeViewer.imageryLayers.addImageryProvider(imagery);
              try {
                const tileset = await Cesium.createOsmBuildingsAsync();
                globeViewer.scene.primitives.add(tileset);
              } catch (e) { /* ignore buildings if blocked */ }
            } else {
              // Fallback public imagery
              globeViewer.imageryLayers.removeAll();
              globeViewer.imageryLayers.addImageryProvider(new Cesium.OpenStreetMapImageryProvider({ url: 'https://tile.openstreetmap.org/' }));
              globeViewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
            }
          } catch (e) {
            // Final fallback if Ion fails
            try {
              globeViewer.imageryLayers.removeAll();
              globeViewer.imageryLayers.addImageryProvider(new Cesium.OpenStreetMapImageryProvider({ url: 'https://tile.openstreetmap.org/' }));
              globeViewer.terrainProvider = new Cesium.EllipsoidTerrainProvider();
            } catch (_) {}
          }
        }
        await setupCesiumScene();
        // Custom controls: arrows adjust pitch (wheel uses Cesium default zoom)
        // (canvas already defined above for input listeners)
        window.addEventListener('keydown', (e) => {
          if (e.key === 'ArrowUp') { camPitchDeg = Math.max(-89, camPitchDeg - 2); }
          if (e.key === 'ArrowDown') { camPitchDeg = Math.min(-1, camPitchDeg + 2); }
        });
      }
      function setTab(which) {
        if (which === '2d') {
          tab2d.classList.add('active'); tab3d.classList.remove('active');
          mapDiv.classList.add('active'); globeDiv.classList.remove('active');
          ensureMap().then(() => setTimeout(() => map && map.invalidateSize(), 50));
        } else {
          tab3d.classList.add('active'); tab2d.classList.remove('active');
          globeDiv.classList.add('active'); mapDiv.classList.remove('active');
          ensureGlobe(() => globeViewer && globeViewer.resize());
        }
      }
      tab2d.addEventListener('click', () => setTab('2d'));
      tab3d.addEventListener('click', () => setTab('3d'));
      window.addEventListener('resize', () => { if (map) map.invalidateSize(); if (globeViewer) globeViewer.resize(); });
      function parseGpx(text) {
        const dom = new DOMParser().parseFromString(text, 'application/xml');
        const toNum = v => Number.parseFloat(v);
        const isValid = (lat, lon) => Number.isFinite(lat) && Number.isFinite(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180;
        const parseTime = (el) => {
          const t = el.querySelector('time');
          if (!t) return null;
          const d = new Date(t.textContent.trim());
          return isNaN(d.getTime()) ? null : d;
        };
        // Split a sequence when jumping over the antimeridian to avoid world-spanning lines
        function splitByMeridian(points) {
          if (points.length < 2) return points.length ? [points] : [];
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
        // Smart-split into flights based on time gaps, stationary periods, and implausible jumps
        function smartSplit(points) {
          if (points.length < 2) return [];
          const segs = [];
          const GAP_S = 300;       // 5 minutes gap splits
          const STOP_SPEED = 0.5;  // m/s consider stationary below this
          const STOP_MIN_S = 60;   // 1 minute stationary splits
          const MAX_SPEED = 150;   // m/s (540 km/h) split if exceeded
          const BIG_JUMP_M = 250000; // 250 km jump also splits
          const R = 6371000.0;
          const toRad = (x) => x * Math.PI / 180;
          function haversine(p1, p2) {
            const lat1 = toRad(p1.lat), lon1 = toRad(p1.lon);
            const lat2 = toRad(p2.lat), lon2 = toRad(p2.lon);
            const dlat = lat2 - lat1, dlon = lon2 - lon1;
            const a = Math.sin(dlat/2)**2 + Math.cos(lat1)*Math.cos(lat2)*Math.sin(dlon/2)**2;
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            return R * c;
          }
          let cur = [points[0]];
          let stopAccum = 0;
          for (let i=1;i<points.length;i++){
            const a = points[i-1], b = points[i];
            const dt = (a.time && b.time) ? Math.max(0, (b.time - a.time) / 1000) : null;
            const dist = haversine(a, b);
            const spd = dt && dt > 0 ? dist / dt : null;
            let split = false; let reason = '';
            if (dt && dt > GAP_S) { split = true; reason = `gap>${GAP_S}s (dt=${dt.toFixed(1)}s)`; }
            if (!split && spd && spd > MAX_SPEED) { split = true; reason = `speed>${MAX_SPEED}m/s (v=${spd.toFixed(1)}m/s)`; }
            if (!split && !dt && dist > BIG_JUMP_M) { split = true; reason = `jump>${BIG_JUMP_M}m without time (d=${Math.round(dist)}m)`; }
            if (spd !== null && spd < STOP_SPEED && dt) {
              stopAccum += dt;
              if (stopAccum >= STOP_MIN_S && cur.length > 1) {
                split = true; reason = `stationary>${STOP_MIN_S}s (v<${STOP_SPEED}m/s, accum=${Math.round(stopAccum)}s)`;
              }
            } else {
              stopAccum = 0;
            }
            if (split) {
              console.info(`[split] rule:${reason} at leg ${i}/${points.length-1}`);
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
        // Tracks and segments
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
                  time: parseTime(tp)
                })).filter(p => isValid(p.lat, p.lon));
                splitByMeridian(pts).forEach(s => {
                  const subs = smartSplit(s);
                  subs.forEach(ss => segs.push(ss));
                });
              });
            } else {
              const pts = Array.from(trk.querySelectorAll('trkpt')).map(tp => ({
                lat: toNum(tp.getAttribute('lat')),
                lon: toNum(tp.getAttribute('lon')),
                ele: (function(){ const _e = tp.querySelector('ele'); return toNum((_e && _e.textContent) || '0') || 0; })(),
                time: parseTime(tp)
              })).filter(p => isValid(p.lat, p.lon));
              splitByMeridian(pts).forEach(s => {
                const subs = smartSplit(s);
                subs.forEach(ss => segs.push(ss));
              });
            }
            // build separate track entries per segment (flight)
            const R = 6371000.0;
            const toRad = (x) => x * Math.PI / 180;
            function haversine(p1, p2) {
              const lat1 = toRad(p1.lat), lon1 = toRad(p1.lon);
              const lat2 = toRad(p2.lat), lon2 = toRad(p2.lon);
              const dlat = lat2 - lat1, dlon = lon2 - lon1;
              const a = Math.sin(dlat/2)**2 + Math.cos(lat1)*Math.cos(lat2)*Math.sin(dlon/2)**2;
              const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
              return R * c;
            }
            let flightIdx = 1;
            for (const seg of segs) {
              if (!seg || seg.length < 2) continue;
              // Filter out tiny/non-moving segments (e.g., 6-pt stationary tails)
              const segDuration = (() => {
                const t0 = seg[0].time, t1 = seg[seg.length-1].time;
                return (t0 && t1) ? Math.max(0, (t1 - t0) / 1000) : null;
              })();
              const segDist = (() => {
                let d=0; for (let i=0;i<seg.length-1;i++) d += haversine(seg[i], seg[i+1]); return d;
              })();
              const minPts = 8, minDist = 200; // meters
              const avgSpd = (segDuration && segDuration>0) ? (segDist/segDuration) : null;
              const isStationary = (avgSpd !== null && avgSpd < 0.5 && (segDuration||0) >= 60);
              if (seg.length < minPts || segDist < minDist || isStationary) {
                console.info(`[filter] drop seg: pts=${seg.length} dist=${Math.round(segDist)}m dur=${segDuration!==null?Math.round(segDuration)+'s':'n/a'} avg=${avgSpd!==null?avgSpd.toFixed(2)+'m/s':'n/a'}`);
                continue;
              }
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
          // Route fallback
          const rpts = Array.from(dom.querySelectorAll('rtept')).map(tp => ({
            lat: toNum(tp.getAttribute('lat')),
            lon: toNum(tp.getAttribute('lon')),
            ele: (function(){ const _e = tp.querySelector('ele'); return toNum((_e && _e.textContent) || '0') || 0; })(),
            time: parseTime(tp)
          })).filter(p => isValid(p.lat, p.lon));
          const segs = [];
          splitByMeridian(rpts).forEach(s => {
            const subs = smartSplit(s);
            subs.forEach(ss => segs.push(ss));
          });
          let flightIdx = 1;
          for (const seg of segs) {
            if (!seg || seg.length < 2) continue;
            const segDuration = (() => {
              const t0 = seg[0].time, t1 = seg[seg.length-1].time;
              return (t0 && t1) ? Math.max(0, (t1 - t0) / 1000) : null;
            })();
            const segDist = (() => { let d=0; for (let i=0;i<seg.length-1;i++) d += haversine(seg[i], seg[i+1]); return d; })();
            const minPts = 8, minDist = 200; // meters
            const avgSpd = (segDuration && segDuration>0) ? (segDist/segDuration) : null;
            const isStationary = (avgSpd !== null && avgSpd < 0.5 && (segDuration||0) >= 60);
            if (seg.length < minPts || segDist < minDist || isStationary) {
              console.info(`[filter] drop seg: pts=${seg.length} dist=${Math.round(segDist)}m dur=${segDuration!==null?Math.round(segDuration)+'s':'n/a'} avg=${avgSpd!==null?avgSpd.toFixed(2)+'m/s':'n/a'}`);
              continue;
            }
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
      async function previewFile(file, preserveSelection=false) {
        const text = await file.text();
        const { tracks, waypoints } = parseGpx(text);
        lastFileBlob = file;
        // Populate track selector
        console.info('[tracks] parsed:', tracks.length, 'entries');
        tracks.forEach((t,i)=>{
          const pts = t.segments.reduce((n,s)=>n+s.length,0);
          console.info(`[tracks] ${i+1}/${tracks.length} '${t.name}' segments:${t.segments.length} points:${pts}`);
          t.segments.forEach((s,si)=>console.debug(`  seg#${si+1} pts:${s.length}`));
        });
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
        function pickTrack() {
          const sel = trackSelect.value;
          if (sel === 'auto') {
            // Pick track with most points
            return tracks.reduce((a,b) => (a.segments.reduce((n,s)=>n+s.length,0) >= b.segments.reduce((n,s)=>n+s.length,0)) ? a : b, tracks[0] || {segments:[], legs:[]});
          }
          const t = tracks[Number(sel)];
          return t || {segments:[], legs:[]};
        }
        const chosen = pickTrack();
        console.info('[tracks] chosen:', chosen && chosen.name ? chosen.name : 'none');
        // 2D
        await ensureMap();
        const includeWpts = document.getElementById('include_waypoints').value === '1';
        const color = document.getElementById('color').value || '#ff0000';
        const width = parseInt(document.getElementById('width').value || '3', 10);
        const group = [];
        if (trackLayer) trackLayer.clearLayers();
        if (wptLayer) wptLayer.clearLayers();
        const colorBy = document.getElementById('color_by').value || 'solid';
        if (chosen.segments.length) {
          if (colorBy === 'speed') {
            // Color each leg by speed gradient
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
              const pl = L.polyline([[a.lat,a.lon],[b.lat,b.lon]], { color: speedColor(speed), weight: width }).addTo(trackLayer);
              group.push(pl);
            });
          } else {
            chosen.segments.forEach(seg => {
              if (seg.length > 1) {
                const latlngs = seg.map(p => [p.lat, p.lon]);
                const pl = L.polyline(latlngs, { color, weight: width }).addTo(trackLayer);
                group.push(pl);
              }
            });
          }
        }
        if (includeWpts && waypoints.length) {
          waypoints.forEach(w => {
            const m = L.marker([w.lat, w.lon]).addTo(wptLayer);
            m.bindPopup(w.name);
            group.push(m);
          });
        }
        if (group.length && map) {
          const fg = L.featureGroup(group);
          map.fitBounds(fg.getBounds().pad(0.2));
        }
        // 3D
        ensureGlobe(() => {
          // Clear previous
          if (globeTracks && globeTracks.length) {
            globeTracks.forEach(ent => globeViewer.entities.remove(ent));
            globeTracks = [];
          }
          // Build entities per segment
          let longestSeg = [];
          const toCesiumColor = (c) => Cesium.Color.fromCssColorString(c);
          const colorBy = document.getElementById('color_by').value || 'solid';
          if (colorBy === 'speed' && chosen.legs && chosen.legs.length) {
            // Per-leg speed color gradient in 3D as well
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
              const positions = [Cesium.Cartesian3.fromDegrees(a.lon, a.lat, a.ele||0), Cesium.Cartesian3.fromDegrees(b.lon, b.lat, b.ele||0)];
              const ent = globeViewer.entities.add({ polyline: { positions, width: width, material: toCesiumColor(speedColor(speed)), clampToGround: false } });
              globeTracks.push(ent);
            });
            // choose a reasonable base segment for fly path
            longestSeg = chosen.segments.reduce((acc, seg) => seg.length > acc.length ? seg : acc, []);
          } else {
            // Solid color per segment
            chosen.segments.forEach(seg => {
              if (seg.length > 1) {
                const positions = seg.map(p => Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.ele || 0));
                const ent = globeViewer.entities.add({ polyline: { positions, width: width, material: toCesiumColor(color), clampToGround: false } });
                globeTracks.push(ent);
                if (seg.length > longestSeg.length) longestSeg = seg;
              }
            });
          }
          // Do not auto-zoom the globe in preview; keep normal map view
          globePositions = longestSeg.map(p => ({lat: p.lat, lon: p.lon, ele: p.ele||0, time: p.time||null}));
        // Build high-resolution fly path for smoothness
        function buildFlyPath(points) {
          const out = [];
          if (points.length < 2) return out;
          const toRad = (x)=>x*Math.PI/180, toDeg=(x)=>x*180/Math.PI;
            // approximate spacing ~150m for smoother playback
          const R=6371000.0; const hav = (a,b)=>{const la1=toRad(a.lat),lo1=toRad(a.lon),la2=toRad(b.lat),lo2=toRad(b.lon);const dlat=la2-la1,dlon=lo2-lo1;const A=Math.sin(dlat/2)**2+Math.cos(la1)*Math.cos(la2)*Math.sin(dlon/2)**2;return 2*R*Math.asin(Math.sqrt(A));};
          for(let i=0;i<points.length-1;i++){
            const a=points[i], b=points[i+1];
            const dist = hav(a,b);
            const steps = Math.max(1, Math.min(30, Math.round(dist/150)));
            for(let s=0;s<steps;s++){
              const t=s/steps;
              // Interpolate time when both endpoints have valid time
              const time = (a.time instanceof Date && b.time instanceof Date && b.time > a.time) ? new Date(a.time.getTime() + t * (b.time - a.time)) : null;
              out.push({
                lat: a.lat + (b.lat-a.lat)*t,
                lon: a.lon + (b.lon-a.lon)*t,
                ele: a.ele + (b.ele-a.ele)*t,
                time: time
              });
            }
          }
            out.push(points[points.length-1]);
            return out;
          }
          globePositions = buildFlyPath(globePositions);
          // Setup flythrough UI
          const canFly = globePositions.length >= 2;
          playBtn.disabled = !canFly;
          pauseBtn.disabled = !canFly;
          stopBtn.disabled = !canFly;
          flyRange.disabled = !canFly;
          flySpeedInput.disabled = !canFly;
          document.getElementById('preset_chase').disabled = !canFly;
          document.getElementById('preset_front').disabled = !canFly;
          document.getElementById('preset_left').disabled = !canFly;
          document.getElementById('preset_right').disabled = !canFly;
          document.getElementById('preset_top').disabled = !canFly;
          flyIndex = 0; flyRange.value = 0; flyRange.max = Math.max(1, globePositions.length - 1);
          // Prepare glider entity placeholder (actual model gets attached when starting flythrough)
          if (gliderEnt) { globeViewer.entities.remove(gliderEnt); gliderEnt = null; }
          // Focus camera on start of selected flight
          try {
            const p0 = globePositions[0];
            const pos0 = Cesium.Cartesian3.fromDegrees(p0.lon, p0.lat, p0.ele || 0);
            const initRange = Math.max(10, Number(camDistInput.value) || 800);
            globeViewer.camera.lookAt(pos0, new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-20), initRange));
          } catch (_) {}
          homeBtn.disabled = false; followBtn.disabled = false; updateFollowBtn();
          // Update overlays after globe is ready
          renderOverlays2D(); ensureGlobe(()=>renderOverlays3D());
        });
      }

      // Disable color input when coloring by speed
      const colorByEl = document.getElementById('color_by');
      const colorEl = document.getElementById('color');
      function syncColorState(){
        const speed = (colorByEl.value === 'speed');
        colorEl.disabled = speed;
        const wrap = colorEl.closest('div'); if (wrap) wrap.style.display = speed ? 'none' : 'block';
      }
      colorByEl.addEventListener('change', syncColorState);
      document.addEventListener('DOMContentLoaded', syncColorState);

      function bearingDeg(lat1, lon1, lat2, lon2) {
        const toRad = x => x * Math.PI / 180; const toDeg = x => x * 180 / Math.PI;
        const dLon = toRad(lon2 - lon1);
        const y = Math.sin(dLon) * Math.cos(toRad(lat2));
        const x = Math.cos(toRad(lat1))*Math.sin(toRad(lat2)) - Math.sin(toRad(lat1))*Math.cos(toRad(lat2))*Math.cos(dLon);
        let brng = Math.atan2(y, x);
        brng = (toDeg(brng) + 360) % 360; return brng;
      }
      // Flythrough with Cesium SampledPosition + RAF-driven clock
      let flyPositionProperty = null;
      let flyStart = null, flyStop = null;
      let flyDurationSec = 0, flyBaseSec = 0, flyLastMs = 0;
      let mediaRecorder = null, mediaChunks = [], mediaStream = null;
      let hudEl = null, hudLastTs = 0, hudFrames = 0, hudFps = 0;
      function buildFlyFromTrack(points) {
        // Build SampledPositionProperty; if no timestamps, synthesize based on distance and nominal speed
        const samples = new Cesium.SampledPositionProperty();
        samples.setInterpolationOptions({ interpolationDegree: 2, interpolationAlgorithm: Cesium.HermitePolynomialApproximation });
        // Check if we have increasing timestamps on any leg
        let hasTime = false;
        for (let i=0;i<points.length-1;i++) {
          const a = points[i], b = points[i+1];
          if (a.time instanceof Date && b.time instanceof Date && b.time > a.time) { hasTime = true; break; }
        }
        const toRad = (x)=>x*Math.PI/180; const R=6371000.0;
        const distm = (a,b)=>{const la1=toRad(a.lat),lo1=toRad(a.lon),la2=toRad(b.lat),lo2=toRad(b.lon);const dlat=la2-la1,dlon=lo2-lo1;const A=Math.sin(dlat/2)**2+Math.cos(la1)*Math.cos(la2)*Math.sin(dlon/2)**2;return 2*R*Math.asin(Math.sqrt(A));};
        let first = null, last = null;
        const dates = [];
        if (hasTime) {
          for (let i=0;i<points.length;i++) {
            let d = points[i].time instanceof Date ? points[i].time : (first || new Date());
            if (!first) first = d; last = d;
            dates.push(d);
            samples.addSample(Cesium.JulianDate.fromDate(d), Cesium.Cartesian3.fromDegrees(points[i].lon, points[i].lat, (points[i].ele||0)));
          }
        } else {
          // Synthesize times proportional to distance using nominal speed (m/s)
          const nominal = 25; // ~90 km/h default; Speed input multiplies this
          let t = new Date();
          first = t;
          dates.push(t);
          samples.addSample(Cesium.JulianDate.fromDate(t), Cesium.Cartesian3.fromDegrees(points[0].lon, points[0].lat, (points[0].ele||0)));
          for (let i=1;i<points.length;i++) {
            const dsec = distm(points[i-1], points[i]) / Math.max(0.1, nominal);
            t = new Date(t.getTime() + dsec*1000);
            dates.push(t);
            samples.addSample(Cesium.JulianDate.fromDate(t), Cesium.Cartesian3.fromDegrees(points[i].lon, points[i].lat, (points[i].ele||0)));
          }
          last = t;
        }
        return { samples, firstTime: first, lastTime: last, dates };
      }
      function startFlythrough() {
        if (!globeViewer || globePositions.length < 2) return;
        const built = buildFlyFromTrack(globePositions);
        flyPositionProperty = built.samples;
        const flyDates = built.dates || [];
        if (gliderEnt) { globeViewer.entities.remove(gliderEnt); gliderEnt = null; }
        // initial camera focus on start
        try {
          const startPos = Cesium.Cartesian3.fromDegrees(globePositions[0].lon, globePositions[0].lat, globePositions[0].ele || 0);
          const startCarto = Cesium.Cartographic.fromDegrees(globePositions[0].lon, globePositions[0].lat);
          const initRange = Math.max(10, Number(camDistInput.value) || 800);
          globeViewer.camera.lookAt(startPos, new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-20), initRange));
        } catch (e) {}
        // Start with a point, then upgrade to model if available
        gliderEnt = globeViewer.entities.add({
          position: flyPositionProperty,
          point: { pixelSize: 10, color: Cesium.Color.YELLOW, outlineColor: Cesium.Color.BLACK, outlineWidth: 2 },
          path: { width: 2, material: Cesium.Color.ORANGE.withAlpha(0.35), leadTime: 0, trailTime: 120 }
        });
        // Build attitude-based orientation (heading/pitch/roll → quaternion)
        try {
          const orient = new Cesium.SampledProperty(Cesium.Quaternion);
          const toRad = (x)=>x*Math.PI/180;
          // Apply a +90° yaw so model nose aligns with motion (front was pointing right)
          const yawCorr = Cesium.Quaternion.fromAxisAngle(Cesium.Cartesian3.UNIT_Z, Cesium.Math.toRadians(90));
          for (let i=0;i<globePositions.length;i++) {
            const d = flyDates[i] || new Date(built.firstTime.getTime() + (i * (Cesium.JulianDate.secondsDifference(Cesium.JulianDate.fromDate(built.lastTime), Cesium.JulianDate.fromDate(built.firstTime))) / Math.max(1, globePositions.length-1)) * 1000);
            const p = globePositions[i];
            const prev = globePositions[Math.max(0, i-1)];
            const next = globePositions[Math.min(globePositions.length-1, i+1)];
            const cartoPrev = Cesium.Cartographic.fromDegrees(prev.lon, prev.lat, prev.ele||0);
            const cartoNow = Cesium.Cartographic.fromDegrees(p.lon, p.lat, p.ele||0);
            const cartoNext = Cesium.Cartographic.fromDegrees(next.lon, next.lat, next.ele||0);
            const geod1 = new Cesium.EllipsoidGeodesic(cartoPrev, cartoNow);
            const geod2 = new Cesium.EllipsoidGeodesic(cartoNow, cartoNext);
            const dist1 = geod1.surfaceDistance || 0, dist2 = geod2.surfaceDistance || 0;
            const dt1 = i>0 ? Math.max(1e-3, (flyDates[i] - flyDates[i-1]) / 1000) : 1;
            const dt2 = i<flyDates.length-1 ? Math.max(1e-3, (flyDates[i+1] - flyDates[i]) / 1000) : 1;
            const gs1 = dist1 / dt1, gs2 = dist2 / dt2; // m/s
            const vs1 = ((p.ele||0) - (prev.ele||0)) / dt1;
            const vs2 = ((next.ele||0) - (p.ele||0)) / dt2;
            const gs = (gs1 + gs2) * 0.5;
            const vs = (vs1 + vs2) * 0.5;
            // heading from now→next
            const heading = Math.atan2(
              Math.sin(cartoNext.longitude - cartoNow.longitude) * Math.cos(cartoNext.latitude),
              Math.cos(cartoNow.latitude)*Math.sin(cartoNext.latitude) - Math.sin(cartoNow.latitude)*Math.cos(cartoNext.latitude)*Math.cos(cartoNext.longitude - cartoNow.longitude)
            );
            // turn rate from prev→now→next
            const hdgPrev = Math.atan2(
              Math.sin(cartoNow.longitude - cartoPrev.longitude) * Math.cos(cartoNow.latitude),
              Math.cos(cartoPrev.latitude)*Math.sin(cartoNow.latitude) - Math.sin(cartoPrev.latitude)*Math.cos(cartoNow.latitude)*Math.cos(cartoNow.longitude - cartoPrev.longitude)
            );
            const dHead = Cesium.Math.negativePiToPi(heading - hdgPrev);
            const dtc = Math.max(1e-3, (dt1 + dt2) * 0.5);
            const turnRate = dHead / dtc; // rad/s
            const pitch = Math.atan2(vs, Math.max(0.1, gs));
            const roll = Cesium.Math.clamp(Math.atan((gs * turnRate) / 9.81), -0.7, 0.7); // ~±40°
            // Swap axes to match model: HPR.pitch controls visual roll; HPR.roll controls visual pitch
            // Map: desiredRoll -> HPR.pitch (negated for right-wing-down), desiredPitch -> HPR.roll
            const hpr = new Cesium.HeadingPitchRoll(heading, -roll, pitch);
            const q = Cesium.Transforms.headingPitchRollQuaternion(Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.ele||0), hpr);
            const qc = Cesium.Quaternion.multiply(q, yawCorr, new Cesium.Quaternion());
            orient.addSample(Cesium.JulianDate.fromDate(d), qc);
          }
          gliderEnt.orientation = orient;
        } catch (e) { console.warn('Orientation build failed', e); gliderEnt.orientation = new Cesium.VelocityOrientationProperty(flyPositionProperty); }
        (async () => {
          try {
            const res = await fetch('/static/models/glider/Glider.glb', { method: 'HEAD' });
            if (res && res.ok && gliderEnt) {
              gliderEnt.point = undefined;
              // Prefer Z-up; rely on orientation correction for forward alignment
              let modelOpts = { uri: '/static/models/glider/Glider.glb', scale: 1.0, minimumPixelSize: 64 };
              try { modelOpts.upAxis = Cesium.Axis.Z; } catch(_) {}
              gliderEnt.model = new Cesium.ModelGraphics(modelOpts);
            }
          } catch (_) {}
        })();
        flyStart = Cesium.JulianDate.fromDate(built.firstTime);
        flyStop = Cesium.JulianDate.fromDate(built.lastTime);
        const clock = globeViewer.clock;
        // We'll drive time manually to avoid any wall-clock jumps
        clock.shouldAnimate = false;
        clock.startTime = flyStart.clone();
        clock.stopTime = flyStop.clone();
        clock.currentTime = flyStart.clone();
        globeViewer.trackedEntity = undefined;
        // Enable follow when playback starts
        setFollowCam(true);
        // Smooth follow camera
        const cam = globeViewer.camera;
        let camHeading = 0; // radians
        let camPitch = Cesium.Math.toRadians(-15);
        function angleLerp(a, b, t) {
          // shortest-path interpolate angles
          let diff = Cesium.Math.negativePiToPi(b - a);
          return a + diff * t;
        }
        flyDurationSec = Math.max(0, Cesium.JulianDate.secondsDifference(flyStop, flyStart));
        hudEl = document.getElementById('fly_hud'); if (hudEl) hudEl.style.display='block';
        const tele = document.getElementById('telemetry'); if (tele) tele.style.display='flex';
        console.log('[fly] start', {points: globePositions.length, durationSec: flyDurationSec.toFixed ? Number(flyDurationSec).toFixed(1) : flyDurationSec});
        flyLastMs = performance.now();
        const step = (ts) => {
          const dt = Math.max(0, (ts - flyLastMs) / 1000);
          flyLastMs = ts;
          const speed = Math.max(0.05, Number(flySpeedInput.value) || 1.0);
          flyBaseSec = Math.min(flyDurationSec, flyBaseSec + dt * speed);
          const now = Cesium.JulianDate.addSeconds(flyStart, flyBaseSec, new Cesium.JulianDate());
          globeViewer.clock.currentTime = now;
          const pos = flyPositionProperty.getValue(now);
          if (!pos) return;
          // Estimate heading by looking slightly ahead in time
          const prevT = Cesium.JulianDate.addSeconds(now, -0.25, new Cesium.JulianDate());
          const nextT = Cesium.JulianDate.addSeconds(now, 0.25, new Cesium.JulianDate());
          const pos1 = flyPositionProperty.getValue(prevT) || pos;
          const pos2 = flyPositionProperty.getValue(nextT) || pos;
          const carto1 = Cesium.Cartographic.fromCartesian(pos1);
          const carto2 = Cesium.Cartographic.fromCartesian(pos2);
          const h = Math.atan2(
            Math.sin(carto2.longitude - carto1.longitude) * Math.cos(carto2.latitude),
            Math.cos(carto1.latitude) * Math.sin(carto2.latitude) - Math.sin(carto1.latitude) * Math.cos(carto2.latitude) * Math.cos(carto2.longitude - carto1.longitude)
          );
          const targetHeading = h; // radians
          camHeading = angleLerp(camHeading, targetHeading, 0.15);
          const range = Math.max(10, Number(camDistInput.value) || 800);
          if (followCam) {
            const target = pos;
            const h = camHeading + (followOffsetHeading || 0);
            const p = camPitch + (followOffsetPitch || 0);
            const r = Math.max(10, Number(camDistInput.value) || 800);
            cam.lookAt(target, new Cesium.HeadingPitchRange(h, p, r));
          }
          // update UI slider
          const frac = Cesium.JulianDate.secondsDifference(now, flyStart) / Math.max(1e-6, flyDurationSec);
          flyRange.value = String(Math.round(frac * 1000));
          // Telemetry + HUD updates (throttled)
          if (hudEl) {
            hudFrames++;
            const elapsed = ts - hudLastTs;
            if (!hudLastTs) { hudLastTs = ts; }
            if (elapsed >= 500) {
              hudFps = Math.round((hudFrames * 1000) / Math.max(1, elapsed));
              hudLastTs = ts; hudFrames = 0;
              const percent = Math.round(frac * 100);
              const ents = globeViewer && globeViewer.entities ? globeViewer.entities.values.length : 0;
              // derive telemetry using samples around current time (prev/next)
              const geod = new Cesium.EllipsoidGeodesic(carto1, carto2);
              const srfDist = geod.surfaceDistance || 0;
              const dt = Math.max(1e-3, Cesium.JulianDate.secondsDifference(nextT, prevT)); // seconds between samples
              const gs = srfDist / dt; // m/s ground speed (horizontal)
              const dh = (carto2.height||0) - (carto1.height||0);
              const vs = dh / dt; // m/s vertical speed
              const altNow = Cesium.Cartographic.fromCartesian(pos).height || 0;
              const s3d = Math.hypot(srfDist, dh) / dt; // m/s total speed in 3D
              hudEl.textContent = `fps:${hudFps}  prog:${percent}%  t:${flyBaseSec.toFixed(1)}/${flyDurationSec.toFixed(1)}s  ents:${ents}  alt:${altNow.toFixed(0)}m  gs:${(gs*3.6).toFixed(0)}km/h  vs:${vs.toFixed(1)}m/s`;
              // Update SpaceX-style telemetry tiles
              const qNow = gliderEnt && gliderEnt.orientation && gliderEnt.orientation.getValue ? gliderEnt.orientation.getValue(now) : null;
              // Heading approximation already computed: targetHeading (rad). Pitch from camera frame: use difference between positions
              const hdgDeg = Math.round((Cesium.Math.toDegrees(targetHeading) + 360) % 360);
              const pitchDeg = Math.round((Math.atan2(vs, Math.max(0.1, gs)) * 180/Math.PI));
              const rollDeg = (function(){
                if (!qNow) return 0;
                const hpr = Cesium.HeadingPitchRoll.fromQuaternion(qNow);
                return Math.round(Cesium.Math.toDegrees(hpr.roll));
              })();
              const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
              setVal('tl_fps', hudFps);
              setVal('tl_elapsed', flyBaseSec.toFixed(1));
              setVal('tl_alt', Math.round(altNow));
              setVal('tl_gs', Math.round(gs*3.6));
              setVal('tl_s3d', Math.round(s3d*3.6));
              setVal('tl_vs', vs.toFixed(1));
              setVal('tl_hdg', hdgDeg);
              setVal('tl_pitch', pitchDeg);
              setVal('tl_roll', rollDeg);
            }
          }
          globeViewer.scene.requestRender();
          if (flyBaseSec >= flyDurationSec) {
            // done
            return;
          }
          flyReq = requestAnimationFrame(step);
        };
        if (flyReq) cancelAnimationFrame(flyReq);
        flyBaseSec = 0;
        flyReq = requestAnimationFrame(step);
        // Set slider to 0..1000 for smooth scrubbing by fraction
        flyRange.max = '1000'; flyRange.value = '0';
        // slider handled in RAF step for accuracy
        // enable recording controls once viewer is active
        document.getElementById('rec3d').disabled = false;
        document.getElementById('stoprec3d').disabled = false;
        updateFollowBtn();
      }
      function pauseFlythrough() {
        if (!globeViewer) return;
        if (flyReq) { cancelAnimationFrame(flyReq); flyReq = null; }
      }
      function stopFlythrough() {
        if (!globeViewer) return;
        globeViewer.clock.shouldAnimate = false;
        globeViewer.trackedEntity = undefined;
        // release camera from lookAt
        try { globeViewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY); } catch (e) {}
        if (globeFlyTimer) { globeViewer.clock.onTick.removeEventListener(globeFlyTimer); globeFlyTimer = null; }
        if (flyReq) { cancelAnimationFrame(flyReq); flyReq = null; }
        flyBaseSec = 0;
        if (hudEl) { hudEl.textContent = 'flythrough: stopped'; hudEl.style.display = 'none'; }
        const tele2 = document.getElementById('telemetry'); if (tele2) tele2.style.display='none';
        console.log('[fly] stop');
      }
      async function startRecording() {
        try {
          if (!globeViewer) return;
          const canvas = globeViewer.scene.canvas;
          mediaStream = canvas.captureStream(30);
          const mime = MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
          mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mime });
          mediaChunks = [];
          mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size) mediaChunks.push(e.data); };
          mediaRecorder.onstop = () => {
            const blob = new Blob(mediaChunks, { type: mediaRecorder.mimeType || 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'flythrough.webm'; a.click();
            URL.revokeObjectURL(url);
            mediaChunks = [];
          };
          mediaRecorder.start();
        } catch (err) {
          console.error('Recording failed', err);
          alert('Recording failed: ' + (err && err.message ? err.message : String(err)));
        }
      }
      function stopRecording() {
        try {
          if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
          if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
        } catch (err) { /* ignore */ }
      }
      function onScrub() {
        if (!globeViewer || !flyStart || !flyStop) return;
        const frac = Math.max(0, Math.min(1, (Number(flyRange.value)||0) / 1000));
        const total = Cesium.JulianDate.secondsDifference(flyStop, flyStart);
        const now = Cesium.JulianDate.addSeconds(flyStart, frac * total, new Cesium.JulianDate());
        globeViewer.clock.currentTime = now;
        flyBaseSec = frac * total;
      }
      function onSpeedChange() { if (globeViewer) globeViewer.clock.multiplier = Math.max(0.1, Number(flySpeedInput.value) || 1.0); }
      function onCamDistChange() { /* tracked camera; distance not applied here */ }
      function onTrackChange() { if (lastFileBlob) previewFile(lastFileBlob, true); }
      playBtn.addEventListener('click', startFlythrough);
      pauseBtn.addEventListener('click', pauseFlythrough);
      stopBtn.addEventListener('click', stopFlythrough);
      homeBtn.addEventListener('click', () => {
        try {
          const p = globePositions && globePositions[0];
          if (!p || !globeViewer) return;
          const pos = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.ele || 0);
          const initRange = Math.max(10, Number(camDistInput.value) || 800);
          globeViewer.camera.lookAt(pos, new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-20), initRange));
        } catch (e) {}
      });
      followBtn.addEventListener('click', () => { setFollowCam(!followCam); });
      const recBtn = document.getElementById('rec3d');
      const stopRecBtn = document.getElementById('stoprec3d');
      recBtn.addEventListener('click', startRecording);
      stopRecBtn.addEventListener('click', stopRecording);
      flyRange.addEventListener('input', onScrub);
      flySpeedInput.addEventListener('change', onSpeedChange);
      camDistInput.addEventListener('change', onCamDistChange);
      trackSelect.addEventListener('change', onTrackChange);
      // Camera presets
      function setPreset(which){
        if (!globeViewer) return;
        if (which==='chase'){ followOffsetHeading=0; followOffsetPitch=Cesium.Math.toRadians(-15); followRangeScale=1; }
        if (which==='front'){ followOffsetHeading=Math.PI; followOffsetPitch=Cesium.Math.toRadians(-5); followRangeScale=0.8; }
        if (which==='left'){ followOffsetHeading=Cesium.Math.toRadians(-90); followOffsetPitch=Cesium.Math.toRadians(-5); followRangeScale=1; }
        if (which==='right'){ followOffsetHeading=Cesium.Math.toRadians(90); followOffsetPitch=Cesium.Math.toRadians(-5); followRangeScale=1; }
        if (which==='top'){ followOffsetHeading=0; followOffsetPitch=Cesium.Math.toRadians(-80); followRangeScale=1.2; }
      }
      document.getElementById('preset_chase').addEventListener('click', ()=>setPreset('chase'));
      document.getElementById('preset_front').addEventListener('click', ()=>setPreset('front'));
      document.getElementById('preset_left').addEventListener('click', ()=>setPreset('left'));
      document.getElementById('preset_right').addEventListener('click', ()=>setPreset('right'));
      document.getElementById('preset_top').addEventListener('click', ()=>setPreset('top'));
      // Fullscreen toggle for preview area
      const fullBtn = document.getElementById('fullbtn');
      const previewEl = document.getElementById('preview');
      function updateFullBtn(){ fullBtn.textContent = (document.fullscreenElement ? 'Exit Full' : 'Full'); }
      fullBtn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
          (previewEl.requestFullscreen && previewEl.requestFullscreen()) || (globeDiv.requestFullscreen && globeDiv.requestFullscreen());
        } else { document.exitFullscreen && document.exitFullscreen(); }
      });
      document.addEventListener('fullscreenchange', updateFullBtn);

      // Live update preview when form settings change (debounced)
      let previewTimer = null;
      function schedulePreview(){
        if (!lastFileBlob) return;
        if (previewTimer) clearTimeout(previewTimer);
        previewTimer = setTimeout(()=>previewFile(lastFileBlob), 150);
      }
      form.addEventListener('input', schedulePreview);

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        statusEl.textContent = '';
        btn.disabled = true;
        try {
          const formData = new FormData(form);
          const file = formData.get('file');
          if (!file || !file.name.endsWith('.gpx')) {
            statusEl.textContent = 'Please choose a .gpx file';
            btn.disabled = false;
            return;
          }
          // Update preview
          previewFile(file);
          statusEl.textContent = 'Converting…';
          const res = await fetch('/api/convert', { method: 'POST', body: formData });
          if (!res.ok) {
            const msg = await res.text();
            throw new Error(msg || 'Conversion failed');
          }
          const blob = await res.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          const dot = file.name.lastIndexOf('.');
          const name = (dot>0? file.name.slice(0,dot): file.name) + '.kml';
          a.href = url; a.download = name; a.click();
          window.URL.revokeObjectURL(url);
          statusEl.textContent = 'Done!';
        } catch (err) {
          statusEl.textContent = (err && err.message) ? err.message : 'Error converting file';
        } finally {
          btn.disabled = false;
        }
      });

      // Open in Google Earth (serves a KML URL and opens it)
      const openGeBtn = document.getElementById('open_ge');
      openGeBtn.addEventListener('click', async () => {
        statusEl.textContent = '';
        try {
          const formData = new FormData(form);
          const file = formData.get('file');
          if (!file || !file.name.endsWith('.gpx')) { statusEl.textContent = 'Please choose a .gpx file'; return; }
          // keep preview in sync
          previewFile(file);
          const res = await fetch('/api/convert_link', { method: 'POST', body: formData });
          if (!res.ok) { const msg = await res.text(); throw new Error(msg || 'Conversion failed'); }
          const data = await res.json();
          const abs = new URL(data.url, window.location.origin).toString();
          // Attempt to open directly in Google Earth Web
          const ge = 'https://earth.google.com/web/?link=' + encodeURIComponent(abs);
          window.open(ge, '_blank') || window.open(abs, '_blank');
        } catch (err) {
          statusEl.textContent = (err && err.message) ? err.message : 'Error creating KML link';
        }
      });

      // Preview on file select
      document.getElementById('file').addEventListener('change', (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) previewFile(f);
      });

      // --- Additional overlays: plan, airspaces, airports ---
      let parsedPlan = null; // { points:[{lat,lon,ele?}]} or null
      let parsedAirspaces = []; // [{class, lower_m, upper_m, polygon:[[lat,lon],...] }]
      let parsedAirports = []; // [{name, ident, lat, lon, elev_m?}]

      async function parsePlanFile(file) {
        if (!file) { parsedPlan = null; return; }
        const name = (file.name||'').toLowerCase();
        const text = await file.text();
        try {
          if (name.endsWith('.lnmpln') || name.endsWith('.pln') || name.endsWith('.xml')) {
            const dom = new DOMParser().parseFromString(text, 'application/xml');
            // LNMPLN: look for Waypoint list
            let pts = [];
            dom.querySelectorAll('Waypoint, ATCWaypoint').forEach(wp => {
              const _posLat = wp.getAttribute('Lat') || (function(){ const el = wp.querySelector('PosLat'); return el && el.textContent; })();
              const _posLon = wp.getAttribute('Lon') || (function(){ const el = wp.querySelector('PosLong'); return el && el.textContent; })();
              const lat = parseFloat(_posLat);
              const lon = parseFloat(_posLon);
              if (!isNaN(lat) && !isNaN(lon)) pts.push({lat,lon});
              const wpos = wp.querySelector('WorldPosition');
              if (wpos && wpos.textContent && wpos.textContent.indexOf(',') !== -1) {
                const parts = wpos.textContent.split(',').map(s=>s.trim());
                const la = parseFloat(parts[0]); const lo = parseFloat(parts[1]);
                if (!isNaN(la) && !isNaN(lo)) pts.push({lat:la, lon:lo});
              }
            });
            if (pts.length<2) {
              // GPX route inside? fallback
              dom.querySelectorAll('rtept').forEach(tp=>{
                const la = parseFloat(tp.getAttribute('lat'));
                const lo = parseFloat(tp.getAttribute('lon'));
                if (!isNaN(la) && !isNaN(lo)) pts.push({lat:la,lon:lo});
              });
            }
            parsedPlan = pts.length>=2 ? { points: dedupeConsecutive(pts) } : null;
          } else if (name.endsWith('.gpx')) {
            const dom = new DOMParser().parseFromString(text, 'application/xml');
            let pts = [];
            dom.querySelectorAll('rtept,trkpt').forEach(tp=>{
              const la = parseFloat(tp.getAttribute('lat'));
              const lo = parseFloat(tp.getAttribute('lon'));
              if (!isNaN(la) && !isNaN(lo)) pts.push({lat:la,lon:lo});
            });
            parsedPlan = pts.length>=2 ? { points: dedupeConsecutive(pts) } : null;
          } else { parsedPlan = null; }
        } catch (_) { parsedPlan = null; }
      }

      function dedupeConsecutive(arr){
        const out=[]; let prev=null; for(const p of arr){ if(!prev||p.lat!==prev.lat||p.lon!==prev.lon){ out.push(p); prev=p; } } return out;
      }

      async function parseAirspaceFile(file) {
        parsedAirspaces = [];
        if (!file) return;
        const name = (file.name||'').toLowerCase();
        const text = await file.text();
        try {
          if (name.endsWith('.txt')) {
            parsedAirspaces = parseOpenAir(text);
          } else if (name.endsWith('.kml')) {
            parsedAirspaces = parseKmlAirspace(text);
          } else if (name.endsWith('.json') || name.endsWith('.geojson')) {
            parsedAirspaces = parseGeoJsonAirspace(JSON.parse(text));
          }
        } catch(_) { parsedAirspaces=[]; }
      }

      function parseAltToMeters(s){
        if (!s) return null;
        s = s.trim().toUpperCase();
        if (s==='SFC') return 0;
        if (s.startsWith('FL')) { const f = parseFloat(s.slice(2)); return isNaN(f)?null: Math.round(f*100*0.3048); }
        let buf=''; for (let i=0;i<s.length;i++){ const ch=s[i]; if ((ch>='0'&&ch<='9') || ch==='.') buf+=ch; }
        const v = parseFloat(buf); if (isNaN(v)) return null;
        const isMeters = s.includes('M') && !s.includes('FT');
        return isMeters ? Math.round(v) : Math.round(v*0.3048);
      }

function parseOpenAir(text){
  const lines = text.split(/\\r?\\n/);
  const out=[]; let cur=null; let currentClass='UNK', al=null, ah=null;
  function push(){ 
    if(cur && cur.length>=3){ 
      out.push({
        "class": currentClass,
        lower_m: (al!=null ? al : 0),
        upper_m: (ah!=null ? ah : 30000),
        polygon: cur.slice()
      }); 
    } 
    cur=null; 
  }
  for (let raw of lines){ 
    const ln=raw.trim(); 
    if(!ln) continue; 
    const tag=ln.substring(0,2);
    if (tag==='AC'){ 
      currentClass = ln.substring(2).trim(); 
    }
    else if (tag==='AL'){ 
      al = parseAltToMeters(ln.substring(2).trim()); 
    }
    else if (tag==='AH'){ 
      ah = parseAltToMeters(ln.substring(2).trim()); 
    }
    else if (tag==='DP'){
      let raw = ln.substring(2).trim();
      let norm = '';
      for (let i=0;i<raw.length;i++){ 
        const ch=raw[i]; 
        norm += (ch===','||ch===';') ? ' ' : ch; 
      }
      const parts = norm.split(' ').filter(Boolean);
      if (parts.length>=2){ 
        const lat=parseFloat(parts[0]); 
        const lon=parseFloat(parts[1]); 
        if (!isNaN(lat)&&!isNaN(lon)){ 
          if(!cur) cur=[]; 
          cur.push([lat,lon]); 
        } 
      }
    }
    else if (tag==='* '|| tag==='**'){ continue; }
    else if (tag==='AN'){ /* name ignored here */ }
  }
  push(); 
  return out;
}
      function parseKmlAirspace(text){
        const dom = new DOMParser().parseFromString(text, 'application/xml');
        const out=[];
        dom.querySelectorAll('Placemark').forEach(pm=>{
          const coords = pm.querySelector('coordinates'); if (!coords) return;
          const toks = coords.textContent.trim().split(' ').filter(Boolean);
          const pairs = toks.map(c=>c.split(',').map(Number)).filter(t=>t.length>=2);
          const poly = pairs.map(t=>[t[1], t[0]]);
          (function(){ const n = pm.querySelector('name'); out.push({"class": (n && n.textContent) || 'KML', lower_m:0, upper_m:30000, polygon:poly}); })();
        });
        return out;
      }

      function parseGeoJsonAirspace(geo){
        const out=[]; const feats = geo.type==='FeatureCollection'? geo.features : [geo];
        for (const f of feats){ const g=f.geometry; if(!g) continue; const props=f.properties||{}; const lower=parseAltToMeters(props.lower||props.floor||'0'); const upper=parseAltToMeters(props.upper||props.ceiling||'30000');
          function pushPoly(coords){ const poly = coords.map(([lon,lat])=>[lat,lon]); out.push({"class":props.class||props.name||'ASP', lower_m:(lower!=null?lower:0), upper_m:(upper!=null?upper:30000), polygon:poly}); }
          if (g.type==='Polygon'){ const rings=g.coordinates; if (rings&&rings[0]) pushPoly(rings[0]); }
          if (g.type==='MultiPolygon'){ for (const poly of g.coordinates){ if(poly&&poly[0]) pushPoly(poly[0]); } }
        }
        return out;
      }

      async function parseAirportsFile(file){
        parsedAirports = [];
        if (!file) return;
        const name=(file.name||'').toLowerCase();
        const text = await file.text();
        try {
          if (name.endsWith('.csv')){
            const lines=text.split(/\\r?\\n/); if(!lines.length) return; const head=lines[0].split(',').map(s=>s.trim().toLowerCase());
            const iLat=head.findIndex(h=>h.includes('lat'));
            const iLon=head.findIndex(h=>h.includes('lon'));
            const iName=head.findIndex(h=>h.includes('name'));
            const iId=head.findIndex(h=>h.includes('ident')||h.includes('icao')||h.includes('iata'));
            const iElev=head.findIndex(h=>h.includes('elev')||h.includes('elevation'));
            for (let i=1;i<lines.length;i++){ const row=lines[i].split(','); const la=parseFloat(row[iLat]); const lo=parseFloat(row[iLon]); if(!isNaN(la)&&!isNaN(lo)){ parsedAirports.push({name: row[iName]||'', ident: row[iId]||'', lat: la, lon: lo, elev_m: row[iElev]? Math.round(parseFloat(row[iElev])): null}); } }
          } else if (name.endsWith('.gpx')){
            const dom=new DOMParser().parseFromString(text,'application/xml');
            dom.querySelectorAll('wpt').forEach(w=>{ const la=parseFloat(w.getAttribute('lat')); const lo=parseFloat(w.getAttribute('lon')); if(!isNaN(la)&&!isNaN(lo)){ const ne=w.querySelector('name'); parsedAirports.push({name: (ne && ne.textContent) || 'wpt', ident:'', lat:la, lon:lo, elev_m: null}); }});
          }
        } catch(_) { parsedAirports=[]; }
      }

      function renderOverlays2D(){ if (!map) return; if (planLayer) planLayer.clearLayers(); if (airspaceLayer) airspaceLayer.clearLayers(); if (airportLayer) airportLayer.clearLayers();
        const showP = showPlan.checked, showA = showAirspaces.checked, showAP = showAirports.checked; const aspAlt = Number(aspAltInput.value)||0;
        if (showP && parsedPlan && parsedPlan.points && parsedPlan.points.length>1){ const latlngs=parsedPlan.points.map(p=>[p.lat,p.lon]); L.polyline(latlngs,{color:'#00d1ff',weight:3,dashArray:'5,6'}).addTo(planLayer); }
        if (showA && parsedAirspaces && parsedAirspaces.length){ parsedAirspaces.forEach(a=>{ if (aspAlt>=a.lower_m && aspAlt<=a.upper_m){ const poly=L.polygon(a.polygon,{color:'#8a2be2',weight:1,fill:true,fillOpacity:0.15}); poly.addTo(airspaceLayer); }}); }
        if (showAP && parsedAirports && parsedAirports.length){ parsedAirports.forEach(ap=>{ const m=L.circleMarker([ap.lat,ap.lon],{radius:5,color:'#10b981',fill:true,fillOpacity:0.9}); m.bindPopup(`${ap.ident?ap.ident+' – ':''}${ap.name||'airport'}`); m.addTo(airportLayer); }); }
      }

      function clearEntities(arr){ if (!globeViewer||!arr) return; arr.forEach(e=>{ try{globeViewer.entities.remove(e);}catch(_){}}); arr.length=0; }
      function renderOverlays3D(){ if (!globeViewer) return; const showP=showPlan.checked, showA=showAirspaces.checked, showAP=showAirports.checked; const aspAlt=Number(aspAltInput.value)||0; const C=Cesium; clearEntities(plan3D); clearEntities(airspace3D); clearEntities(airport3D);
        if (showP && parsedPlan && parsedPlan.points && parsedPlan.points.length>1){ const positions=parsedPlan.points.map(p=>C.Cartesian3.fromDegrees(p.lon,p.lat,0)); plan3D.push(globeViewer.entities.add({ polyline:{ positions, width:3, material: C.Color.CYAN } })); }
        if (showA && parsedAirspaces && parsedAirspaces.length){ parsedAirspaces.forEach(a=>{ if (aspAlt>=a.lower_m && aspAlt<=a.upper_m){ const hierarchy = a.polygon.map(([lat,lon])=>C.Cartesian3.fromDegrees(lon,lat, a.lower_m)); const extrudedHeight=a.upper_m; airspace3D.push(globeViewer.entities.add({ polygon:{ hierarchy: new C.PolygonHierarchy(hierarchy), extrudedHeight, height: a.lower_m, material: C.Color.PURPLE.withAlpha(0.2), outline:true, outlineColor: C.Color.PURPLE } })); }}); }
        if (showAP && parsedAirports && parsedAirports.length){ parsedAirports.forEach(ap=>{ airport3D.push(globeViewer.entities.add({ position:C.Cartesian3.fromDegrees(ap.lon,ap.lat, ap.elev_m||0), point:{ pixelSize:6, color: C.Color.CHARTREUSE }, label:{ text: (ap.ident?ap.ident+' – ':'')+(ap.name||'Airport'), font:'12px sans-serif', pixelOffset: new C.Cartesian2(12,-12), showBackground:true, backgroundColor:C.Color.BLACK.withAlpha(0.5) } })); }); }
      }

      async function onPlanChange(e){ await parsePlanFile(e.target.files && e.target.files[0]); renderOverlays2D(); ensureGlobe(()=>renderOverlays3D()); }
      async function onAirspaceChange(e){ await parseAirspaceFile(e.target.files && e.target.files[0]); renderOverlays2D(); ensureGlobe(()=>renderOverlays3D()); }
      async function onAirportsChange(e){ await parseAirportsFile(e.target.files && e.target.files[0]); renderOverlays2D(); ensureGlobe(()=>renderOverlays3D()); }
      planFileInput.addEventListener('change', onPlanChange);
      airspaceFileInput.addEventListener('change', onAirspaceChange);
      airportsFileInput.addEventListener('change', onAirportsChange);
      showPlan.addEventListener('change', ()=>{ renderOverlays2D(); ensureGlobe(()=>renderOverlays3D()); });
      showAirspaces.addEventListener('change', ()=>{ renderOverlays2D(); ensureGlobe(()=>renderOverlays3D()); });
      showAirports.addEventListener('change', ()=>{ renderOverlays2D(); ensureGlobe(()=>renderOverlays3D()); });
      aspAltInput.addEventListener('change', ()=>{ renderOverlays2D(); ensureGlobe(()=>renderOverlays3D()); });
    </script>
  </body>
  </html>
"""
