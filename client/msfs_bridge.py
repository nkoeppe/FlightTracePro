#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import sys
import time
from typing import Optional


def meters_from_feet(ft: Optional[float]) -> Optional[float]:
    return None if ft is None else ft * 0.3048


class MSFSSource:
    def __init__(self):
        self.sim = None
        self.areq = None

    def start(self) -> bool:
        print("[msfs] Attempting SimConnect initialization...", file=sys.stderr)
        
        # Step 0: Check if MSFS process is running
        try:
            import subprocess
            result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq FlightSimulator.exe'], 
                                  capture_output=True, text=True, shell=True)
            if 'FlightSimulator.exe' in result.stdout:
                print("[msfs] ✓ Flight Simulator process found", file=sys.stderr)
            else:
                print("[msfs] ⚠ Flight Simulator process NOT found", file=sys.stderr)
                print("[msfs] Make sure MSFS 2020 is running", file=sys.stderr)
        except Exception as e:
            print(f"[msfs] Could not check MSFS process: {e}", file=sys.stderr)
        
        # Step 1: Try to import SimConnect
        try:
            print("[msfs] Importing SimConnect module...", file=sys.stderr)
            from SimConnect import SimConnect, AircraftRequests
            print("[msfs] ✓ SimConnect module imported successfully", file=sys.stderr)
        except ImportError as e:
            print(f"[msfs] ✗ SimConnect module not found: {e}", file=sys.stderr)
            print("[msfs] Install with: pip install SimConnect==0.4.26", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[msfs] ✗ SimConnect import failed: {e}", file=sys.stderr)
            return False
        
        # Step 2: Try to create SimConnect instance with different approaches
        connection_attempts = [
            ("Default connection", lambda: SimConnect()),
            ("Named connection", lambda: SimConnect(auto_connect=False)),
            ("Local connection", lambda: SimConnect(auto_connect=True)),
        ]
        
        for attempt_name, connect_func in connection_attempts:
            try:
                print(f"[msfs] Trying {attempt_name}...", file=sys.stderr)
                self.sim = connect_func()
                print(f"[msfs] ✓ {attempt_name} successful", file=sys.stderr)
                break
            except Exception as e:
                print(f"[msfs] ✗ {attempt_name} failed: {e}", file=sys.stderr)
                continue
        
        if not self.sim:
            print("[msfs] ✗ All SimConnect connection attempts failed", file=sys.stderr)
            print("[msfs] Troubleshooting:", file=sys.stderr)
            print("[msfs] 1. Make sure MSFS 2020 is running and you're in a flight", file=sys.stderr)
            print("[msfs] 2. Check MSFS Options > General > Developers > Enable SimConnect", file=sys.stderr)
            print("[msfs] 3. Try restarting MSFS completely", file=sys.stderr)
            print("[msfs] 4. Check Windows permissions", file=sys.stderr)
            return False
        
        # Step 3: Try to create AircraftRequests
        try:
            print("[msfs] Creating AircraftRequests...", file=sys.stderr)
            self.areq = AircraftRequests(self.sim, _time=50)
            print("[msfs] ✓ AircraftRequests ready", file=sys.stderr)
            return True
        except Exception as e:
            print(f"[msfs] ✗ Failed to create AircraftRequests: {e}", file=sys.stderr)
            try:
                self.sim.quit()
            except:
                pass
            self.sim = None
            return False

    def read(self):
        if not self.areq:
            return None
        # Many variables are documented at https://docs.flightsimulator.com/html/Programming_Tools/SimVars/Simulation_Variables.htm
        try:
            def gv(name, unit=None, alt=None):
                # Prefer explicit unit when provided, then try without unit; also try underscore variant
                names = [name]
                if " " in name:
                    names.append(name.replace(" ", "_"))
                for nm in names:
                    if unit is not None:
                        try:
                            v = self.areq.get(nm, unit)
                            if v is not None:
                                return v
                        except Exception:
                            pass
                    try:
                        v = self.areq.get(nm)
                        if v is not None:
                            return v
                    except Exception:
                        pass
                return alt

            lat = gv("PLANE LATITUDE", unit="degrees")
            lon = gv("PLANE LONGITUDE", unit="degrees")
            alt_ft = gv("PLANE ALTITUDE", unit="feet")
            hdg = gv("PLANE HEADING DEGREES TRUE", unit="degrees")
            # Ground velocity comes in feet per second in some setups; try getting KNOTS directly
            try:
                spd_kt = gv("AIRSPEED TRUE", unit="knots")
            except Exception:
                gv_fps = gv("GROUND VELOCITY", unit="feet per second")
                spd_kt = gv_fps * 0.592484 if gv_fps is not None else None
            vsi_fpm = gv("VERTICAL SPEED", unit="feet per minute")
            pitch = gv("PLANE PITCH DEGREES", unit="degrees")
            roll = gv("PLANE BANK DEGREES", unit="degrees")
            # Fallback to GPS vars if plane vars are not ready (e.g., in menus)
            if lat is None or lon is None:
                lat = gv("GPS POSITION LAT", unit="degrees") or gv("GPS_POSITION_LAT", unit="degrees")
                lon = gv("GPS POSITION LON", unit="degrees") or gv("GPS_POSITION_LON", unit="degrees")
            if alt_ft is None:
                alt_ft = gv("INDICATED ALTITUDE", unit="feet")
            # Convert radians to degrees if SimConnect returned radians by default
            def rad_to_deg_if_needed(x):
                try:
                    if x is None:
                        return None
                    # treat absolute radians range
                    if -6.5 <= float(x) <= 6.5:
                        return float(x) * 180.0 / math.pi
                    return float(x)
                except Exception:
                    return x
            hdg_deg = rad_to_deg_if_needed(hdg)
            pitch_deg = rad_to_deg_if_needed(pitch)
            roll_deg = rad_to_deg_if_needed(roll)

            return {
                "lat": float(lat) if lat is not None else None,
                "lon": float(lon) if lon is not None else None,
                "alt_m": meters_from_feet(alt_ft),
                "spd_kt": float(spd_kt) if spd_kt is not None else None,
                "vsi_ms": (vsi_fpm * 0.00508) if vsi_fpm is not None else None,  # ft/min -> m/s
                "hdg_deg": float(hdg_deg) if hdg_deg is not None else None,
                "pitch_deg": float(pitch_deg) if pitch_deg is not None else None,
                "roll_deg": float(roll_deg) if roll_deg is not None else None,
            }
        except Exception as e:
            print(f"[msfs] read error: {e}", file=sys.stderr)
            return None


async def run_ws_loop(server: str, channel: str, callsign: str, key: Optional[str], rate_hz: float):
    import websockets
    url = f"{server.rstrip('/')}/ws/live/{channel}?mode=feeder" + (f"&key={key}" if key else "")
    src = MSFSSource()
    print(f"[bridge] WS connecting to {url}")
    while True:
        # Ensure SimConnect is available; wait until sim is running
        if not src.areq:
            if not src.start():
                print("[bridge] Waiting for MSFS/SimConnect… (retry in 3s)")
                await asyncio.sleep(3)
                continue
        try:
            async with websockets.connect(url, max_size=1_000_000, compression=None, ping_interval=20, ping_timeout=20) as ws:
                print("[bridge] connected")
                dt = 1.0 / max(1.0, rate_hz)
                while True:
                    s = src.read()
                    if s is not None and (s.get("lat") is None or s.get("lon") is None):
                        try:
                            print(f"[bridge] raw sample (no send): {s}")
                        except Exception:
                            pass
                    if s and (s.get("lat") is not None) and (s.get("lon") is not None):
                        s["callsign"] = callsign
                        s["ts"] = time.time()
                        await ws.send(json.dumps({"type": "state", "payload": s}))
                        # lightweight debug every ~2s
                        if int(time.time()) % 2 == 0:
                            latv = s.get('lat'); lonv = s.get('lon'); altv = s.get('alt_m')
                            latf = f"{latv:.5f}" if isinstance(latv, (int,float)) else "n/a"
                            lonf = f"{lonv:.5f}" if isinstance(lonv, (int,float)) else "n/a"
                            altf = f"{altv:.0f}" if isinstance(altv, (int,float)) else "n/a"
                            print(f"[bridge] sent lat={latf} lon={lonf} alt={altf}")
                    await asyncio.sleep(dt)
        except Exception as e:
            print(f"[bridge] WS error: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)
            # Drop SimConnect to force re-init if sim was closed
            src.areq = None


async def run_http_loop(server: str, channel: str, callsign: str, key: Optional[str], rate_hz: float):
    import requests
    url = f"{server.rstrip('/')}/api/live/{channel}"
    src = MSFSSource()
    print(f"[bridge] HTTP posting to {url}")
    dt = 1.0 / max(1.0, rate_hz)
    while True:
        if not src.areq:
            if not src.start():
                print("[bridge] Waiting for MSFS/SimConnect… (retry in 3s)")
                await asyncio.sleep(3)
                continue
        s = src.read()
        if s and (s.get("lat") is not None) and (s.get("lon") is not None):
            s["callsign"] = callsign
            s["ts"] = time.time()
            params = {"key": key} if key else None
            try:
                r = requests.post(url, params=params, json=s, timeout=3)
                if not r.ok:
                    print(f"[bridge] post failed: {r.status_code} {r.text}")
                elif int(time.time()) % 2 == 0:
                    latv = s.get('lat'); lonv = s.get('lon'); altv = s.get('alt_m')
                    latf = f"{latv:.5f}" if isinstance(latv, (int,float)) else "n/a"
                    lonf = f"{lonv:.5f}" if isinstance(lonv, (int,float)) else "n/a"
                    altf = f"{altv:.0f}" if isinstance(altv, (int,float)) else "n/a"
                    print(f"[bridge] posted lat={latf} lon={lonf} alt={altf}")
            except Exception as e:
                print(f"[bridge] post error: {e}")
        await asyncio.sleep(dt)


def simulate_sample(t: float, origin_lat: float = 47.3769, origin_lon: float = 8.5417, origin_alt_m: float = 500.0):
    # Simple circle path around origin
    R = 0.02  # approx 2km radius in degrees
    ang = (t * 0.05) % (2 * 3.14159)
    return {
        "lat": origin_lat + R * (0.8 * __import__('math').sin(ang)),
        "lon": origin_lon + R * (1.2 * __import__('math').cos(ang)),
        "alt_m": origin_alt_m + 50 * __import__('math').sin(ang * 2),
        "spd_kt": 60.0,
        "vsi_ms": 0.0,
        "hdg_deg": (ang * 180.0 / 3.14159) % 360,
    }


async def run_ws_demo(server: str, channel: str, callsign: str, rate_hz: float, origin_lat: float, origin_lon: float, origin_alt_m: float):
    import websockets
    url = f"{server.rstrip('/')}/ws/live/{channel}?mode=feeder"
    print(f"[demo] WS connecting to {url}")
    dt = 1.0 / max(1.0, rate_hz)
    while True:
        try:
            async with websockets.connect(url, max_size=1_000_000, compression=None, ping_interval=20, ping_timeout=20) as ws:
                print("[demo] connected")
                while True:
                    s = simulate_sample(time.time(), origin_lat, origin_lon, origin_alt_m)
                    s["callsign"] = callsign
                    s["ts"] = time.time()
                    await ws.send(json.dumps({"type": "state", "payload": s}))
                    await asyncio.sleep(dt)
        except Exception as e:
            print(f"[demo] WS error: {e}. Reconnecting in 3s…")
            await asyncio.sleep(3)


async def run_http_demo(server: str, channel: str, callsign: str, rate_hz: float, origin_lat: float, origin_lon: float, origin_alt_m: float):
    import requests
    url = f"{server.rstrip('/')}/api/live/{channel}"
    print(f"[demo] HTTP posting to {url}")
    dt = 1.0 / max(1.0, rate_hz)
    while True:
        s = simulate_sample(time.time(), origin_lat, origin_lon, origin_alt_m)
        s["callsign"] = callsign
        s["ts"] = time.time()
        try:
            requests.post(url, json=s, timeout=3)
        except Exception as e:
            print(f"[demo] post error: {e}")
        await asyncio.sleep(dt)


def main():
    ap = argparse.ArgumentParser(description="MSFS 2020 → FlightTracePro live bridge")
    ap.add_argument("--server", required=True, help="Server base URL, e.g. ws://host:8000 or http://host:8000")
    ap.add_argument("--channel", default="default", help="Channel name")
    ap.add_argument("--callsign", default="N123AB", help="Your callsign/label")
    ap.add_argument("--key", default=None, help="Optional post key if server requires")
    ap.add_argument("--mode", choices=["ws", "http"], default="ws", help="Transport: ws (websocket) or http")
    ap.add_argument("--rate", type=float, default=10.0, help="Update rate in Hz (e.g. 10 = every 0.1s for smooth flight)")
    ap.add_argument("--demo", action="store_true", help="Simulate data without MSFS")
    ap.add_argument("--origin-lat", type=float, default=47.3769)
    ap.add_argument("--origin-lon", type=float, default=8.5417)
    ap.add_argument("--origin-alt", type=float, default=500.0)
    args = ap.parse_args()

    if args.demo:
        if args.mode == "ws":
            if not args.server.startswith("ws://") and not args.server.startswith("wss://"):
                print("[demo] --server must start with ws:// or wss://", file=sys.stderr)
                sys.exit(2)
            asyncio.run(run_ws_demo(args.server, args.channel, args.callsign, args.rate, args.origin_lat, args.origin_lon, args.origin_alt))
        else:
            if not args.server.startswith("http://") and not args.server.startswith("https://"):
                print("[demo] --server must start with http:// or https://", file=sys.stderr)
                sys.exit(2)
            asyncio.run(run_http_demo(args.server, args.channel, args.callsign, args.rate, args.origin_lat, args.origin_lon, args.origin_alt))
    else:
        if args.mode == "ws":
            if not args.server.startswith("ws://") and not args.server.startswith("wss://"):
                print("[bridge] --server must start with ws:// or wss:// for ws mode", file=sys.stderr)
                sys.exit(2)
            asyncio.run(run_ws_loop(args.server, args.channel, args.callsign, args.key, args.rate))
        else:
            if not args.server.startswith("http://") and not args.server.startswith("https://"):
                print("[bridge] --server must start with http:// or https:// for http mode", file=sys.stderr)
                sys.exit(2)
            asyncio.run(run_http_loop(args.server, args.channel, args.callsign, args.key, args.rate))


if __name__ == "__main__":
    main()
