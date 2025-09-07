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

            # Aircraft identification - get REAL aircraft title from SimConnect
            aircraft_title = None
            
            # Try to get the actual aircraft title using SimConnect's direct data access
            try:
                # The Python SimConnect library supports direct string variable access through the sim object
                # Try multiple approaches to get the real aircraft name
                
                # Method 1: Use proper SimConnect data definition with callback (advanced method)
                if hasattr(self, '_aircraft_title_cache'):
                    # Use cached title if we have one from previous requests
                    aircraft_title = getattr(self, '_aircraft_title_cache', None)
                    if aircraft_title:
                        print(f"[msfs] Using cached aircraft title: {aircraft_title}", file=sys.stderr)
                
                # Method 2: Try to use SimConnect's built-in string variable support
                if not aircraft_title:
                    try:
                        # Some SimConnect libraries support direct string access
                        from SimConnect import DWORD
                        
                        # Try to request aircraft title using SimConnect event system
                        # This is a more complex approach that requires event handling
                        req_id = 9999
                        
                        # Try different approaches based on SimConnect library version
                        if hasattr(self.sim, 'request_system_state'):
                            result = self.sim.request_system_state(req_id, "AircraftLoaded")
                            if result:
                                print(f"[msfs] System state result: {result}", file=sys.stderr)
                        
                        # Alternative: Try to get aircraft file path and extract name
                        if hasattr(self.sim, 'request_data_on_sim_object') and hasattr(self.sim, 'add_data_definition'):
                            # Create a simple data definition for aircraft info
                            def_id = 1001
                            try:
                                # Add definition for string data (if supported)
                                self.sim.add_data_definition(def_id, "TITLE", None, 0)  # SIMCONNECT_DATATYPE_STRING256
                                print(f"[msfs] Added data definition for TITLE", file=sys.stderr)
                                
                                # This would require proper event loop handling to get response
                                # For now, we'll note the attempt
                                
                            except Exception as sub_e:
                                print(f"[msfs] Data definition creation failed: {sub_e}", file=sys.stderr)
                                
                    except Exception as e:
                        print(f"[msfs] Advanced SimConnect approach failed: {e}", file=sys.stderr)
                
                # Fallback: Use AircraftRequests with enhanced aircraft identification
                if not aircraft_title:
                    engine_type = gv("ENGINE TYPE")
                    num_engines = gv("NUMBER OF ENGINES")
                    is_retractable = gv("IS GEAR RETRACTABLE")
                    max_weight = gv("MAX GROSS WEIGHT")
                    
                    print(f"[msfs] Aircraft characteristics: Engine={engine_type}, Engines={num_engines}, Retractable={is_retractable}, Weight={max_weight}", file=sys.stderr)
                    
                    # Try to identify specific aircraft using database
                    try:
                        from aircraft_database import identify_aircraft
                        identified_aircraft = identify_aircraft(engine_type, num_engines, is_retractable, max_weight)
                        if identified_aircraft:
                            aircraft_title = identified_aircraft
                            print(f"[msfs] Identified aircraft from database: {aircraft_title}", file=sys.stderr)
                    except Exception as e:
                        print(f"[msfs] Aircraft database lookup failed: {e}", file=sys.stderr)
                    
                    # Fallback to generic categorization if database lookup failed
                    if not aircraft_title:
                        if engine_type == 0:  # Piston
                            if num_engines == 1:
                                aircraft_title = "Single Engine Piston"
                            elif num_engines == 2:
                                aircraft_title = "Twin Engine Piston"
                            else:
                                aircraft_title = f"Piston Aircraft ({num_engines}E)" if num_engines else "Piston Aircraft"
                        elif engine_type == 1:  # Jet
                            if max_weight and max_weight > 400000:
                                aircraft_title = "Wide Body Airliner"
                            elif max_weight and max_weight > 150000:
                                aircraft_title = "Narrow Body Airliner"
                            elif max_weight and max_weight > 50000:
                                aircraft_title = "Heavy Business Jet"
                            elif max_weight and max_weight > 20000:
                                aircraft_title = "Medium Business Jet"
                            elif num_engines == 1:
                                aircraft_title = "Single Engine Jet"
                            elif num_engines == 2:
                                aircraft_title = "Light Business Jet"
                            else:
                                aircraft_title = f"Jet Aircraft ({num_engines}E)" if num_engines else "Jet Aircraft"
                        elif engine_type == 2:  # Turboprop
                            if max_weight and max_weight > 40000:
                                aircraft_title = "Regional Turboprop"
                            elif num_engines == 1:
                                aircraft_title = "Single Engine Turboprop"
                            elif num_engines == 2:
                                aircraft_title = "Twin Turboprop"
                            else:
                                aircraft_title = f"Turboprop Aircraft ({num_engines}E)" if num_engines else "Turboprop Aircraft"
                        elif engine_type == 3:  # Helicopter
                            if max_weight and max_weight > 15000:
                                aircraft_title = "Heavy Helicopter"
                            elif max_weight and max_weight > 8000:
                                aircraft_title = "Medium Helicopter"
                            else:
                                aircraft_title = "Light Helicopter"
                        elif engine_type == 4:  # Turbine
                            aircraft_title = "Turbine Aircraft"
                        elif engine_type == 5:  # Unsupported
                            aircraft_title = "Experimental Aircraft"
                        else:
                            # Use characteristics to make educated guess
                            if is_retractable:
                                aircraft_title = "Complex Aircraft"
                            elif max_weight and max_weight > 12500:
                                aircraft_title = "Large Aircraft"
                            else:
                                aircraft_title = "General Aviation Aircraft"
                                
                        # Add gear type info for generic categories
                        if aircraft_title and not any(keyword in aircraft_title for keyword in ["Airliner", "Business", "Regional", "Heavy", "Medium", "Light"]):
                            if is_retractable:
                                aircraft_title += " (Retractable)"
                            elif is_retractable is False:
                                aircraft_title += " (Fixed Gear)"
                        
            except Exception as e:
                aircraft_title = "Aircraft"
                print(f"[msfs] Aircraft identification error: {e}", file=sys.stderr)
            
            # Final fallback
            if not aircraft_title:
                aircraft_title = "Aircraft"
            
            # Engine RPM - comprehensive approach to find working RPM variables
            rpm_percent = None
            try:
                # List all possible RPM variables to try
                rpm_variables_to_try = [
                    ("PROP RPM:1", "rpm", 2500.0),  # Indexed prop RPM
                    ("PROP RPM", "rpm", 2500.0),    # Non-indexed prop RPM  
                    ("ENG RPM:1", "rpm", 2500.0),   # Indexed engine RPM
                    ("ENG RPM", "rpm", 2500.0),     # Non-indexed engine RPM
                    ("TURB ENG N1:1", "percent", 1.0),  # Indexed N1 (already percentage)
                    ("TURB ENG N1", "percent", 1.0),    # Non-indexed N1
                    ("GENERAL ENG RPM:1", "percent", 1.0),  # Indexed general RPM
                    ("GENERAL ENG RPM", "percent", 1.0),    # Non-indexed general RPM
                    ("ENGINE RPM:1", "rpm", 2500.0),       # Alternative engine RPM
                    ("ENGINE RPM", "rpm", 2500.0),         # Alternative engine RPM
                ]
                
                print(f"[msfs] === RPM DEBUG: Testing RPM variables ===", file=sys.stderr)
                
                for var_name, unit, scale_factor in rpm_variables_to_try:
                    try:
                        rpm_value = gv(var_name, unit=unit)
                        print(f"[msfs] RPM Test: {var_name} ({unit}) = {rpm_value}", file=sys.stderr)
                        
                        if rpm_value is not None and rpm_value > 0:
                            if unit == "percent":
                                rpm_percent = float(rpm_value)
                            else:  # rpm unit
                                rpm_percent = min(100.0, (float(rpm_value) / scale_factor) * 100.0)
                            
                            print(f"[msfs] ✓ Found working RPM: {var_name} = {rpm_value} {unit}, converted to {rpm_percent}%", file=sys.stderr)
                            break
                    except Exception as e:
                        print(f"[msfs] RPM Test: {var_name} failed - {e}", file=sys.stderr)
                
                # If no RPM found, try without units
                if rpm_percent is None:
                    print(f"[msfs] === RPM DEBUG: Trying RPM variables without units ===", file=sys.stderr)
                    for var_name, _, scale_factor in rpm_variables_to_try:
                        try:
                            rpm_value = gv(var_name.split(':')[0])  # Remove index for non-unit test
                            print(f"[msfs] RPM No-Unit Test: {var_name} = {rpm_value}", file=sys.stderr)
                            
                            if rpm_value is not None and rpm_value > 0:
                                # Try to detect if it's already a percentage (0-100 range) or RPM (>100)
                                if float(rpm_value) <= 100:
                                    rpm_percent = float(rpm_value)
                                    print(f"[msfs] ✓ Found RPM as percentage: {rpm_value}%", file=sys.stderr)
                                else:
                                    rpm_percent = min(100.0, (float(rpm_value) / scale_factor) * 100.0)
                                    print(f"[msfs] ✓ Found RPM value: {rpm_value}, converted to {rpm_percent}%", file=sys.stderr)
                                break
                        except Exception as e:
                            print(f"[msfs] RPM No-Unit Test: {var_name} failed - {e}", file=sys.stderr)
                
                if rpm_percent is None:
                    print(f"[msfs] ⚠ No working RPM variables found", file=sys.stderr)
                    
            except Exception as e:
                print(f"[msfs] RPM detection error: {e}", file=sys.stderr)
            
            # Fuel flow - simplified approach 
            fuel_flow_gph = None
            try:
                # Try indexed engine fuel flow (engine 1)
                ff1 = gv("ENG FUEL FLOW GPH:1", unit="gallons per hour")
                if ff1 and ff1 > 0:
                    fuel_flow_gph = float(ff1)
                    print(f"[msfs] Got fuel flow engine 1: {fuel_flow_gph} GPH", file=sys.stderr)
                else:
                    # Try non-indexed
                    ff = gv("ENG FUEL FLOW GPH", unit="gallons per hour")
                    if ff and ff > 0:
                        fuel_flow_gph = float(ff)
                        print(f"[msfs] Got fuel flow: {fuel_flow_gph} GPH", file=sys.stderr)
                    else:
                        # Try PPH and convert
                        ff_pph = gv("ENG FUEL FLOW PPH:1", unit="pounds per hour")
                        if ff_pph and ff_pph > 0:
                            fuel_flow_gph = float(ff_pph) / 6.0  # Rough PPH to GPH conversion
                            print(f"[msfs] Got fuel flow PPH: {ff_pph}, converted to {fuel_flow_gph} GPH", file=sys.stderr)
            except Exception as e:
                print(f"[msfs] Failed to get fuel flow: {e}", file=sys.stderr)
                
            # Throttle position - simplified
            throttle_pct = None
            try:
                # Try indexed throttle (engine 1)
                thr1 = gv("GENERAL ENG THROTTLE LEVER POSITION:1", unit="percent")
                if thr1 is not None:
                    throttle_pct = float(thr1)
                    print(f"[msfs] Got throttle: {throttle_pct}%", file=sys.stderr)
                else:
                    # Try non-indexed
                    thr = gv("GENERAL ENG THROTTLE LEVER POSITION", unit="percent") 
                    if thr is not None:
                        throttle_pct = float(thr)
                        print(f"[msfs] Got throttle (non-indexed): {throttle_pct}%", file=sys.stderr)
            except Exception as e:
                print(f"[msfs] Failed to get throttle: {e}", file=sys.stderr)
            
            # Autopilot information - more resilient variable access
            ap_master = gv("AUTOPILOT MASTER") or gv("AUTOPILOT ON")
            ap_hdg_lock = gv("AUTOPILOT HEADING LOCK") or gv("AUTOPILOT HDG LOCK")
            ap_alt_lock = gv("AUTOPILOT ALTITUDE LOCK") or gv("AUTOPILOT ALT LOCK")
            ap_speed_hold = gv("AUTOPILOT AIRSPEED HOLD") or gv("AUTOPILOT IAS HOLD")
            ap_vs_hold = gv("AUTOPILOT VERTICAL HOLD") or gv("AUTOPILOT VS HOLD")
            ap_nav_lock = gv("AUTOPILOT NAV1 LOCK") or gv("AUTOPILOT NAV LOCK")
            ap_approach_arm = gv("AUTOPILOT APPROACH ARM") or gv("AUTOPILOT APR ARM")
            
            # Target values for autopilot - try without units first
            ap_hdg_target = gv("AUTOPILOT HEADING LOCK DIR") or gv("AUTOPILOT HEADING LOCK DIR", unit="degrees")
            ap_alt_target = gv("AUTOPILOT ALTITUDE LOCK VAR") or gv("AUTOPILOT ALTITUDE LOCK VAR", unit="feet") 
            ap_speed_target = gv("AUTOPILOT AIRSPEED HOLD VAR") or gv("AUTOPILOT AIRSPEED HOLD VAR", unit="knots")
            ap_vs_target = gv("AUTOPILOT VERTICAL HOLD VAR") or gv("AUTOPILOT VERTICAL HOLD VAR", unit="feet per minute")

            return {
                "lat": float(lat) if lat is not None else None,
                "lon": float(lon) if lon is not None else None,
                "alt_m": meters_from_feet(alt_ft),
                "spd_kt": float(spd_kt) if spd_kt is not None else None,
                "vsi_ms": (vsi_fpm * 0.00508) if vsi_fpm is not None else None,  # ft/min -> m/s
                "hdg_deg": float(hdg_deg) if hdg_deg is not None else None,
                "pitch_deg": float(pitch_deg) if pitch_deg is not None else None,
                "roll_deg": float(roll_deg) if roll_deg is not None else None,
                # Aircraft information
                "aircraft": str(aircraft_title) if aircraft_title else "Aircraft",
                "rpm_pct": float(rpm_percent) if rpm_percent is not None else None,
                "fuel_flow_gph": float(fuel_flow_gph) if fuel_flow_gph is not None else None,
                "throttle_pct": float(throttle_pct) if throttle_pct is not None else None,
                # Autopilot status
                "ap_master": bool(ap_master) if ap_master is not None else None,
                "ap_hdg_lock": bool(ap_hdg_lock) if ap_hdg_lock is not None else None,
                "ap_alt_lock": bool(ap_alt_lock) if ap_alt_lock is not None else None,
                "ap_speed_hold": bool(ap_speed_hold) if ap_speed_hold is not None else None,
                "ap_vs_hold": bool(ap_vs_hold) if ap_vs_hold is not None else None,
                "ap_nav_lock": bool(ap_nav_lock) if ap_nav_lock is not None else None,
                "ap_approach_arm": bool(ap_approach_arm) if ap_approach_arm is not None else None,
                # Autopilot targets
                "ap_hdg_target": float(rad_to_deg_if_needed(ap_hdg_target)) if ap_hdg_target is not None else None,
                "ap_alt_target": meters_from_feet(ap_alt_target) if ap_alt_target is not None else None,
                "ap_speed_target": float(ap_speed_target) if ap_speed_target is not None else None,
                "ap_vs_target": (ap_vs_target * 0.00508) if ap_vs_target is not None else None,  # ft/min -> m/s
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


# --- Realistic demo flight simulation (deterministic per callsign) ---
from math import sin, cos, radians, degrees, atan2, sqrt
import random

_SIM_ROUTES: dict[str, dict] = {}

def _hash_callsign(callsign: str) -> int:
    h = 0
    for ch in callsign or "ACFT":
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def _dest_ll(lat, lon, bearing_deg, distance_m):
    R = 6371000.0
    br = radians(bearing_deg)
    lat1 = radians(lat)
    lon1 = radians(lon)
    dr = distance_m / R
    lat2 = atan2(
        sin(lat1)*cos(dr) + cos(lat1)*sin(dr)*cos(br),
        sqrt(1 - (sin(lat1)*cos(dr) + cos(lat1)*sin(dr)*cos(br))**2)
    )
    # Use standard formula for lon2 avoiding precision issues
    lat2s = sin(lat1)*cos(dr) + cos(lat1)*sin(dr)*cos(br)
    lat2v = atan2(lat2s, sqrt(max(0.0, 1.0 - lat2s*lat2s)))
    lat2v = lat2  # keep lat2 from above
    lon2 = lon1 + atan2(sin(br)*sin(dr)*cos(lat1), cos(dr) - sin(lat1)*sin(lat2v))
    return degrees(lat2v), (degrees(lon2) + 540) % 360 - 180

def _bearing_deg(lat1, lon1, lat2, lon2):
    y = sin(radians(lon2 - lon1)) * cos(radians(lat2))
    x = cos(radians(lat1))*sin(radians(lat2)) - sin(radians(lat1))*cos(radians(lat2))*cos(radians(lon2 - lon1))
    br = (degrees(atan2(y, x)) + 360.0) % 360.0
    return br

def _build_route(callsign: str, origin_lat: float, origin_lon: float):
    rnd = random.Random(_hash_callsign(callsign))
    # Generate 5–9 waypoints 5–30 km from origin, forming a closed loop
    n = rnd.randint(5, 9)
    wps = []
    bearing = rnd.uniform(0, 360)
    for i in range(n):
        distance_km = rnd.uniform(6.0, 25.0)
        # vary bearing smoothly to avoid sharp zig-zags
        bearing = (bearing + rnd.uniform(25, 90)) % 360
        lat2, lon2 = _dest_ll(origin_lat, origin_lon, bearing, distance_km * 1000.0)
        wps.append((lat2, lon2))
    # Close loop back toward the first point to form circuit
    wps.append(wps[0])
    # Per-segment speeds in knots (GA-ish)
    speeds = [rnd.uniform(80, 140) for _ in range(len(wps) - 1)]
    # Precompute distances and durations
    segs = []
    total_t = 0.0
    for i in range(len(wps) - 1):
        a = wps[i]; b = wps[i+1]
        d = _haversine_m(a[0], a[1], b[0], b[1])
        spd_mps = speeds[i] * 0.514444
        dur = max(30.0, d / max(30.0, spd_mps))
        segs.append({"a": a, "b": b, "d": d, "spd_kt": speeds[i], "dur": dur, "t0": total_t, "t1": total_t + dur})
        total_t += dur
    return {"wps": wps, "segs": segs, "loop_dur": total_t, "seed": rnd.random()}

def _get_route(callsign: str, origin_lat: float, origin_lon: float):
    r = _SIM_ROUTES.get(callsign)
    if not r:
        r = _build_route(callsign, origin_lat, origin_lon)
        r["t_start"] = time.time()
        _SIM_ROUTES[callsign] = r
    return r

def simulate_sample(t: float, origin_lat: float = 47.3769, origin_lon: float = 8.5417, origin_alt_m: float = 500.0, callsign: str = "ACFT"):
    """Return a realistic-ish simulated flight sample.
    Deterministic per callsign; smooth headings, speeds, and altitude with light noise.
    """
    route = _get_route(callsign, origin_lat, origin_lon)
    elapsed = (t - route.get("t_start", t))
    loop = route["loop_dur"] or 1.0
    tm = elapsed % loop

    # Find active segment
    segs = route["segs"]
    seg = segs[-1]
    for s in segs:
        if s["t0"] <= tm <= s["t1"]:
            seg = s
            break
    u = (tm - seg["t0"]) / max(1e-6, (seg["t1"] - seg["t0"]))
    # Ease in/out for turns (smooth course changes)
    ue = 3*u*u - 2*u*u*u

    # Interpolate position linearly across segment (short enough to ignore GC curvature)
    a = seg["a"]; b = seg["b"]
    lat = a[0] + (b[0] - a[0]) * ue
    lon = a[1] + (b[1] - a[1]) * ue

    # Heading from bearing, with slight smoothing and light noise
    hdg = _bearing_deg(a[0], a[1], b[0], b[1])
    hdg += 2.5*sin((elapsed+route["seed"]) * 0.05)  # gentle weave
    hdg = (hdg + 360.0) % 360.0

    # Speed: base per segment with a small sinusoidal variation
    spd_kt = float(seg["spd_kt"]) + 5.0*sin((elapsed+route["seed"]) * 0.1)
    spd_mps = spd_kt * 0.514444

    # Altitude: climb to cruise early, then small oscillations
    cruise_extra = 800.0 + 600.0*sin(route["seed"] * 6.28)
    climb_time = 180.0
    if elapsed < climb_time:
        alt_m = origin_alt_m + (cruise_extra * (elapsed/climb_time))
        vsi_ms = cruise_extra / climb_time
    else:
        wobble = 80.0 * sin((elapsed - climb_time) * 2*3.14159 / 120.0)
        alt_m = origin_alt_m + cruise_extra + wobble
        vsi_ms = (2*3.14159/120.0) * 80.0 * cos((elapsed - climb_time) * 2*3.14159 / 120.0)

    # Bank/roll estimated from turn rate (deg/s approx change in heading)
    hdg_future = (hdg + 5.0*sin((elapsed+0.2+route["seed"]) * 0.05)) % 360.0
    turn_rate = ((hdg_future - hdg + 540.0) % 360.0) - 180.0
    roll_deg = max(-25.0, min(25.0, turn_rate * 3.0))

    # Pitch approx from vertical speed vs forward speed
    pitch_deg = max(-10.0, min(10.0, degrees(atan2(vsi_ms, max(1.0, spd_mps)))))

    # Demo engine data
    rpm_pct = 75.0 + 15.0 * sin(elapsed * 0.3)  # 60-90% RPM variation
    fuel_flow_gph = 8.5 + 2.0 * sin(elapsed * 0.2)  # 6.5-10.5 GPH variation
    throttle_pct = 70.0 + 10.0 * sin(elapsed * 0.1)  # 60-80% throttle variation
    
    # Demo autopilot data (cycling through states)
    ap_cycle = int(elapsed / 30) % 3  # Change every 30 seconds
    ap_master = ap_cycle > 0
    ap_hdg_lock = ap_cycle == 1 or ap_cycle == 2
    ap_alt_lock = ap_cycle == 2
    
    return {
        "lat": lat,
        "lon": lon,
        "alt_m": alt_m,
        "spd_kt": spd_kt,
        "vsi_ms": vsi_ms,
        "hdg_deg": hdg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        # Aircraft information
        "aircraft": f"Demo Aircraft {callsign[-2:]}",
        "rpm_pct": rpm_pct,
        "fuel_flow_gph": fuel_flow_gph,
        "throttle_pct": throttle_pct,
        # Autopilot status
        "ap_master": ap_master,
        "ap_hdg_lock": ap_hdg_lock,
        "ap_alt_lock": ap_alt_lock,
        "ap_speed_hold": ap_cycle == 2,
        "ap_vs_hold": False,
        "ap_nav_lock": False,
        "ap_approach_arm": False,
        # Autopilot targets
        "ap_hdg_target": hdg + 10.0 if ap_hdg_lock else None,
        "ap_alt_target": alt_m + 50.0 if ap_alt_lock else None,
        "ap_speed_target": spd_kt if ap_cycle == 2 else None,
        "ap_vs_target": 0.0 if ap_cycle == 2 else None,
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
                    s = simulate_sample(time.time(), origin_lat, origin_lon, origin_alt_m, callsign)
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
        s = simulate_sample(time.time(), origin_lat, origin_lon, origin_alt_m, callsign)
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
    ap.add_argument("--debug", action="store_true", help="Enable debug logging for SimConnect variables")
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
