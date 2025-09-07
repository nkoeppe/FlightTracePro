#!/usr/bin/env python3
import sys
import time
import json
import asyncio
import threading
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot, QDateTime, QSettings
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QHBoxLayout, QVBoxLayout, QSystemTrayIcon,
    QMenu, QMessageBox, QStyle, QTextEdit, QDialog, QCheckBox, QDialogButtonBox,
    QFrame, QProgressBar
)
from PySide6.QtNetwork import QLocalServer, QLocalSocket


def meters_from_feet(ft: Optional[float]) -> Optional[float]:
    return None if ft is None else ft * 0.3048


class MSFSSource:
    def __init__(self, log_cb=None):
        self.sim = None
        self.areq = None
        self._log = (lambda level, msg: None) if log_cb is None else log_cb

    def start(self) -> bool:
        self._log('INFO', "Attempting SimConnect initialization...")
        
        # When frozen (PyInstaller), ensure the SimConnect DLL search path is available
        try:
            import os as _os, sys as _sys
            if getattr(_sys, 'frozen', False):
                # Add common extraction/installation dirs to the DLL search path
                cand_dirs = []
                cand_dirs.append(_os.path.dirname(_sys.executable))
                cand_dirs.append(getattr(_sys, '_MEIPASS', None))
                # Also check for a bundled SimConnect package directory next to the exe
                cand_dirs.append(_os.path.join(_os.path.dirname(_sys.executable), 'SimConnect'))
                seen = set()
                for d in cand_dirs:
                    if d and d not in seen and _os.path.isdir(d):
                        seen.add(d)
                        try:
                            # Python 3.8+ secure DLL loading API
                            from os import add_dll_directory as _add_dll_directory  # type: ignore
                            _add_dll_directory(d)
                        except Exception:
                            # Fallback: prepend to PATH for legacy search
                            try:
                                _os.environ['PATH'] = d + _os.pathsep + _os.environ.get('PATH', '')
                            except Exception:
                                pass
                # Best-effort pre-load if DLL sits next to the exe
                try:
                    import ctypes as _ct
                    for _nm in ('SimConnect.dll', 'SimConnect64.dll'):
                        _p = _os.path.join(_os.path.dirname(_sys.executable), _nm)
                        if _os.path.exists(_p):
                            try:
                                _ct.WinDLL(_p)
                                break
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        
        # Step 1: Try to import SimConnect
        try:
            self._log('DEBUG', "Importing SimConnect module...")
            from SimConnect import SimConnect, AircraftRequests
            self._log('INFO', "✓ SimConnect module imported successfully")
        except ImportError as e:
            self._log('ERROR', f"✗ SimConnect module not found: {e}")
            self._log('ERROR', "Install with: pip install SimConnect==0.4.26")
            return False
        except Exception as e:
            self._log('ERROR', f"✗ SimConnect import failed: {e}")
            return False
        
        # Step 2: Try to create SimConnect instance with different approaches
        connection_attempts = [
            ("Default connection", lambda: SimConnect()),
            ("Named connection", lambda: SimConnect(auto_connect=False)),
            ("Local connection", lambda: SimConnect(auto_connect=True)),
        ]
        
        for attempt_name, connect_func in connection_attempts:
            try:
                self._log('DEBUG', f"Trying {attempt_name}...")
                self.sim = connect_func()
                self._log('INFO', f"✓ {attempt_name} successful")
                break
            except Exception as e:
                self._log('DEBUG', f"✗ {attempt_name} failed: {e}")
                continue
        
        if not self.sim:
            self._log('ERROR', "✗ All SimConnect connection attempts failed")
            self._log('ERROR', "Troubleshooting:")
            self._log('ERROR', "1. Make sure MSFS 2020 is running and you're in a flight")
            self._log('ERROR', "2. Check MSFS Options > General > Developers > Enable SimConnect")
            self._log('ERROR', "3. Try restarting MSFS completely")
            self._log('ERROR', "4. Run as Administrator")
            return False
        
        # Step 3: Try to create AircraftRequests
        try:
            self._log('DEBUG', "Creating AircraftRequests...")
            self.areq = AircraftRequests(self.sim, _time=50)
            self._log('INFO', "✓ AircraftRequests ready - SimConnect fully initialized!")
            return True
        except Exception as e:
            self._log('ERROR', f"✗ Failed to create AircraftRequests: {e}")
            try:
                self.sim.quit()
            except:
                pass
            self.sim = None
            return False

    def read(self):
        if not self.areq:
            return None
        try:
            def gv(name, unit=None, alt=None):
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
            spd_kt = gv("AIRSPEED TRUE", unit="knots")
            if spd_kt is None:
                gv_fps = gv("GROUND VELOCITY", unit="feet per second")
                spd_kt = gv_fps * 0.592484 if gv_fps is not None else None
            vsi_fpm = gv("VERTICAL SPEED", unit="feet per minute")
            pitch = gv("PLANE PITCH DEGREES", unit="degrees")
            roll = gv("PLANE BANK DEGREES", unit="degrees")
            if lat is None or lon is None:
                lat = gv("GPS POSITION LAT", unit="degrees") or gv("GPS_POSITION_LAT", unit="degrees")
                lon = gv("GPS POSITION LON", unit="degrees") or gv("GPS_POSITION_LON", unit="degrees")
            if alt_ft is None:
                alt_ft = gv("INDICATED ALTITUDE", unit="feet")
            import math
            def rad_to_deg_if_needed(x):
                try:
                    if x is None:
                        return None
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
                "vsi_ms": (vsi_fpm * 0.00508) if vsi_fpm is not None else None,
                "hdg_deg": float(hdg_deg) if hdg_deg is not None else None,
                "pitch_deg": float(pitch_deg) if pitch_deg is not None else None,
                "roll_deg": float(roll_deg) if roll_deg is not None else None,
            }
        except Exception:
            return None


from math import sin, cos, radians, degrees, atan2, sqrt, pi
import random, time as _time

_SIM_ROUTES_GUI = {}

def _hash_cs(cs: str) -> int:
    h = 0
    for ch in cs or 'ACFT':
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h

def _hav_m(a,b,c,d):
    R=6371000.0
    dlat=radians(c-a); dlon=radians(d-b)
    x=sin(dlat/2)**2 + cos(radians(a))*cos(radians(c))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(x), sqrt(1-x))

def _dest(a,b,brg,dist):
    R=6371000.0; br=radians(brg); lat1=radians(a); lon1=radians(b); dr=dist/R
    sinLat = sin(lat1)*cos(dr) + cos(lat1)*sin(dr)*cos(br)
    lat2 = atan2(sinLat, sqrt(max(0.0,1.0 - sinLat*sinLat)))
    lon2 = lon1 + atan2(sin(br)*sin(dr)*cos(lat1), cos(dr) - sin(lat1)*sin(lat2))
    return degrees(lat2), (degrees(lon2)+540)%360-180

def _brg(a,b,c,d):
    y = sin(radians(d-b))*cos(radians(c))
    x = cos(radians(a))*sin(radians(c)) - sin(radians(a))*cos(radians(c))*cos(radians(d-b))
    return (degrees(atan2(y,x))+360)%360

def _route(cs, olat, olon):
    rnd = random.Random(_hash_cs(cs))
    n = rnd.randint(5,9)
    wps=[]; bearing=rnd.uniform(0,360)
    for _ in range(n):
        bearing=(bearing + rnd.uniform(25,90))%360
        dist_km=rnd.uniform(6,25)
        wps.append(_dest(olat, olon, bearing, dist_km*1000))
    wps.append(wps[0])
    speeds=[rnd.uniform(80,140) for _ in range(len(wps)-1)]
    segs=[]; t0=0.0
    for i in range(len(wps)-1):
        a=wps[i]; b=wps[i+1]; d=_hav_m(a[0],a[1],b[0],b[1])
        dur=max(30.0, d/ max(30.0, speeds[i]*0.514444))
        segs.append(dict(a=a,b=b,d=d,spd_kt=speeds[i],dur=dur,t0=t0,t1=t0+dur))
        t0+=dur
    return dict(wps=wps,segs=segs,loop=t0,seed=rnd.random(),start=_time.time())

def _get_route(cs, olat, olon):
    r=_SIM_ROUTES_GUI.get(cs)
    if not r:
        r=_route(cs,olat,olon)
        _SIM_ROUTES_GUI[cs]=r
    return r

def simulate_sample(t: float, origin_lat: float = 47.3769, origin_lon: float = 8.5417, origin_alt_m: float = 500.0, callsign: str = 'ACFT'):
    r=_get_route(callsign, origin_lat, origin_lon)
    el=(t - r['start']); loop=r['loop'] or 1.0; tm=el%loop
    seg=r['segs'][-1]
    for s in r['segs']:
        if s['t0']<=tm<=s['t1']:
            seg=s; break
    u=(tm - seg['t0'])/max(1e-6,(seg['t1']-seg['t0']))
    ue=3*u*u - 2*u*u*u
    a=seg['a']; b=seg['b']
    lat=a[0] + (b[0]-a[0])*ue
    lon=a[1] + (b[1]-a[1])*ue
    hdg=_brg(a[0],a[1],b[0],b[1])
    hdg=(hdg + 2.5*sin((el+r['seed'])*0.05))%360
    spd_kt=float(seg['spd_kt']) + 5.0*sin((el+r['seed'])*0.1)
    spd_mps=spd_kt*0.514444
    cruise_extra=800.0 + 600.0*sin(r['seed']*6.28)
    climb_time=180.0
    if el < climb_time:
        alt_m = origin_alt_m + (cruise_extra * (el/climb_time))
        vsi_ms = cruise_extra/climb_time
    else:
        wobble = 80.0 * sin((el - climb_time) * 2*pi / 120.0)
        alt_m = origin_alt_m + cruise_extra + wobble
        vsi_ms = (2*pi/120.0) * 80.0 * cos((el - climb_time) * 2*pi / 120.0)
    hdg_future=(hdg + 5.0*sin((el+0.2+r['seed'])*0.05))%360
    turn_rate=((hdg_future-hdg+540)%360)-180
    roll_deg=max(-25.0,min(25.0,turn_rate*3.0))
    pitch_deg=max(-10.0,min(10.0,degrees(atan2(vsi_ms,max(1.0,spd_mps)))))
    return {
        'lat':lat,'lon':lon,'alt_m':alt_m,
        'spd_kt':spd_kt,'vsi_ms':vsi_ms,'hdg_deg':hdg,
        'pitch_deg':pitch_deg,'roll_deg':roll_deg,
    }

LEVELS = {"DEBUG":10, "INFO":20, "WARN":30, "ERROR":40}


class BridgeWorker(QThread):
    status = Signal(str)
    connected = Signal(bool)
    log = Signal(str)

    def __init__(self, mode: str, server: str, channel: str, callsign: str, key: Optional[str], rate_hz: float, demo: bool=False, origin_lat: float=47.3769, origin_lon: float=8.5417, origin_alt: float=500.0, level: str = "INFO"):
        super().__init__()
        self._mode = mode
        self._server = server.rstrip('/')
        self._channel = channel
        self._callsign = callsign
        self._key = key
        self._rate_hz = max(0.2, rate_hz)
        self._demo = demo
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._origin_alt = origin_alt
        self._min_level = LEVELS.get(level.upper(), 20)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()
    
    def cleanup(self):
        """Clean up the worker thread properly"""
        self.stop()
        if self.isRunning():
            self.wait(2000)  # Wait up to 2 seconds
        
        # Disconnect all signals
        try:
            self.status.disconnect()
            self.connected.disconnect()
            self.log.disconnect()
        except:
            pass

    @Slot(str)
    def set_level(self, level: str):
        self._min_level = LEVELS.get(level.upper(), 20)

    def _log(self, level: str, msg: str):
        if LEVELS.get(level.upper(), 20) >= self._min_level:
            self.log.emit(f"[{level}] {msg}")

    async def _run_ws(self):
        try:
            import websockets
        except Exception:
            self.status.emit("websockets library not installed")
            self.log.emit("websockets library not installed")
            return
        src = MSFSSource(log_cb=self._log)
        self._log("DEBUG", "Attempting SimConnect/WebSocket loop start")
        url = f"{self._server}/ws/live/{self._channel}?mode=feeder" + (f"&key={self._key}" if self._key else "")
        dt = 1.0 / self._rate_hz
        while not self._stop.is_set():
            # Demo mode: synthesize samples without SimConnect
            if self._demo:
                try:
                    async with websockets.connect(url, max_size=1_000_000, compression=None, ping_interval=20, ping_timeout=20) as ws:
                        self.status.emit("connected (demo)")
                        self._log("INFO", f"WS connected (demo) to {url}")
                        self.connected.emit(True)
                        while not self._stop.is_set():
                            s = simulate_sample(time.time(), self._origin_lat, self._origin_lon, self._origin_alt, self._callsign)
                            s["callsign"] = self._callsign
                            s["ts"] = time.time()
                            await ws.send(json.dumps({"type": "state", "payload": s}))
                            self._log("DEBUG", f"sent (demo) lat={s['lat']:.5f} lon={s['lon']:.5f} alt={s['alt_m']:.0f}")
                            await asyncio.sleep(dt)
                except Exception as e:
                    self.status.emit(f"ws error: {e}")
                    self._log("ERROR", f"WS error (demo): {e}")
                    self.connected.emit(False)
                    await asyncio.sleep(2)
                continue
            # ensure simconnect is up for non-demo
            if not src.areq:
                if not src.start():
                    self.status.emit("waiting for MSFS/SimConnect…")
                    self._log("INFO", "waiting for MSFS/SimConnect…")
                    await asyncio.sleep(2)
                    continue
                else:
                    self._log("INFO", "SimConnect ready")
            try:
                async with websockets.connect(url, max_size=1_000_000, compression=None, ping_interval=20, ping_timeout=20) as ws:
                    self.status.emit("connected")
                    self._log("INFO", f"WS connected to {url}")
                    self.connected.emit(True)
                    while not self._stop.is_set():
                        s = src.read()
                        if s:
                            # Log raw sample for debugging (structure/values)
                            try:
                                self._log("DEBUG", f"raw sample: {s}")
                            except Exception:
                                pass
                        if s and (s.get('lat') is not None) and (s.get('lon') is not None):
                            s["callsign"] = self._callsign
                            s["ts"] = time.time()
                            await ws.send(json.dumps({"type": "state", "payload": s}))
                            try:
                                self._log("DEBUG", f"sent lat={s['lat']:.5f} lon={s['lon']:.5f} alt={(s['alt_m'] or 0):.0f}")
                            except Exception:
                                pass
                        else:
                            # Simplified debug when no valid sample present
                            self._log("DEBUG", "sample unavailable (lat/lon missing)")
                        await asyncio.sleep(dt)
            except Exception as e:
                self.status.emit(f"ws error: {e}")
                self._log("ERROR", f"WS error: {e}")
                self.connected.emit(False)
                await asyncio.sleep(2)
                src.areq = None
        self.connected.emit(False)

    async def _run_http(self):
        try:
            import requests
        except Exception:
            self.status.emit("requests library not installed")
            return
        src = MSFSSource(log_cb=self._log)
        url = f"{self._server}/api/live/{self._channel}"
        dt = 1.0 / self._rate_hz
        while not self._stop.is_set():
            if self._demo:
                self.connected.emit(True)
                try:
                    s = simulate_sample(time.time(), self._origin_lat, self._origin_lon, self._origin_alt, self._callsign)
                    s["callsign"] = self._callsign
                    s["ts"] = time.time()
                    params = {"key": self._key} if self._key else None
                    requests.post(url, params=params, json=s, timeout=3)
                    self._log("DEBUG", f"posted (demo) lat={s['lat']:.5f} lon={s['lon']:.5f} alt={s['alt_m']:.0f}")
                except Exception as e:
                    self.status.emit(f"http error: {e}")
                    self._log("ERROR", f"HTTP error (demo): {e}")
                await asyncio.sleep(dt)
                continue
            if not src.areq:
                if not src.start():
                    self.status.emit("waiting for MSFS/SimConnect…")
                    self._log("INFO", "waiting for MSFS/SimConnect…")
                    await asyncio.sleep(2)
                    continue
                else:
                    self._log("INFO", "SimConnect ready")
            self.connected.emit(True)
            try:
                s = src.read()
                if s:
                    try:
                        self._log("DEBUG", f"raw sample: {s}")
                    except Exception:
                        pass
                if s and (s.get('lat') is not None) and (s.get('lon') is not None):
                    s["callsign"] = self._callsign
                    s["ts"] = time.time()
                    params = {"key": self._key} if self._key else None
                    requests.post(url, params=params, json=s, timeout=3)
                    try:
                        self._log("DEBUG", f"posted lat={s['lat']:.5f} lon={s['lon']:.5f} alt={(s['alt_m'] or 0):.0f}")
                    except Exception:
                        pass
                else:
                    # Simplified debug when no valid sample present
                    self._log("DEBUG", "sample unavailable (lat/lon missing)")
            except Exception as e:
                self.status.emit(f"http error: {e}")
                self._log("ERROR", f"HTTP error: {e}")
            await asyncio.sleep(dt)
        self.connected.emit(False)

    def run(self):
        try:
            if self._mode == 'ws':
                asyncio.run(self._run_ws())
            else:
                asyncio.run(self._run_http())
        except Exception as e:
            self.status.emit(str(e))
            self.connected.emit(False)


class MainWindow(QMainWindow):
    # Signal for update notifications from background thread
    update_info_signal = Signal(str, str)  # version, download_url
    # Signal for deferred update processing from main thread
    deferred_update_signal = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlightTracePro - Bridge")
        self.setMinimumWidth(520)
        self.worker: Optional[BridgeWorker] = None
        self.log_entries = []  # Store all log entries with their levels

        w = QWidget(); self.setCentralWidget(w)
        lay = QGridLayout(); w.setLayout(lay)

        row = 0
        lay.addWidget(QLabel("Server"), row, 0)
        self.server = QComboBox(); self.server.setEditable(True); self.server.setInsertPolicy(QComboBox.NoInsert); lay.addWidget(self.server, row, 1, 1, 2)
        self.btnSaveServer = QPushButton("★ Save"); self.btnSaveServer.setToolTip("Save server to recents"); lay.addWidget(self.btnSaveServer, row, 3); row += 1

        lay.addWidget(QLabel("Channel"), row, 0)
        self.channel = QLineEdit("default"); lay.addWidget(self.channel, row, 1)
        lay.addWidget(QLabel("Callsign"), row, 2)
        self.callsign = QLineEdit("N123AB"); lay.addWidget(self.callsign, row, 3); row += 1

        lay.addWidget(QLabel("Post Key"), row, 0)
        self.key = QLineEdit(""); self.key.setEchoMode(QLineEdit.Password); lay.addWidget(self.key, row, 1)
        lay.addWidget(QLabel("Mode"), row, 2)
        self.mode = QComboBox(); self.mode.addItems(["ws", "http"]); lay.addWidget(self.mode, row, 3); row += 1

        lay.addWidget(QLabel("Rate (Hz)"), row, 0)
        self.rate = QSpinBox(); self.rate.setRange(1, 30); self.rate.setValue(10); lay.addWidget(self.rate, row, 1)
        self.status = QLabel("disconnected"); self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter); lay.addWidget(self.status, row, 2, 1, 2); row += 1

        btns = QHBoxLayout()
        self.demoBtn = QPushButton("Demo Mode: Off"); self.demoBtn.setCheckable(True); self.demoBtn.clicked.connect(lambda: self.demoBtn.setText(f"Demo Mode: {'On' if self.demoBtn.isChecked() else 'Off'}")); btns.addWidget(self.demoBtn)
        self.btnConnect = QPushButton("Connect"); btns.addWidget(self.btnConnect)
        self.btnDisconnect = QPushButton("Disconnect"); btns.addWidget(self.btnDisconnect)
        lay.addLayout(btns, row, 0, 1, 4); row += 1

        # Log viewer
        lay.addWidget(QLabel("Logs"), row, 0)
        self.logView = QTextEdit(); self.logView.setReadOnly(True); self.logView.setMinimumHeight(200)
        lay.addWidget(self.logView, row, 1, 1, 2)
        
        # Log controls in vertical layout
        logControls = QVBoxLayout()
        self.logLevel = QComboBox(); self.logLevel.addItems(["DEBUG","INFO","WARN","ERROR"]); self.logLevel.setCurrentText("INFO"); logControls.addWidget(self.logLevel)
        self.btnClearLogs = QPushButton("Clear Logs"); logControls.addWidget(self.btnClearLogs)
        # Hide test update button (keep for debugging)
        self.btnTestUpdate = QPushButton("Test Update"); self.btnTestUpdate.setToolTip("Test update mechanism with fake update"); self.btnTestUpdate.hide()
        # Add proper Check for Updates button
        self.btnCheckUpdates = QPushButton("Check for Updates"); self.btnCheckUpdates.setToolTip("Check for new versions on GitHub"); logControls.addWidget(self.btnCheckUpdates)
        # Add Options button
        self.btnOptions = QPushButton("Options"); self.btnOptions.setToolTip("Open application settings"); logControls.addWidget(self.btnOptions)
        logControlsWidget = QWidget(); logControlsWidget.setLayout(logControls)
        lay.addWidget(logControlsWidget, row, 3); row += 1
        
        self.logLevel.currentTextChanged.connect(self.on_level_changed)
        self.btnClearLogs.clicked.connect(self.clear_logs)
        self.btnTestUpdate.clicked.connect(self.test_update_mechanism)
        self.btnCheckUpdates.clicked.connect(self.check_update_async)
        self.btnOptions.clicked.connect(self.show_options_dialog)

        self.btnConnect.clicked.connect(self.on_connect)
        self.btnDisconnect.clicked.connect(self.on_disconnect)
        self.btnSaveServer.clicked.connect(self.on_save_server)

        # Tray
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.tray_menu = QMenu()
        act_show = QAction("Open", self); act_show.triggered.connect(self.showNormal); self.tray_menu.addAction(act_show)
        act_quit = QAction("Quit", self); act_quit.triggered.connect(self.quit_application); self.tray_menu.addAction(act_quit)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self._on_tray)
        self.tray.show()

        # Connect update signals
        self.update_info_signal.connect(self._show_update_prompt)
        self.deferred_update_signal.connect(self.check_for_deferred_update)

        # Recents storage
        self.settings = QSettings("FlightTracePro", "Bridge")
        self.load_recents()
        
        # Check for deferred updates first, then auto-update if no deferred update
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1000, self.check_startup_updates)
    
    def check_startup_updates(self):
        """Check for deferred updates first, then auto-updates if none pending"""
        self.append_log("[update] Checking for updates on startup...")
        
        # First check for deferred updates
        import os, tempfile
        tmpdir = tempfile.gettempdir()
        marker_file = os.path.join(tmpdir, 'flighttracepro_deferred_update.marker')
        
        if os.path.exists(marker_file):
            self.append_log("[update] Deferred update found, processing...")
            # Use signal to ensure GUI thread execution
            self.deferred_update_signal.emit()
        else:
            self.append_log("[update] No deferred update, checking for new updates...")
            # Auto-update check (best-effort) - only if no deferred update
            try:
                self.check_update_async()
            except Exception:
                pass

    def _on_tray(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            if self.isMinimized() or not self.isVisible():
                self.showNormal(); self.activateWindow()
            else:
                self.hide()

    def closeEvent(self, e):
        # Check if user has set a remembered preference
        close_action = self.settings.value("closeAction", "ask", type=str)
        
        if close_action == "minimize":
            self._minimize_to_tray()
            e.ignore()
        elif close_action == "quit":
            self.quit_application()
            e.accept()
        else:
            # Show custom close dialog
            dialog = CloseDialog(self)
            result = dialog.exec()
            
            if result == QDialog.DialogCode.Accepted:
                action, remember = dialog.get_result()
                
                if remember:
                    self.settings.setValue("closeAction", action)
                
                if action == "minimize":
                    self._minimize_to_tray()
                    e.ignore()
                else:  # quit
                    self.quit_application()
                    e.accept()
            else:
                # Cancel - do nothing
                e.ignore()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, 'worker') and self.worker:
                self.worker.cleanup()
        except:
            pass
    
    def _minimize_to_tray(self):
        """Minimize to tray with notification"""
        self.hide()
        
        # Show tray notification
        if self.tray and self.tray.supportsMessages():
            self.tray.showMessage(
                "FlightTracePro",
                "Application is running in the background. Click the tray icon to restore.",
                QSystemTrayIcon.MessageIcon.Information,
                3000  # 3 seconds
            )

    def quit_application(self):
        """Properly quit the application"""
        # Stop worker if running
        if self.worker:
            self.worker.cleanup()
            self.worker = None
        
        # Disconnect all signals to prevent thread storage issues
        try:
            self.update_info_signal.disconnect()
            self.deferred_update_signal.disconnect()
        except:
            pass
        
        # Hide tray icon
        if self.tray:
            self.tray.hide()
        
        # Process pending events before cleanup
        QApplication.processEvents()
        
        # Force cleanup of Qt objects
        self.deleteLater()
        
        # Process events again to handle deleteLater
        QApplication.processEvents()
        
        # Quit application
        QApplication.quit()

    @Slot()
    def on_connect(self):
        if self.worker and self.worker.isRunning():
            return
        server = self.server.currentText().strip()
        mode = self.mode.currentText()
        if mode == 'ws' and not (server.startswith('ws://') or server.startswith('wss://')):
            QMessageBox.warning(self, "Server", "For ws mode, server must start with ws:// or wss://")
            return
        if mode == 'http' and not (server.startswith('http://') or server.startswith('https://')):
            QMessageBox.warning(self, "Server", "For http mode, server must start with http:// or https://")
            return
        self.worker = BridgeWorker(
            mode=mode,
            server=server,
            channel=self.channel.text().strip() or 'default',
            callsign=self.callsign.text().strip() or 'N123AB',
            key=(self.key.text().strip() or None),
            rate_hz=float(self.rate.value()),
            demo=self.demoBtn.isChecked(),
            level=self.logLevel.currentText(),
        )
        self.worker.status.connect(self.on_status)
        self.worker.log.connect(self.append_log)
        self.worker.connected.connect(self.on_connected)
        self.worker.start()
        self.on_status("connecting…")
        # Save server MRU
        self.save_recent_server(server)

    @Slot()
    def on_disconnect(self):
        if self.worker:
            self.worker.cleanup()
            self.worker = None
        self.on_connected(False)

    @Slot(str)
    def on_status(self, msg: str):
        self.status.setText(msg)
        self.tray.setToolTip(f"FlightTracePro – {msg}")
        self.append_log(msg)

    @Slot(bool)
    def on_connected(self, ok: bool):
        self.status.setText("connected" if ok else "disconnected")
        self.tray.setToolTip(f"FlightTracePro – {'connected' if ok else 'disconnected'}")
        self.append_log("connected" if ok else "disconnected")

    def append_log(self, msg: str):
        try:
            ts = QDateTime.currentDateTime().toString("hh:mm:ss")
        except Exception:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%H:%M:%S")
        
        # Extract log level from message if present
        level_value = 20  # Default to INFO level
        if msg.startswith('[') and '] ' in msg:
            level_part = msg.split('] ', 1)[0][1:]
            level_value = LEVELS.get(level_part.upper(), 20)
        
        display_text = f"[{ts}] {msg}"
        
        # Store the log entry
        log_entry = {
            'timestamp': ts,
            'message': msg,
            'level_value': level_value,
            'display_text': display_text
        }
        self.log_entries.append(log_entry)
        
        # Only display if it matches current filter level
        current_level = self.logLevel.currentText()
        min_level = LEVELS.get(current_level.upper(), 20)
        if level_value >= min_level:
            self.logView.append(display_text)

    def load_recents(self):
        rec = self.settings.value("recentServers", [], type=list)
        last = self.settings.value("lastServer", "ws://localhost:8000", type=str)
        self.server.clear()
        for item in rec:
            self.server.addItem(item)
        if last and last not in rec:
            self.server.insertItem(0, last)
        self.server.setCurrentText(last or (rec[0] if rec else "ws://localhost:8000"))

    def save_recent_server(self, url: str):
        url = (url or '').strip()
        if not url:
            return
        rec = self.settings.value("recentServers", [], type=list)
        rec = [x for x in rec if x != url]
        rec.insert(0, url)
        rec = rec[:10]
        self.settings.setValue("recentServers", rec)
        self.settings.setValue("lastServer", url)
        self.server.blockSignals(True)
        self.server.clear()
        for item in rec:
            self.server.addItem(item)
        self.server.setCurrentText(url)
        self.server.blockSignals(False)

    @Slot()
    def on_save_server(self):
        self.save_recent_server(self.server.currentText().strip())

    @Slot()
    def clear_logs(self):
        self.log_entries.clear()
        self.logView.clear()
        self.append_log("Logs cleared")

    @Slot()
    def on_level_changed(self):
        if self.worker:
            self.worker.set_level(self.logLevel.currentText())
        # Filter logs based on new level
        self.filter_logs()
        
    def filter_logs(self):
        """Filter displayed logs based on current log level"""
        current_level = self.logLevel.currentText()
        min_level = LEVELS.get(current_level.upper(), 20)
        
        self.logView.clear()
        for entry in self.log_entries:
            if entry['level_value'] >= min_level:
                self.logView.append(entry['display_text'])

    # --- Self-update (GitHub Releases) ---
    def check_update_async(self):
        # Disable button and show checking status
        self.btnCheckUpdates.setEnabled(False)
        self.btnCheckUpdates.setText("Checking...")
        self.append_log("[update] Manually checking for updates...")
        
        import threading
        t = threading.Thread(target=self._check_update_with_ui_reset, daemon=True)
        t.start()
    
    def _check_update_with_ui_reset(self):
        """Wrapper for _check_update that resets UI afterwards"""
        try:
            self._check_update()
        finally:
            # Reset button state
            if hasattr(self, 'btnCheckUpdates'):
                self.btnCheckUpdates.setEnabled(True)
                self.btnCheckUpdates.setText("Check for Updates")

    def _get_app_version(self):
        """Get app version from VERSION file or fallback to hardcoded"""
        return get_app_version()

    def _check_update(self):
        import os, sys, json, tempfile, time
        try:
            import requests
        except ImportError:
            self.append_log("[update] requests library not available for auto-update")
            return
        APP_VERSION = self._get_app_version()
        REPO = os.environ.get('FLIGHTTRACEPRO_REPO', 'nkoeppe/FlightTracePro')  # set to 'owner/repo'
        if REPO == 'your-user/your-repo':
            return
        
        self.append_log(f"[update] Checking for updates from {REPO} (current: {APP_VERSION})")
        try:
            r = requests.get(f'https://api.github.com/repos/{REPO}/releases/latest', timeout=5)
            if r.status_code != 200:
                return
            rel = r.json()
            tag = rel.get('tag_name','')
            # Handle multiple tag formats: v1.0.0, bridge-v1.0.0, flighttracepro-v1.0.0-10
            ver = tag
            # Strip common prefixes
            for prefix in ['flighttracepro-v', 'bridge-v', 'v']:
                if ver.startswith(prefix):
                    ver = ver[len(prefix):]
                    break
            
            def parse(v):
                # Extract version numbers from string like "0.1.0-10" -> ["0", "1", "0", "10"]
                import re
                numbers = re.findall(r'\d+', v)
                return tuple(int(n) for n in numbers) if numbers else (0,)
            
            def compare_versions(current, latest):
                # Pad shorter version with zeros for proper comparison
                # e.g., (0, 2) becomes (0, 2, 0) to compare with (0, 2, 27)
                max_len = max(len(current), len(latest))
                current_padded = current + (0,) * (max_len - len(current))
                latest_padded = latest + (0,) * (max_len - len(latest))
                return latest_padded > current_padded
            
            current_version = parse(APP_VERSION)
            latest_version = parse(ver)
            self.append_log(f"[update] Current: {APP_VERSION} ({current_version}), Latest: {ver} ({latest_version})")
            
            if not compare_versions(current_version, latest_version):
                self.append_log(f"[update] ✅ Already up to date (v{APP_VERSION} is latest)")
                return
            # find exe asset
            assets = rel.get('assets', [])
            url = None
            for a in assets:
                n = a.get('name','')
                if n.lower().endswith('.exe'):
                    url = a.get('browser_download_url')
                    break
            if not url:
                self.append_log("[update] No .exe asset found in latest release")
                return
            
            # Prompt - need to use signal to show from main thread
            self.update_info_signal.emit(ver, url)
            
        except Exception as e:
            self.append_log(f"[update] Update check failed: {e}")
    
    def _show_update_prompt(self, version, download_url):
        """Show update prompt in main thread"""
        # Show modern update dialog
        dialog = UpdateDialog(version, download_url, self)
        result = dialog.exec()
        
        if result == UpdateDialog.UpdateAction.INSTALL_NOW:
            # Download and install immediately
            self.append_log("[update] User chose to install update now - starting download...")
            import threading
            t = threading.Thread(target=self._download_and_install, args=(version, download_url, False), daemon=True)
            t.start()
        elif result == UpdateDialog.UpdateAction.INSTALL_LATER:
            # Download and defer restart
            self.append_log("[update] User chose to install later - downloading for deferred installation...")
            import threading
            t = threading.Thread(target=self._download_and_install, args=(version, download_url, True), daemon=True)
            t.start()
    
    def test_update_mechanism(self):
        """Test the update mechanism with the current executable"""
        import os, sys, tempfile, shutil
        
        self.append_log("[test] Testing update mechanism...")
        
        if not getattr(sys, 'frozen', False):
            self.append_log("[test] Not running as frozen executable - test not applicable")
            return
            
        try:
            # Create fake "new" version by copying current exe
            cur = os.path.abspath(sys.executable)
            tmpdir = tempfile.gettempdir()
            newexe = os.path.join(tmpdir, 'FlightTracePro_test.exe')
            
            self.append_log(f"[test] Current exe: {cur}")
            self.append_log(f"[test] Test exe: {newexe}")
            
            # Copy current exe as test update
            shutil.copy2(cur, newexe)
            self.append_log(f"[test] Created test update file")
            
            # Create test batch file
            bat = os.path.join(tmpdir, 'flighttracepro_test_update.bat')
            log_file = os.path.join(tmpdir, 'flighttracepro_test_update.log')
            
            with open(bat, 'w') as bf:
                bf.write(f"""@echo off
title FlightTracePro Test Updater

REM Define paths as variables
set "TARGET_EXE={cur}"
set "NEW_EXE={newexe}"
set "LOGFILE={log_file}"

echo [%date% %time%] FlightTracePro TEST Auto-Updater Started >> %LOGFILE%
echo [%date% %time%] Target: %TARGET_EXE% >> %LOGFILE%
echo [%date% %time%] New EXE: %NEW_EXE% >> %LOGFILE%

echo.
echo ========================================
echo FlightTracePro TEST Auto-Updater
echo ========================================
echo.
echo This is a TEST - no actual update will occur
echo Log file: %LOGFILE%
echo.

echo Step 1: Waiting for application to close...
echo [%date% %time%] Step 1: Waiting for app to close... >> %LOGFILE%
timeout /t 2 /nobreak >nul

echo Step 2: TEST - Would create backup here...
echo [%date% %time%] Step 2: TEST backup creation >> %LOGFILE%

echo Step 3: TEST - Would install new version here...
echo [%date% %time%] Step 3: TEST installation >> %LOGFILE%

echo Step 4: TEST - Would clean up here...
echo [%date% %time%] Step 4: TEST cleanup >> %LOGFILE%

echo Step 5: Test complete - restarting application...
echo [%date% %time%] Step 5: Test complete - restarting application... >> %LOGFILE%

echo Starting: %TARGET_EXE%
echo [%date% %time%] Starting: %TARGET_EXE% >> %LOGFILE%
start "" %TARGET_EXE%
echo [%date% %time%] TEST UPDATE COMPLETE! Application restarted >> %LOGFILE%

echo [%date% %time%] TEST update process completed >> %LOGFILE%

REM Cleanup test files
echo [%date% %time%] Cleaning up test files... >> %LOGFILE%
del %NEW_EXE% >nul 2>&1
del "%~f0" >nul 2>&1
""")
            
            self.append_log(f"[test] Created test batch file: {bat}")
            self.append_log(f"[test] Test log will be: {log_file}")
            
            # Now test the same launch mechanism as the real updater
            launched = False
            
            # Method 1: subprocess
            try:
                import subprocess
                self.append_log(f"[test] Testing subprocess launch...")
                
                if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                    proc = subprocess.Popen(
                        ['cmd.exe', '/c', bat],
                        shell=False,
                        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
                        cwd=os.path.dirname(bat),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    proc = subprocess.Popen(bat, shell=True, cwd=os.path.dirname(bat))
                
                self.append_log(f"[test] Subprocess started with PID: {proc.pid}")
                launched = True
            except Exception as e:
                self.append_log(f"[test] Subprocess failed: {e}")
                
                # Try cmd wrapper
                try:
                    proc = subprocess.Popen(
                        ['cmd.exe', '/c', 'start', '/min', bat],
                        shell=False,
                        creationflags=subprocess.DETACHED_PROCESS if hasattr(subprocess, 'DETACHED_PROCESS') else 0,
                        cwd=os.path.dirname(bat)
                    )
                    self.append_log(f"[test] CMD wrapper started with PID: {proc.pid}")
                    launched = True
                except Exception as cmd_error:
                    self.append_log(f"[test] CMD wrapper failed: {cmd_error}")
            
            # Method 2: os.startfile
            if not launched:
                try:
                    os.startfile(bat)
                    self.append_log(f"[test] Started with os.startfile")
                    launched = True
                except Exception as e:
                    self.append_log(f"[test] os.startfile failed: {e}")
            
            if launched:
                self.append_log(f"[test] Test batch launched successfully!")
                self.append_log(f"[test] Check console window and log file: {log_file}")
                self.append_log(f"[test] Application will restart in test mode")
            else:
                self.append_log(f"[test] All launch methods failed!")
                self.append_log(f"[test] Try running manually: {bat}")
                
        except Exception as e:
            self.append_log(f"[test] Test failed: {e}")

    def _download_and_install(self, version, url, install_later=False):
        """Download and install update"""
        try:
            import os, sys, tempfile
            try:
                import requests
            except ImportError:
                self.append_log("[update] requests library not available for download")
                return
                
            self.append_log(f"[update] Downloading {version}...")
            
            # Download
            tmpdir = tempfile.gettempdir()
            newexe = os.path.join(tmpdir, 'FlightTracePro_new.exe')
            with requests.get(url, stream=True, timeout=30) as rr:
                rr.raise_for_status()
                with open(newexe, 'wb') as f:
                    for chunk in rr.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            self.append_log(f"[update] Download completed: {newexe}")
            
            # Only works when frozen
            if not getattr(sys, 'frozen', False):
                self.append_log(f"[update] Not running as frozen executable, manual restart required")
                # Can't use QMessageBox from background thread, use log instead
                return
            cur = os.path.abspath(sys.executable)
            bat = os.path.join(tmpdir, 'flighttracepro_update.bat')
            
            # Create batch file with logging
            log_file = os.path.join(tmpdir, 'flighttracepro_update.log')
            
            # Escape paths for Windows batch - replace problematic characters
            def batch_escape(path):
                # Convert forward slashes to backslashes and wrap in quotes
                return f'"{path.replace("/", "\\")}"'
            
            cur_safe = batch_escape(cur)
            newexe_safe = batch_escape(newexe)  
            log_safe = batch_escape(log_file)
            
            # Create enhanced batch file for better deferred update support
            deferred_mode = "true" if install_later else "false"
            
            with open(bat, 'w', encoding='cp1252') as bf:
                bf.write(f"""@echo off
setlocal enabledelayedexpansion
title FlightTracePro Updater v{version}

REM Define paths as variables  
set TARGET_EXE={cur_safe}
set NEW_EXE={newexe_safe}
set LOGFILE={log_safe}
set DEFERRED_MODE={deferred_mode}
set VERSION={version}

REM Normalize paths for operations needing suffixes
set TARGET_NOQUOTE=%TARGET_EXE:"=%
set NEW_NOQUOTE=%NEW_EXE:"=%
set BACKUP_EXE=%TARGET_NOQUOTE%.backup

echo [%date% %time%] FlightTracePro Auto-Updater v%VERSION% Started >> %LOGFILE%
echo [%date% %time%] Target: %TARGET_EXE% >> %LOGFILE%
echo [%date% %time%] New EXE: %NEW_EXE% >> %LOGFILE%
echo [%date% %time%] Deferred Mode: %DEFERRED_MODE% >> %LOGFILE%
echo [%date% %time%] Batch: %~f0 >> %LOGFILE%

echo.
echo ========================================
echo FlightTracePro Auto-Updater v%VERSION%
echo ========================================
echo.
if "%DEFERRED_MODE%"=="true" (
    echo Mode: Deferred Update ^(triggered by user restart^)
) else (
    echo Mode: Immediate Update ^(triggered by update check^)
)
echo Log file: %LOGFILE%
echo.

echo Step 1: Waiting for application to close...
echo [%date% %time%] Step 1: Waiting for app to close... >> %LOGFILE%
if "%DEFERRED_MODE%"=="true" (
    REM For deferred updates, wait a bit longer to ensure proper shutdown
    echo Waiting for deferred update startup to complete...
    timeout /t 3 /nobreak >nul
) else (
    REM Quick check that process has exited for immediate updates
    timeout /t 1 /nobreak >nul
)

echo Step 2: Verifying update environment...
echo [%date% %time%] Step 2: Verifying update environment... >> %LOGFILE%

REM Check if we're in the right directory
if not exist %TARGET_EXE% (
    echo [%date% %time%] ERROR: Target EXE not found: %TARGET_EXE% >> %LOGFILE%
    echo ERROR: Target application not found!
    echo This may happen if FlightTracePro was moved after the update was prepared.
    echo Please download and install the update manually.
    echo.
    pause
    exit /b 1
)

echo Step 3: Creating backup...
echo [%date% %time%] Step 3: Creating backup... >> %LOGFILE%
copy %TARGET_EXE% %BACKUP_EXE% >nul 2>&1
if errorlevel 1 (
    echo [%date% %time%] ERROR: Could not create backup! >> %LOGFILE%
    echo ERROR: Could not create backup!
    echo This may be due to file permissions or antivirus interference.
    echo Please run as Administrator or temporarily disable antivirus.
    echo.
    pause
    exit /b 1
) else (
    echo [%date% %time%] Backup created successfully >> %LOGFILE%
    echo Backup created successfully
)

echo Step 4: Installing new version...
echo [%date% %time%] Step 4: Installing new version... >> %LOGFILE%

REM Wait a moment for any file system delays
timeout /t 2 /nobreak >nul

REM Check if new EXE exists with retry logic
set FILE_CHECK_RETRY=0
:check_new_exe
set /a FILE_CHECK_RETRY+=1
if exist %NEW_EXE% (
    echo [%date% %time%] New EXE found on check attempt !FILE_CHECK_RETRY! >> %LOGFILE%
    goto file_check_success
)

echo [%date% %time%] New EXE check attempt !FILE_CHECK_RETRY!/5: File not found >> %LOGFILE%
if !FILE_CHECK_RETRY! LSS 5 (
    timeout /t 1 /nobreak >nul
    goto check_new_exe
)

echo [%date% %time%] ERROR: New EXE not found after 5 attempts: %NEW_EXE% >> %LOGFILE%
echo ERROR: New EXE not found after 5 attempts!
pause
exit /b 1

:file_check_success

echo [%date% %time%] New EXE found, attempting to install with aggressive retry... >> %LOGFILE%

REM Simple but aggressive retry with longer waits
set UPDATE_RETRY=0
:update_retry
set /a UPDATE_RETRY+=1
echo [%date% %time%] Update attempt !UPDATE_RETRY!/10 >> %LOGFILE%
echo Installing new version (attempt !UPDATE_RETRY!/10)

REM Delete target first to reduce conflicts
if exist %TARGET_EXE% del %TARGET_EXE% >nul 2>&1

REM No wait needed - Windows handles this well

REM Copy with /Y flag for overwrite
copy /Y %NEW_EXE% %TARGET_EXE% >nul 2>&1

REM Verify the copy worked by checking if file exists and has correct size
if exist %TARGET_EXE% (
    echo [%date% %time%] Copy appeared successful, verifying file... >> %LOGFILE%
    REM Compare file sizes to ensure copy was complete (handle quotes safely)
    for %%A in ("%NEW_NOQUOTE%") do set NEWSIZE=%%~zA
    for %%B in ("%TARGET_NOQUOTE%") do set CURSIZE=%%~zB
    
    if "!NEWSIZE!"=="!CURSIZE!" (
        echo [%date% %time%] File sizes match - update successful! >> %LOGFILE%
        echo Update successful! File sizes match.
        goto update_success
    ) else (
        echo [%date% %time%] File size mismatch - copy incomplete (New: !NEWSIZE!, Current: !CURSIZE!) >> %LOGFILE%
        echo File size mismatch - copy incomplete
    )
) else (
    echo [%date% %time%] Target file does not exist after copy >> %LOGFILE%
    echo Copy failed - target file missing
)

REM Fast retry logic - no waiting needed
if !UPDATE_RETRY! LSS 10 (
    echo [%date% %time%] Copy failed, retrying immediately... >> %LOGFILE%
    echo Copy failed, retrying (attempt !UPDATE_RETRY!/10)
    goto update_retry
) else (
    echo [%date% %time%] ERROR: Update failed after 10 attempts! >> %LOGFILE%
    echo ERROR: Update failed after 10 attempts!
    echo.  
    echo This is likely caused by:
    echo - Windows Defender real-time protection
    echo - Antivirus software blocking file replacement
    echo - Windows file system indexing
    echo.
    echo MANUAL UPDATE REQUIRED:
    echo 1. Temporarily disable Windows Defender real-time protection
    echo 2. Download the latest release from GitHub
    echo 3. Replace the executable manually
    echo.
    echo Restoring backup...
    copy "%BACKUP_EXE%" %TARGET_EXE% >nul 2>&1
    echo Backup restored.
    pause
    exit /b 1
)

:update_success

echo Step 5: Cleaning up...
echo [%date% %time%] Step 5: Cleaning up... >> %LOGFILE%
del %BACKUP_EXE% >nul 2>&1
del %NEW_EXE% >nul 2>&1

echo Step 6: Update complete - restarting application...
echo [%date% %time%] Step 6: Update complete - restarting application... >> %LOGFILE%

echo Restarting FlightTracePro...
echo [%date% %time%] Restarting FlightTracePro... >> %LOGFILE%

if "%DEFERRED_MODE%"=="true" (
    echo [%date% %time%] Deferred update completed successfully >> %LOGFILE%
    echo Deferred update completed! Starting the new version...
    REM Clean up the deferred update marker file if it exists
    del "%TEMP%\\flighttracepro_deferred_update.marker" >nul 2>&1
) else (
    echo [%date% %time%] Immediate update completed successfully >> %LOGFILE%
    echo Immediate update completed! Starting the new version...
)

REM Start application exactly like double-clicking using explorer
echo [%date% %time%] Launching application via explorer... >> %LOGFILE%
explorer "%TARGET_NOQUOTE%"
echo [%date% %time%] Launch command issued >> %LOGFILE%

echo.
if "%DEFERRED_MODE%"=="true" (
    echo [SUCCESS] Deferred update to v%VERSION% completed successfully!
) else (
    echo [SUCCESS] Update to v%VERSION% completed successfully!
)
echo FlightTracePro is starting with the new version.
echo [%date% %time%] Update successful - application restarted >> %LOGFILE%

echo [%date% %time%] Update process completed >> %LOGFILE%

REM Add a small delay before cleanup for deferred updates
if "%DEFERRED_MODE%"=="true" (
    echo Cleaning up deferred update files...
    timeout /t 2 /nobreak >nul
)

REM For instant updates, give user option to keep window open to review results
if "%DEFERRED_MODE%"=="false" (
    echo.
    echo ========================================
    echo Update completed! Press any key to keep this window open,
    echo or it will auto-close in 5 seconds...
    echo ========================================
    timeout /t 5 /nobreak >nul
    if !errorlevel! equ 0 (
        echo Auto-closing...
        REM Self-delete and exit immediately - use exit 0 to close the entire command window
        echo [%date% %time%] Self-deleting batch file and closing window... >> %LOGFILE%
        start "" /b cmd /c "timeout /t 1 /nobreak >nul & del /f /q "%~f0" >nul 2>&1"
        exit 0
    ) else (
        echo Press any key to close...
        pause >nul
        REM Self-delete and exit after user presses key - use exit 0 to close the entire command window
        echo [%date% %time%] Self-deleting batch file and closing window... >> %LOGFILE%
        start "" /b cmd /c "timeout /t 1 /nobreak >nul & del /f /q "%~f0" >nul 2>&1"
        exit 0
    )
) else (
    REM Deferred mode - silent cleanup and exit - use exit /b to preserve silent behavior
    echo [%date% %time%] Self-deleting batch file... >> %LOGFILE%
    start "" /b cmd /c "timeout /t 1 /nobreak >nul & del /f /q "%~f0" >nul 2>&1"
    exit /b 0
)
""")
            
            self.append_log(f"[update] Created batch file: {bat}")
            self.append_log(f"[update] Log file will be: {log_file}")
            
            # Verify batch file was created
            if not os.path.exists(bat):
                self.append_log(f"[update] ERROR: Failed to create batch file!")
                return
                
            self.append_log(f"[update] Batch file size: {os.path.getsize(bat)} bytes")
            
            # Show batch file contents for debugging
            try:
                with open(bat, 'r') as f:
                    content_preview = f.read()[:200] + "..." if len(f.read()) > 200 else f.read()
                    f.seek(0)  # Reset file pointer
                    content_preview = f.read()[:200]
                self.append_log(f"[update] Batch preview: {content_preview}")
            except Exception as preview_error:
                self.append_log(f"[update] Could not preview batch file: {preview_error}")
            
            self.append_log(f"[update] Starting update process...")
            
            # For deferred updates, don't run the batch file immediately
            launched = True  # Consider it successful since we're deferring
            
            if not install_later:
                # For immediate updates, try multiple launch methods
                launched = False
                
                # Method 1: subprocess with proper detachment
                try:
                    import subprocess
                    self.append_log(f"[update] Attempting subprocess launch...")
                    
                    # Try different subprocess approaches
                    if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                        # Windows: Create new console window - must use cmd.exe for batch files
                        proc = subprocess.Popen(
                            ['cmd.exe', '/c', bat],
                            shell=False,
                            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
                            cwd=os.path.dirname(bat),
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    else:
                        # Fallback - use shell=True for batch files
                        proc = subprocess.Popen(
                            bat,
                            shell=True,
                            cwd=os.path.dirname(bat),
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    
                    self.append_log(f"[update] Subprocess started with PID: {proc.pid}")
                    
                    # Don't wait for the process - let it run independently
                    launched = True
                except Exception as subprocess_error:
                    self.append_log(f"[update] Subprocess failed: {subprocess_error}")
                    
                    # Try the cmd.exe wrapper approach
                    try:
                        self.append_log(f"[update] Trying cmd.exe wrapper...")
                        proc = subprocess.Popen(
                            ['cmd.exe', '/c', 'start', '/min', bat],
                            shell=False,
                            creationflags=subprocess.DETACHED_PROCESS if hasattr(subprocess, 'DETACHED_PROCESS') else 0,
                            cwd=os.path.dirname(bat)
                        )
                        self.append_log(f"[update] CMD wrapper started with PID: {proc.pid}")
                        launched = True
                    except Exception as cmd_error:
                        self.append_log(f"[update] CMD wrapper failed: {cmd_error}")
                
                # Method 2: os.startfile fallback
                if not launched:
                    try:
                        self.append_log(f"[update] Attempting startfile launch...")
                        os.startfile(bat)
                        launched = True
                        self.append_log(f"[update] Started with os.startfile")
                    except Exception as startfile_error:
                        self.append_log(f"[update] Startfile failed: {startfile_error}")
                
                # Method 3: system command fallback
                if not launched:
                    try:
                        self.append_log(f"[update] Attempting system command launch...")
                        os.system(f'start "" "{bat}"')
                        launched = True
                        self.append_log(f"[update] Started with os.system")
                    except Exception as system_error:
                        self.append_log(f"[update] System command failed: {system_error}")
            
            if launched:
                if install_later:
                    # For deferred installation, create marker file and show completion dialog
                    self.append_log(f"[update] Update downloaded and prepared for restart later")
                    self.append_log(f"[update] Update will install on next application start")
                    self.append_log(f"[update] Check update log after restart: {log_file}")
                    
                    # Create deferred update marker file
                    marker_file = os.path.join(tmpdir, 'flighttracepro_deferred_update.marker')
                    with open(marker_file, 'w') as mf:
                        mf.write(f"version={version}\n")
                        mf.write(f"batch_file={bat}\n")
                        mf.write(f"log_file={log_file}\n")
                        mf.write(f"new_exe={newexe}\n")
                    
                    # Show completion dialog in main thread
                    from PySide6.QtCore import QTimer
                    def show_completion():
                        self._show_deferred_update_completion(version)
                    QTimer.singleShot(100, show_completion)
                else:
                    # Immediate installation - exit application
                    self.append_log(f"[update] Update process started - application will restart in 5 seconds")
                    self.append_log(f"[update] Check update log after restart: {log_file}")
                    # Give the batch file time to start
                    import time
                    time.sleep(3)
                    self.append_log(f"[update] Exiting application for update...")
                    time.sleep(1)
                    os._exit(0)
            else:
                self.append_log(f"[update] ERROR: All launch methods failed!")
                self.append_log(f"[update] Manual update required:")
                self.append_log(f"[update] 1. Download new version from GitHub releases")
                self.append_log(f"[update] 2. Or run this batch file manually: {bat}")
                self.append_log(f"[update] 3. Check the log file: {log_file}")
                
                # Try to run the batch file manually as a test
                try:
                    self.append_log(f"[update] Attempting manual test run...")
                    import subprocess
                    result = subprocess.run([bat], shell=True, capture_output=True, text=True, timeout=10)
                    self.append_log(f"[update] Manual test - return code: {result.returncode}")
                    if result.stdout:
                        self.append_log(f"[update] Manual test - stdout: {result.stdout[:200]}")
                    if result.stderr:
                        self.append_log(f"[update] Manual test - stderr: {result.stderr[:200]}")
                except Exception as test_error:
                    self.append_log(f"[update] Manual test failed: {test_error}")
        except Exception as e:
            self.append_log(f"[update] {e}")
    
    def _show_deferred_update_completion(self, version):
        """Show dialog when deferred update is ready"""
        dialog = DeferredUpdateCompletionDialog(version, self)
        dialog.exec()
    
    def show_options_dialog(self):
        """Show options dialog"""
        dialog = OptionsDialog(self)
        dialog.exec()
    
    @Slot()
    def check_for_deferred_update(self):
        """Check for and handle deferred updates on startup"""
        import os, tempfile
        try:
            tmpdir = tempfile.gettempdir()
            marker_file = os.path.join(tmpdir, 'flighttracepro_deferred_update.marker')
            
            self.append_log(f"[update] Checking for deferred update marker: {marker_file}")
            
            if os.path.exists(marker_file):
                self.append_log("[update] Found deferred update marker - processing...")
                
                # Read marker file
                marker_data = {}
                with open(marker_file, 'r') as mf:
                    for line in mf:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            marker_data[key] = value
                
                self.append_log(f"[update] Marker data: {marker_data}")
                
                version = marker_data.get('version', 'Unknown')
                batch_file = marker_data.get('batch_file', '')
                new_exe = marker_data.get('new_exe', '')
                
                self.append_log(f"[update] Batch file: {batch_file}")
                self.append_log(f"[update] New exe: {new_exe}")
                self.append_log(f"[update] Batch exists: {os.path.exists(batch_file) if batch_file else False}")
                self.append_log(f"[update] New exe exists: {os.path.exists(new_exe) if new_exe else False}")
                
                if batch_file and os.path.exists(batch_file) and new_exe and os.path.exists(new_exe):
                    # Auto-apply deferred update on startup
                    self.append_log("[update] Auto-applying deferred update...")
                    
                    # Show progress dialog
                    self.append_log("[update] Showing progress dialog...")
                    progress_dialog = DeferredUpdateProgressDialog(version)
                    progress_dialog.show()
                    progress_dialog.start_animation()
                    
                    # Process events to show the dialog
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents()
                    
                    # Give user a moment to see the dialog, then execute
                    from PySide6.QtCore import QTimer
                    
                    def execute_update_delayed():
                        """Execute the update after showing progress dialog"""
                        # Execute the batch file and exit immediately
                        try:
                            import subprocess
                            self.append_log(f"[update] Executing batch file: {batch_file}")
                            
                            # Debug: Check batch file properties
                            try:
                                batch_size = os.path.getsize(batch_file)
                                self.append_log(f"[update] Batch file size: {batch_size} bytes")
                                
                                # Check if we can read the batch file
                                with open(batch_file, 'r', encoding='cp1252') as bf:
                                    first_line = bf.readline().strip()
                                self.append_log(f"[update] Batch first line: {repr(first_line)}")
                            except Exception as debug_e:
                                self.append_log(f"[update] Batch file debug failed: {debug_e}")
                            
                            # Try different execution methods for Windows compatibility
                            launched = False
                            
                            # Method 1: Direct batch execution with shell=True
                            try:
                                proc = subprocess.Popen(
                                    batch_file,
                                    shell=True,
                                    cwd=os.path.dirname(batch_file),
                                    creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
                                )
                                self.append_log(f"[update] Method 1 success - PID: {proc.pid}")
                                launched = True
                            except Exception as e1:
                                self.append_log(f"[update] Method 1 failed: {e1}")
                            
                            # Method 2: cmd.exe with proper quoting
                            try:
                                proc = subprocess.Popen(
                                    f'cmd.exe /c "{batch_file}"',
                                    shell=True,
                                    cwd=os.path.dirname(batch_file)
                                )
                                self.append_log(f"[update] Method 2 success - PID: {proc.pid}")
                                launched = True
                            except Exception as e2:
                                self.append_log(f"[update] Method 2 failed: {e2}")
                                
                                # Method 3: os.startfile as last resort
                                try:
                                    os.startfile(batch_file)
                                    self.append_log("[update] Method 3 (os.startfile) success")
                                    launched = True
                                except Exception as e3:
                                    self.append_log(f"[update] Method 3 failed: {e3}")
                            
                            if launched:
                                # Remove marker file
                                os.remove(marker_file)
                                self.append_log("[update] Auto-applying deferred update - application will restart...")
                                # Exit application
                                import time
                                time.sleep(1)
                                os._exit(0)
                            else:
                                raise Exception("All execution methods failed")
                        except ImportError:
                            self.append_log(f"[update] subprocess not available, using fallback...")
                            try:
                                os.startfile(batch_file)
                                os.remove(marker_file) 
                                import time
                                time.sleep(1)
                                os._exit(0)
                            except Exception as fallback_e:
                                self.append_log(f"[update] Fallback failed: {fallback_e}")
                        except Exception as e:
                            self.append_log(f"[update] Failed to execute deferred update: {e}")
                            # If auto-execution fails, show the dialog as fallback
                            self.append_log("[update] Auto-execution failed, showing restart dialog...")
                            progress_dialog.stop_animation()
                            progress_dialog.close()
                            dialog = DeferredUpdateRestartDialog(version, batch_file, self)
                            dialog.exec()
                    
                    # Start the delayed execution using QTimer (2 seconds to show progress)
                    QTimer.singleShot(2000, execute_update_delayed)
                else:
                    if batch_file and not os.path.exists(batch_file):
                        self.append_log(f"[update] Deferred update batch file missing: {batch_file}")
                    if new_exe and not os.path.exists(new_exe):
                        self.append_log(f"[update] Deferred update exe file missing: {new_exe}")
                    self.append_log("[update] Deferred update files not found, removing marker")
                    os.remove(marker_file)
            else:
                self.append_log("[update] No deferred update marker found")
        except Exception as e:
            self.append_log(f"[update] Error checking for deferred updates: {e}")


class CloseDialog(QDialog):
    """Custom close dialog with better styling"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FlightTracePro")
        self.setFixedSize(400, 200)
        self.setModal(True)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Icon and message
        msg_layout = QHBoxLayout()
        
        # Icon label
        icon_label = QLabel()
        icon_label.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon).pixmap(48, 48))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        msg_layout.addWidget(icon_label)
        
        # Message text
        msg_label = QLabel("What would you like to do when closing FlightTracePro?")
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("font-size: 12px; font-weight: bold; margin-left: 10px;")
        msg_layout.addWidget(msg_label, 1)
        
        layout.addLayout(msg_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Options
        options_layout = QVBoxLayout()
        options_layout.setSpacing(10)
        
        self.minimize_btn = QPushButton("🗗 Minimize to system tray")
        self.minimize_btn.setStyleSheet("QPushButton { text-align: left; padding: 10px; font-size: 11px; }")
        self.minimize_btn.setToolTip("Keep FlightTracePro running in the background")
        self.minimize_btn.clicked.connect(lambda: self.accept_with_action("minimize"))
        options_layout.addWidget(self.minimize_btn)
        
        self.quit_btn = QPushButton("❌ Exit application")
        self.quit_btn.setStyleSheet("QPushButton { text-align: left; padding: 10px; font-size: 11px; }")
        self.quit_btn.setToolTip("Completely close FlightTracePro")
        self.quit_btn.clicked.connect(lambda: self.accept_with_action("quit"))
        options_layout.addWidget(self.quit_btn)
        
        layout.addLayout(options_layout)
        
        # Remember choice checkbox
        self.remember_cb = QCheckBox("Remember my choice for next time")
        self.remember_cb.setStyleSheet("margin-top: 10px;")
        layout.addWidget(self.remember_cb)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Set default focus
        self.minimize_btn.setFocus()
        
        # Instance variables
        self.selected_action = None
    
    def accept_with_action(self, action):
        self.selected_action = action
        self.accept()
    
    def get_result(self):
        return self.selected_action, self.remember_cb.isChecked()


class OptionsDialog(QDialog):
    """Options dialog for application settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("FlightTracePro Options")
        self.setFixedSize(450, 300)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("Application Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Settings section
        settings_group = QFrame()
        settings_group.setFrameStyle(QFrame.Shape.Box)
        settings_group.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 5px; padding: 10px; }")
        
        settings_layout = QVBoxLayout(settings_group)
        
        # Close behavior setting
        close_label = QLabel("Close button behavior:")
        close_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        settings_layout.addWidget(close_label)
        
        self.close_combo = QComboBox()
        self.close_combo.addItem("Ask me every time", "ask")
        self.close_combo.addItem("Always minimize to tray", "minimize")
        self.close_combo.addItem("Always exit application", "quit")
        
        # Set current value
        current_action = parent.settings.value("closeAction", "ask", type=str)
        for i in range(self.close_combo.count()):
            if self.close_combo.itemData(i) == current_action:
                self.close_combo.setCurrentIndex(i)
                break
        
        settings_layout.addWidget(self.close_combo)
        
        # Future settings can be added here
        settings_layout.addStretch()
        
        layout.addWidget(settings_group)
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
    
    def accept(self):
        # Save settings
        selected_action = self.close_combo.currentData()
        self.parent_window.settings.setValue("closeAction", selected_action)
        super().accept()


class UpdateDialog(QDialog):
    """Modern update dialog with Material Design styling"""
    
    class UpdateAction:
        CANCEL = 0
        INSTALL_NOW = 1
        INSTALL_LATER = 2
    
    def __init__(self, version, download_url, parent=None):
        super().__init__(parent)
        self.version = version
        self.download_url = download_url
        self.result_action = self.UpdateAction.CANCEL
        
        self.setWindowTitle("FlightTracePro Update Available")
        self.setFixedSize(500, 400)
        self.setModal(True)
        
        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f8f9fa, stop: 1 #e9ecef);
                border-radius: 10px;
            }
            QLabel#titleLabel {
                font-size: 20px;
                font-weight: bold;
                color: #1976d2;
                margin-bottom: 10px;
            }
            QLabel#versionLabel {
                font-size: 16px;
                font-weight: 600;
                color: #2e7d32;
                margin-bottom: 5px;
            }
            QLabel#descLabel {
                font-size: 12px;
                color: #555;
                line-height: 1.4;
            }
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2196f3, stop: 1 #1976d2);
                border: none;
                border-radius: 6px;
                color: white;
                font-weight: 600;
                font-size: 13px;
                padding: 12px 24px;
                margin: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #42a5f5, stop: 1 #1e88e5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #1565c0, stop: 1 #0d47a1);
            }
            QPushButton#primaryButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4caf50, stop: 1 #388e3c);
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #66bb6a, stop: 1 #43a047);
            }
            QPushButton#secondaryButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ff9800, stop: 1 #f57c00);
            }
            QPushButton#secondaryButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ffb74d, stop: 1 #fb8c00);
            }
            QPushButton#cancelButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #757575, stop: 1 #424242);
            }
            QPushButton#cancelButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #9e9e9e, stop: 1 #616161);
            }
            QFrame#separatorFrame {
                background-color: #e0e0e0;
                border: none;
            }
        """)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header section with icon and title
        header_layout = QHBoxLayout()
        
        # Update icon
        icon_label = QLabel()
        update_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
        icon_label.setPixmap(update_icon.pixmap(48, 48))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        header_layout.addWidget(icon_label)
        
        # Title and version info
        info_layout = QVBoxLayout()
        
        title_label = QLabel("🚀 Update Available!")
        title_label.setObjectName("titleLabel")
        info_layout.addWidget(title_label)
        
        version_label = QLabel(f"Version {version}")
        version_label.setObjectName("versionLabel")
        info_layout.addWidget(version_label)
        
        header_layout.addLayout(info_layout, 1)
        layout.addLayout(header_layout)
        
        # Separator
        separator = QFrame()
        separator.setObjectName("separatorFrame")
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        
        # Description
        desc_text = (
            "A new version of FlightTracePro is available with improvements and bug fixes.\n\n"
            "Choose how you'd like to proceed:\n\n"
            "• Install Now: Download and install immediately (app will restart)\n"
            "• Install Later: Download now, auto-install on next app start\n"
            "• Cancel: Skip this update for now"
        )
        desc_label = QLabel(desc_text)
        desc_label.setObjectName("descLabel")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        desc_label.setMargin(10)
        layout.addWidget(desc_label, 1)
        
        # Button section
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)
        
        # Install Now button
        self.install_now_btn = QPushButton("🔄 Install Now & Restart")
        self.install_now_btn.setObjectName("primaryButton")
        self.install_now_btn.setToolTip("Download and install the update immediately. The app will restart automatically.")
        self.install_now_btn.clicked.connect(lambda: self._accept_with_action(self.UpdateAction.INSTALL_NOW))
        button_layout.addWidget(self.install_now_btn)
        
        # Install Later button
        self.install_later_btn = QPushButton("⏰ Download Now, Auto-Install Later")
        self.install_later_btn.setObjectName("secondaryButton")
        self.install_later_btn.setToolTip("Download the update now, then automatically install it the next time you start the app.")
        self.install_later_btn.clicked.connect(lambda: self._accept_with_action(self.UpdateAction.INSTALL_LATER))
        button_layout.addWidget(self.install_later_btn)
        
        # Horizontal layout for cancel button
        cancel_layout = QHBoxLayout()
        cancel_layout.addStretch()
        
        # Cancel button
        cancel_btn = QPushButton("✖️ Skip This Update")
        cancel_btn.setObjectName("cancelButton")
        cancel_btn.setToolTip("Skip this update and continue using the current version.")
        cancel_btn.clicked.connect(lambda: self._accept_with_action(self.UpdateAction.CANCEL))
        cancel_layout.addWidget(cancel_btn)
        
        button_layout.addLayout(cancel_layout)
        layout.addLayout(button_layout)
        
        # Set focus and default
        self.install_later_btn.setFocus()
    
    def _accept_with_action(self, action):
        self.result_action = action
        if action == self.UpdateAction.CANCEL:
            self.reject()
        else:
            self.accept()
    
    def exec(self):
        super().exec()
        return self.result_action


class DeferredUpdateCompletionDialog(QDialog):
    """Dialog shown when a deferred update download is complete"""
    
    def __init__(self, version, parent=None):
        super().__init__(parent)
        self.version = version
        
        self.setWindowTitle("FlightTracePro Update Ready")
        self.setFixedSize(400, 250)
        self.setModal(True)
        
        # Apply styling
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f1f8e9, stop: 1 #e8f5e8);
                border-radius: 10px;
            }
            QLabel#titleLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2e7d32;
                margin-bottom: 10px;
            }
            QLabel#descLabel {
                font-size: 12px;
                color: #555;
                line-height: 1.4;
            }
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4caf50, stop: 1 #388e3c);
                border: none;
                border-radius: 6px;
                color: white;
                font-weight: 600;
                font-size: 13px;
                padding: 12px 24px;
                margin: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #66bb6a, stop: 1 #43a047);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header with checkmark icon
        header_layout = QHBoxLayout()
        
        # Success icon
        icon_label = QLabel("✅")
        icon_label.setStyleSheet("font-size: 32px;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        header_layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel(f"Update v{version} Downloaded!")
        title_label.setObjectName("titleLabel")
        header_layout.addWidget(title_label, 1)
        
        layout.addLayout(header_layout)
        
        # Description
        desc_text = (
            f"The update to version {version} has been downloaded and is ready to install.\n\n"
            "The update will automatically install and restart the application the next time you start FlightTracePro."
        )
        desc_label = QLabel(desc_text)
        desc_label.setObjectName("descLabel")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(desc_label, 1)
        
        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("👍 Got It!")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)


class DeferredUpdateRestartDialog(QDialog):
    """Dialog shown on startup when a deferred update is ready to install"""
    
    class RestartAction:
        CANCEL = 0
        RESTART_NOW = 1
        RESTART_LATER = 2
    
    def __init__(self, version, batch_file, parent=None):
        super().__init__(parent)
        self.version = version
        self.batch_file = batch_file
        self.result_action = self.RestartAction.CANCEL
        
        self.setWindowTitle("FlightTracePro Update Ready")
        self.setFixedSize(450, 350)
        self.setModal(True)
        
        # Apply styling
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #fff3e0, stop: 1 #ffe0b2);
                border-radius: 10px;
            }
            QLabel#titleLabel {
                font-size: 18px;
                font-weight: bold;
                color: #ef6c00;
                margin-bottom: 10px;
            }
            QLabel#descLabel {
                font-size: 12px;
                color: #555;
                line-height: 1.4;
            }
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ff9800, stop: 1 #f57c00);
                border: none;
                border-radius: 6px;
                color: white;
                font-weight: 600;
                font-size: 13px;
                padding: 12px 24px;
                margin: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ffb74d, stop: 1 #fb8c00);
            }
            QPushButton#primaryButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4caf50, stop: 1 #388e3c);
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #66bb6a, stop: 1 #43a047);
            }
            QPushButton#cancelButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #757575, stop: 1 #424242);
            }
            QPushButton#cancelButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #9e9e9e, stop: 1 #616161);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header with update icon
        header_layout = QHBoxLayout()
        
        # Update pending icon
        icon_label = QLabel("🔄")
        icon_label.setStyleSheet("font-size: 32px;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        header_layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel(f"Update v{version} Ready to Install")
        title_label.setObjectName("titleLabel")
        header_layout.addWidget(title_label, 1)
        
        layout.addLayout(header_layout)
        
        # Description
        desc_text = (
            f"A previously downloaded update (version {version}) is ready to install.\n\n"
            "Would you like to install it now or continue with the current version?\n\n"
            "The installation process will:\n"
            "• Close the current application\n"
            "• Install the new version\n"
            "• Restart FlightTracePro automatically"
        )
        desc_label = QLabel(desc_text)
        desc_label.setObjectName("descLabel")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(desc_label, 1)
        
        # Button section
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)
        
        # Install now button
        install_btn = QPushButton("🚀 Install Update Now")
        install_btn.setObjectName("primaryButton")
        install_btn.setToolTip("Install the update and restart the application now")
        install_btn.clicked.connect(lambda: self._accept_with_action(self.RestartAction.RESTART_NOW))
        button_layout.addWidget(install_btn)
        
        # Horizontal layout for other buttons
        other_layout = QHBoxLayout()
        
        # Continue button
        continue_btn = QPushButton("⏭️ Continue Without Update")
        continue_btn.setObjectName("cancelButton")
        continue_btn.setToolTip("Continue using the current version (update will remain available)")
        continue_btn.clicked.connect(lambda: self._accept_with_action(self.RestartAction.RESTART_LATER))
        other_layout.addWidget(continue_btn)
        
        button_layout.addLayout(other_layout)
        layout.addLayout(button_layout)
        
        # Set default focus
        install_btn.setFocus()
    
    def _accept_with_action(self, action):
        self.result_action = action
        if action == self.RestartAction.RESTART_LATER:
            self.reject()
        else:
            self.accept()
    
    def exec(self):
        super().exec()
        return self.result_action


class DeferredUpdateProgressDialog(QDialog):
    """Steam-like progress dialog for deferred updates"""
    
    def __init__(self, version=""):
        super().__init__()
        self.version = version
        self.init_ui()
        self.setup_animation()
        
    def init_ui(self):
        self.setWindowTitle("FlightTracePro")
        self.setFixedSize(400, 200)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)  # No close button
        self.setModal(True)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 20, 30, 20)
        
        # Modern gradient background
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1e3c72, stop:1 #2a5298);
                border-radius: 10px;
            }
            QLabel {
                color: white;
                background: transparent;
            }
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                text-align: center;
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #4CAF50, stop:1 #81C784);
                border-radius: 6px;
                margin: 2px;
            }
        """)
        
        # Icon and header
        header_layout = QHBoxLayout()
        
        # Update icon
        icon_label = QLabel()
        icon_pixmap = self.style().standardPixmap(QStyle.StandardPixmap.SP_BrowserReload)
        icon_label.setPixmap(icon_pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        header_layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel("Installing Update")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Version info
        version_text = f"Updating to version {self.version}..." if self.version else "Installing update..."
        self.version_label = QLabel(version_text)
        self.version_label.setStyleSheet("font-size: 12px; color: #E3F2FD;")
        layout.addWidget(self.version_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Please wait...")
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Preparing update installation...")
        self.status_label.setStyleSheet("font-size: 11px; color: #BBDEFB;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Note
        note_label = QLabel("The application will restart automatically when complete.")
        note_label.setStyleSheet("font-size: 10px; color: #90CAF9; font-style: italic;")
        note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note_label)
        
        self.setLayout(layout)
        
        # Center on screen
        self.center_on_screen()
        
    def center_on_screen(self):
        """Center the dialog on the screen"""
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        self.move(screen.center() - self.rect().center())
        
    def setup_animation(self):
        """Setup status text animation"""
        from PySide6.QtCore import QTimer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_status_animation)
        self.animation_step = 0
        self.status_messages = [
            "Preparing update installation...",
            "Stopping application services...",
            "Installing new version...",
            "Updating configuration...",
            "Finalizing installation...",
            "Starting updated application..."
        ]
        
    def start_animation(self):
        """Start the status animation"""
        self.animation_timer.start(2000)  # Update every 2 seconds
        
    def update_status_animation(self):
        """Update the status text to simulate progress"""
        if self.animation_step < len(self.status_messages):
            self.status_label.setText(self.status_messages[self.animation_step])
            self.animation_step += 1
        else:
            # Loop back or stop
            self.status_label.setText("Completing installation...")
            
    def stop_animation(self):
        """Stop the animation"""
        if hasattr(self, 'animation_timer'):
            self.animation_timer.stop()


def get_app_version():
    """Get app version from VERSION_BUILD or VERSION file or fallback to hardcoded"""
    import os
    import sys
    try:
        # First try VERSION_BUILD (created during CI build with run number)
        for version_name in ['VERSION_BUILD', 'VERSION']:
            version_file = os.path.join(os.path.dirname(__file__), version_name)
            if not os.path.exists(version_file) and getattr(sys, 'frozen', False):
                # If frozen, version files might be in the same directory as the exe
                version_file = os.path.join(os.path.dirname(sys.executable), version_name)
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    return f.read().strip()
    except Exception:
        pass
    # Fallback to environment variable or hardcoded
    return os.environ.get('FLIGHTTRACEPRO_APP_VERSION', '0.2.0')

# ---------------- Single Instance Support ----------------
SINGLE_INSTANCE_NAME = 'FlightTracePro_Bridge_SingleInstance'


class _SingleInstanceServer:
    """Listens for activation requests from subsequent instances"""

    def __init__(self, name: str, on_activate):
        self._name = name
        self._on_activate = on_activate
        self._server = QLocalServer()
        # Remove stale server (in case of previous crash)
        try:
            QLocalServer.removeServer(self._name)
        except Exception:
            pass
        self._server.newConnection.connect(self._handle_new_connection)
        self._server.listen(self._name)

    def _handle_new_connection(self):
        try:
            sock = self._server.nextPendingConnection()
            if sock is None:
                return
            # Read the message to determine action
            message = 'activate'  # default action
            try:
                if sock.waitForReadyRead(50):
                    message = bytes(sock.readAll()).decode('utf-8', 'ignore').strip()
            except Exception:
                pass
            try:
                sock.disconnectFromServer()
            except Exception:
                pass
            
            # Only activate the window if explicitly requested
            # For 'check' messages, we just respond that we exist (by accepting the connection)
            if message == 'activate':
                try:
                    self._on_activate()
                except Exception:
                    pass
        except Exception:
            pass


def _notify_existing_instance(name: str, message: str = 'activate') -> bool:
    """Try to notify a running instance. Returns True if one was found."""
    try:
        sock = QLocalSocket()
        sock.connectToServer(name)
        if not sock.waitForConnected(150):
            return False
        try:
            sock.write(message.encode('utf-8'))
            sock.flush()
            sock.waitForBytesWritten(50)
        except Exception:
            pass
        try:
            sock.disconnectFromServer()
        except Exception:
            pass
        return True
    except Exception:
        return False

def main():
    import sys
    
    # Handle command line arguments for updater testing
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--version', '-v', '/version']:
            # Return version info for updater verification
            version = get_app_version()
            print(f"FlightTracePro Bridge v{version}")
            sys.exit(0)
        elif arg in ['--help', '-h', '/?']:
            print("FlightTracePro Bridge Client")
            print("Options:")
            print("  --version, -v     Show version")
            print("  --help, -h        Show this help")
            sys.exit(0)
    
    # Create application with proper cleanup
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running when window is closed (for tray)

    # Check if another instance exists
    if _notify_existing_instance(SINGLE_INSTANCE_NAME, 'check'):
        # Show warning dialog asking user what to do
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("FlightTracePro Bridge Already Running")
        msg.setText("There is already an instance of FlightTracePro Bridge running.")
        msg.setInformativeText("What would you like to do?")
        
        # Add custom buttons
        show_existing_btn = msg.addButton("Show Existing Instance", QMessageBox.ActionRole)
        run_anyway_btn = msg.addButton("Run Another Instance Anyway", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.setDefaultButton(show_existing_btn)
        msg.exec()
        
        if msg.clickedButton() == show_existing_btn:
            # Activate the existing instance and exit
            _notify_existing_instance(SINGLE_INSTANCE_NAME, 'activate')
            return 0
        elif msg.clickedButton() == cancel_btn:
            # User chose to cancel, exit
            return 0
        # If "Run Another Instance Anyway" was clicked, continue with startup
    
    try:
        win = MainWindow()

        # Setup single-instance server to handle future activation requests
        def _activate_main_window():
            try:
                if win.isMinimized():
                    win.showNormal()
                else:
                    win.show()
                win.raise_()
                win.activateWindow()
            except Exception:
                pass

        win._single_instance = _SingleInstanceServer(SINGLE_INSTANCE_NAME, _activate_main_window)
        win.show()
        
        # Run the application event loop
        exit_code = app.exec()
        
        # Ensure proper cleanup
        try:
            win.quit_application()
        except:
            pass
            
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)
    finally:
        # Force cleanup of any remaining Qt objects
        try:
            QApplication.processEvents()
        except:
            pass


if __name__ == '__main__':
    main()
