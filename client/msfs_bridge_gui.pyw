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
    QMenu, QMessageBox, QStyle, QTextEdit
)


def meters_from_feet(ft: Optional[float]) -> Optional[float]:
    return None if ft is None else ft * 0.3048


class MSFSSource:
    def __init__(self, log_cb=None):
        self.sim = None
        self.areq = None
        self._log = (lambda level, msg: None) if log_cb is None else log_cb

    def start(self) -> bool:
        self._log('INFO', "Attempting SimConnect initialization...")
        
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


def simulate_sample(t: float, origin_lat: float = 47.3769, origin_lon: float = 8.5417, origin_alt_m: float = 500.0):
    import math
    R = 0.02
    ang = (t * 0.05) % (2 * math.pi)
    return {
        "lat": origin_lat + R * (0.8 * math.sin(ang)),
        "lon": origin_lon + R * (1.2 * math.cos(ang)),
        "alt_m": origin_alt_m + 50 * math.sin(ang * 2),
        "spd_kt": 60.0,
        "vsi_ms": 0.0,
        "hdg_deg": (ang * 180.0 / math.pi) % 360,
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
        src = MSFSSource()
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
                            s = simulate_sample(time.time(), self._origin_lat, self._origin_lon, self._origin_alt)
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
                            # emit raw snapshot for debugging
                            try:
                                snap_lat = gv("PLANE LATITUDE") or gv("GPS POSITION LAT")
                                snap_lon = gv("PLANE LONGITUDE") or gv("GPS POSITION LON")
                                self._log("DEBUG", f"sample unavailable (lat={snap_lat} lon={snap_lon})")
                            except Exception:
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
        src = MSFSSource()
        url = f"{self._server}/api/live/{self._channel}"
        dt = 1.0 / self._rate_hz
        while not self._stop.is_set():
            if self._demo:
                self.connected.emit(True)
                try:
                    s = simulate_sample(time.time(), self._origin_lat, self._origin_lon, self._origin_alt)
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
                    try:
                        snap_lat = gv("PLANE LATITUDE") or gv("GPS POSITION LAT")
                        snap_lon = gv("PLANE LONGITUDE") or gv("GPS POSITION LON")
                        self._log("DEBUG", f"sample unavailable (lat={snap_lat} lon={snap_lon})")
                    except Exception:
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
        self.status = QLabel("disconnected"); self.status.setAlignment(Qt.AlignRight | Qt.AlignVCenter); lay.addWidget(self.status, row, 2, 1, 2); row += 1

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
        self.btnTestUpdate = QPushButton("Test Update"); self.btnTestUpdate.setToolTip("Test update mechanism with fake update"); logControls.addWidget(self.btnTestUpdate)
        logControlsWidget = QWidget(); logControlsWidget.setLayout(logControls)
        lay.addWidget(logControlsWidget, row, 3); row += 1
        
        self.logLevel.currentTextChanged.connect(self.on_level_changed)
        self.btnClearLogs.clicked.connect(self.clear_logs)
        self.btnTestUpdate.clicked.connect(self.test_update_mechanism)

        self.btnConnect.clicked.connect(self.on_connect)
        self.btnDisconnect.clicked.connect(self.on_disconnect)
        self.btnSaveServer.clicked.connect(self.on_save_server)

        # Tray
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.tray_menu = QMenu()
        act_show = QAction("Open", self); act_show.triggered.connect(self.showNormal); self.tray_menu.addAction(act_show)
        act_quit = QAction("Quit", self); act_quit.triggered.connect(self.close); self.tray_menu.addAction(act_quit)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self._on_tray)
        self.tray.show()

        # Connect update signal
        self.update_info_signal.connect(self._show_update_prompt)

        # Auto-update check (best-effort)
        try:
            self.check_update_async()
        except Exception:
            pass

        # Recents storage
        self.settings = QSettings("FlightTracePro", "Bridge")
        self.load_recents()

    def _on_tray(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            if self.isMinimized() or not self.isVisible():
                self.showNormal(); self.activateWindow()
            else:
                self.hide()

    def closeEvent(self, e):
        # Minimize to tray
        self.hide()
        e.ignore()

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
            self.worker.stop()
            self.worker.wait(1000)
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
        import threading
        t = threading.Thread(target=self._check_update, daemon=True)
        t.start()

    def _check_update(self):
        import os, sys, json, tempfile, time
        try:
            import requests
        except ImportError:
            self.append_log("[update] requests library not available for auto-update")
            return
        APP_VERSION = os.environ.get('FLIGHTTRACEPRO_APP_VERSION', '0.1.0')
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
            
            current_version = parse(APP_VERSION)
            latest_version = parse(ver)
            self.append_log(f"[update] Current: {APP_VERSION} ({current_version}), Latest: {ver} ({latest_version})")
            
            if latest_version <= current_version:
                self.append_log("[update] Already up to date")
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
        from PySide6.QtWidgets import QMessageBox
        ret = QMessageBox.information(self, 'Update Available', 
                                    f'New version {version} available. Update now?', 
                                    QMessageBox.Yes | QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        
        # Download and install in background thread
        import threading
        t = threading.Thread(target=self._download_and_install, args=(version, download_url), daemon=True)
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

echo Step 5: Restarting application...
echo [%date% %time%] Step 5: Restarting application... >> %LOGFILE%
timeout /t 2 /nobreak >nul

echo Starting: %TARGET_EXE%
echo [%date% %time%] Starting: %TARGET_EXE% >> %LOGFILE%
start "" %TARGET_EXE%

echo.
echo TEST UPDATE COMPLETE! Application should restart now.
echo [%date% %time%] TEST update process completed >> %LOGFILE%
timeout /t 3 /nobreak >nul

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
                        [bat],
                        shell=False,
                        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
                        cwd=os.path.dirname(bat),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    proc = subprocess.Popen([bat], shell=True, cwd=os.path.dirname(bat))
                
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

    def _download_and_install(self, version, url):
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
                return f'"{path.replace("/", "\\\\")}"'
            
            cur_safe = batch_escape(cur)
            newexe_safe = batch_escape(newexe)  
            log_safe = batch_escape(log_file)
            
            with open(bat, 'w') as bf:
                bf.write(f"""@echo off
setlocal enabledelayedexpansion
title FlightTracePro Updater

REM Define paths as variables
set TARGET_EXE={cur_safe}
set NEW_EXE={newexe_safe}
set LOGFILE={log_safe}

REM Normalize paths for operations needing suffixes
set TARGET_NOQUOTE=%TARGET_EXE:"=%
set NEW_NOQUOTE=%NEW_EXE:"=%
set BACKUP_EXE=%TARGET_NOQUOTE%.backup

echo [%date% %time%] FlightTracePro Auto-Updater Started >> %LOGFILE%
echo [%date% %time%] Target: %TARGET_EXE% >> %LOGFILE%
echo [%date% %time%] New EXE: %NEW_EXE% >> %LOGFILE%
echo [%date% %time%] Batch: %~f0 >> %LOGFILE%

echo.
echo ========================================
echo FlightTracePro Auto-Updater
echo ========================================
echo.
echo Log file: %LOGFILE%

echo Step 1: Waiting for application to close...
echo [%date% %time%] Step 1: Waiting for app to close... >> %LOGFILE%
timeout /t 5 /nobreak >nul

echo Step 2: Extended wait for file system...
echo [%date% %time%] Step 2: Extended wait for file system... >> %LOGFILE%
echo Waiting 3 more seconds for Windows to fully release file locks...
timeout /t 3 /nobreak >nul
echo [%date% %time%] Extended wait completed >> %LOGFILE%

echo Step 3: Creating backup...
echo [%date% %time%] Step 3: Creating backup... >> %LOGFILE%
if exist %TARGET_EXE% (
    copy %TARGET_EXE% %BACKUP_EXE% >nul 2>&1
    if errorlevel 1 (
        echo [%date% %time%] ERROR: Could not create backup! >> %LOGFILE%
        echo ERROR: Could not create backup!
        pause
        exit /b 1
    ) else (
        echo [%date% %time%] Backup created successfully >> %LOGFILE%
    )
) else (
    echo [%date% %time%] ERROR: Original EXE not found: %TARGET_EXE% >> %LOGFILE%
    echo ERROR: Original EXE not found!
    pause
    exit /b 1
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
echo [%date% %time%] Update attempt !UPDATE_RETRY!/20... >> %LOGFILE%
echo Installing new version... (attempt !UPDATE_RETRY!/20)

REM Delete target first to reduce conflicts
if exist %TARGET_EXE% del %TARGET_EXE% >nul 2>&1

REM Wait a moment for filesystem
timeout /t 1 /nobreak >nul

REM Copy with /Y flag for overwrite
copy /Y %NEW_EXE% %TARGET_EXE% >nul 2>&1

REM Verify the copy worked by checking if file exists and has correct size
if exist %TARGET_EXE% (
    echo [%date% %time%] Copy appeared successful, verifying file... >> %LOGFILE%
    REM Compare file sizes to ensure copy was complete
    for %%A in (%NEW_EXE%) do set NEWSIZE=%%~zA
    for %%B in (%TARGET_EXE%) do set CURSIZE=%%~zB
    
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

REM Retry logic with exponential backoff
if !UPDATE_RETRY! LSS 20 (
    if !UPDATE_RETRY! LSS 5 (
        set WAIT_TIME=1
    ) else if !UPDATE_RETRY! LSS 10 (
        set WAIT_TIME=2
    ) else (
        set WAIT_TIME=3
    )
    
    echo [%date% %time%] Copy failed, waiting !WAIT_TIME! seconds before retry... >> %LOGFILE%
    echo Copy failed, waiting !WAIT_TIME! seconds before retry...
    timeout /t !WAIT_TIME! /nobreak >nul
    goto update_retry
) else (
    echo [%date% %time%] ERROR: Update failed after 20 attempts! >> %LOGFILE%
    echo ERROR: Update failed after 20 attempts!
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
    copy "{cur}.backup" "{cur}" >nul 2>&1
    echo Backup restored.
    pause
    exit /b 1
)

:update_success

echo Step 5: Cleaning up...
echo [%date% %time%] Step 5: Cleaning up... >> %LOGFILE%
del %BACKUP_EXE% >nul 2>&1
del %NEW_EXE% >nul 2>&1

echo Step 6: Restarting application...
echo [%date% %time%] Step 6: Restarting application... >> %LOGFILE%
timeout /t 2 /nobreak >nul

echo Starting: %TARGET_EXE%
echo [%date% %time%] Starting: %TARGET_EXE% >> %LOGFILE%
start "" %TARGET_EXE%
if errorlevel 1 (
    echo [%date% %time%] ERROR: Failed to start application >> %LOGFILE%
    echo ERROR: Failed to start application!
    pause
    exit /b 1
) else (
    echo [%date% %time%] Application started successfully >> %LOGFILE%
)

echo.
echo Update complete! Application should restart now.
echo [%date% %time%] Update process completed >> %LOGFILE%
timeout /t 3 /nobreak >nul

REM Self-delete (but keep log file for debugging)
echo [%date% %time%] Self-deleting batch file... >> %LOGFILE%
del "%~f0" >nul 2>&1
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
            
            # Try multiple launch methods
            launched = False
            
            # Method 1: subprocess with proper detachment
            try:
                import subprocess
                self.append_log(f"[update] Attempting subprocess launch...")
                
                # Try different subprocess approaches
                if hasattr(subprocess, 'CREATE_NEW_CONSOLE'):
                    # Windows: Create new console window
                    proc = subprocess.Popen(
                        [bat],
                        shell=False,
                        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
                        cwd=os.path.dirname(bat),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    # Fallback
                    proc = subprocess.Popen(
                        [bat],
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


def main():
    import sys
    
    # Handle command line arguments for updater testing
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--version', '-v', '/version']:
            # Return version info for updater verification
            version = os.environ.get('FLIGHTTRACEPRO_APP_VERSION', '0.1.0')
            print(f"FlightTracePro Bridge v{version}")
            sys.exit(0)
        elif arg in ['--help', '-h', '/?']:
            print("FlightTracePro Bridge Client")
            print("Options:")
            print("  --version, -v     Show version")
            print("  --help, -h        Show this help")
            sys.exit(0)
    
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
