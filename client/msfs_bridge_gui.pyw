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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlightTracePro Bridge")
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
        self.rate = QSpinBox(); self.rate.setRange(1, 30); self.rate.setValue(2); lay.addWidget(self.rate, row, 1)
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
        logControlsWidget = QWidget(); logControlsWidget.setLayout(logControls)
        lay.addWidget(logControlsWidget, row, 3); row += 1
        
        self.logLevel.currentTextChanged.connect(self.on_level_changed)
        self.btnClearLogs.clicked.connect(self.clear_logs)

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
        import requests
        APP_VERSION = os.environ.get('FLIGHTTRACEPRO_APP_VERSION', '0.1.0')
        REPO = os.environ.get('FLIGHTTRACEPRO_REPO', 'nkoeppe/FlightTracePro')  # set to 'owner/repo'
        if REPO == 'your-user/your-repo':
            return
        try:
            r = requests.get(f'https://api.github.com/repos/{REPO}/releases/latest', timeout=5)
            if r.status_code != 200:
                return
            rel = r.json()
            tag = rel.get('tag_name','')
            ver = tag.lstrip('v').lstrip('bridge-v')
            def parse(v):
                parts = ''.join(c if (c.isdigit() or c=='.') else ' ' for c in v).split()
                return tuple(int(p) for p in (parts[0].split('.') if parts else ['0']))
            if parse(ver) <= parse(APP_VERSION):
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
                return
            # Prompt
            from PySide6.QtWidgets import QMessageBox
            ret = QMessageBox.information(self, 'Update Available', f'New version {ver} available. Update now?', QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                return
            # Download
            tmpdir = tempfile.gettempdir()
            newexe = os.path.join(tmpdir, 'FlightTracePro_new.exe')
            with requests.get(url, stream=True, timeout=30) as rr:
                rr.raise_for_status()
                with open(newexe, 'wb') as f:
                    for chunk in rr.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            # Only works when frozen
            if not getattr(sys, 'frozen', False):
                QMessageBox.information(self, 'Update', f'Downloaded new EXE to {newexe}. Please restart manually.')
                return
            cur = os.path.abspath(sys.executable)
            bat = os.path.join(tmpdir, 'flighttracepro_update.bat')
            with open(bat, 'w') as bf:
                bf.write(f"""@echo off
echo Updating...
set TARGET="{cur}"
set NEW="{newexe}"
:loop
copy /Y %NEW% %TARGET% >nul
if errorlevel 1 (
  timeout /t 1 >nul
  goto loop
)
start "" %TARGET%
del %NEW%
del "%~f0"
""")
            # Spawn updater and exit
            os.startfile(bat)
            os._exit(0)
        except Exception as e:
            self.append_log(f"[update] {e}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
