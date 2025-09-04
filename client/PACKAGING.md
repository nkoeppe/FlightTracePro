Build a Windows EXE for the NavMap Bridge (GUI).

1) Install Python 3.10+ and dependencies:

   pip install -r client/requirements.txt

2) Build with PyInstaller:

   Option A (PowerShell):

     # Release build (no console)
     ./client/build_exe.ps1

     # Debug build (with console to see import/SimConnect errors)
     ./client/build_exe.ps1 -Debug

   Option B (cmd):

     client\build_exe.bat

   Option C (manual):

     .venv\Scripts\pip.exe install pyinstaller
     .venv\Scripts\pyinstaller.exe --clean --noconsole --onefile --name NavMapBridge ^
       --hidden-import SimConnect ^
       --collect-all SimConnect ^
       --collect-submodules websockets ^
       --collect-submodules requests ^
       --collect-data certifi ^
       client\msfs_bridge_gui.pyw --specpath client

3) The EXE will be at `dist/NavMapBridge.exe`.

Notes
- The app minimizes to the system tray; right-click the tray icon for menu.
- Ensure Microsoft Flight Simulator 2020 is running.
- Ensure `SimConnect` is installed (pip package installed by requirements).
- For secure posting, your server should set LIVE_POST_KEY; enter the same key in the client.
 - Recent servers are stored via Windows registry (QSettings) under NavMap/Bridge.
