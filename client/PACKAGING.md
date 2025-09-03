Build a Windows EXE for the bridge GUI.

1) Install Python 3.10+ and dependencies:

   pip install -r client/requirements.txt

2) Build with PyInstaller:

   pyinstaller --noconsole --name NavMapBridge --onefile --icon NONE client/msfs_bridge_gui.pyw

3) Run the generated `dist/NavMapBridge.exe`.

Notes
- The app minimizes to the system tray; right-click the tray icon for menu.
- Ensure Microsoft Flight Simulator 2020 is running.
- Ensure `SimConnect` is installed (pip package installed by requirements).
- For secure posting, your server should set LIVE_POST_KEY; enter the same key in the client.
