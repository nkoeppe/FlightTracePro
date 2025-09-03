Windows bridge script to stream MSFS 2020 position to the server.

Quick start (Windows, Python 3.10+):

- Install dependencies:
- `pip install SimConnect==0.4.26 websockets requests`
- Run in WebSocket mode (recommended):
  - `python msfs_bridge.py --server ws://YOUR_HOST:8000 --channel default --callsign N123AB --mode ws`
- Or HTTP mode (simpler, slightly more overhead):
  - `python msfs_bridge.py --server http://YOUR_HOST:8000 --channel default --callsign N123AB --mode http`

Notes

- If `pip install -r client/requirements.txt` fails on SimConnect version, ensure it is `SimConnect==0.4.26`.
- PySide6 is large; first install can take a minute on Windows.

Optional post key for secured posting:

- Server can set `LIVE_POST_KEY` env var.
- Bridge then passes `--key YOUR_KEY`.

Notes:

- Requires MSFS 2020 running. Little Navmap is not required.
- If `SimConnect` is not installed, the script prints guidance.
- Use `--rate 10` to send 10 Hz updates if your machine allows it.
