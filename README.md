NavMap – GPX→KML + Live Map (Web)

Overview
- Minimal FastAPI service to convert uploaded `.gpx` files to `.kml`.
- Simple, responsive HTML upload page at `/`.
- API endpoint at `/api/convert` returning KML with proper content type.

Run locally (Python)
1. Create a venv and install deps:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Start the server:
   - `uvicorn server.app:app --reload`
   - Optional: set `LIVE_POST_KEY=yoursecret` to require a key for posting live positions.
3. Open `http://localhost:8000` in your browser.

Docker (dev and prod)

- Development (no external networks/secrets):
  - `docker compose -f docker-compose.dev.yml up --build`
  - Hot reload enabled; edits to `server/app.py` and `static/` reflect immediately.
- Production (uses external `proxy` network and Docker secret for Cesium token):
  - Ensure Docker network `proxy` exists and secret file at `./secrets/cesium_ion_token`.
  - `docker compose up --build -d`

API
- `POST /api/convert`
  - Form fields: `file` (required, .gpx), optional `name`, `width`, `color` (web hex), `altitude_mode` (absolute|relativeToGround|clampToGround), `extrude` (0|1)
  - Returns: `application/vnd.google-earth.kml+xml` with `Content-Disposition: attachment`.

Live Map
- WebSocket: `/ws/live/{channel}`
  - Viewers: connect as-is.
  - Feeders: use `?mode=feeder&key=...` when `LIVE_POST_KEY` is set on server.
- HTTP POST: `/api/live/{channel}?key=...`
  - JSON body: `{ lat, lon, alt_m?, spd_kt?, vsi_ms?, hdg_deg?, pitch_deg?, roll_deg?, callsign?, aircraft?, ts? }`
- Recent: `GET /api/live/{channel}/recent`

Health check
- `GET /healthz` → `{ "status": "ok" }`

Client bridge (Windows)
- See `client/msfs_bridge.py` to stream MSFS 2020 position to the server in real-time.
- Install on Windows: `pip install SimConnect websockets requests`
- Run (WebSocket): `python msfs_bridge.py --server ws://YOUR_HOST:8000 --channel default --callsign N123AB --mode ws`
- Run (HTTP): `python msfs_bridge.py --server http://YOUR_HOST:8000 --channel default --callsign N123AB --mode http`

Notes
- Color is converted from web hex to KML’s `aabbggrr` (opaque).
- If no tracks are found, routes and waypoints are attempted.
- The original `gpx_to_kml.py` remains for CLI use in containers if needed.

Reverse proxy (example Nginx)
```
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```
