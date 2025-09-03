GPX → KML Converter (Web)

Overview
- Minimal FastAPI service to convert uploaded `.gpx` files to `.kml`.
- Simple, responsive HTML upload page at `/`.
- API endpoint at `/api/convert` returning KML with proper content type.

Run locally (Python)
1. Create a venv and install deps:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Start the server:
   - `uvicorn app:app --reload`
3. Open `http://localhost:8000` in your browser.

Docker
1. Build:
   - `docker build -t gpx2kml-web .`
2. Run:
   - `docker run --rm -p 8000:8000 gpx2kml-web`
3. Browse:
   - `http://localhost:8000`

API
- `POST /api/convert`
  - Form fields: `file` (required, .gpx), optional `name`, `width`, `color` (web hex), `altitude_mode` (absolute|relativeToGround|clampToGround), `extrude` (0|1)
  - Returns: `application/vnd.google-earth.kml+xml` with `Content-Disposition: attachment`.

Health check
- `GET /healthz` → `{ "status": "ok" }`

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

