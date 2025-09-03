from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import gpxpy
import simplekml
from io import StringIO, BytesIO
from typing import Optional


app = FastAPI(title="GPX → KML Converter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_gpx_to_kml(
    gpx_text: str,
    line_name: str = "Flight Trail 3D",
    line_color: str = simplekml.Color.red,
    line_width: int = 3,
    altitude_mode: str = "absolute",
    extrude: int = 0,
) -> bytes:
    try:
        gpx = gpxpy.parse(StringIO(gpx_text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid GPX: {e}")

    coords = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coords.append((point.longitude, point.latitude, point.elevation))

    if not coords:
        # Also try routes and waypoints if no tracks
        for route in gpx.routes:
            for point in route.points:
                coords.append((point.longitude, point.latitude, point.elevation))
        if not coords:
            for wpt in gpx.waypoints:
                coords.append((wpt.longitude, wpt.latitude, wpt.elevation))

    if not coords:
        raise HTTPException(status_code=400, detail="No coordinates found in GPX")

    kml = simplekml.Kml()
    ls = kml.newlinestring(name=line_name)
    ls.coords = coords

    # Altitude mode mapping
    mode_map = {
        "absolute": simplekml.AltitudeMode.absolute,
        "relativeToGround": simplekml.AltitudeMode.relativetoground,
        "clampToGround": simplekml.AltitudeMode.clamptoground,
    }
    ls.altitudemode = mode_map.get(altitude_mode, simplekml.AltitudeMode.absolute)
    ls.extrude = extrude
    ls.style.linestyle.width = line_width
    ls.style.linestyle.color = line_color

    kml_bytes = kml.kml().encode("utf-8")
    return kml_bytes


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/api/convert")
async def api_convert(
    file: UploadFile = File(..., description="GPX file"),
    name: Optional[str] = Form(default="Flight Trail 3D"),
    width: Optional[int] = Form(default=3),
    color: Optional[str] = Form(default="#ff0000"),  # hex in RRGGBB or AABBGGRR? simplekml expects aabbggrr
    altitude_mode: Optional[str] = Form(default="absolute"),
    extrude: Optional[int] = Form(default=0),
):
    if not file.filename or not file.filename.lower().endswith(".gpx"):
        raise HTTPException(status_code=400, detail="Please upload a .gpx file")

    # Convert web hex color to simplekml color (aabbggrr). Use full opacity.
    def web_hex_to_simplekml(hex_color: str) -> str:
        c = hex_color.strip().lstrip('#')
        if len(c) == 3:  # expand shorthand like f00
            c = ''.join([ch * 2 for ch in c])
        if len(c) != 6:
            return simplekml.Color.red
        rr, gg, bb = c[0:2], c[2:4], c[4:6]
        # simplekml uses aabbggrr; use ff for alpha (opaque)
        return f"ff{bb}{gg}{rr}"

    gpx_text = (await file.read()).decode("utf-8", errors="ignore")
    kml_bytes = convert_gpx_to_kml(
        gpx_text,
        line_name=name or "Flight Trail 3D",
        line_color=web_hex_to_simplekml(color or "#ff0000"),
        line_width=int(width) if width else 3,
        altitude_mode=altitude_mode or "absolute",
        extrude=int(extrude) if extrude is not None else 0,
    )

    filename_base = file.filename.rsplit('.', 1)[0]
    out_name = f"{filename_base}.kml"

    return StreamingResponse(
        BytesIO(kml_bytes),
        media_type="application/vnd.google-earth.kml+xml",
        headers={"Content-Disposition": f"attachment; filename={out_name}"},
    )


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GPX → KML Converter</title>
    <style>
      :root { color-scheme: light dark; }
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; }
      .container { max-width: 720px; margin: 40px auto; padding: 24px; }
      .card { border: 1px solid #ddd; border-radius: 12px; padding: 24px; }
      h1 { margin-top: 0; }
      label { display: block; margin: 12px 0 4px; font-weight: 600; }
      input[type="file"], input[type="text"], input[type="number"], select {
        width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #ccc;
      }
      .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .actions { margin-top: 16px; display: flex; gap: 12px; align-items: center; }
      button { background: #2563eb; color: white; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
      button:disabled { opacity: 0.6; cursor: default; }
      .note { font-size: 12px; color: #666; }
      footer { margin-top: 24px; font-size: 12px; color: #666; text-align: center; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h1>GPX → KML Converter</h1>
        <p>Upload a <strong>.gpx</strong> file and download a <strong>.kml</strong>.</p>
        <form id="form">
          <label for="file">GPX file</label>
          <input id="file" name="file" type="file" accept=".gpx,application/gpx+xml" required />

          <div class="row">
            <div>
              <label for="name">Line name</label>
              <input id="name" name="name" type="text" value="Flight Trail 3D" />
            </div>
            <div>
              <label for="color">Line color</label>
              <input id="color" name="color" type="color" value="#ff0000" />
            </div>
          </div>

          <div class="row">
            <div>
              <label for="width">Line width</label>
              <input id="width" name="width" type="number" min="1" max="10" value="3" />
            </div>
            <div>
              <label for="altitude_mode">Altitude mode</label>
              <select id="altitude_mode" name="altitude_mode">
                <option value="absolute" selected>absolute</option>
                <option value="relativeToGround">relativeToGround</option>
                <option value="clampToGround">clampToGround</option>
              </select>
            </div>
          </div>

          <div class="row">
            <div>
              <label for="extrude">Extrude</label>
              <select id="extrude" name="extrude">
                <option value="0" selected>no</option>
                <option value="1">yes</option>
              </select>
            </div>
          </div>

          <div class="actions">
            <button id="convert" type="submit">Convert to KML</button>
            <span class="note" id="status"></span>
          </div>
        </form>
      </div>
      <footer>
        Powered by FastAPI, GPXPy, and simplekml.
      </footer>
    </div>

    <script>
      const form = document.getElementById('form');
      const statusEl = document.getElementById('status');
      const btn = document.getElementById('convert');

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        statusEl.textContent = '';
        btn.disabled = true;
        try {
          const formData = new FormData(form);
          const file = formData.get('file');
          if (!file || !file.name.endsWith('.gpx')) {
            statusEl.textContent = 'Please choose a .gpx file';
            btn.disabled = false;
            return;
          }
          statusEl.textContent = 'Converting…';
          const res = await fetch('/api/convert', { method: 'POST', body: formData });
          if (!res.ok) {
            const msg = await res.text();
            throw new Error(msg || 'Conversion failed');
          }
          const blob = await res.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          const name = file.name.replace(/\.[^.]+$/, '') + '.kml';
          a.href = url; a.download = name; a.click();
          window.URL.revokeObjectURL(url);
          statusEl.textContent = 'Done!';
        } catch (err) {
          statusEl.textContent = (err && err.message) ? err.message : 'Error converting file';
        } finally {
          btn.disabled = false;
        }
      });
    </script>
  </body>
  </html>
"""

