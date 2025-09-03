import gpxpy
import simplekml
import os

# Input/output files (use mounted volume paths)
gpx_file = os.getenv("GPX_FILE", "flight.gpx")
kml_file = os.getenv("KML_FILE", "flight_3d.kml")

with open(gpx_file, "r") as f:
    gpx = gpxpy.parse(f)

coords = []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            coords.append((point.longitude, point.latitude, point.elevation))

kml = simplekml.Kml()
ls = kml.newlinestring(name="Flight Trail 3D")
ls.coords = coords
ls.altitudemode = simplekml.AltitudeMode.absolute
ls.extrude = 0
ls.style.linestyle.width = 3
ls.style.linestyle.color = simplekml.Color.red

kml.save(kml_file)

print(f"✅ KML saved: {kml_file}")

