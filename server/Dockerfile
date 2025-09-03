FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application into /app (module name is 'app')
COPY server/app.py /app/app.py
COPY server/gpx_to_kml.py /app/gpx_to_kml.py
COPY static /app/static

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
