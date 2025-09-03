

FROM python:3.11-slim

# Install dependencies
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py /app/
COPY gpx_to_kml.py /app/

# Expose port
EXPOSE 8000

# Default command: serve FastAPI app
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
