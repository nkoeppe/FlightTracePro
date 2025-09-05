FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application into /app (module name is 'app')
COPY server/app.py /app/app.py
COPY server/gpx_to_kml.py /app/gpx_to_kml.py
COPY server/alembic.ini /app/alembic.ini
COPY server/alembic /app/alembic
COPY static /app/static
COPY server/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
