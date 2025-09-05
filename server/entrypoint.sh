#!/usr/bin/env bash
set -euo pipefail

# Wait for DB if DATABASE_URL provided
if [[ -n "${DATABASE_URL:-}" ]]; then
  echo "Running DB migrations..."
  alembic upgrade head || {
    echo "Alembic migration failed" >&2
    exit 1
  }
fi

echo "Starting server..."
exec uvicorn app:app "$@"

