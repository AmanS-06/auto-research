#!/bin/bash
set -e

echo "Running database migrations..."
alembic upgrade head

if [ "$RELOAD" = "true" ]; then
    echo "Starting API server with hot reload..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Starting API server..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
fi
