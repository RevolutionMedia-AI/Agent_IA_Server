#!/bin/sh
set -e
# Default to 8080 when PORT is not set
PORT="${PORT:-8080}"
echo "Starting app on port $PORT"
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
