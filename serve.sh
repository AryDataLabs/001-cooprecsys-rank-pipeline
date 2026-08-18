#!/usr/bin/env bash
# serve.sh — Start the FastAPI recommendation server
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
echo "🚀 Starting cooprecsys API on $HOST:$PORT"
uvicorn cooprecsys.api.serve:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
