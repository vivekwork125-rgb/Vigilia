#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
.venv/bin/pip install -q -r backend/requirements.txt
if [[ ! -d frontend/node_modules ]]; then (cd frontend && npm ci --no-fund); fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH="$PWD/backend"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8100}"
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8100 &
backend_pid=$!
(cd frontend && npm run dev) &
frontend_pid=$!
trap 'kill "$backend_pid" "$frontend_pid" 2>/dev/null || true' EXIT INT TERM
printf '\nVIGILIA: http://localhost:3100\nAPI docs: http://127.0.0.1:8100/docs\n\n'
wait
