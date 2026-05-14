#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

pick_port() {
  python - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
while True:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        port += 1
    else:
        print(port)
        break
    finally:
        sock.close()
PY
}

wait_for_url() {
  python - "$1" "$2" <<'PY'
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

url = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
last_err = "no response"

while time.time() < deadline:
    try:
        with urlopen(url, timeout=2) as resp:
            if 200 <= resp.status < 500:
                print(url)
                raise SystemExit(0)
            last_err = f"http {resp.status}"
    except URLError as exc:
        last_err = str(exc)
    except Exception as exc:  # pragma: no cover - shell helper
        last_err = str(exc)
    time.sleep(0.5)

print(f"timeout waiting for {url}: {last_err}", file=sys.stderr)
raise SystemExit(1)
PY
}

show_logs() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo
    echo "--- tail: $file ---"
    tail -n 40 "$file" || true
  fi
}

cleanup() {
  local status="$?"
  trap - EXIT INT TERM
  for pid in "${FRONTEND_PID:-}" "${API_PID:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    fi
  done
  exit "$status"
}

need_cmd python
need_cmd npm

if [[ ! -d frontend/node_modules ]]; then
  echo "frontend/node_modules is missing. run: cd frontend && npm install" >&2
  exit 1
fi

API_HOST="${API_HOST:-127.0.0.1}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
REQUESTED_API_PORT="${API_PORT:-8000}"
REQUESTED_FRONTEND_PORT="${FRONTEND_PORT:-5173}"
LOG_DIR="${LOG_DIR:-runs/dev}"
mkdir -p "$LOG_DIR"

API_PORT="$(pick_port "$REQUESTED_API_PORT")"
FRONTEND_PORT="$(pick_port "$REQUESTED_FRONTEND_PORT")"
API_LOG="$LOG_DIR/api.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
PROXY_TARGET="http://${API_HOST}:${API_PORT}"

if [[ "$API_PORT" != "$REQUESTED_API_PORT" ]]; then
  echo "api port $REQUESTED_API_PORT was busy, using $API_PORT"
fi
if [[ "$FRONTEND_PORT" != "$REQUESTED_FRONTEND_PORT" ]]; then
  echo "frontend port $REQUESTED_FRONTEND_PORT was busy, using $FRONTEND_PORT"
fi

trap cleanup EXIT INT TERM

python -c "from api.server import run; run(host='${API_HOST}', port=${API_PORT})" >"$API_LOG" 2>&1 &
API_PID=$!

(
  cd frontend
  VITE_API_PROXY_TARGET="$PROXY_TARGET" \
  VITE_DEV_HOST="$FRONTEND_HOST" \
  VITE_DEV_PORT="$FRONTEND_PORT" \
  npm run dev
) >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

if ! wait_for_url "${PROXY_TARGET}/api/health" 30 >/dev/null; then
  echo "backend did not become ready" >&2
  show_logs "$API_LOG"
  show_logs "$FRONTEND_LOG"
  exit 1
fi

if ! wait_for_url "http://${FRONTEND_HOST}:${FRONTEND_PORT}/" 60 >/dev/null; then
  echo "frontend did not become ready" >&2
  show_logs "$API_LOG"
  show_logs "$FRONTEND_LOG"
  exit 1
fi

if ! wait_for_url "http://${FRONTEND_HOST}:${FRONTEND_PORT}/api/health" 30 >/dev/null; then
  echo "frontend proxy did not become ready" >&2
  show_logs "$API_LOG"
  show_logs "$FRONTEND_LOG"
  exit 1
fi

cat <<EOF
services are up

api:      ${PROXY_TARGET}
frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}
logs:     ${API_LOG}
          ${FRONTEND_LOG}

press ctrl+c to stop both services
EOF

set +e
wait -n "$API_PID" "$FRONTEND_PID"
status="$?"
set -e

if kill -0 "$API_PID" >/dev/null 2>&1 || kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
  echo "one service exited, shutting the rest down" >&2
fi

exit "$status"
