#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# run.sh — start the RVU Voice Assistant (FastAPI backend + Vite frontend)
#
#   ./run.sh            warm up all models, then open the app (best for demos)
#   ./run.sh --fast     skip model pre-loading (quick start, slow first query)
#   ./run.sh --no-open  don't open the browser automatically
#
# Ctrl+C stops both servers.
# ──────────────────────────────────────────────────────────────────────────
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT=8000
WEB_PORT=5173
API_URL="http://127.0.0.1:${API_PORT}"
WEB_URL="http://localhost:${WEB_PORT}"
STARTUP_TIMEOUT_S=600   # cold model warm-up can take a few minutes

FAST=false
OPEN_BROWSER=true
for arg in "$@"; do
  case "$arg" in
    --fast)    FAST=true ;;
    --no-open) OPEN_BROWSER=false ;;
    -h|--help) sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (see ./run.sh --help)"; exit 1 ;;
  esac
done

say()  { printf '\033[1;33m▸\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────
command -v python3 >/dev/null || fail "python3 not found"
command -v npm     >/dev/null || fail "npm not found"
command -v ffmpeg  >/dev/null || say "ffmpeg not found — voice input will fail (brew install ffmpeg)"
[[ -f "$ROOT/.env" ]] || fail ".env missing — run: cp .env.example .env and set HF_TOKEN"
[[ -f "$ROOT/data/processed/chunks.jsonl" ]] || fail "Knowledge base missing — run: python3 scripts/ingest.py"

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  say "Installing frontend dependencies …"
  (cd "$ROOT/frontend" && npm install) || fail "npm install failed"
fi

# Stop anything already listening on our ports (e.g. a previous run).
for port in "$API_PORT" "$WEB_PORT"; do
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    say "Port $port is busy — stopping old process(es): $(echo $pids)"
    kill $pids 2>/dev/null
    sleep 2
    kill -9 $pids 2>/dev/null || true
  fi
done

# ── Shutdown handling ─────────────────────────────────────────────────────
API_PID=""
WEB_PID=""
cleanup() {
  trap - INT TERM EXIT
  echo
  say "Shutting down …"
  for pid in $WEB_PID $API_PID; do
    pkill -TERM -P "$pid" 2>/dev/null
    kill -TERM "$pid" 2>/dev/null
  done
  wait 2>/dev/null
  say "Stopped."
}
trap 'cleanup; exit 0' INT TERM
trap cleanup EXIT

# Prefix each output line so the two logs are easy to tell apart.
tag() { awk -v t="$1" '{ print t " " $0; fflush() }'; }

# ── Backend ───────────────────────────────────────────────────────────────
if $FAST; then
  say "Starting backend (fast mode: models load on first use) …"
  export KEEP_LLM_RESIDENT=false KEEP_TTS_RESIDENT=false KEEP_EMBED_RESIDENT=false
else
  say "Starting backend and warming up models (first run can take a few minutes) …"
fi
export TQDM_DISABLE=1
(cd "$ROOT" && exec python3 -m uvicorn backend.app.main:app \
    --host 127.0.0.1 --port "$API_PORT" --log-level info) \
  > >(tag $'\033[36m[api]\033[0m') 2>&1 &
API_PID=$!

# ── Frontend ──────────────────────────────────────────────────────────────
say "Starting frontend …"
(cd "$ROOT/frontend" && exec ./node_modules/.bin/vite --port "$WEB_PORT" --strictPort) \
  > >(tag $'\033[35m[web]\033[0m') 2>&1 &
WEB_PID=$!

# ── Wait until both are ready ─────────────────────────────────────────────
deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -sf "${API_URL}/api/health" >/dev/null && curl -sf "$WEB_URL" >/dev/null; do
  kill -0 "$API_PID" 2>/dev/null || fail "Backend exited during startup — see [api] log above"
  kill -0 "$WEB_PID" 2>/dev/null || fail "Frontend exited during startup — see [web] log above"
  (( SECONDS < deadline )) || fail "Timed out after ${STARTUP_TIMEOUT_S}s waiting for servers"
  sleep 2
done

echo
printf '\033[1;32m✓ RVU Voice Assistant is ready\033[0m\n'
echo "    App:      $WEB_URL"
echo "    API docs: ${API_URL}/docs"
echo "    Press Ctrl+C to stop both servers."
echo
$OPEN_BROWSER && command -v open >/dev/null && open "$WEB_URL"

# Exit (and clean up) as soon as either server stops.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 2
done
say "A server stopped unexpectedly — shutting down the other one."
exit 1
