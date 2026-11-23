#!/usr/bin/env bash
# Start (or restart) the canopy-cover web tool in the background on this machine, listening on 127.0.0.1 only.
#   bash app/run.sh            # start on port 8765
#   bash app/run.sh stop       # stop it
# Log: <pipeline data>/app/server.log. Reach it through a port forward (VS Code "Ports" tab, or ssh -L 8765:localhost:8765).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8765}
DATA=$(.venv/bin/python -I -c "import sys; sys.path.insert(0, 'pipeline'); import common; print(common.DATA)")
mkdir -p "$DATA/app"
pkill -f "app/server.py --port $PORT" 2>/dev/null && sleep 2 || true
[ "${1:-}" = stop ] && { echo "stopped"; exit 0; }
setsid nohup .venv/bin/python -I app/server.py --port "$PORT" >> "$DATA/app/server.log" 2>&1 < /dev/null &
for i in $(seq 1 60); do
    if grep -q "running: http" "$DATA/app/server.log" 2>/dev/null && curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then break; fi
    sleep 1
done
grep "running: http" "$DATA/app/server.log" | tail -1
