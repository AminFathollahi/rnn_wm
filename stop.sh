#!/usr/bin/env bash
# Stop the running job.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
pidfile=logs/current.pid
if [[ ! -f $pidfile ]] || ! kill -0 "$(cat $pidfile)" 2>/dev/null; then
    echo "nothing running"
    exit 0
fi
pgid=$(ps -o pgid= -p "$(cat $pidfile)" | tr -d ' ')
kill -TERM -- "-$pgid" 2>/dev/null || true
for _ in $(seq 20); do
    kill -0 "$(cat $pidfile)" 2>/dev/null || { echo "stopped"; rm -f $pidfile; exit 0; }
    sleep 1
done
kill -KILL -- "-$pgid" 2>/dev/null || true
rm -f $pidfile
echo "stopped"
