#!/usr/bin/env bash
# Start one detached job, refusing to run a second alongside it.
set -Eeuo pipefail
name=$1; shift
mkdir -p logs
pidfile=logs/current.pid
if [[ -f $pidfile ]] && kill -0 "$(cat $pidfile)" 2>/dev/null; then
    echo "already running: $(cat logs/current.name 2>/dev/null || echo job) (pid $(cat $pidfile))"
    echo "watch with: tail -f logs/current.log      stop with: ./stop.sh"
    exit 1
fi
printf '%s' "$name" > logs/current.name
nohup setsid env PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" "$@" > "logs/current.log" 2>&1 < /dev/null &
echo $! > $pidfile
sleep 20
if ! kill -0 "$(cat $pidfile)" 2>/dev/null; then
    rm -f $pidfile
    echo "$name exited immediately:"
    tail -n 15 logs/current.log
    exit 1
fi
echo "$name running (pid $(cat $pidfile))"
tail -n 3 logs/current.log
echo "watch with: tail -f logs/current.log      stop with: ./stop.sh"
