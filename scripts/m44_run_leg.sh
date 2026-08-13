#!/usr/bin/env bash
# M44 Step 5 — run one leg with heartbeat + stall watchdog + failure alert.
# usage: m44_run_leg.sh <pair K|L|O|G> [extra m44_leg.py args, e.g. --resume]
set -u
cd "$(dirname "$0")/.."
PAIR=${1:?pair}; shift || true
TAG="m44_${PAIR}_r1"
LOG="runs/${TAG}.log"

uv run python scripts/m44_leg.py "$PAIR" --iterations 15 "$@" >> "$LOG" 2>&1 &
PID=$!
hermes send -t telegram "M44 leg $PAIR START (pid $PID, 15 it, tag $TAG). Pause: touch checkpoints/ppo_pause_${TAG}"

bash scripts/m44_heartbeat.sh "$PID" "$LOG" "$TAG" &
HB=$!
(
  while kill -0 "$PID" 2>/dev/null; do
    sleep 1200
    kill -0 "$PID" 2>/dev/null || break
    if [ $(( $(date +%s) - $(stat -c %Y "$LOG") )) -gt 1200 ]; then
      hermes send -t telegram "M44 STALL? [$TAG] no log output for 20+ min — possible mp.Pool wedge (M43: all-futex_wait; check and bounce the exact PID $PID)"
    fi
  done
) &
WD=$!

wait "$PID"; RC=$?
kill "$HB" "$WD" 2>/dev/null
if [ "$RC" -ne 0 ]; then
  hermes send -t telegram "M44 FAILURE: leg $PAIR exited rc=$RC. Tail: $(tail -2 "$LOG" | tr '\n' ' | ')"
else
  BEST=$(grep -E "PROMOTED" "$LOG" | tail -1)
  hermes send -t telegram "M44 leg $PAIR DONE rc=0. ${BEST:-No promotion (pair keeps its seed).} Last eval: $(grep -E 'vs_teacher' "$LOG" | tail -1)"
fi
exit "$RC"
