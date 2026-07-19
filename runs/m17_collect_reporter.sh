#!/usr/bin/env bash
# Watches the M17 10k policy collection (or its remainder run): pings Telegram
# on a FIXED time interval (regardless of whether shards increased) AND on
# completion/crash. This guarantees the user is never left in the dark if the
# run stalls (the earlier version only pinged on shard-count increase, so a
# frozen run went silent for 2h).
OUT=/home/falkhan/Documents/python_projects/pokemon-tcg-competition/data/plan_m17b
TARGET=10000
PING_EVERY=900   # 15 min, no matter what
LAST_PING=$(date +%s)

PING() { hermes send -t telegram -f - 2>/dev/null <<<"[M17 collect] $1"; }

while true; do
  SHARDS=$(ls "$OUT"/shard_*.npz 2>/dev/null | wc -l)
  GAMES=$((SHARDS * 200))   # plan_iter writes no .progress_* beacons; derive
  NOW=$(date +%s)
  if [ $((NOW - LAST_PING)) -ge $PING_EVERY ]; then
    if pgrep -f 'plan_iter collect' >/dev/null; then
      PING "alive: ~$GAMES/$TARGET games ($SHARDS shards, $((SHARDS*100/50))% of 50) — still running"
    else
      if grep -q '^done:' runs/m16_collect_10k.log runs/m16_collect_10kb.log 2>/dev/null; then
        PING "DONE — collection finished. Orchestrator will train + ship."
      else
        PING "STOPPED (no 'done:' line) — check runs/m16_collect_10kb.log for errors."
      fi
      break
    fi
    LAST_PING=$NOW
  fi
  sleep 60
done
