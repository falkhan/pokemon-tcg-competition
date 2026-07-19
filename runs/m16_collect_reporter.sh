#!/usr/bin/env bash
# Watches the M17 10k policy collection: polls per-worker progress beacons
# + shard count, pings Telegram every ~30 min and on completion/crash.
OUT=/home/falkhan/Documents/python_projects/pokemon-tcg-competition/data/plan_m17
TARGET=10000
LAST_SHARDS=0
LAST_PING=$(date +%s)
PING_EVERY=1800   # 30 min

ping() { hermes send -t telegram -f - 2>/dev/null <<<"[M17 collect] $1"; }

while true; do
  SHARDS=$(ls "$OUT"/shard_*.npz 2>/dev/null | wc -l)
  # plan_iter writes no .progress_* beacons, so derive games from shards
  # (shard_size=200). This is a lower bound (last partial shard not counted).
  GAMES=$((SHARDS * 200))
  NOW=$(date +%s)
  if [ "$SHARDS" -gt "$LAST_SHARDS" ]; then
    LAST_SHARDS=$SHARDS
    # only ping on real progress every PING_EVERY, not every shard
    if [ $((NOW - LAST_PING)) -ge $PING_EVERY ]; then
      ping "running: ~$GAMES/$TARGET games ($SHARDS shards, $((SHARDS*100/50))% of 50)"
      LAST_PING=$NOW
    fi
  fi
  if ! pgrep -f 'plan_iter collect' >/dev/null; then
    # give it a moment in case it's between the collect finish and the tail flush
    sleep 3
    if ! pgrep -f 'plan_iter collect' >/dev/null; then
      if grep -q '^done:' "$OUT"/../runs/m16_collect_10k.log 2>/dev/null; then
        ping "DONE — 10k collection finished (see runs/m16_collect_10k.log). Next: train + ship."
      else
        ping "STOPPED (no 'done:' line) — check runs/m16_collect_10k.log for errors."
      fi
      break
    fi
  fi
  sleep 60
done
