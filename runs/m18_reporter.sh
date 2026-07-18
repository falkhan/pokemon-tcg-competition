#!/usr/bin/env bash
# M18 heartbeat (hard rule): watches a long-running pipeline stage, pings
# Telegram every ~30 min with the log tail and once on completion/crash.
# usage: m18_reporter.sh <tag> <log> <proc_pattern> <done_regex>
#   e.g. m18_reporter.sh "M18 collect" runs/m18_collect.log \
#          'plan_iter collect' '^done:'
TAG=$1
LOG=$2
PAT=$3
DONE_RE=$4
PING_EVERY=1800   # 30 min

ping() { hermes send -t telegram -f - 2>/dev/null <<<"[$TAG] $1"; }

LAST_PING=$(date +%s)
while true; do
  NOW=$(date +%s)
  if [ $((NOW - LAST_PING)) -ge $PING_EVERY ]; then
    ping "running: $(tail -1 "$LOG" 2>/dev/null | cut -c1-200)"
    LAST_PING=$NOW
  fi
  if ! pgrep -f "$PAT" >/dev/null; then
    sleep 3
    if ! pgrep -f "$PAT" >/dev/null; then
      if grep -qE "$DONE_RE" "$LOG" 2>/dev/null; then
        ping "DONE: $(grep -E "$DONE_RE" "$LOG" | tail -1 | cut -c1-200)"
      else
        ping "STOPPED without done-marker — check $LOG"
      fi
      break
    fi
  fi
  sleep 60
done
