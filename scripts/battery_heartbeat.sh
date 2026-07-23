#!/usr/bin/env bash
# Hourly Hermes heartbeat for a long battery run (CLAUDE.md standing rule).
# Usage: battery_heartbeat.sh <tag> <glob_prefix> <done_marker_log>
# Reports completed-run count and the newest jsonl; exits when the done
# marker line appears in the log.
set -u
cd "$(dirname "$0")/.."
TAG="${1:?tag}"; PREFIX="${2:?glob prefix}"; LOG="${3:?log file}"
while true; do
  sleep 3600
  if grep -q "DONE" "$LOG" 2>/dev/null; then
    hermes send -t telegram "[$TAG] battery finished ($(date +%H:%M))." || true
    exit 0
  fi
  n=$(ls ${PREFIX}*.jsonl 2>/dev/null | wc -l)
  latest=$(ls -t ${PREFIX}*.jsonl 2>/dev/null | head -1)
  hermes send -t telegram \
    "[$TAG] heartbeat $(date +%H:%M): ${n} result files, latest ${latest##*/}" \
    || true
done
