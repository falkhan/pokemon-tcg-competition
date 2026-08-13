#!/usr/bin/env bash
# Hourly Hermes heartbeat for a long-running leg (CLAUDE.md standing rule).
# usage: m44_heartbeat.sh <pid> <logfile> <tag>
PID=$1; LOG=$2; TAG=$3
while kill -0 "$PID" 2>/dev/null; do
  sleep 3600
  kill -0 "$PID" 2>/dev/null || break
  LAST=$(grep -E "^iter [0-9]+: (defect|vs_)" "$LOG" | tail -1)
  hermes send -t telegram "M44 heartbeat [$TAG]: ${LAST:-collecting, no iter line yet} ($(date +%H:%M))"
done
