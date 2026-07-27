#!/usr/bin/env bash
# Reusable hourly heartbeat -> Hermes Telegram for a long-running training log.
#
# Usage: hermes_heartbeat.sh <tag> <logfile> <progress_grep_re> [interval_secs]
#   tag    short label for the run (e.g. m32b)
#   log    file the run writes to (tail is grepped)
#   re     egrep pattern selecting the progress line to report (e.g. '^iter ')
#   interval  seconds between pulses (default 3600 = hourly)
#
# Sends the latest matching progress line every <interval>s. If the log stops
# growing for 2 consecutive intervals the run has ended/stalled: a final line is
# sent and the reporter exits. Launch with run_in_background so its sleeps do not
# block. Confirm completion against the run's own exit, not this reporter
# (per the stale-line caveat in the campaign memory).
set -u
tag="${1:?tag}"; log="${2:?logfile}"; re="${3:?grep re}"; interval="${4:-3600}"
last_size=-1; stale=0
while true; do
  sleep "$interval"
  line=$(grep -E "$re" "$log" 2>/dev/null | tail -1)
  size=$(wc -c < "$log" 2>/dev/null || echo 0)
  hermes send -t telegram "[$tag] heartbeat: ${line:-<no progress line yet>} (log ${size}B)" >/dev/null 2>&1
  if [ "$size" = "$last_size" ]; then stale=$((stale + 1)); else stale=0; fi
  last_size="$size"
  if [ "$stale" -ge 2 ]; then
    hermes send -t telegram "[$tag] log stopped growing for 2 intervals — run ended/stalled. Last: ${line:-<none>}" >/dev/null 2>&1
    break
  fi
done
