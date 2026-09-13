#!/usr/bin/env bash
# Stall-detected band-gate runner — the M43 pool-wedge bounce procedure,
# automated (docs/M43.md "Gate infrastructure note": the m17-era native
# libcg hang fires intermittently at workers 8; all workers futex_wait,
# zero progress, no error). Detector: no *.jsonl under <out> changes for
# <interval> seconds while the run is alive -> kill the process group,
# DELETE the newest (suspect, chunk-partial) cell file, relaunch — resume
# is chunk-granular and completed cells return instantly.
#
# Usage: m46_run_watch.sh <band-spec.json> <out-dir> [interval-secs=180]
# Exit code = the underlying m46_band_gate run's exit code.
set -u
cd "$(dirname "$0")/.."
spec="${1:?band spec}"; out="${2:?out dir}"; interval="${3:-180}"
log="${out%/}.watch.log"
mkdir -p "$out"

hermes_ping () { hermes send -t telegram "[m46-watch] $1" 2>/dev/null || true; }

run_once () {
  setsid uv run python scripts/m46_band_gate.py run "$spec" --out "$out" \
    --workers 8 >>"$log" 2>&1 &
  PID=$!
  echo "launched pid $PID $(date)" >>"$log"
}

snapshot () {
  find "$out" -name '*.jsonl' -exec stat -c '%n %s %Y' {} + 2>/dev/null \
    | sort | md5sum | cut -d' ' -f1
}

run_once
bounces=0
while true; do
  s1=$(snapshot)
  sleep "$interval"
  if ! kill -0 "$PID" 2>/dev/null; then
    wait "$PID"; rc=$?
    echo "run exited rc=$rc after $bounces bounce(s) $(date)" >>"$log"
    exit "$rc"
  fi
  s2=$(snapshot)
  if [ "$s1" = "$s2" ]; then
    newest=$(ls -t "$out"/s*/*.jsonl 2>/dev/null | head -1)
    echo "WEDGE: zero growth for ${interval}s — bouncing (suspect: ${newest:-none}) $(date)" >>"$log"
    kill -TERM -- "-$PID" 2>/dev/null; sleep 5
    kill -KILL -- "-$PID" 2>/dev/null; sleep 2
    [ -n "$newest" ] && rm -f "$newest"
    bounces=$((bounces + 1))
    hermes_ping "pool wedge #$bounces bounced on $(basename "$spec") (deleted $(basename "${newest:-none}"), resuming)"
    run_once
  fi
done
