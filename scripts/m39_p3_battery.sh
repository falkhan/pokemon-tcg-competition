#!/usr/bin/env bash
# M39 P3 — gate the policy-lane arms on the G-13 panel roster.
#
# Every arm is a NET, run under the LIVE rule config (`conserve`), so the
# single variable against the `conserve` control battery is the checkpoint.
# That keeps P2 (rules) and P3 (policy) separable inside one milestone, which
# is what M38 failed to do and what G-7 exists to enforce at ship time.
#
#   vsloss      corpus A, alpha=0     harvested vs-loss seats, winners only
#   vsloss_a25  corpus A, alpha=0.25  do the kept LOSER rows carry signal?
#   bestresp    corpus B, alpha=0     filtered self-imitation vs the beds
#   retain_a    corpus A + champion shards  } P3-C retention (BACKLOG sweep
#   retain_b    corpus B + champion shards  } #3): does data mixing break the
#                                             ~1-epoch dose law? Both run
#                                             because A and B fail independently.
#
# NOT gated: `bestresp_lo` (the single-pass corpus-B net). The n=400 screen put
# it at -8.1pp mean and NEGATIVE on all four screen beds; spending 72 panel
# cells to rank a net that lost on every screen bed is not a measurement, it is
# a formality. Recorded as a kill in the diary with its numbers.
#
# The 900+ ceiling panel RUNS for these arms (unlike P2, where no bed can
# fire a rule): per the plan it does not gate Ship B, it is the slot-3
# selection signal, and m39_decide.py prints it apart from the weighted pool.
#
# P3-B exploiter-overfit control, pre-registered: `bestresp` must win on the
# FULL WEIGHTED ROSTER. A win on the beds it was collected against together
# with weighted-pool harm is a KILL, not a ship (0.90 at 32 decks -> 0.54 at
# 1024, arXiv 2404.16689 — and our beds are three clones).
set -u
cd "$(dirname "$0")/.."

LOG="${WAIT_FOR:-runs/m39_p2_battery.log}"
if [ -f "$LOG" ] && ! grep -q "P2 BATTERY DONE" "$LOG"; then
  echo "[p3] waiting for $LOG to finish  $(date)"
  while ! grep -q "P2 BATTERY DONE" "$LOG"; do sleep 30; done
fi

ARGS=()
# Unquoted on purpose: the default list must word-split into separate arms.
# shellcheck disable=SC2068
for arm in ${@:-vsloss vsloss_a25 bestresp retain_a retain_b}; do
  ck="checkpoints/m39_${arm}.pt"
  if [ ! -f "$ck" ]; then
    echo "[p3] SKIP $arm - $ck not built" >&2
    continue
  fi
  ARGS+=("$arm" "model-conserve:$ck:alakazam_v2_h4")
done
[ "${#ARGS[@]}" -eq 0 ] && { echo "[p3] no arms to gate" >&2; exit 2; }

bash scripts/m39_panel_gate.sh "${ARGS[@]}"
echo "=== M39 P3 BATTERY DONE $(date) ==="
