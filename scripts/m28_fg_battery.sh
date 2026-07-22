#!/usr/bin/env bash
# M28 F+G strength battery. val_acc CANNOT rank these: the winners-only arms
# hold out a different (easier, winner-biased) val set — 1289 rows vs 2800 — so
# 0.699 vs 0.652 compares two different questions. Strength is the only common
# yardstick, as M26/M27 both concluded for different reasons.
# Pins (m27_both, plain `model:`): lucario 0.6625 n=400, dragapult 0.3425 n=400.
set -euo pipefail
cd "$(dirname "$0")/.."
for arm in m28_control m28_phase m28_winners m28_wphase; do
  for opp in lucario dragapult; do
    for s in 1 2; do
      uv run python -m rl.matchrunner play \
        --a "model:checkpoints/${arm}.pt:clone54618168" --b "rule:${opp}" \
        -n 200 --workers 8 --seed "$s" \
        --checkpoint "runs/m28fg_${arm#m28_}_${opp}_s${s}.jsonl"
    done
  done
done
echo "=== M28 FG BATTERY DONE $(date) ==="
