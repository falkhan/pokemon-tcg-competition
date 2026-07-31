#!/usr/bin/env bash
# M38 G2 — re-pin every solver-backed baseline AFTER the G1 winner landed
# (docs/M38-plan.md: pins made against the wrong bar go stale immediately;
# fresh controls everywhere before any retrain arm is judged).
#
# Pins (2-seed pooled, n=400/seed = 800/bed, workers 8):
#   champ_lucario : champion (m28_winners + gacfr3) vs solver:lucario
#                   -- the campaign-bar bed, corrected + semantic gate
#   plain_lucario : m28_winners WITHOUT rules vs solver:lucario
#                   -- the G5 no-rules baseline, pinned same-day
#   champ_grim    : champion vs the grim BC clone (M26-era pin was 0.650)
#   champ_rocket  : champion vs the rocket BC clone (M30-era pin was 0.610)
#
# Usage: bash scripts/m38_repin.sh   (resume-safe: run_pairs checkpoints)
set -u
cd "$(dirname "$0")/.."

CHAMP="modelt-gacfr3:checkpoints/m28_winners.pt:alakazam_v2_h4"
PLAIN="model:checkpoints/m28_winners.pt:alakazam_v2_h4"
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"

run() {  # run <name> <spec_a> <spec_b>
  local name="$1" a="$2" b="$3"
  for seed in 1 2; do
    echo "[m38 G2] $name seed $seed"
    uv run python -m rl.matchrunner play --a "$a" --b "$b" -n 400 \
      --workers 8 --seed "$seed" \
      --checkpoint "runs/m38_repin_${name}_s${seed}.jsonl"
  done
}

run champ_lucario "$CHAMP" "solver:lucario"
run plain_lucario "$PLAIN" "solver:lucario"
run champ_grim "$CHAMP" "$GRIM"
run champ_rocket "$CHAMP" "$ROCKET"
echo "DONE"
