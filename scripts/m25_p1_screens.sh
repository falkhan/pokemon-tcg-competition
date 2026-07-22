#!/usr/bin/env bash
# M25 Phase 1.1 screens: each Grimmsnarl clone on ITS OWN deck (deck+pilot are
# a unit — M24 proved zero-shot transfer dead) vs rule:lucario, n=200 seed 1.
# Decode: 0 = side-a WIN. Full battery (seed 2, dragapult, floors) follows on
# the winner via the measure-agent protocol.
set -euo pipefail
cd "$(dirname "$0")/.."
GRIM_DECK="data/kaggle/grimmsnarl_3121746f_deck.csv"
for sub in 54861775 54863653; do
  tag="m25_scr_grim_${sub}"
  echo "=== ${tag} $(date) ==="
  uv run python -m rl.matchrunner play \
    --a "model:checkpoints/m25_bc_grim_${sub}.pt:${GRIM_DECK}" --b rule:lucario \
    -n 200 --workers 8 --seed 1 \
    --checkpoint "runs/${tag}.jsonl"
done
echo "=== M25 P1 SCREENS DONE $(date) ==="
