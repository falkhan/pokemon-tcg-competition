#!/usr/bin/env bash
# M29 P3: ship-config battery (modelt: — O1 on, as live) for both candidates.
# Pins = live 54903635 battery: lucario 0.671 n=800 · dragapult 0.3375 n=400 ·
# grim 0.6875 n=400 (advisory) · kyogre >=0.95. Decode: 0=WIN.
set -euo pipefail
cd "$(dirname "$0")/.."
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
for arm in m28_winners m29_pooled_winners; do
  A="modelt:checkpoints/${arm}.pt:clone54618168"
  for s in 1 2 3 4; do
    uv run python -m rl.matchrunner play --a "$A" --b rule:lucario -n 200 --workers 8 --seed "$s" --checkpoint "runs/m29_${arm}_luc_s${s}.jsonl"
  done
  for s in 1 2; do
    uv run python -m rl.matchrunner play --a "$A" --b rule:dragapult -n 200 --workers 8 --seed "$s" --checkpoint "runs/m29_${arm}_drag_s${s}.jsonl"
    uv run python -m rl.matchrunner play --a "$A" --b "$GRIM" -n 200 --workers 8 --seed "$s" --checkpoint "runs/m29_${arm}_grim_s${s}.jsonl"
  done
  uv run python -m rl.matchrunner play --a "$A" --b random:kyogre -n 200 --workers 8 --seed 1 --checkpoint "runs/m29_${arm}_kyo_s1.jsonl"
done
echo "=== M29 P3 BATTERY DONE $(date) ==="
