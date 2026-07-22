#!/usr/bin/env bash
# M24 Phase 1 full battery on the screen winner (m24_bc_54618168, clone deck):
# rule:lucario seed 2 (seed 1 = screen 0.660), rule:dragapult n=200 x 2 seeds,
# random:kyogre floor n=200 x 2 seeds, then latency. Decode 0=WIN.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="checkpoints/m24_bc_54618168.pt"
DECK="data/kaggle/clone_54618168_deck.csv"
run() {
  echo "=== $3 $(date) ==="
  uv run python -m rl.matchrunner play \
    --a "model:${CKPT}:${DECK}" --b "$1" \
    -n 200 --workers 8 --seed "$2" \
    --checkpoint "runs/$3.jsonl"
}
run rule:lucario 2 m24_bat_lucario_s2
run rule:dragapult 1 m24_bat_dragapult_s1
run rule:dragapult 2 m24_bat_dragapult_s2
run random:kyogre 1 m24_bat_random_s1
run random:kyogre 2 m24_bat_random_s2
echo "=== latency $(date) ==="
uv run python -m rl.matchrunner play \
  --a "model:${CKPT}:${DECK}" --b rule:lucario \
  -n 20 --workers 1 --seed 3 --latency
echo "=== M24 P1 BATTERY DONE $(date) ==="
