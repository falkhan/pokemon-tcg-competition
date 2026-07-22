#!/usr/bin/env bash
# M23 audit E1 gate battery: V3-as-BC clone strength probes (n=200 x 2 seeds
# per opponent, sequential — 8-worker hard cap is per-box, not per-run).
# Pre-registered: pooled >=~0.54 vs rule:lucario = arch exonerated; <=~0.45 =
# implicated. Decode: jsonl results 0=WIN (rl/matchrunner.py::series_wr).
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="checkpoints/bc_clone_v3_54618168.pt"
DECK="data/kaggle/clone_54618168_deck.csv"
for opp in rule:lucario rule:dragapult; do
  for seed in 1 2; do
    tag="m23_e1_${opp##*:}_s${seed}"
    echo "=== ${tag} $(date) ==="
    uv run python -m rl.matchrunner play \
      --a "model:${CKPT}:${DECK}" --b "${opp}" \
      -n 200 --workers 8 --seed "${seed}" \
      --checkpoint "runs/${tag}.jsonl"
  done
done
echo "=== E1 GATES DONE $(date) ==="
