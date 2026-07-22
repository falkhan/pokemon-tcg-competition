#!/usr/bin/env bash
# M25 Phase 1.1: train Grimmsnarl-mill teacher clones (fresh V3-as-BC, E1/M24
# recipe: fresh init, epochs 10). One checkpoint per single-teacher corpus —
# the M24 law says single-teacher beats pooled; the pooled corpus is the backup.
# Record val_acc in docs/M25.md immediately (checkpoints store no metadata).
set -euo pipefail
cd "$(dirname "$0")/.."
for sub in 54861775 54863653; do
  echo "=== train m25_bc_grim_${sub} $(date) ==="
  uv run python -m rl.plan_iter train \
    --data "data/bc_m25_grimmsnarl_${sub}" \
    --name "m25_bc_grim_${sub}" \
    --epochs 10 --lr 1e-4
done
echo "=== M25 P1 TRAIN DONE $(date) ==="
