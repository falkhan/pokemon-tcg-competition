#!/usr/bin/env bash
# M33 Stage 1 gate: same-corpus A/B of the setup value.
#   baseline = current recipe (extra_dim=0, deck_idx-matched pairs)
#   m33      = late-game features + coarse late-bucket pairs
# Both from scratch (no warm-start) so the ONLY difference is the changes.
# Read the per-bucket accuracy in the last epoch's [t..] detail; the GATE is
# whether t32+ climbs from ~0.46-0.66 toward the ~0.80 early bar.
# NOTE: torch training only (no mp workers) — run AFTER the collect frees CPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-data/setupval_m33}"
EPOCHS="${2:-8}"
echo "=== BASELINE (extra_dim=0, old pairs) ==="
uv run python -m rl.setup_value train --data "$DATA" --name m33_sv_base \
  --epochs "$EPOCHS" 2>&1 | tee runs/m33_sv_base.log \
  | grep -E "recipe:|states,|train pairs|epoch|best"
echo
echo "=== M33 (late feats + coarse pairs) ==="
uv run python -m rl.setup_value train --data "$DATA" --name m33_sv_late \
  --epochs "$EPOCHS" --late --coarse-pairs 2>&1 | tee runs/m33_sv_late.log \
  | grep -E "recipe:|states,|train pairs|epoch|best"
echo
echo "=== M33 Stage 1 A/B DONE $(date) ==="
