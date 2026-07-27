#!/usr/bin/env bash
# M28 Track E: train a REAL value function for the v3 lineage.
#
# Why: B (MCTS) and C (PPO) both died, and both failures trace to the value
# estimate. rl/value_train.py's own docstring measured the auxiliary head at
# 0.62 sign-accuracy (~chance) vs ~0.87 for a supervised one, and named the
# mechanism as "documented-but-never-executed": the head is OOD on determinized
# search states, so train it ON those states (--search-plies).
#
# Two arms so the search-state half is attributable:
#   onpolicy  game states only            (--search-plies 0)
#   search    + determinized search states (--search-plies 4)
#
# Gate (pre-registered, docs/M28-plan.md E3): held-out sign-accuracy >= 0.75,
# split BY GAME (a per-state split leaks the outcome label and inflates it).
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=checkpoints/m27_both.pt
DECK=clone54618168

echo "=== collect on-policy $(date) ==="
uv run python -m rl.value_train collect --checkpoint "$CKPT" --deck "$DECK" \
  --games 300 --out data/value_m28_onpolicy.npz

echo "=== collect with search states $(date) ==="
uv run python -m rl.value_train collect --checkpoint "$CKPT" --deck "$DECK" \
  --games 150 --search-plies 4 --search-samples 2 \
  --out data/value_m28_search.npz

for arm in onpolicy search; do
  echo "=== train $arm $(date) ==="
  uv run python -m rl.value_train train --checkpoint m27_both.pt \
    --data "data/value_m28_${arm}.npz" --out "m28_value_${arm}.pt"
done
echo "=== M28 TRACK E DONE $(date) ==="
