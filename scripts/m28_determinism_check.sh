#!/usr/bin/env bash
# M28: is matchrunner variance BINOMIAL or overdispersed?
#
# cg.game.battle_start(deck0, deck1) takes NO seed and cg exposes no seeding
# hook, so --seed never controlled game randomness (rl/matchrunner.py:728 says
# so: "the engine's own RNG drives game variance either way -- repeated runs are
# independent samples"). Consequence: our "seed 1 / seed 2" were never paired
# replicates, they were 2x200 independent games -- which makes pooling valid and
# the binomial MDE correct, PROVIDED the variance really is binomial.
#
# This runs the SAME spec 6 times at n=200 and compares the observed spread with
# sqrt(p(1-p)/n). If observed sd > binomial sd, every MDE we have quoted is
# understated and the campaign's gates are weaker than advertised.
set -euo pipefail
cd "$(dirname "$0")/.."
for r in 1 2 3 4 5 6; do
  uv run python -m rl.matchrunner play \
    --a model:checkpoints/m27_both.pt:clone54618168 --b rule:lucario \
    -n 200 --workers 8 --seed "$r" \
    --checkpoint "runs/m28_det_r${r}.jsonl"
done
echo "=== M28 DETERMINISM CHECK DONE $(date) ==="
