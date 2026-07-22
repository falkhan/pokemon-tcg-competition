#!/usr/bin/env bash
# M28 Track B2: the M8.4 sims ladder, re-run on the v3h value head.
#
# M8.4 killed MCTS on a FLAT ladder (0.515/0.490/0.495) -- on the v1 value head,
# a caveat docs/M22.md states explicitly. This re-runs it on m27_both's v3 head
# via rl/mcts.V3Evaluator.
#
# Rung 0 is the SAME net with no search (plain `model:`, no O1 override, since
# MCTS does not apply the override either) -- without it the ladder measures
# nothing. Pre-registered kill bar: if 0->64 sims moves less than the ~10pp MDE
# at n=400, the M8.4 kill reproduces on the new head and MCTS closes for good.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=checkpoints/m27_both.pt
for rung in "0:model:${CKPT}:clone54618168" \
            "16:mcts:${CKPT}:clone54618168:16" \
            "32:mcts:${CKPT}:clone54618168:32" \
            "64:mcts:${CKPT}:clone54618168:64"; do
  sims=${rung%%:*}; spec=${rung#*:}
  for s in 1 2; do
    echo "=== sims=${sims} seed ${s} $(date) ==="
    uv run python -m rl.matchrunner play --a "${spec}" --b rule:lucario \
      -n 200 --workers 8 --seed "${s}" \
      --checkpoint "runs/m28_mcts_${sims}_s${s}.jsonl"
  done
done
echo "=== M28 B2 LADDER DONE $(date) ==="
