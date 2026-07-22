#!/usr/bin/env bash
# M28 Tracks F+G as a 2x2: demonstration quality x phase conditioning.
#
#   F  winners-only corpus (203 games) vs the full one (370, 45% of them the
#      teacher's LOSSES cloned at uniform weight)
#   G  phase conditioning = the M28 option block (in the corpus, both arms) PLUS
#      --phase-weight on late-game rows. M27's lesson: features alone do
#      nothing; features x weighting is what moved the target.
#
# All arms warm-start from m27_both via --init-v3m (97 -> 100, new option
# columns zero-init, so init is EXACTLY m27_both) and keep its card-kind
# weighting, so the only variables are the corpus and the phase weight.
set -euo pipefail
cd "$(dirname "$0")/.."
KIND=(--card-kind-weight SUPPORTER:5 --card-kind-weight STADIUM:5)
run() {  # run <name> <data> [extra...]
  local name=$1 data=$2; shift 2
  echo "=== train ${name} $(date) ==="
  uv run python -m rl.plan_iter train --data "${data}" --name "${name}" \
    --init-v3m checkpoints/m27_both.pt --epochs 10 --lr 1e-4 \
    "${KIND[@]}" "$@" 2>&1 | tee "runs/${name}.log"
}
run m28_control data/bc_m28_phase
run m28_phase   data/bc_m28_phase   --phase-weight 4
run m28_winners data/bc_m28_winners
run m28_wphase  data/bc_m28_winners --phase-weight 4
echo "=== M28 FG TRAIN DONE $(date) ==="
