#!/usr/bin/env bash
# M27 Phase F: strength battery for the Phase D candidate.
#
# Exactly the M26 battery (same opponents, same specs, same n) so the M26 pins
# are directly comparable — the candidate is measured with the O1 telepath
# override ON (`modelt:`), because that is what is live as sub 54903635 and the
# supporter weighting cost ATTACH fidelity, which O1 partly covers.
#
# M26 pins to beat (all on `modelt:m25_bc_alakazam_v3h:clone54618168`):
#   rule:lucario     0.671  pooled n=800  (4 seeds)
#   rule:dragapult   0.3375 pooled n=400  <- out-of-loop, the one that matters
#   grim mill        0.6875 pooled n=400  (advisory only)
#   random:kyogre    0.975  n=200         (floor >=0.95)
#
# POWER (from the measure-agent skill): pooled n=800 resolves 7.0pp, n=400 ~10pp.
# Anything smaller is NOT a result. 4 seeds on lucario to match the pin's power.
# Decode results with the 0=WIN rule: 0 = side-a win, 1 = loss, 2 = draw.
set -euo pipefail
cd "$(dirname "$0")/.."

CAND=${1:-m27_both}
A="modelt:checkpoints/${CAND}.pt:clone54618168"
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"

play() {  # play <tag> <opponent-spec> <seed>
  local tag=$1 opp=$2 seed=$3
  echo "=== ${CAND} vs ${tag} seed ${seed} $(date) ==="
  uv run python -m rl.matchrunner play --a "${A}" --b "${opp}" \
    -n 200 --workers 8 --seed "${seed}" \
    --checkpoint "runs/m27_${CAND#m27_}_${tag}_s${seed}.jsonl"
}

for s in 1 2 3 4; do play luc  "rule:lucario"   "$s"; done
for s in 1 2;       do play drag "rule:dragapult" "$s"; done
for s in 1 2;       do play grim "${GRIM}"        "$s"; done
play kyo "random:kyogre" 1

echo "=== M27 P6 BATTERY DONE $(date) ==="
