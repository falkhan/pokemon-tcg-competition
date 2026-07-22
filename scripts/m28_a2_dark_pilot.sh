#!/usr/bin/env bash
# M28 Track A2: pick the canonical Darkness sparring pilot BY MEASUREMENT.
#
# The M26 grim gate is a strawman: we read 0.6875 against it offline while going
# 2W-9L vs the same family live, and the clone behind it scores 0.148 vs
# rule:dragapult (M27). So instead of assuming a pilot, put every pilot we can
# on decks/grimmsnarl_dark.csv and measure them against ONE fixed, non-Darkness
# yardstick (rule:lucario). Strongest wins and becomes the pin.
# Decode with the 0=WIN rule. Workers 8 (hard cap).
set -euo pipefail
cd "$(dirname "$0")/.."
DECK=grimmsnarl_dark
REF=rule:lucario

for spec_tag in \
  "clone:model:checkpoints/m25_bc_grim_54861775.pt:${DECK}" \
  "clone2:model:checkpoints/m25_bc_grim_54863653.pt:${DECK}" \
  "solver:solver:${DECK}" \
  "generic:generic:${DECK}" \
  "rulelu:rule:lucario:${DECK}" \
  "ruleio:rule:iono:${DECK}" \
  ; do
  tag=${spec_tag%%:*}; spec=${spec_tag#*:}
  for s in 1 2; do
    echo "=== ${tag} (${spec}) vs ${REF} seed ${s} $(date) ==="
    uv run python -m rl.matchrunner play --a "${spec}" --b "${REF}" \
      -n 200 --workers 8 --seed "${s}" \
      --checkpoint "runs/m28_dark_${tag}_s${s}.jsonl"
  done
done
echo "=== M28 A2 DONE $(date) ==="
