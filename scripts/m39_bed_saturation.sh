#!/usr/bin/env bash
# M39 — is bed saturation real, or is it an artefact of OUR matchup?
#
# P0.9 found that cloning grim from 1000+ pilots produces a bed no harder
# for us than cloning from 750 pilots (+0.5 / -0.9pp, inside noise at
# n=2400). Two very different explanations fit that:
#
#   (a) SATURATION: BC cloning cannot capture the skill difference, so the
#       two beds really are the same strength. Our whole offline apparatus
#       then measures against opponents weaker than live.
#   (b) MATCHUP: the beds DO differ in strength, but the grim-vs-us matchup
#       is decided by something the skill gap does not touch, so both lose
#       to us at the same rate.
#
# The discriminating test is head-to-head: play the two clones against EACH
# OTHER, same deck both sides. Under (a) it is ~0.50. Under (b) the 1000+
# clone wins clearly. Nothing else in the campaign separates these.
set -u
cd "$(dirname "$0")/.."
A="model:checkpoints/m39_bc_topgrim.pt:grim_live"
B="model:checkpoints/m39_bc_grim.pt:grim_live"
for seed in 1 2 3 4 5 6; do
  f="runs/m39_bedsat_topgrim_vs_grim_s${seed}.jsonl"
  [ -f "$f" ] && continue
  echo "[bedsat] topgrim vs grim seed $seed  $(date)"
  uv run python -m rl.matchrunner play --a "$A" --b "$B" -n 400 \
    --workers 8 --seed "$seed" --checkpoint "$f"
done
echo "=== M39 BED SATURATION TEST DONE $(date) ==="
