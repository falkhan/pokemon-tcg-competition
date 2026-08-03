#!/usr/bin/env bash
# M40 Phase 5 — S2 arm 2: two trains off ONE corpus, then the gate.
#
# Pre-registered in docs/M40.md (2026-08-03, before collection):
#   2a = outcome-alpha 0.25 (arm-1 recipe on the WIDE pool -> isolates pool)
#   2b = advantage-weighted off the E0-validated value head, same corpus
#        (-> isolates the weighting variable; --advantage-ckpt, plan_iter)
# Both: init cont3 (retain_b retired after the live regression), retention =
# champion shards, SUPPORTER/STADIUM x5 (the cont3 recipe), 3 epochs lr 1e-4.
# Critic for 2b = m39_retain_b's value head: the head E0 measured (0.642
# matched-pair, 68% within-game). A discredited POLICY can still carry the
# campaign's only validated CRITIC — gradients never touch it.
#
# Gate runs all three arms under the resumable m40_pan prefix. The control
# keeps the S6 battery's arm name `cont3_pkgz` ON PURPOSE: same name ->
# same run files -> its 96 old-roster cells are re-used and only the new
# loss-family cells are paid (matchrunner is unchanged since that battery,
# so the vintages mix legitimately). Decode afterwards:
#   uv run python scripts/m40_decide.py --arm s2b_a --control cont3_pkgz
#   uv run python scripts/m40_decide.py --arm s2b_b --control cont3_pkgz
set -eu
cd "$(dirname "$0")/.."

CORPUS="data/m40_s2_b"
RETAIN="data/bc_m38_w9294"
INIT="checkpoints/m38_w9294_cont3.pt"
COMMON=(--data "$CORPUS" "$RETAIN" --init "$INIT" --epochs 3 --lr 1e-4
        --card-kind-weight SUPPORTER:5 --card-kind-weight STADIUM:5)

# Completeness check = re-run the collector with the FULL pre-registered
# args; a complete dir answers "nothing to do" and never plays a game.
# (--status alone computes its denominator from the CURRENT --beds flag, so
# it under-reports a pool wider than the default — learned the hard way.)
uv run python scripts/m40_s2_collect.py --out "$CORPUS" --games 300 --tau 0.6 \
    --beds mirror m28 wall_d1 wall_d2 grim_d1 grim_d2 arch_d1 arch_d2 topgrim \
           dragapult_d1 dragapult_d2 garchomp_d1 garchomp_d2 rocket_d1 rocket_d2 \
  | grep -q "nothing to do" \
  || { echo "collection incomplete — refusing to train on a partial corpus"; exit 2; }

if [ ! -f checkpoints/m40_s2b_a.pt ]; then
  echo "=== train 2a (outcome-alpha 0.25) $(date)"
  uv run python -m rl.plan_iter train "${COMMON[@]}" --name m40_s2b_a \
    --outcome-weight 0.25 2>&1 | tail -6
fi
if [ ! -f checkpoints/m40_s2b_b.pt ]; then
  echo "=== train 2b (advantage-weighted) $(date)"
  uv run python -m rl.plan_iter train "${COMMON[@]}" --name m40_s2b_b \
    --advantage-ckpt checkpoints/m39_retain_b.pt --advantage-beta 1.0 2>&1 | tail -6
fi

echo "=== gate (m40_pan, resumable) $(date)"
bash scripts/m40_panel_gate.sh \
  cont3_pkgz "model-c-pkgz:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4" \
  s2b_a      "model-c-pkgz:checkpoints/m40_s2b_a.pt:alakazam_v2_h4" \
  s2b_b      "model-c-pkgz:checkpoints/m40_s2b_b.pt:alakazam_v2_h4"
echo "=== S2 ARM 2 PIPELINE DONE $(date)"
