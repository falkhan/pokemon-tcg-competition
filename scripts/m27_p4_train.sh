#!/usr/bin/env bash
# M27 Phase D: the supporter/stadium arms, as a 2x2.
#
# Two interventions, measured separately and together, all warm-started from
# the SHIPPED checkpoint so the only variables are the corpus and the weighting:
#
#   features  data/bc_m27_alakazam (OPTION_M27_DIM: supporter-legal /
#             stadium-legal / gust-wanted) via --init-v3m, whose new option
#             columns are zero-init -> init is EXACTLY the shipped net
#   weighting --card-kind-weight SUPPORTER:5 STADIUM:5 (6.5% + 1.4% of rows;
#             --class-weight PLAY:k cannot separate these from items, which
#             the clone already plays at teacher rate -- docs/M27.md Probe 3)
#
# The no-op control is mandatory: M26 showed a control retrain of an IDENTICAL
# recipe swings per-class numbers by +-5-10pp, so any per-class claim needs it.
# Recipe = M25/M26 (epochs 10, lr 1e-4). Record val_acc in docs/M27.md
# immediately -- checkpoints store no metadata.
set -euo pipefail
cd "$(dirname "$0")/.."

SHIPPED=checkpoints/m25_bc_alakazam_v3h.pt
OLD=data/bc_m25_alakazam_v3h        # 94-wide options
NEW=data/bc_m27_alakazam            # 97-wide options (M27 block appended)
KIND=(--card-kind-weight SUPPORTER:5 --card-kind-weight STADIUM:5)

run() {  # run <name> <init-flag> <init> <data> [extra...]
  local name=$1 flag=$2 init=$3 data=$4; shift 4
  echo "=== train ${name} $(date) ==="
  uv run python -m rl.plan_iter train \
    --data "${data}" --name "${name}" "${flag}" "${init}" \
    --epochs 10 --lr 1e-4 "$@" 2>&1 | tee "runs/${name}.log"
}

# --- control arm: same recipe, old corpus, no weighting (run-noise calibration)
run m27_control --init "${SHIPPED}" "${OLD}"
# --- weighting only (old corpus, so the M27 features are absent)
run m27_kind    --init "${SHIPPED}" "${OLD}" "${KIND[@]}"
# --- features only (new corpus, no weighting)
run m27_feat    --init-v3m "${SHIPPED}" "${NEW}"
# --- both
run m27_both    --init-v3m "${SHIPPED}" "${NEW}" "${KIND[@]}"

echo "=== M27 P4 TRAIN DONE $(date) ==="
