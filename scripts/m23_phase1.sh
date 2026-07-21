#!/usr/bin/env bash
# M23 Phase 1: continue the M22c strong-teacher recipe to convergence.
#   usage: scripts/m23_phase1.sh [TAG] [ITERS] [GAMES_PER_ITER] [EVAL_GAMES]
#   smoke: scripts/m23_phase1.sh m23smoke 1 40 40
# Deltas vs the M22c leg (docs/M23-plan.md): KL omitted (pure PPO), gust-boost
# dropped, solver slices out of the pool, promotion gates on vs_teacher
# (rl/ppo.py M23 edit), per-iter eval for the convergence curve.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-m23p1}"
ITERS="${2:-20}"
GPI="${3:-400}"
EVAL="${4:-200}"
LOG="runs/${TAG}_pipeline.log"
{
  echo "=== M23 PHASE1 TRAIN $(date) tag=${TAG} iters=${ITERS} gpi=${GPI} eval=${EVAL} ==="
  uv run python -m rl.ppo \
    --start ppo_current_m22cRL.pt \
    --tag "${TAG}" \
    --iterations "${ITERS}" --games-per-iter "${GPI}" --workers 8 \
    --eval-every 1 --eval-games "${EVAL}" \
    --learn-deck lucario --eval-deck lucario \
    --plan-coef 1.0 --plan-tau 1.5 \
    --opponents rule:lucario=0.45 rule:iono=0.15 rule:tuned:lucario=0.05 \
                mirror=0.20 past=0.10 random:kyogre=0.05
  echo "=== TRAIN DONE $(date) ==="
} 2>&1 | tee "${LOG}"
