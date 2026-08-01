#!/usr/bin/env bash
# M38 G3 gate battery — retrain arms vs the champion control on NON-SOLVER
# primary beds (docs/M38-plan.md Phase 1: a student judged mainly by its own
# teacher inherits teacher-student correlation; solver bed = diagnostic only).
#
# Candidates (all PLAIN, no rule stack — one variable per comparison; the
# rules question is G5's):
#   ft      model:checkpoints/m38_ft.pt        (fine-tune from m28_winners)
#   scratch model:checkpoints/m38_scratch.pt
#   control model:checkpoints/m28_winners.pt   (the champion net, plain)
# Beds: the E0a non-solver set + solver:lucario as the diagnostic leg.
# n=400 x 2 seeds = 800/cell (the pre-registered MDE size).
#
# Usage: bash scripts/m38_gate.sh [candidates...]   (default: all three)
set -u
cd "$(dirname "$0")/.."

declare -A CAND=(
  [ft]="model:checkpoints/m38_ft.pt:alakazam_v2_h4"
  [scratch]="model:checkpoints/m38_scratch.pt:alakazam_v2_h4"
  [control]="model:checkpoints/m28_winners.pt:alakazam_v2_h4"
)
declare -A BED=(
  [tuned]="rule:tuned:lucario"
  [iono]="rule:iono"
  [dragapult]="rule:dragapult"
  [grim]="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
  [rocket]="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
  [m28]="model:checkpoints/m28_winners.pt:clone54618168"
  [mirror]="model:checkpoints/m28_winners.pt:alakazam_v2_h4"
  [lucsolver]="solver:lucario"
)
BEDS_ORDER=(tuned iono dragapult grim rocket m28 mirror lucsolver)

for cand in "${@:-ft scratch control}"; do :; done
CANDS=("$@"); [ ${#CANDS[@]} -eq 0 ] && CANDS=(ft scratch control)

for cand in "${CANDS[@]}"; do
  for bed in "${BEDS_ORDER[@]}"; do
    for seed in 1 2; do
      echo "[m38 gate] $cand vs $bed seed $seed"
      uv run python -m rl.matchrunner play --a "${CAND[$cand]}" \
        --b "${BED[$bed]}" -n 400 --workers 8 --seed "$seed" \
        --checkpoint "runs/m38_gate_${cand}_${bed}_s${seed}.jsonl"
    done
  done
done
echo "DONE"
