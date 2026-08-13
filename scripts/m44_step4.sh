#!/usr/bin/env bash
# M44 Step 4 — pair L seed pick + seed floors (docs/M44-plan.md r4).
# All runs are resumable (run_pairs jsonl checkpoints under runs/).
# Decode law: results 0=a-win 1=a-loss 2=draw; wr=(w+0.5d)/n (measure-agent).
set -euo pipefail
cd "$(dirname "$0")/.."

# M43 box lesson: un-pinned BLAS = 15x slower matchrunner. Always pin.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

N=400
W=8

pick () {  # ckpt-stem
  uv run python -m rl.matchrunner play \
    --a "model-pz:checkpoints/$1.pt:lucario" --b rule:tuned:lucario \
    -n $N --workers $W --seed 1 --checkpoint "runs/m44_L_pick_$1.jsonl"
}

floor () {  # name spec anchor-spec anchor-name
  uv run python -m rl.matchrunner play \
    --a "$2" --b "$3" \
    -n $N --workers $W --seed 1 --checkpoint "runs/m44_floor_$1_vs_$4.jsonl"
}

case "${1:-all}" in
  pick)
    pick ppo_best_m43a_base
    pick m41b_wide_prod
    pick m28_winners
    ;;
  floors)  # $2 = L's chosen checkpoint stem (its vs-tuned leg = its pick run)
    L="${2:?floors needs the picked L checkpoint stem}"
    floor K model-c-pkgz:checkpoints/ppo_best_m43a_base.pt:alakazam_v2_h4 rule:tuned:lucario tuned
    floor K model-c-pkgz:checkpoints/ppo_best_m43a_base.pt:alakazam_v2_h4 rule:iono iono
    floor L "model-pz:checkpoints/${L}.pt:lucario" rule:iono iono
    floor O model-pz:checkpoints/m41_ogerpon.pt:ogerpon rule:tuned:lucario tuned
    floor O model-pz:checkpoints/m41_ogerpon.pt:ogerpon rule:iono iono
    floor G model-pz:checkpoints/m39_bc_grim.pt:grim_live rule:tuned:lucario tuned
    floor G model-pz:checkpoints/m39_bc_grim.pt:grim_live rule:iono iono
    ;;
  *) echo "usage: $0 pick|floors <L-stem>"; exit 2 ;;
esac
echo "M44-STEP4-$1-DONE"
