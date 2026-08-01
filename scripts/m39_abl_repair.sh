#!/usr/bin/env bash
# M39 P0.7 repair — re-run the ablation cells that died mid-battery.
#
# Cause, recorded so it is not repeated: the operator ran `git stash` on
# rl/matchrunner.py (to verify some unrelated test failures were
# pre-existing) WHILE this battery was running. The stash removed the
# abl-no-* spec kinds the battery depends on, so every cell launched inside
# that window failed with "cannot parse opponent spec". Lesson: never stash
# or edit a file a running background battery imports.
#
# Idempotent: skips any cell whose checkpoint already holds a full battery.
# Usage: bash scripts/m39_abl_repair.sh
set -u
cd "$(dirname "$0")/.."

NET=checkpoints/m38_w9294_cont3.pt
declare -A ARM=(
  [no_telepath]="abl-no-telepath:$NET:alakazam_v2_h4"
  [no_deckguard]="abl-no-deckguard:$NET:alakazam_v2_h4"
  [no_ash]="abl-no-ash:$NET:alakazam_v2_h4"
  [no_conserve]="abl-no-conserve:$NET:alakazam_v2_h4"
  [no_benchfloor]="abl-no-benchfloor:$NET:alakazam_v2_h4"
  [plain]="model:$NET:alakazam_v2_h4"
  [full]="modelt-gacfr3:$NET:alakazam_v2_h4"
  [no_racemode3]="modelt-gacf:$NET:alakazam_v2_h4"
)
declare -A BED=(
  [wall]="model:checkpoints/m38_bc_wall.pt:greattusk_wall"
  [top]="model:checkpoints/m39_bc_top.pt:clone54618168"
)
# A complete n=400 cell writes 16 `results` lines (the header line has none).
COMPLETE_LINES=16

for arm in "${!ARM[@]}"; do
  for bed in wall top; do
    for seed in 1 2; do
      f="runs/m39_abl_${arm}_${bed}_s${seed}.jsonl"
      n=0; [ -f "$f" ] && n=$(grep -c results "$f" || true)
      if [ "$n" -ge "$COMPLETE_LINES" ]; then continue; fi
      echo "[m39 abl repair] $arm vs $bed seed $seed (had $n lines)  $(date)"
      rm -f "$f"
      uv run python -m rl.matchrunner play --a "${ARM[$arm]}" \
        --b "${BED[$bed]}" -n 400 --workers 8 --seed "$seed" --checkpoint "$f"
    done
  done
done
echo "=== M39 ABLATION REPAIR DONE $(date) ==="
