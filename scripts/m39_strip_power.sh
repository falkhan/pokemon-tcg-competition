#!/usr/bin/env bash
# M39 P0.7b — the decision-1 contrast at ADEQUATE power.
#
# Why this exists: the P0.7 ablation ran n=800/cell (2 seeds). Per
# docs/VALIDATION.md the n needed to detect a 5pp difference at 80% power is
# ~1560 PER ARM, so every "no signal" verdict it produced was preordained —
# the battery could not have found the effect it was looking for. Worse, two
# independent n=800 measurements of the SAME cell (plain vs top) came back
# 0.516 and 0.466: matchrunner at workers=8 is not run-reproducible (M37
# established this — engine RNG is per worker process), so each run is an
# independent sample and seeds do not pin the result.
#
# This adds 4 more seeds to the two arms decision 1 actually turns on,
# taking them to n=2400/cell pooled with the existing runs.
set -u
cd "$(dirname "$0")/.."
NET=checkpoints/m38_w9294_cont3.pt
declare -A ARM=([plain]="model:$NET:alakazam_v2_h4"
                [full]="modelt-gacfr3:$NET:alakazam_v2_h4")
declare -A BED=([wall]="model:checkpoints/m38_bc_wall.pt:greattusk_wall"
                [top]="model:checkpoints/m39_bc_top.pt:clone54618168")
for arm in plain full; do
  for bed in wall top; do
    for seed in 3 4 5 6; do
      f="runs/m39_abl_${arm}_${bed}_s${seed}.jsonl"
      [ -f "$f" ] && continue
      echo "[m39 power] $arm vs $bed seed $seed  $(date)"
      uv run python -m rl.matchrunner play --a "${ARM[$arm]}" --b "${BED[$bed]}" \
        -n 400 --workers 8 --seed "$seed" --checkpoint "$f"
    done
  done
done
echo "=== M39 STRIP POWER DONE $(date) ==="
