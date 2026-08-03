#!/usr/bin/env bash
# M40 Phase 3 — positive control for the NEW loss-family beds
# (dragapult / garchomp / rocket, cloned at min-score 900 on 2026-08-03).
#
# Live truth these instruments must approach (forensics, pooled 158 games):
#   dragapult 1-11 (0.08)   garchomp 2-6 (0.25)   rocket 1-7 (0.13)
# Pre-registered validation bar: a family's bed panel is live-faithful iff
# the live configs read <= 0.40 against it. Plain clones run FIRST (~20 s a
# cell — every read lands early); composites after (~8 min a cell).
#
# NOTE the standing caveat applies to the composites doubly: the first
# comp_grim cell read 0.718 vs Ship A — the solver wrap may add nothing
# AGAINST US even where it adds +125 ELO clone-vs-clone. The plain cells are
# the primary instrument candidates until the composite question settles.
#
# Usage: bash scripts/m40_lossfam_panels.sh   (resumable; SEEDS="1" default)
set -u
cd "$(dirname "$0")/.."

PREFIX="${PREFIX:-m40_lf}"
SEEDS="${SEEDS:-1}"
N=400
COMPLETE_LINES=16

declare -A BED=(
  [dragapult_d1]="model:checkpoints/m40_bed_dragapult_d1.pt:data/kaggle/dragapult_3631d393_deck.csv"
  [dragapult_d2]="model:checkpoints/m40_bed_dragapult_d2.pt:data/kaggle/dragapult_3631d393_deck.csv"
  [dragapult_d3]="model:checkpoints/m40_bed_dragapult_d3.pt:data/kaggle/dragapult_3631d393_deck.csv"
  [garchomp_d1]="model:checkpoints/m40_bed_garchomp_d1.pt:data/kaggle/garchomp_c7b3253f_deck.csv"
  [garchomp_d2]="model:checkpoints/m40_bed_garchomp_d2.pt:data/kaggle/garchomp_c7b3253f_deck.csv"
  [garchomp_d3]="model:checkpoints/m40_bed_garchomp_d3.pt:data/kaggle/garchomp_c7b3253f_deck.csv"
  [rocket_d1]="model:checkpoints/m40_bed_rocket_d1.pt:data/kaggle/rocket_59e27a5e_deck.csv"
  [rocket_d2]="model:checkpoints/m40_bed_rocket_d2.pt:data/kaggle/rocket_59e27a5e_deck.csv"
  [rocket_d3]="model:checkpoints/m40_bed_rocket_d3.pt:data/kaggle/rocket_59e27a5e_deck.csv"
  [comp_dragapult_d1]="solved:checkpoints/m40_bed_dragapult_d1.pt:data/kaggle/dragapult_3631d393_deck.csv:800:400"
  [comp_dragapult_d2]="solved:checkpoints/m40_bed_dragapult_d2.pt:data/kaggle/dragapult_3631d393_deck.csv:800:400"
  [comp_dragapult_d3]="solved:checkpoints/m40_bed_dragapult_d3.pt:data/kaggle/dragapult_3631d393_deck.csv:800:400"
  [comp_garchomp_d1]="solved:checkpoints/m40_bed_garchomp_d1.pt:data/kaggle/garchomp_c7b3253f_deck.csv:800:400"
  [comp_garchomp_d2]="solved:checkpoints/m40_bed_garchomp_d2.pt:data/kaggle/garchomp_c7b3253f_deck.csv:800:400"
  [comp_garchomp_d3]="solved:checkpoints/m40_bed_garchomp_d3.pt:data/kaggle/garchomp_c7b3253f_deck.csv:800:400"
  [comp_rocket_d1]="solved:checkpoints/m40_bed_rocket_d1.pt:data/kaggle/rocket_59e27a5e_deck.csv:800:400"
  [comp_rocket_d2]="solved:checkpoints/m40_bed_rocket_d2.pt:data/kaggle/rocket_59e27a5e_deck.csv:800:400"
  [comp_rocket_d3]="solved:checkpoints/m40_bed_rocket_d3.pt:data/kaggle/rocket_59e27a5e_deck.csv:800:400"
)
ORDER=(dragapult_d1 dragapult_d2 dragapult_d3
       garchomp_d1 garchomp_d2 garchomp_d3
       rocket_d1 rocket_d2 rocket_d3
       comp_dragapult_d1 comp_dragapult_d2 comp_dragapult_d3
       comp_garchomp_d1 comp_garchomp_d2 comp_garchomp_d3
       comp_rocket_d1 comp_rocket_d2 comp_rocket_d3)

declare -A ARM=(
  [shipA]="model-conserve:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
  [shipB]="model-c-pkg:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
)

for bed in "${ORDER[@]}"; do
  ck="${BED[$bed]#*:}"; ck="${ck%%:*}"
  [ -f "$ck" ] || { echo "[lf-panel] $bed checkpoint missing - skip"; continue; }
  for arm in shipA shipB; do
    for seed in $SEEDS; do
      f="runs/${PREFIX}_${arm}_${bed}_s${seed}.jsonl"
      n=0; [ -f "$f" ] && n=$(grep -c results "$f" || true)
      [ "$n" -ge "$COMPLETE_LINES" ] && continue
      echo "[lf-panel $PREFIX] $arm vs $bed seed $seed  $(date)"
      rm -f "$f"
      uv run python -m rl.matchrunner play --a "${ARM[$arm]}" --b "${BED[$bed]}" \
        -n "$N" --workers 8 --seed "$seed" --checkpoint "$f"
    done
  done
done
echo "=== M40 LOSS-FAMILY PANELS ($PREFIX) DONE $(date) ==="
