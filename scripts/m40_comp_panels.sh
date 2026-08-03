#!/usr/bin/env bash
# M40 Phase 3 — COMPOSITE panels: the positive control and the ceiling.
#
# Two questions, both pre-registered in the M40 phase-2 plan (2026-08-03):
#
#   1. POSITIVE CONTROL (comp_grim_d1..d3): live truth vs grim is 2W-13L
#      (0.13); the plain grim clone panel reads 0.65+. A live-faithful grim
#      instrument must hold our live configs at <= 0.40. If even the composite
#      reads >= 0.5, grim's live losses are NOT reproducible offline and that
#      family must not enter the gate wearing a "live-faithful" label.
#   2. CEILING (comp_topgrim_d1..d3): the "+125 ELO from wrapping a clone in
#      the solver" number is ONE draw, one cell (n=400). G-13 says draw
#      variance is 2.8x sampling variance — this panel is what lets composites
#      anchor anything.
#
# Both arms are the LIVE configs the forensics measured (Ship A = model-c,
# Ship B = model-c-pkg on cont3), so cells read directly against the live
# family records. Solver budget 800:400 — the budget the +125 was measured at.
#
# ~10x clone-cell wall-clock per cell (solver latency); the whole run is an
# overnight job. Resumable exactly like m40_panel_gate.sh: a cell with >=16
# result lines is skipped, so re-running the script continues where it died.
#
# Usage:  bash scripts/m40_comp_panels.sh          # both arms, seeds 1 2 3
#         SEEDS="1" bash scripts/m40_comp_panels.sh
set -u
cd "$(dirname "$0")/.."

PREFIX="${PREFIX:-m40_comp}"
SEEDS="${SEEDS:-1 2 3}"
N=400
COMPLETE_LINES=16

declare -A BED=(
  [comp_grim_d1]="solved:checkpoints/m39_bc_grim.pt:grim_live:800:400"
  [comp_grim_d2]="solved:checkpoints/m39_bc_grim_b.pt:grim_live:800:400"
  [comp_grim_d3]="solved:checkpoints/m39_bed_grim_d3.pt:grim_live:800:400"
  [comp_topgrim_d1]="solved:checkpoints/m39_bc_topgrim.pt:grim_live:800:400"
  [comp_topgrim_d2]="solved:checkpoints/m40_bed_topgrim_d2.pt:grim_live:800:400"
  [comp_topgrim_d3]="solved:checkpoints/m40_bed_topgrim_d3.pt:grim_live:800:400"
)
# Positive control first: if the run dies overnight, the decision-relevant
# half (is the instrument live-faithful at all?) is the half that completed.
ORDER=(comp_grim_d1 comp_grim_d2 comp_grim_d3
       comp_topgrim_d1 comp_topgrim_d2 comp_topgrim_d3)

declare -A ARM=(
  [shipA]="model-conserve:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
  [shipB]="model-c-pkg:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
)

for bed in "${ORDER[@]}"; do
  for arm in shipA shipB; do
    for seed in $SEEDS; do
      f="runs/${PREFIX}_${arm}_${bed}_s${seed}.jsonl"
      n=0; [ -f "$f" ] && n=$(grep -c results "$f" || true)
      [ "$n" -ge "$COMPLETE_LINES" ] && continue
      echo "[comp-panel $PREFIX] $arm vs $bed seed $seed  $(date)"
      rm -f "$f"
      uv run python -m rl.matchrunner play --a "${ARM[$arm]}" --b "${BED[$bed]}" \
        -n "$N" --workers 8 --seed "$seed" --checkpoint "$f"
    done
  done
done
echo "=== M40 COMPOSITE PANELS ($PREFIX) DONE $(date) ==="
