#!/usr/bin/env bash
# M39 gate battery — the roster the WEIGHTED pool (G-3) is computed over.
# Decode with: uv run python scripts/m39_decide.py --arm <arm> --control <ctl>
#
# Roster changes vs m38_gate.sh, both forced by measurement (docs/M39.md):
#   + archaludon  NEW bed (m39_bc_archaludon, 137 seats, val 0.744). Measured
#                 at 12.2% of live games with NO bed — the largest uncovered
#                 family. Without it the roster covers 77.5% of live-mix mass
#                 and G-3 makes the whole gate INVALID.
#   ~ grim        m25_bc_grim_54861775 (M26-era, 600-band, val 0.565) RETIRED
#                 per G-6 -> m39_bc_grim (195 seats at >=700, val 0.662). The
#                 relic is the bed that lied in M38 (offline +4pp, live 1-6).
#   - stall       PLANNED AND KILLED: 9 seats at band, 22 across all bands.
#                 Unbuildable, 2.7% live share, zero 800+ winner seats
#                 against it in the census. Not deferred — killed.
#   + top         P0.8's 900+ instrument. Reported SEPARATELY, never pooled:
#                 it is not part of the live mix, it is the slot-3 signal.
#
# n=400 x 2 seeds = 800/cell.
# Usage: bash scripts/m39_gate.sh <arm-name> <arm-spec> [more pairs...]
#   e.g. bash scripts/m39_gate.sh cont3 "model:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
set -u
cd "$(dirname "$0")/.."

declare -A BED=(
  [tuned]="rule:tuned:lucario"
  [mirror]="model:checkpoints/m28_winners.pt:alakazam_v2_h4"
  [m28]="model:checkpoints/m28_winners.pt:clone54618168"
  [wall]="model:checkpoints/m38_bc_wall.pt:greattusk_wall"
  [archaludon]="model:checkpoints/m39_bc_archaludon.pt:archaludon"
  [grim]="model:checkpoints/m39_bc_grim.pt:grim_live"
  [dragapult]="rule:dragapult"
  [rocket]="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
  [iono]="rule:iono"
  [top]="model:checkpoints/m39_bc_top.pt:clone54618168"
)
# Order = descending live share, so a battery killed early still covers mass.
BEDS_ORDER=(tuned mirror m28 wall archaludon grim dragapult rocket iono top)

if [ "$#" -lt 2 ] || [ $(($# % 2)) -ne 0 ]; then
  echo "usage: bash scripts/m39_gate.sh <arm-name> <arm-spec> [<name> <spec>...]" >&2
  exit 2
fi

while [ "$#" -gt 0 ]; do
  arm="$1"; spec="$2"; shift 2
  for bed in "${BEDS_ORDER[@]}"; do
    for seed in 1 2; do
      echo "[m39 gate] $arm vs $bed seed $seed  $(date)"
      uv run python -m rl.matchrunner play --a "$spec" --b "${BED[$bed]}" \
        -n 400 --workers 8 --seed "$seed" \
        --checkpoint "runs/m39_gate_${arm}_${bed}_s${seed}.jsonl"
    done
  done
done
echo "=== M39 GATE DONE $(date) ==="
