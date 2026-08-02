#!/usr/bin/env bash
# M39 — the G-13 PANEL gate battery. Supersedes scripts/m39_gate.sh.
#
# Difference from m39_gate.sh, and the only one: every BC-cloned family is a
# PANEL of three independently trained draws instead of a single clone.
# D1 (docs/M39.md) measured bed strength as a draw from a training lottery —
# .671 / .623 / .546 for three clones of the same grim list at n=2400 each,
# a 12.5pp spread against a 2.0pp binomial CI. A single-clone cell reports
# one lottery ticket as if it were the family.
#
# The budget is REDISTRIBUTED, not increased: 3 seeds (n=1200) against each
# of three draws instead of 6 seeds (n=2400) against one, so a panel family
# ends at n=3600 for slightly more machine time and a number that averages
# over bed construction. Families with no re-drawable corpus (rule agents,
# our own champion lineage, the M30 rocket relic) stay single and are
# labelled as such by m39_decide.py.
#
# Cells are keyed <arm>_<bed>_s<seed>, and the panel draws are their own bed
# names (wall_d1..d3), so m39_decide.py pools them inside the family and the
# weighted number stays exactly the G-3 quantity it always was.
#
# Usage:
#   bash scripts/m39_panel_gate.sh <arm-name> <arm-spec> [<name> <spec>...]
#   PREFIX=m39_p2 bash scripts/m39_panel_gate.sh conserve "model-conserve:..."
# Env:
#   PREFIX      run-file prefix           (default m39_pan)
#   PANEL_SEEDS seeds per panel draw      (default "1 2 3"  -> n=1200/draw)
#   SOLO_SEEDS  seeds per single-draw bed (default "1 2 3 4 5 6" -> n=2400)
#   SKIP_TOP=1  skip the 900+ ceiling panel (it does not gate anything)
set -u
cd "$(dirname "$0")/.."

PREFIX="${PREFIX:-m39_pan}"
PANEL_SEEDS="${PANEL_SEEDS:-1 2 3}"
SOLO_SEEDS="${SOLO_SEEDS:-1 2 3 4 5 6}"
SKIP_TOP="${SKIP_TOP:-0}"
N=400
COMPLETE_LINES=16   # matchrunner writes 16 result lines per completed n=400 cell

declare -A BED=(
  [tuned]="rule:tuned:lucario"
  [mirror]="model:checkpoints/m28_winners.pt:alakazam_v2_h4"
  [m28]="model:checkpoints/m28_winners.pt:clone54618168"
  [dragapult]="rule:dragapult"
  [rocket]="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
  [iono]="rule:iono"

  [wall_d1]="model:checkpoints/m38_bc_wall.pt:greattusk_wall"
  [wall_d2]="model:checkpoints/m39_bed_wall_d2.pt:greattusk_wall"
  [wall_d3]="model:checkpoints/m39_bed_wall_d3.pt:greattusk_wall"
  [arch_d1]="model:checkpoints/m39_bc_archaludon.pt:archaludon"
  [arch_d2]="model:checkpoints/m39_bed_arch_d2.pt:archaludon"
  [arch_d3]="model:checkpoints/m39_bed_arch_d3.pt:archaludon"
  [grim_d1]="model:checkpoints/m39_bc_grim.pt:grim_live"
  [grim_d2]="model:checkpoints/m39_bc_grim_b.pt:grim_live"
  [grim_d3]="model:checkpoints/m39_bed_grim_d3.pt:grim_live"

  [top_d1]="model:checkpoints/m39_bc_top.pt:clone54618168"
  [top_d2]="model:checkpoints/m39_bed_top_d2.pt:clone54618168"
  [top_d3]="model:checkpoints/m39_bed_top_d3.pt:clone54618168"
)
# Order = descending live share, so a battery killed part-way still covers
# the mass that decides the verdict. Panel draws are interleaved d1/d2/d3 so
# a partial family is still a partial PANEL rather than one draw.
SOLO_ORDER=(tuned mirror m28 dragapult rocket iono)
PANEL_ORDER=(wall_d1 wall_d2 wall_d3 arch_d1 arch_d2 arch_d3
             grim_d1 grim_d2 grim_d3)
TOP_ORDER=(top_d1 top_d2 top_d3)

if [ "$#" -lt 2 ] || [ $(($# % 2)) -ne 0 ]; then
  echo "usage: bash scripts/m39_panel_gate.sh <arm-name> <arm-spec> [...]" >&2
  exit 2
fi

run_cell () {   # arm  spec  bed  seeds...
  local arm="$1" spec="$2" bed="$3"; shift 3
  for seed in "$@"; do
    local f="runs/${PREFIX}_${arm}_${bed}_s${seed}.jsonl"
    local n=0; [ -f "$f" ] && n=$(grep -c results "$f" || true)
    [ "$n" -ge "$COMPLETE_LINES" ] && continue
    echo "[panel-gate $PREFIX] $arm vs $bed seed $seed  $(date)"
    rm -f "$f"
    uv run python -m rl.matchrunner play --a "$spec" --b "${BED[$bed]}" \
      -n "$N" --workers 8 --seed "$seed" --checkpoint "$f"
  done
}

while [ "$#" -gt 0 ]; do
  arm="$1"; spec="$2"; shift 2
  for bed in "${SOLO_ORDER[@]}";  do run_cell "$arm" "$spec" "$bed" $SOLO_SEEDS;  done
  for bed in "${PANEL_ORDER[@]}"; do run_cell "$arm" "$spec" "$bed" $PANEL_SEEDS; done
  [ "$SKIP_TOP" = "1" ] && continue
  for bed in "${TOP_ORDER[@]}";   do run_cell "$arm" "$spec" "$bed" $PANEL_SEEDS; done
done
echo "=== M39 PANEL GATE ($PREFIX) DONE $(date) ==="
