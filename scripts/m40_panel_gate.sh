#!/usr/bin/env bash
# M40 — the G-13 PANEL gate battery. Fork of scripts/m39_panel_gate.sh.
#
# Forked rather than extended (Piotr, 2026-08-02): m39_panel_gate.sh produced
# the verdict that shipped M39's Ship A and Ship B, and it stays frozen as the
# record of that verdict. This file is free to grow new beds.
#
# Identical to the M39 script in every statistical respect — same BED map for
# the shared beds, PANEL_SEEDS="1 2 3" (n=1200/draw), SOLO_SEEDS 6 (n=2400),
# N=400, COMPLETE_LINES=16, --workers 8 (the CLAUDE.md parallelism cap; 12 has
# deadlocked repeatedly). Two additions:
#
#   1. PREFIX defaults to m40_pan. M40 runs FRESH controls rather than
#      resuming M39 cells: `rl/matchrunner.py` changed this milestone (the
#      `planzero` serve fix), and `_engine_game` calls battle_start with no
#      seed, so cell-level reproducibility across runs is not established and
#      a mixed-vintage control could not be defended. At ~20 s per n=400 cell
#      a full 72-cell arm costs ~24 min — the honest option is the cheap one.
#
#   2. TOPGRIM_ORDER — the S3 NON-MIRROR high-band ceiling panel (grim seats
#      at >=1000). This is the H-B test: if we are at parity with top MIRROR
#      clones but well under against top GRIM clones, matchup structure
#      explains the live record without any compression. d1 exists
#      (checkpoints/m39_bc_topgrim.pt, val 0.674); d2/d3 are S3's training
#      draws. The block SKIPS ITSELF until those checkpoints exist, so this
#      script is correct both before and after S3 lands.
#
# Usage:
#   bash scripts/m40_panel_gate.sh <arm-name> <arm-spec> [<name> <spec>...]
#   PREFIX=m40_s6 bash scripts/m40_panel_gate.sh pkgz "model-c-pkgz:..."
# Env:
#   PREFIX        run-file prefix             (default m40_pan)
#   PANEL_SEEDS   seeds per panel draw        (default "1 2 3"  -> n=1200/draw)
#   SOLO_SEEDS    seeds per single-draw bed   (default "1 2 3 4 5 6" -> n=2400)
#   SKIP_TOP=1    skip the 900+ mirror ceiling panel
#   SKIP_TOPGRIM=1  skip the non-mirror ceiling panel (auto-skipped if unbuilt)
set -u
cd "$(dirname "$0")/.."

PREFIX="${PREFIX:-m40_pan}"
PANEL_SEEDS="${PANEL_SEEDS:-1 2 3}"
SOLO_SEEDS="${SOLO_SEEDS:-1 2 3 4 5 6}"
SKIP_TOP="${SKIP_TOP:-0}"
SKIP_TOPGRIM="${SKIP_TOPGRIM:-0}"
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

  [topgrim_d1]="model:checkpoints/m39_bc_topgrim.pt:grim_live"
  [topgrim_d2]="model:checkpoints/m40_bed_topgrim_d2.pt:grim_live"
  [topgrim_d3]="model:checkpoints/m40_bed_topgrim_d3.pt:grim_live"

  # M40 phase 4 (2026-08-03) — loss-family panels replacing the weak proxies
  # (rule:dragapult sample agent, M30-era rocket clone). Cloned from 900-1150
  # seats; positive control FAILED (0.56-0.90 vs live 0.08-0.25) so these
  # cells measure RELATIVE deltas only, never live-faithful absolutes.
  [dragapult_d1]="model:checkpoints/m40_bed_dragapult_d1.pt:data/kaggle/dragapult_3631d393_deck.csv"
  [dragapult_d2]="model:checkpoints/m40_bed_dragapult_d2.pt:data/kaggle/dragapult_3631d393_deck.csv"
  [dragapult_d3]="model:checkpoints/m40_bed_dragapult_d3.pt:data/kaggle/dragapult_3631d393_deck.csv"
  [garchomp_d1]="model:checkpoints/m40_bed_garchomp_d1.pt:data/kaggle/garchomp_c7b3253f_deck.csv"
  [garchomp_d2]="model:checkpoints/m40_bed_garchomp_d2.pt:data/kaggle/garchomp_c7b3253f_deck.csv"
  [garchomp_d3]="model:checkpoints/m40_bed_garchomp_d3.pt:data/kaggle/garchomp_c7b3253f_deck.csv"
  [rocket_d1]="model:checkpoints/m40_bed_rocket_d1.pt:data/kaggle/rocket_59e27a5e_deck.csv"
  [rocket_d2]="model:checkpoints/m40_bed_rocket_d2.pt:data/kaggle/rocket_59e27a5e_deck.csv"
  [rocket_d3]="model:checkpoints/m40_bed_rocket_d3.pt:data/kaggle/rocket_59e27a5e_deck.csv"
)
# Order = descending live share, so a battery killed part-way still covers
# the mass that decides the verdict. Panel draws are interleaved d1/d2/d3 so
# a partial family is still a partial PANEL rather than one draw.
# dragapult (rule agent) and rocket (M30 clone) retired from SOLO_ORDER
# 2026-08-03 — replaced by the trained panels below.
SOLO_ORDER=(tuned mirror m28 iono)
PANEL_ORDER=(wall_d1 wall_d2 wall_d3 arch_d1 arch_d2 arch_d3
             grim_d1 grim_d2 grim_d3
             dragapult_d1 dragapult_d2 dragapult_d3
             garchomp_d1 garchomp_d2 garchomp_d3
             rocket_d1 rocket_d2 rocket_d3)
TOP_ORDER=(top_d1 top_d2 top_d3)
TOPGRIM_ORDER=(topgrim_d1 topgrim_d2 topgrim_d3)

if [ "$#" -lt 2 ] || [ $(($# % 2)) -ne 0 ]; then
  echo "usage: bash scripts/m40_panel_gate.sh <arm-name> <arm-spec> [...]" >&2
  exit 2
fi

# The non-mirror ceiling panel is only a PANEL once all three draws exist; a
# partial one would report one lottery ticket as the family (the exact defect
# G-13 exists to prevent), so it is all-or-nothing.
for d in "${TOPGRIM_ORDER[@]}"; do
  ck="${BED[$d]#model:}"; ck="${ck%%:*}"
  if [ ! -f "$ck" ]; then
    [ "$SKIP_TOPGRIM" = "0" ] && \
      echo "[panel-gate $PREFIX] topgrim panel incomplete ($ck missing) - SKIPPING"
    SKIP_TOPGRIM=1
  fi
done

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
  [ "$SKIP_TOP" != "1" ] && \
    for bed in "${TOP_ORDER[@]}"; do run_cell "$arm" "$spec" "$bed" $PANEL_SEEDS; done
  [ "$SKIP_TOPGRIM" != "1" ] && \
    for bed in "${TOPGRIM_ORDER[@]}"; do run_cell "$arm" "$spec" "$bed" $PANEL_SEEDS; done
done
echo "=== M40 PANEL GATE ($PREFIX) DONE $(date) ==="
