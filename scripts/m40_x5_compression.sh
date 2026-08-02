#!/usr/bin/env bash
# M40 S3.2 / X5 — an ERROR BAR on the compression factor.
#
# docs/M40-plan.md §1 rests H-A on one number: M39's D1 head-to-head put the
# 1000-band grim clone over the 750-band clone at 0.623 (z=+12.04), which the
# plan reads as ~86 ELO of separation from demonstrators ~250 ELO apart, i.e.
# "a BC clone retains roughly a third to a half of its demonstrator's edge".
# Five milestones' worth of bed interpretation hangs off that fraction.
#
# It comes from ONE draw-pair. G-13 measured bed construction as a lottery with
# ~+/-6pp per draw — larger than most effects the campaign gates on — so a
# single pairing cannot separate "the band transmits at 40%" from "one clone
# drew well and the other drew badly". The plan says so itself: "the honest
# range on the retention factor is wide (X5 measures it properly)".
#
# X5 runs the full PANEL x PANEL cross: 3 high-band draws x 3 low-band draws,
# n=400 each = 3600 games. Every cell holds the DECK fixed (both sides are the
# same grim list) and varies only the demonstrator band, so the cross measures
# the price of pilot skill with deck effects cancelled. The spread ACROSS the
# nine cells is the construction lottery; the mean is the transmission signal.
#
# Decode with scripts/m40_x5_decode.py.
#
# Usage: bash scripts/m40_x5_compression.sh
set -u
cd "$(dirname "$0")/.."

DECK=grim_live
N=400
COMPLETE_LINES=16

declare -A HI=(
  [d1]=checkpoints/m39_bc_topgrim.pt        # >=1000 seats, val 0.674
  [d2]=checkpoints/m40_bed_topgrim_d2.pt    # same corpus, 9 epochs
  [d3]=checkpoints/m40_bed_topgrim_d3.pt    # same corpus, 8 epochs
)
declare -A LO=(
  [d1]=checkpoints/m39_bc_grim.pt           # 700-850 seats, val 0.662
  [d2]=checkpoints/m39_bc_grim_b.pt         # same corpus, 9 epochs
  [d3]=checkpoints/m39_bed_grim_d3.pt       # same corpus, 8 epochs
)

for h in d1 d2 d3; do
  for l in d1 d2 d3; do
    f="runs/m40_x5_hi${h}_lo${l}.jsonl"
    n=0; [ -f "$f" ] && n=$(grep -c results "$f" || true)
    [ "$n" -ge "$COMPLETE_LINES" ] && continue
    echo "[x5] topgrim_${h} vs grim_${l}  $(date)"
    rm -f "$f"
    uv run python -m rl.matchrunner play \
      --a "model:${HI[$h]}:${DECK}" --b "model:${LO[$l]}:${DECK}" \
      -n "$N" --workers 8 --seed 1 --checkpoint "$f"
  done
done
echo "=== M40 X5 DONE $(date) ==="
