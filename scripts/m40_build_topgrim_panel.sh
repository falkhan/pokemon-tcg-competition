#!/usr/bin/env bash
# M40 S3.1 — turn the NON-MIRROR high-band ceiling bed into a PANEL.
#
# Why this is the H-A/H-B discriminator. `m39_bc_top` clones 900+ pilots of OUR
# OWN list, so parity there is parity with a MIRROR and says nothing about the
# families that actually beat us live (grim, dragapult, archaludon).
# `m39_bc_topgrim` clones the SAME grim list as `m39_bc_grim` but from >=1000
# seats instead of 700-850 — identical deck, two demonstrator altitudes ~250
# ELO apart. The delta between our WR vs each is the price of opponent SKILL
# with the deck held fixed, which no other instrument can separate.
#
# It has to be a panel and not the single draw M39 built, and today's read is
# exactly why: at n=2400 we beat topgrim 0.699 and grim_d1 0.695 — a 0.4pp gap
# suggesting the demonstrator band does not transmit at all. But G-13 prices
# bed construction at ~+/-6pp per draw, and grim_d1 is the grim family's EASY
# ticket (0.695 against a 0.654 family mean). A single-draw-vs-single-draw
# comparison cannot tell "no compression" from "two lucky draws". Three draws
# each can.
#
# Draw recipe is m39_build_panels.sh's, unchanged, so the topgrim panel is
# constructed exactly like every other panel in the roster: same init, same lr,
# epochs varied because D1 measured epoch count as the biggest lottery knob.
# d1 is the existing bed, by the same convention the other families use.
#
# Usage: bash scripts/m40_build_topgrim_panel.sh
set -u
cd "$(dirname "$0")/.."

INIT=checkpoints/m28_winners.pt
LR=3e-4
DATA=data/bc_m39_topgrim          # grim hash 3121746f, min_score 1000

[ -d "$DATA" ] || { echo "no $DATA" >&2; exit 2; }

train_draw () {   # name  epochs
  local name="$1" epochs="$2"
  if [ -f "checkpoints/${name}.pt" ]; then
    echo "[panel] ${name} exists - skip"; return
  fi
  echo "[panel] training ${name} (epochs=${epochs})  $(date)"
  uv run python -m rl.plan_iter train --data "$DATA" \
    --name "$name" --init "$INIT" --epochs "$epochs" --lr "$LR"
}

# d1 = checkpoints/m39_bc_topgrim.pt (10 epochs, val 0.674) — already built.
train_draw m40_bed_topgrim_d2 9
train_draw m40_bed_topgrim_d3 8

echo "=== M40 TOPGRIM PANEL BUILD DONE $(date) ==="
