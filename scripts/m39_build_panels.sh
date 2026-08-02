#!/usr/bin/env bash
# M39 G-13 — build the BED PANELS.
#
# Why (docs/M39.md, D1): bed play-strength is not a property of a corpus, it
# is a draw from a lottery over training runs. Three clones of the same grim
# list from near-identical data scored .671 / .623 / .546 against the same
# net at n=2400 each -- a 12.5pp spread against a 2.0pp binomial CI, and ONE
# fewer training epoch on identical data moved it 4.8pp.
#
# G-13 therefore forbids a single-clone bed in any gate roster. Each family
# gets >=3 draws, the cell value is the panel MEAN, and the spread across
# draws is reported. This costs no extra compute: it is a redistribution of
# match budget (n=800 against each of 3 draws instead of n=2400 against one).
#
# Draw dimension = training run. `plan_iter train` sets no torch seed, so two
# runs of the identical command already differ; we ALSO vary the epoch count
# because D1 measured that as the knob with the largest observed effect. A
# draw is therefore (corpus, epochs, run) and the panel averages over all of
# it -- which is the construction variance we actually want averaged out.
#
# d1 of every family is an EXISTING bed, reused so the panel costs only the
# two new draws per family and so historical cells stay comparable:
#   wall d1 = m38_bc_wall   grim d1 = m39_bc_grim   (d2 = m39_bc_grim_b)
#   archaludon d1 = m39_bc_archaludon               top d1 = m39_bc_top
#
# No panel for: tuned / dragapult / iono (rule agents, no training draw),
# mirror + m28 (m28_winners is our own champion lineage, a historical
# artifact rather than a clone re-drawable from a corpus), rocket (the M30
# corpus is not on this box -- flagged in the diary as a single-draw cell).
#
# Usage: bash scripts/m39_build_panels.sh
set -u
cd "$(dirname "$0")/.."

INIT=checkpoints/m28_winners.pt
LR=3e-4

train_draw () {   # name  epochs  data-dirs...
  local name="$1"; shift
  local epochs="$1"; shift
  if [ -f "checkpoints/${name}.pt" ]; then
    echo "[panel] ${name} exists - skip"
    return
  fi
  echo "[panel] training ${name} (epochs=${epochs})  $(date)"
  uv run python -m rl.plan_iter train --data "$@" \
    --name "$name" --init "$INIT" --epochs "$epochs" --lr "$LR"
}

train_draw m39_bed_wall_d2       9 data/bc_m38_wall
train_draw m39_bed_wall_d3       8 data/bc_m38_wall
train_draw m39_bed_arch_d2       9 data/bc_m39_arch_a data/bc_m39_arch_b
train_draw m39_bed_arch_d3       8 data/bc_m39_arch_a data/bc_m39_arch_b
train_draw m39_bed_grim_d3       8 data/bc_m39_grim
train_draw m39_bed_top_d2        9 data/bc_m39_top
train_draw m39_bed_top_d3        8 data/bc_m39_top

echo "=== M39 PANEL BUILD DONE $(date) ==="
