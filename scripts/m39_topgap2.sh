#!/usr/bin/env bash
# M39 P0.9 — extend the ceiling instrument off the mirror family.
# (Piotr, 2026-08-01: "extend".)
#
# The limitation P0.8 flagged: m39_bc_top is cloned from OUR OWN list's 900+
# pilots, so it answers "can we play top-band mirrors to parity" and says
# nothing about grim/wall/archaludon pilots at that altitude. G-8 forbids
# claiming evidence about a band we have no bed for, and slot 3's whole
# thesis rests on the 900+ band.
#
# m39_bc_topgrim = grim hash 3121746f at score >=1000 (513 seats cached, 200
# extracted, val 0.674). It is the SAME LIST as m39_bc_grim (700-850 seats,
# val 0.662), so the pair is a controlled pilot-skill ladder: identical deck,
# two altitudes. The delta between our WR vs each is the price of opponent
# skill with the deck held fixed -- something no previous instrument could
# separate.
#
# NOT BUILT: a wall high-band bed. Wall at >=850 is 57 seats on 3d0a44b6 and
# 35 on 6f87e9d4 -- both under the 100-seat floor that killed the stall bed.
# Reported rather than built thin, per G-6.
#
# 6 seeds = n=2400/cell, per G-12 (n=800 resolves only ~10pp).
set -u
cd "$(dirname "$0")/.."

declare -A CAND=(
  [champion]="model:checkpoints/m28_winners.pt:alakazam_v2_h4"
  [cont3]="model:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
  [shipcfg]="modelt-gacfr3:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
)
BED="model:checkpoints/m39_bc_topgrim.pt:grim_live"
COMPLETE_LINES=16

for cand in champion cont3 shipcfg; do
  for seed in 1 2 3 4 5 6; do
    f="runs/m39_topgrim_${cand}_s${seed}.jsonl"
    n=0; [ -f "$f" ] && n=$(grep -c results "$f" || true)
    [ "$n" -ge "$COMPLETE_LINES" ] && continue
    echo "[m39 topgrim] $cand seed $seed  $(date)"
    rm -f "$f"
    uv run python -m rl.matchrunner play --a "${CAND[$cand]}" --b "$BED" \
      -n 400 --workers 8 --seed "$seed" --checkpoint "$f"
  done
done
echo "=== M39 TOPGRIM DONE $(date) ==="
