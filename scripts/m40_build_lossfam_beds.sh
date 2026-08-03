#!/usr/bin/env bash
# M40 Phase 2 — clone beds for the three LIVE LOSS FAMILIES the roster
# measured with weak proxies (forensics 2026-08-03: dragapult 1-11 live vs a
# rule-agent bed, rocket 1-7 vs an M30-era clone, garchomp 2-6 vs no bed).
#
# Ceiling-band recipe: the 08-03 snowball made 1000+ seats the RICHEST band
# for all three families (dragapult 133 / garchomp 371 / rocket 337), so
# these clone at min-score 900 rather than the 700-850 the plan priced —
# same choice topgrim made. X5 says the clone will still compress (~0.20
# retention); the composite wrap is what carries it back up.
#
# One dominant deck hash per family (exported by scripts/export_opp_deck.py,
# most-played at >=900), so corpus decks == bed deck exactly:
#   dragapult 3631d393 (106 seats)  garchomp c7b3253f (405)  rocket 59e27a5e (146)
#
# Corpus: ALL seats, not winners-only (M39 law: a bed clones how an opponent
# plays, not how they win), hand-aware. Draws d1/d2/d3 differ by EPOCHS
# 10/9/8 (M39: epoch count is the largest bed-lottery knob) + torch's
# unseeded init. Idempotent: existing corpora/checkpoints are skipped.
set -eu
cd "$(dirname "$0")/.."

declare -A HASH=( [dragapult]=3631d393 [garchomp]=c7b3253f [rocket]=59e27a5e )
declare -A EPOCHS=( [d1]=10 [d2]=9 [d3]=8 )

for fam in dragapult garchomp rocket; do
  corpus="data/bc_m40_${fam}"
  if [ ! -f "$corpus/deck_registry.json" ]; then
    echo "=== build $corpus (hash ${HASH[$fam]}, min-score 900) $(date)"
    uv run python -m rl.replay_bc build --deck-hash "${HASH[$fam]}" \
      --min-score 900 --hand-aware --out "$corpus"
  else
    echo "=== $corpus exists - skipping build"
  fi
  for d in d1 d2 d3; do
    ck="checkpoints/m40_bed_${fam}_${d}.pt"
    if [ -f "$ck" ]; then echo "=== $ck exists - skipping"; continue; fi
    echo "=== train $ck (${EPOCHS[$d]} epochs) $(date)"
    uv run python -m rl.plan_iter train --data "$corpus" \
      --name "m40_bed_${fam}_${d}" --init checkpoints/m28_winners.pt \
      --epochs "${EPOCHS[$d]}" --lr 3e-4 2>&1 | tail -3
  done
done
echo "=== LOSS-FAMILY BEDS DONE $(date)"
