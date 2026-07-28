#!/usr/bin/env bash
# M36 P0.1b — wall-bed strawman check (docs/M36-plan.md W1).
# The M26/M30 strawman law: before any fix is measured on the new
# decks/greattusk_wall.csv bed, the m35 SHIP config must reproduce the live
# loss mode on it (live: 0-4, all deck-outs at t30-39 with 3-5 prizes left).
# Arms: generic:/solver: wall pilots, n=200 x 2 seeds, workers 8 (hard cap).
# Decode: matchrunner jsonl results 0 = side-A (us) WIN, 1 = loss, 2 = draw.
# If generic is too weak (M32-B warning: generic:archaludon sat 0.86 vs live
# 0.47), climb the pilot ladder: generic2 -> solver2 -> solver-dev -> BC clone.
set -euo pipefail
cd "$(dirname "$0")/.."

A="modelt-gacf:checkpoints/m28_winners.pt:decks/alakazam_v2.csv"
WALL="decks/greattusk_wall.csv"

for pilot in generic solver; do
  for s in 1 2; do
    out="runs/m36_strawman_${pilot}wall_s${s}.jsonl"
    echo ">>> pilot=$pilot seed=$s -> $out"
    uv run python -m rl.matchrunner play --a "$A" --b "${pilot}:${WALL}" \
      -n 200 --workers 8 --seed "$s" --checkpoint "$out"
  done
done
echo "M36 strawman done:$(date -u +%H:%M:%S)"
