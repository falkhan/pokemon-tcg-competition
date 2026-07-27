#!/usr/bin/env bash
# M36 P2 — hammer-4 deck-tech A/B (docs/M36-plan.md W1 + the E4 mechanism:
# Enhanced Hammer strips the wall's special energy = unlocks Powerful Hand).
# Deck is the SINGLE variable: modelt-gacf rules (the live ship set) piloting
# decks/alakazam_v2_h4.csv (+1 Enhanced Hammer, -1 Hilda) vs the m36 battery's
# gacf+v2 control jsonls (same session, same code state).
# Usage: m36_decktech.sh w      # decisive: wall bed only
#        m36_decktech.sh full   # non-inferiority: the other 6 beds
set -euo pipefail
cd "$(dirname "$0")/.."

A="modelt-gacf:checkpoints/m28_winners.pt:decks/alakazam_v2_h4.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
MIRROR="model:checkpoints/m28_winners.pt:clone54618168"
ARCH="model:checkpoints/m28_winners.pt:decks/archaludon.csv"

bed_spec() { case "$1" in
  rocket) echo "$ROCKET" ;; grim) echo "$GRIM" ;; luc) echo "rule:lucario" ;;
  mirror) echo "$MIRROR" ;; arch) echo "$ARCH" ;; kyo) echo "random:kyogre" ;;
  wall) echo "solver:decks/greattusk_wall.csv" ;;
  *) echo "unknown bed $1" >&2; exit 1 ;; esac; }

phase="${1:-w}"
case "$phase" in
  w) beds=(wall) ;;
  full) beds=(rocket grim luc mirror arch kyo) ;;
  *) echo "phase must be w or full" >&2; exit 1 ;;
esac

echo "M36 decktech phase $phase | beds=${beds[*]}"
for bed in "${beds[@]}"; do
  B="$(bed_spec "$bed")"
  for s in 1 2; do
    out="runs/m36_h4_${bed}_s${s}.jsonl"
    echo ">>> h4 bed=$bed seed=$s -> $out"
    uv run python -m rl.matchrunner play --a "$A" --b "$B" -n 200 --workers 8 \
      --seed "$s" --checkpoint "$out"
  done
done
echo "M36 decktech phase $phase DONE:$(date -u +%H:%M:%S)"
