#!/usr/bin/env bash
# M37 strawman screens: the M36 ship config (m28_winners + gacf + alakazam_v2_h4)
# vs the NEW stall beds extracted from the 55030954 live cache, plus reference
# beds — multi-opponent per the leaderboard-sim rule.
#
# Beds:
#   hop      solver:decks/hops_stall.csv        (live 0-2, deck-out losses)
#   garchomp solver:decks/cynthia_garchomp.csv  (live 0-2, deck-out losses)
#   grimlive solver:decks/grim_live.csv         (live 1-4 vs this exact list)
#   wall     solver:decks/greattusk_wall.csv    (cross-machine ref: m36 pin 0.42)
#   mirror   model:m28_winners:clone54618168    (ref: m36 pin ~0.53)
#
# Strawman law (M26/M30): a bed is only valid for measuring a fix if the
# current config LOSES on it the way it loses live (deck-out). If solver: is
# too weak a pilot, escalate rungs (generic2/solver2/solver-dev) before use.
# Usage: m37_strawman.sh [seed] [bed ...]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="checkpoints/m28_winners.pt"; DECK="decks/alakazam_v2_h4.csv"
A="modelt-gacf:${CKPT}:${DECK}"

bed_spec() { case "$1" in
  hop)      echo "solver:decks/hops_stall.csv" ;;
  garchomp) echo "solver:decks/cynthia_garchomp.csv" ;;
  grimlive) echo "solver:decks/grim_live.csv" ;;
  wall)     echo "solver:decks/greattusk_wall.csv" ;;
  mirror)   echo "model:checkpoints/m28_winners.pt:clone54618168" ;;
  luc)      echo "rule:lucario" ;;
  arch)     echo "model:checkpoints/m28_winners.pt:decks/archaludon.csv" ;;
  *) echo "unknown bed $1" >&2; exit 1 ;; esac; }

seed="${1:-1}"; shift || true
beds=("${@:-hop garchomp grimlive wall mirror}"); beds=(${beds[@]})

for bed in "${beds[@]}"; do
  out="runs/m37_strawman_${bed}_s${seed}.jsonl"
  echo ">>> bed=$bed seed=$seed -> $out"
  uv run python -m rl.matchrunner play --a "$A" --b "$(bed_spec "$bed")" \
    -n 200 --workers 8 --seed "$seed" --checkpoint "$out"
done
echo "M37 strawman seed=$seed DONE $(date -u +%H:%M:%S)"
