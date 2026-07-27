#!/usr/bin/env bash
# M36 rules matrix: O11 gustveto A/B (docs/M36-plan.md P1). Pilot weights =
# m28_winners, deck = alakazam_v2; the RULE SET is the only variable:
#   gacf   modelt-gacf   telepath,deckguard,ash,conserve,benchfloor  (M35 SHIP, CONTROL)
#   gacfv  modelt-gacfv  gacf + gustveto (no Boss's Orders at opp-prizes <= 1)
#
# Beds = the m35 six + the NEW wall bed (strawman-validated 2026-07-27:
# solver:greattusk_wall reproduces the live deck-out farm, wr 0.330 pooled):
#   rocket m30_bc_rocket + rocket deck   grim m25_bc_grim + grim deck
#   luc rule:lucario   mirror m28+clone   arch m28+archaludon   kyo random:kyogre
#   wall solver:decks/greattusk_wall.csv
# Fresh gacf CONTROL runs everywhere (no cross-battery reuse of the m35
# jsonls — code drift between batteries voided such comparisons before).
# matchrunner result 0 == side-A (our arm) WIN. workers 8. 2-prop z vs gacf.
# Usage: m36_battery.sh 2a [arm ...]   # decisive: wall rocket grim luc
#        m36_battery.sh 2b [arm ...]   # full: mirror arch kyo
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="checkpoints/m28_winners.pt"; DECK="decks/alakazam_v2.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
MIRROR="model:checkpoints/m28_winners.pt:clone54618168"
ARCH="model:checkpoints/m28_winners.pt:decks/archaludon.csv"
WALL="solver:decks/greattusk_wall.csv"

arm_kind() { case "$1" in
  gacf) echo modelt-gacf ;; gacfv) echo modelt-gacfv ;;
  *) echo "unknown arm $1" >&2; exit 1 ;; esac; }
bed_spec() { case "$1" in
  rocket) echo "$ROCKET" ;; grim) echo "$GRIM" ;; luc) echo "rule:lucario" ;;
  mirror) echo "$MIRROR" ;; arch) echo "$ARCH" ;; kyo) echo "random:kyogre" ;;
  wall) echo "$WALL" ;;
  *) echo "unknown bed $1" >&2; exit 1 ;; esac; }

phase="${1:-2a}"; shift || true
arms=("${@:-gacf gacfv}"); arms=(${arms[@]})
case "$phase" in
  2a) beds=(wall rocket grim luc) ;;
  2b) beds=(mirror arch kyo) ;;
  *) echo "phase must be 2a or 2b" >&2; exit 1 ;;
esac

echo "M36 battery phase $phase | arms=${arms[*]} | beds=${beds[*]}"
for a in "${arms[@]}"; do
  A="$(arm_kind "$a"):${CKPT}:${DECK}"
  for bed in "${beds[@]}"; do
    B="$(bed_spec "$bed")"
    for s in 1 2; do
      out="runs/m36_${a}_${bed}_s${s}.jsonl"
      echo ">>> arm=$a bed=$bed seed=$s -> $out"
      uv run python -m rl.matchrunner play --a "$A" --b "$B" -n 200 --workers 8 \
        --seed "$s" --checkpoint "$out"
    done
  done
done
echo "M36 battery phase $phase DONE:$(date -u +%H:%M:%S)"
