#!/usr/bin/env bash
# M35 rules matrix: bench-floor rule + poffinfloor revert probe. Pilot weights =
# m28_winners, deck = alakazam_v2 (the M33 live deck); the RULE SET is the only
# variable. 4 arms via matchrunner _MODEL_FIX_KINDS:
#   gac    modelt-gac   telepath,deckguard,ash,conserve            (M30 rules)
#   gacb   modelt-gacb  + poffinfloor                              (M33 ship, CONTROL)
#   gacf   modelt-gacf  gac + benchfloor                           (bf replaces poffinfloor)
#   gacbf  modelt-gacbf gacb + benchfloor                          (new rule on ship)
#
# Beds (hold rocket/grim = the M33 deck-out wins; non-inferior elsewhere):
#   rocket m30_bc_rocket + rocket deck   grim m25_bc_grim + grim deck
#   luc rule:lucario   mirror m28+clone   arch m28+archaludon   kyo random:kyogre
# matchrunner result 0 == side-A (our arm) WIN. workers 8. 2-prop z vs gacb.
# Usage: m35_battery.sh 2a [arm ...]   # decisive: rocket grim luc
#        m35_battery.sh 2b [arm ...]   # full: mirror arch kyo
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="checkpoints/m28_winners.pt"; DECK="decks/alakazam_v2.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
MIRROR="model:checkpoints/m28_winners.pt:clone54618168"
ARCH="model:checkpoints/m28_winners.pt:decks/archaludon.csv"

arm_kind() { case "$1" in
  gac) echo modelt-gac ;; gacb) echo modelt-gacb ;;
  gacf) echo modelt-gacf ;; gacbf) echo modelt-gacbf ;;
  *) echo "unknown arm $1" >&2; exit 1 ;; esac; }
bed_spec() { case "$1" in
  rocket) echo "$ROCKET" ;; grim) echo "$GRIM" ;; luc) echo "rule:lucario" ;;
  mirror) echo "$MIRROR" ;; arch) echo "$ARCH" ;; kyo) echo "random:kyogre" ;;
  *) echo "unknown bed $1" >&2; exit 1 ;; esac; }

phase="${1:-2a}"; shift || true
arms=("${@:-gacb gac gacf gacbf}"); arms=(${arms[@]})
case "$phase" in
  2a) beds=(rocket grim luc) ;;
  2b) beds=(mirror arch kyo) ;;
  *) echo "phase must be 2a or 2b" >&2; exit 1 ;;
esac

echo "M35 battery phase $phase | arms=${arms[*]} | beds=${beds[*]}"
for a in "${arms[@]}"; do
  A="$(arm_kind "$a"):${CKPT}:${DECK}"
  for bed in "${beds[@]}"; do
    B="$(bed_spec "$bed")"
    for s in 1 2; do
      out="runs/m35_${a}_${bed}_s${s}.jsonl"
      echo ">>> arm=$a bed=$bed seed=$s -> $out"
      uv run python -m rl.matchrunner play --a "$A" --b "$B" -n 200 --workers 8 \
        --seed "$s" --checkpoint "$out"
    done
  done
done
echo "M35 battery phase $phase done:$(date -u +%H:%M:%S)"
