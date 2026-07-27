#!/usr/bin/env bash
# M34 battery: consistency deck-tech A/B. Pilot = m28_winners (UNCHANGED);
# the DECK is the only variable. Compares 3 legal search-add variants against the
# shipped alakazam_v2 baseline, all piloted by m28_winners, on the live-faithful beds.
#
# Variants (Tier-0 opening-brick, scripts/brick_sim.py):
#   v2   decks/alakazam_v2.csv       15.4%  (shipped baseline)
#   mine decks/alakazam_v2_mine.csv  13.3%  -1 Nighttime Mine  +1 Ultra Ball
#   ace  decks/alakazam_v2_ace.csv   13.3%  -1 Enriching Energy +1 Master Ball (ACE swap; NO disruption cut)
#   agg  decks/alakazam_v2_agg.csv   11.5%  -1 Enh Hammer -1 Xerosic +2 Ultra Ball (aggressive)
#
# Beds (must HOLD the M33 deck-out gains on rocket/grim; non-inferior elsewhere):
#   rocket  m30_bc_rocket + rocket deck    (v2 pin 0.640 -- MUST HOLD)
#   grim    m25_bc_grim   + grim deck      (v2 pin 0.739 -- MUST HOLD)
#   luc     rule:lucario                   (aggro bed; v2 pin 0.708)
#   mirror  m28_winners   + base clone     (v2 pin 0.487)
#   arch    m28_winners   + archaludon deck (v2 pin 0.865)
#   kyo     random:kyogre                  (floor >=0.90; v2 pin 0.970)
#
# matchrunner jsonl result 0 == side-A (our deck) WIN. workers 8 (CLAUDE.md cap).
# Usage: m34_battery.sh 2a [deck ...]   # decisive: rocket grim luc
#        m34_battery.sh 2b [deck ...]   # full: mirror arch kyo
set -euo pipefail
cd "$(dirname "$0")/.."

PILOT="checkpoints/m28_winners.pt"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
MIRROR="model:checkpoints/m28_winners.pt:clone54618168"
ARCH="model:checkpoints/m28_winners.pt:decks/archaludon.csv"

deck_path() { case "$1" in
  v2) echo decks/alakazam_v2.csv ;;  mine) echo decks/alakazam_v2_mine.csv ;;
  ace) echo decks/alakazam_v2_ace.csv ;; agg) echo decks/alakazam_v2_agg.csv ;;
  *) echo "unknown deck $1" >&2; exit 1 ;; esac; }

bed_spec() { case "$1" in
  rocket) echo "$ROCKET" ;; grim) echo "$GRIM" ;; luc) echo "rule:lucario" ;;
  mirror) echo "$MIRROR" ;; arch) echo "$ARCH" ;; kyo) echo "random:kyogre" ;;
  *) echo "unknown bed $1" >&2; exit 1 ;; esac; }

phase="${1:-2a}"; shift || true
decks=("${@:-v2 mine ace agg}"); decks=(${decks[@]})
case "$phase" in
  2a) beds=(rocket grim luc) ;;
  2b) beds=(mirror arch kyo) ;;
  *) echo "phase must be 2a or 2b" >&2; exit 1 ;;
esac

echo "M34 battery phase $phase | decks=${decks[*]} | beds=${beds[*]}"
for d in "${decks[@]}"; do
  A="model:${PILOT}:$(deck_path "$d")"
  for bed in "${beds[@]}"; do
    B="$(bed_spec "$bed")"
    for s in 1 2; do
      out="runs/m34_${d}_${bed}_s${s}.jsonl"
      echo ">>> deck=$d bed=$bed seed=$s -> $out"
      uv run python -m rl.matchrunner play --a "$A" --b "$B" -n 200 --workers 8 \
        --seed "$s" --checkpoint "$out"
    done
  done
done
echo "M34 battery phase $phase done:$(date -u +%H:%M:%S)"
