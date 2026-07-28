#!/usr/bin/env bash
# M37 racemode battery (docs/M37-plan.md P1): O12 A/B on TRIGGER beds only.
# Pilot weights = m28_winners, deck = alakazam_v2_h4; the RULE SET is the
# only variable:
#   gacf    modelt-gacf    telepath,deckguard,ash,conserve,benchfloor (SHIP, CONTROL)
#   gacfr   modelt-gacfr   gacf + racemode  (O12 margin-gated)
#   gacfrr  modelt-gacfrr  gacf + racemoder (O12 blanket)
#
# Trigger beds (opponent deck contains a _RACEMODE_OPP_IDS Pokémon):
#   hop      solver:decks/hops_stall.csv         (strawman-valid, 0.470)
#   wall     solver:decks/greattusk_wall.csv     (strawman-valid, 0.395)
#   grim     m25 grim clone                      (blocked on laptop sync)
#   garchomp m37 garchomp clone                  (blocked on Phase 1 build)
# Non-trigger beds are NOT run per-arm — O12 is provably inert without a
# trigger id on the opponent board; the `inert` phase verifies that claim
# (bar B4) instead of burning battery games on it.
# Fresh gacf CONTROL runs everywhere (drift law — no cross-battery reuse).
# matchrunner result 0 == side-A WIN. workers 8 (hard cap).
#
# Usage: m37_battery.sh beds  [seed ...]     # default seeds 1 2 3
#        m37_battery.sh inert [seed]         # B4 mirror inertness smoke
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="checkpoints/m28_winners.pt"; DECK="decks/alakazam_v2_h4.csv"

arm_kind() { case "$1" in
  gacf) echo modelt-gacf ;; gacfr) echo modelt-gacfr ;;
  gacfrr) echo modelt-gacfrr ;; gacfr2) echo modelt-gacfr2 ;;
  *) echo "unknown arm $1" >&2; exit 1 ;; esac; }
bed_spec() { case "$1" in
  hop)      echo "solver:decks/hops_stall.csv" ;;
  wall)     echo "solver:decks/greattusk_wall.csv" ;;
  grim)     echo "model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv" ;;
  garchomp) echo "model:checkpoints/m37_bc_garchomp.pt:data/kaggle/garchomp_m37_deck.csv" ;;
  mirror)   echo "model:checkpoints/m28_winners.pt:clone54618168" ;;
  *) echo "unknown bed $1" >&2; exit 1 ;; esac; }
bed_ready() { case "$1" in
  grim)     [[ -f checkpoints/m25_bc_grim_54861775.pt ]] ;;
  garchomp) [[ -f checkpoints/m37_bc_garchomp.pt ]] ;;
  *) true ;; esac; }

phase="${1:-beds}"; shift || true

if [[ "$phase" == "inert" ]]; then
  s="${1:-1}"
  for a in gacf gacfr; do
    out="runs/m37_inert_${a}_mirror_s${s}.jsonl"
    echo ">>> inert arm=$a seed=$s -> $out"
    uv run python -m rl.matchrunner play --a "$(arm_kind "$a"):${CKPT}:${DECK}" \
      --b "$(bed_spec mirror)" -n 100 --workers 8 --seed "$s" --checkpoint "$out"
  done
  exit 0
fi

seeds=("${@:-1 2 3}"); seeds=(${seeds[@]})
beds=(${M37_BEDS:-hop wall grim garchomp})   # env override for parallel invocations
arms=(${M37_ARMS:-gacf gacfr gacfrr})        # v2 battery: "gacf gacfr2"
prefix="${M37_PREFIX:-m37}"                  # v2 battery: m37v2 (fresh controls)
for bed in "${beds[@]}"; do
  if ! bed_ready "$bed"; then
    echo "=== bed $bed NOT READY (checkpoint missing) — skipped ==="
    continue
  fi
  for a in "${arms[@]}"; do
    for s in "${seeds[@]}"; do
      out="runs/${prefix}_${a}_${bed}_s${s}.jsonl"
      echo ">>> arm=$a bed=$bed seed=$s -> $out"
      uv run python -m rl.matchrunner play --a "$(arm_kind "$a"):${CKPT}:${DECK}" \
        --b "$(bed_spec "$bed")" -n 200 --workers 8 --seed "$s" --checkpoint "$out"
    done
  done
done
echo "M37 battery DONE"
