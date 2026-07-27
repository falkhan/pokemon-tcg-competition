#!/usr/bin/env bash
# M31 P3: ship-config battery for the bench-economy / supporter rule arms,
# floors FIRST (m29 post-mortem rec #5 — don't spend the n=800 mirror before
# the out-of-loop floors), mirror phase LAST. Only arms still alive after the
# P2 screens are batteried.
# Arms: gacb (gac + O7 poffinfloor), gacd (gac + O8 drawfloor) — both RULE
# arms, weights frozen at m28_winners => non-inferiority bar.
# Pins = LIVE gac (54929991, M30 ship): mirror 0.6750 n=800 · dragapult 0.3750
# n=400 · kyogre >=0.95 · rocket 0.5675 n=400 (advisory + deck-out guard) ·
# grim 0.6687 n=400 (advisory). Decode: 0=WIN.
# Usage: m31_battery.sh floors [arm ...]   # then, for arms still alive:
#        m31_battery.sh mirror [arm ...]
set -euo pipefail
cd "$(dirname "$0")/.."
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"

arm_spec() {
  case "$1" in
    gacb) echo "modelt-gacb:checkpoints/m28_winners.pt:clone54618168" ;;
    gacd) echo "modelt-gacd:checkpoints/m28_winners.pt:clone54618168" ;;
    gac)  echo "modelt-gac:checkpoints/m28_winners.pt:clone54618168" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

phase="${1:-floors}"; shift || true
arms=("${@:-}")
[ -z "${arms[0]:-}" ] && arms=(gacb gacd)

for arm in "${arms[@]}"; do
  A="$(arm_spec "$arm")"
  if [ "$phase" = floors ]; then
    for s in 1 2; do
      uv run python -m rl.matchrunner play --a "$A" --b rule:dragapult -n 200 --workers 8 --seed "$s" --checkpoint "runs/m31_${arm}_drag_s${s}.jsonl"
      uv run python -m rl.matchrunner play --a "$A" --b "$GRIM" -n 200 --workers 8 --seed "$s" --checkpoint "runs/m31_${arm}_grim_s${s}.jsonl"
      uv run python -m rl.matchrunner play --a "$A" --b "$ROCKET" -n 200 --workers 8 --seed "$s" --checkpoint "runs/m31_${arm}_rocket_s${s}.jsonl"
    done
    uv run python -m rl.matchrunner play --a "$A" --b random:kyogre -n 200 --workers 8 --seed 1 --checkpoint "runs/m31_${arm}_kyo_s1.jsonl"
  else
    for s in 1 2 3 4; do
      uv run python -m rl.matchrunner play --a "$A" --b rule:lucario -n 200 --workers 8 --seed "$s" --checkpoint "runs/m31_${arm}_luc_s${s}.jsonl"
    done
  fi
done
echo "=== M31 P3 $phase DONE $(date) ==="
