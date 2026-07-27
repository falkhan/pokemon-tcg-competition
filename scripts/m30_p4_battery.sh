#!/usr/bin/env bash
# M30 P4: ship-config battery, floors FIRST (m29 post-mortem rec #5 — don't
# spend the n=800 mirror before the out-of-loop floors), mirror phase LAST.
# Arms: B1/B2 + guard+ash (P3 survivors), B1 + tempo-only (pre-registered
# fallback), Arm D weights + O1. Pins = live 54903635 (M26 ship): lucario
# 0.671 n=800 · dragapult 0.3375 n=400 · grim 0.6875 n=400 (advisory) ·
# kyogre >=0.95 · rocket 0.458 n=120 offline_behavior read (advisory
# candidate, validated 2026-07-23). Decode: 0=WIN.
# Usage: m30_p4_battery.sh floors   # then, for arms still alive:
#        m30_p4_battery.sh mirror [arm ...]
set -euo pipefail
cd "$(dirname "$0")/.."
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"

arm_spec() {
  case "$1" in
    b1ga)    echo "modelt-guardash:checkpoints/m25_bc_alakazam_v3h.pt:clone54618168" ;;
    b1tempo) echo "modelt-tempo:checkpoints/m25_bc_alakazam_v3h.pt:clone54618168" ;;
    b2ga)    echo "modelt-guardash:checkpoints/m28_winners.pt:clone54618168" ;;
    gac)     echo "modelt-gac:checkpoints/m28_winners.pt:clone54618168" ;;
    darm)    echo "modelt:checkpoints/m30_54773249_winners.pt:clone54618168" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

phase="${1:-floors}"; shift || true
arms=("${@:-}")
[ -z "${arms[0]:-}" ] && arms=(b1ga b1tempo b2ga darm)

for arm in "${arms[@]}"; do
  A="$(arm_spec "$arm")"
  if [ "$phase" = floors ]; then
    for s in 1 2; do
      uv run python -m rl.matchrunner play --a "$A" --b rule:dragapult -n 200 --workers 8 --seed "$s" --checkpoint "runs/m30_${arm}_drag_s${s}.jsonl"
      uv run python -m rl.matchrunner play --a "$A" --b "$GRIM" -n 200 --workers 8 --seed "$s" --checkpoint "runs/m30_${arm}_grim_s${s}.jsonl"
      uv run python -m rl.matchrunner play --a "$A" --b "$ROCKET" -n 200 --workers 8 --seed "$s" --checkpoint "runs/m30_${arm}_rocket_s${s}.jsonl"
    done
    uv run python -m rl.matchrunner play --a "$A" --b random:kyogre -n 200 --workers 8 --seed 1 --checkpoint "runs/m30_${arm}_kyo_s1.jsonl"
  else
    for s in 1 2 3 4; do
      uv run python -m rl.matchrunner play --a "$A" --b rule:lucario -n 200 --workers 8 --seed "$s" --checkpoint "runs/m30_${arm}_luc_s${s}.jsonl"
    done
  fi
done
echo "=== M30 P4 $phase DONE $(date) ==="
