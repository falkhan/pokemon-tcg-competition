#!/usr/bin/env bash
# M32-B battery: base (m28_winners) vs ppo (ppo_current_m32b), same deck
# clone54618168, on the re-spec'd beds. This is a TWO-ARM comparison (ppo vs
# base, two-proportion z) — NOT a comparison to a pinned live value.
#
# Beds:
#   arch    generic:archaludon  -- SEALED transfer gate (PRIMARY). archaludon is
#           in NO training pool and NO prior bed: the clean out-of-loop gauge
#           (replaces rule:dragapult, which sat at 0.78 live and could not
#           resolve an improvement).
#   luc     rule:lucario        -- out-of-loop non-inferiority GUARD (Mega
#           Lucario; catastrophic-forgetting check). NOT in the training pool.
#   rocket  } trained-matchup SANITY -- did the training take? (both in the pool)
#   grim    }
#   mirror  model:m28_winners:clone54618168 -- did PPO gain in the (trained)
#           mirror. base-as-A is m28-vs-m28 (~0.5 symmetry sanity); ppo-as-A is
#           the real signal.
#   kyo     random:kyogre       -- FLOOR (>= 0.90).
#
# Do NOT run while the PPO collect is using workers 8 (16 workers deadlocks —
# CLAUDE.md hard rule). Run only after the run completes.
# Usage: m32b_battery.sh primary [arm ...]   # arch + luc (the decisive gates)
#        m32b_battery.sh sanity  [arm ...]    # rocket + grim + mirror + kyo
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="model:checkpoints/m28_winners.pt:clone54618168"
PPO="model:checkpoints/ppo_current_m32b.pt:clone54618168"
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
MIRROR="model:checkpoints/m28_winners.pt:clone54618168"

arm_spec() {
  case "$1" in
    base) echo "$BASE" ;;
    ppo)  echo "$PPO" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

run_bed() {  # arm A_spec bed B_spec nseeds
  local arm="$1" A="$2" bed="$3" B="$4" nseeds="$5" s
  for s in $(seq 1 "$nseeds"); do
    uv run python -m rl.matchrunner play --a "$A" --b "$B" -n 200 --workers 8 \
      --seed "$s" --checkpoint "runs/m32b_${arm}_${bed}_s${s}.jsonl"
  done
}

phase="${1:-primary}"; shift || true
arms=("${@:-}")
[ -z "${arms[0]:-}" ] && arms=(base ppo)

for arm in "${arms[@]}"; do
  A="$(arm_spec "$arm")"
  if [ "$phase" = primary ]; then
    run_bed "$arm" "$A" arch   "generic:archaludon" 2
    run_bed "$arm" "$A" luc    "rule:lucario"       2
  else
    run_bed "$arm" "$A" rocket "$ROCKET"            2
    run_bed "$arm" "$A" grim   "$GRIM"              2
    run_bed "$arm" "$A" mirror "$MIRROR"            2
    run_bed "$arm" "$A" kyo    "random:kyogre"      2
  fi
done
echo "=== M32-B battery $phase DONE $(date) ==="
