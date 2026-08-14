#!/usr/bin/env bash
# M46 08-15 battery — floors + B2-B4 panel probes (docs/M46-plan.md B/C).
#
# Order (registered): floors are cheap and gate the EARLY BANK, so they run
# first; the three panel probes follow sequentially (each ~55-65 min at 8
# pinned workers). Probes decode ONLY after A0v has passed — this script
# refuses to decode them otherwise (the A0v gate is checked by exit code).
#
#   part 1  floors: model-cz-ashw-dg0 (the bank pair) + model-cz-ashw-t0
#           (Tier-0 composite screen) vs tuned + iono, n=200 x 2 seeds each
#           side (floor-gate-sample-size law: n>=200 + second seed).
#   part 2  panel probes: m46_probe_af / _rg / _bc (pre-registered bands
#           42a0afb94c566951 / 2b7b66d543dcf10b / c317b32b155aaa71).
#
# Usage: bash scripts/m46_probes.sh [floors|probes|all]   (default all)
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

MODE="${1:-all}"
M41B=checkpoints/m41b_wide_prod.pt
hermes_ping () { hermes send -t telegram "[m46-probes] $1" 2>/dev/null || true; }

floor_cell () {  # tag  arm-spec  bed-spec  seed
  local out="runs/m46_floor_${1}_s${4}.jsonl"
  [ -f "$out" ] && { echo "=== $out exists — skipping"; return; }
  uv run python -m rl.matchrunner play --a "$2" --b "$3" \
    -n 200 --workers 8 --seed "$4" --checkpoint "$out" 2>&1 | tail -1
}

if [ "$MODE" != "probes" ]; then
  hermes_ping "floors: bank pair (dg0) + Tier-0 composite vs tuned/iono"
  for seed in 1 2; do
    for anchor in "rule:tuned:lucario" "rule:iono"; do
      tag=$(echo "$anchor" | cut -d: -f2)
      floor_cell "dg0_${tag}"  "model-cz-ashw-dg0:$M41B:alakazam_v2_h4" "$anchor" "$seed"
      floor_cell "ctl_${tag}"  "model-cz-ashw:$M41B:alakazam_v2_h4"     "$anchor" "$seed"
      floor_cell "t0_${tag}"   "model-cz-ashw-t0:$M41B:alakazam_v2_h4"  "$anchor" "$seed"
    done
  done
  echo "=== floors done — decode with scripts/m46_floor_decode.py (or by eye:"
  echo "    arm wr within noise of ctl wr per anchor = PASS)"
  hermes_ping "floors DONE"
fi

if [ "$MODE" != "floors" ]; then
  # A0v must have PASSED (acceptance MET = exit 0) before probe decodes are
  # believed; re-decode it here and stop if not.
  if ! uv run python scripts/m46_band_gate.py decode docs/specs/m46_A0v.json \
      --out runs/m46_A0v >/dev/null 2>&1; then
    echo "=== A0v acceptance NOT MET — probes must not be decoded (plan A0v"
    echo "    kill rule: panel demoted to advisory, guards-only fallback)."
    hermes_ping "REFUSED probe run: A0v not accepted"
    exit 2
  fi
  for g in af rg bc; do
    hermes_ping "panel probe $g starting (~1h)"
    uv run python scripts/m46_band_gate.py run "docs/specs/m46_probe_$g.json" \
      --out "runs/m46_probe_$g" --workers 8 2>&1 | tail -2
    uv run python scripts/m46_band_gate.py decode "docs/specs/m46_probe_$g.json" \
      --out "runs/m46_probe_$g" 2>&1 | tail -8
    hermes_ping "panel probe $g decoded — see runs/m46_probe_$g"
  done
fi
