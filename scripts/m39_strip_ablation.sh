#!/usr/bin/env bash
# M39 P0.7 — per-rule strip ablation, the evidence base for decision 1.
#
# G5 (M38) measured the shipped stack only in AGGREGATE:
#     plain .323  >  gacf .278  >  gacfr3 .254     (wall bed)
# and its own diary entry says per-rule attribution beyond racemode3 is
# INCOMPLETE. So "strip everything" was an inference from an aggregate, not
# a measurement. This decomposes it: leave-one-out from the shipped stack,
# plus the plain and full pins.
#
# Shipped stack (submission/main.py:113):
#     telepath,deckguard,ash,conserve,benchfloor,racemode3
# NOTE gacf = deckGuard/Ash/Conserve/benchFloor — the "keep the economy
# rules" option is NOT a conservative reading of G5, it is an UNMEASURED
# config. That is what this battery exists to fix.
#
# Net: m38_w9294_cont3 (the M38 ship net) — we are deciding ITS fix string.
# Beds: wall (where G5 found the harm) + top (P0.8's 900+ instrument, which
# independently reproduced the stack cost at -5.0pp, z=+2.00).
#
# Expected outcome per G5: strip all. This battery exists to catch the case
# where that expectation is wrong for a specific rule.
#
# Usage: bash scripts/m39_strip_ablation.sh
set -u
cd "$(dirname "$0")/.."

NET=checkpoints/m38_w9294_cont3.pt
declare -A ARM=(
  [plain]="model:$NET:alakazam_v2_h4"
  [full]="modelt-gacfr3:$NET:alakazam_v2_h4"
  [no_racemode3]="modelt-gacf:$NET:alakazam_v2_h4"
  [no_telepath]="abl-no-telepath:$NET:alakazam_v2_h4"
  [no_deckguard]="abl-no-deckguard:$NET:alakazam_v2_h4"
  [no_ash]="abl-no-ash:$NET:alakazam_v2_h4"
  [no_conserve]="abl-no-conserve:$NET:alakazam_v2_h4"
  [no_benchfloor]="abl-no-benchfloor:$NET:alakazam_v2_h4"
)
ARMS_ORDER=(plain full no_racemode3 no_telepath no_deckguard no_ash
            no_conserve no_benchfloor)
declare -A BED=(
  [wall]="model:checkpoints/m38_bc_wall.pt:greattusk_wall"
  [top]="model:checkpoints/m39_bc_top.pt:clone54618168"
)

for arm in "${ARMS_ORDER[@]}"; do
  for bed in wall top; do
    for seed in 1 2; do
      echo "[m39 abl] $arm vs $bed seed $seed  $(date)"
      uv run python -m rl.matchrunner play --a "${ARM[$arm]}" \
        --b "${BED[$bed]}" -n 400 --workers 8 --seed "$seed" \
        --checkpoint "runs/m39_abl_${arm}_${bed}_s${seed}.jsonl"
    done
  done
done
echo "=== M39 STRIP ABLATION DONE $(date) ==="
