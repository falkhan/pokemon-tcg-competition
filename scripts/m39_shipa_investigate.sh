#!/usr/bin/env bash
# M39 P1-inv — resolve the Ship A question, and settle `conserve` with a
# measurement instead of a judgement call. (Piotr, 2026-08-01: "investigate
# further".)
#
# What P0 left unresolved: the strip read -1.10pp +/-1.73 (z=-1.24) on the
# weighted pool at n=800/cell -- no detectable difference, with a roster MDE
# of 2.5pp. Ship A's premise ("the cheapest real gain available") is neither
# confirmed nor refuted at that power, and G-12 says a delta inside the MDE
# is "no signal", never "no effect".
#
# THREE ARMS = the three candidate ship configs, nothing else varies:
#   plain     strip everything            (decision 1 as settled)
#   conserve  strip everything BUT conserve
#   shipcfg   keep the full gacfr3 stack  (status quo, what is live)
#
# All three across the full 9-bed weighted roster at seeds 1-6 => n=2400 per
# cell, which per docs/VALIDATION.md resolves ~5pp per cell and takes the
# weighted-pool MDE from 2.5pp to ~1.4pp.
#
# Idempotent: existing complete cells are skipped, so plain/shipcfg only pay
# for the four new seeds.
set -u
cd "$(dirname "$0")/.."

NET=checkpoints/m38_w9294_cont3.pt
declare -A ARM=(
  [plain]="model:$NET:alakazam_v2_h4"
  [conserve]="model-conserve:$NET:alakazam_v2_h4"
  [shipcfg]="modelt-gacfr3:$NET:alakazam_v2_h4"
)
declare -A BED=(
  [tuned]="rule:tuned:lucario"
  [mirror]="model:checkpoints/m28_winners.pt:alakazam_v2_h4"
  [m28]="model:checkpoints/m28_winners.pt:clone54618168"
  [wall]="model:checkpoints/m38_bc_wall.pt:greattusk_wall"
  [archaludon]="model:checkpoints/m39_bc_archaludon.pt:archaludon"
  [grim]="model:checkpoints/m39_bc_grim.pt:grim_live"
  [dragapult]="rule:dragapult"
  [rocket]="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
  [iono]="rule:iono"
)
BEDS_ORDER=(tuned mirror m28 wall archaludon grim dragapult rocket iono)
COMPLETE_LINES=16

for arm in plain conserve shipcfg; do
  for bed in "${BEDS_ORDER[@]}"; do
    for seed in 1 2 3 4 5 6; do
      f="runs/m39_gate_${arm}_${bed}_s${seed}.jsonl"
      n=0; [ -f "$f" ] && n=$(grep -c results "$f" || true)
      [ "$n" -ge "$COMPLETE_LINES" ] && continue
      echo "[shipA] $arm vs $bed seed $seed  $(date)"
      rm -f "$f"
      uv run python -m rl.matchrunner play --a "${ARM[$arm]}" --b "${BED[$bed]}" \
        -n 400 --workers 8 --seed "$seed" --checkpoint "$f"
    done
  done
done
echo "=== M39 SHIP A INVESTIGATION DONE $(date) ==="
