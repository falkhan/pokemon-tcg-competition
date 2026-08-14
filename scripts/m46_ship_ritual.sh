#!/usr/bin/env bash
# M46 export→verify→QC ritual (NO SUBMIT — the stop for Piotr's replay
# review + explicit go + deck confirmation is AFTER this).
# usage: m46_ship_ritual.sh bank|cand [extra-fixes]
#
#   bank  the early guards-only ship (Piotr's 08-14 decision): the live-810
#         m41b bundle + dudguard0 as the single variable.
#   cand  the panel candidate: bank string + the guards the B2-B4 probes
#         adopted — pass them as $2 (comma-joined, e.g. "attackfloor"),
#         with the matching gate-arm token in $3 (ship_verify check 3b is
#         literal set equality, so the token must exist in
#         rl/matchrunner._MODEL_FIX_KINDS).
#
# Template: scripts/m44_ship_ritual.sh. M46 differences: prev-ship mirror
# leg PINNED to the K alakazam tarball (docs/M46-plan.md ship path — the
# sorted(dist)[-2] default would self-mirror after our own export), corpus
# = the w143 champion BC corpus (planzero bundle, check 6a needs <=1%
# non-zero-plan rows — NOT the PPO shards), and a bespoke STALL leg (first
# ever; live stall wr 0.00) vs the m46 stall bed, labeled uncalibrated.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

ARM_KIND=${1:?bank or cand}
CKPT=m41b_wide_prod.pt
DECK=alakazam_v2_h4
CORPUS=data/bc_m38_w9294
PREV=dist/submission_neural_20260813_145528.tar.gz   # K alakazam — PINNED

case "$ARM_KIND" in
  bank) FIXES="conserve,planzero,ash,ashguard,dudguard0"
        ARM="model-cz-ashw-dg0:checkpoints/m41b_wide_prod.pt:alakazam_v2_h4" ;;
  cand) EXTRA=${2:?cand needs the adopted fix list}
        TOKEN=${3:?cand needs the gate-arm token}
        FIXES="conserve,planzero,ash,ashguard,dudguard0,${EXTRA}"
        ARM="${TOKEN}:checkpoints/m41b_wide_prod.pt:alakazam_v2_h4" ;;
  *) echo "unknown arm $ARM_KIND"; exit 2 ;;
esac

hermes_ping () { hermes send -t telegram "[m46-ritual] $1" 2>/dev/null || true; }
hermes_ping "$ARM_KIND ritual starting (fixes: $FIXES)"

echo "=== 0. ci_gate (early, per plan — not at 22:00)"
uv run python scripts/ci_gate.py

echo "=== 1. export + bundle gates + tarball (NO submit)"
./build_submission.sh --checkpoint "$CKPT" --deck "$DECK" --fixes "$FIXES"

TAR=$(ls -t dist/submission_neural_*.tar.gz | head -1)
echo "=== 2. tarball deck md5-check ($TAR)"
tar -xzf "$TAR" -O deck.csv > /tmp/m46_tar_deck.csv
if ! cmp -s /tmp/m46_tar_deck.csv "decks/${DECK}.csv"; then
  echo "FATAL: tarball deck.csv != decks/${DECK}.csv"; hermes_ping "FATAL: tarball deck mismatch"; exit 1
fi
md5sum "$TAR" "decks/${DECK}.csv"

echo "=== 3. ship_verify (--gate-arm = the gated spec token)"
uv run python scripts/ship_verify.py --checkpoint "$CKPT" \
  --deck "$DECK" --corpus "$CORPUS" --gate-arm "$ARM"

echo "=== 4. QC battery (mirror leg pinned to the 2026-08-13 K alakazam ship)"
uv run python scripts/qc_battery.py --prefix "m46_qc_${ARM_KIND}" --prev "$PREV" --skip-ci-gate

echo "=== 5. bespoke STALL leg (uncalibrated opponent — bed built 08-14)"
uv run python - "$ARM_KIND" <<'EOF'
import sys
sys.path.insert(0, ".")
from tcg.evaluation import play_games
wr, results = play_games("submission/main.py", "dist/qc_beds/stall/main.py", 3,
                         replay_prefix=f"m46_qc_{sys.argv[1]}_vsstall")
w = sum(1 for r in results if r[0] > r[1]); l = sum(1 for r in results if r[0] < r[1])
print(f"bespoke vs stall bed (UNCALIBRATED): {w}W-{l}L ({results})")
EOF

hermes_ping "$ARM_KIND ritual DONE (tar=$(basename "$TAR")) — WAITING ON PIOTR: replay review + explicit go + deck confirmation"
echo "M46-RITUAL-${ARM_KIND}-DONE tar=$TAR"
echo "STOP: replays in replays/ — Piotr's manual review + explicit go + deck"
echo "confirmation required before ANY submit."
