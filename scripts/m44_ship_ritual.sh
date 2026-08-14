#!/usr/bin/env bash
# M44 Step 8 — one candidate's export→verify→QC ritual (NO SUBMIT — the stop
# for Piotr's review is after this). usage: m44_ship_ritual.sh O|K
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PAIR=${1:?pair O or K}
case "$PAIR" in
  O) CKPT=ppo_best_m44_O_r1.pt; DECK=ogerpon
     FIXES="planzero"
     ARM="model-pz:checkpoints/ppo_best_m44_O_r1.pt:ogerpon"
     # the ogerpon-lineage live-replay corpus (state width matches serve,
     # no plans -> honest 6a for a planzero bundle). REGISTERED EXCEPTION:
     # the leg's data/ppo shards carry 58.3% non-zero plan rows — the PPO
     # phase trained plan-conditioned while the bundle serves plan=0; same
     # structural property as every planzero ship since m43a. Surfaced in
     # the diary + Piotr's review packet, not hidden.
     CORPUS=data/bc_m43_oger_w143 ;;
  K) CKPT=ppo_best_m44_K_r1.pt; DECK=alakazam_v2_h4
     FIXES="conserve,racemode2,racemode4,planzero"
     ARM="model-c-pkgz:checkpoints/ppo_best_m44_K_r1.pt:alakazam_v2_h4"
     # K's net is the m41b_wide lineage (serve option width 143) — the full
     # w9294 corpus, not the w100 slice
     CORPUS=data/bc_m38_w9294 ;;
  *) echo "unknown pair $PAIR"; exit 2 ;;
esac
PREV=dist/submission_neural_20260812_184453.tar.gz   # r4: the mirror leg

echo "=== 1. export + bundle gates + tarball (NO submit)"
./build_submission.sh --checkpoint "$CKPT" --deck "$DECK" --fixes "$FIXES"

TAR=$(ls -t dist/submission_neural_*.tar.gz | head -1)
echo "=== 2. tarball deck md5-check ($TAR)"
tar -xzf "$TAR" -O deck.csv > /tmp/m44_tar_deck.csv
if ! cmp -s /tmp/m44_tar_deck.csv "decks/${DECK}.csv"; then
  echo "FATAL: tarball deck.csv != decks/${DECK}.csv"; exit 1
fi
md5sum "$TAR" "decks/${DECK}.csv"

echo "=== 3. ship_verify (--gate-arm = the gated spec token)"
# NB --checkpoint takes a BASENAME (ship_verify prepends checkpoints/)
uv run python scripts/ship_verify.py --checkpoint "$CKPT" \
  --deck "$DECK" --corpus "$CORPUS" --gate-arm "$ARM"

echo "=== 4. QC battery (mirror leg pinned to the 2026-08-12 ogerpon ship)"
uv run python scripts/qc_battery.py --prefix "m44_qc_${PAIR}" --prev "$PREV"

echo "=== 5. bespoke leg: vs the m43a alakazam ship (183326) — the matchup"
echo "===    the M44 league training targeted"
rm -rf dist/bespoke_opp && mkdir -p dist/bespoke_opp
tar -xzf dist/submission_neural_20260812_183326.tar.gz -C dist/bespoke_opp
uv run python - "$PAIR" <<'EOF'
import sys
sys.path.insert(0, ".")
from tcg.evaluation import play_games
wr, results = play_games("submission/main.py", "dist/bespoke_opp/main.py", 3,
                         replay_prefix=f"m44_qc_{sys.argv[1]}_vsm43a")
w = sum(1 for r in results if r[0] > r[1]); l = sum(1 for r in results if r[0] < r[1])
print(f"bespoke vs m43a-ship: {w}W-{l}L ({results})")
EOF
echo "M44-RITUAL-${PAIR}-DONE tar=$TAR"
