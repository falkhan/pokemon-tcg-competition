#!/usr/bin/env bash
# M33 Stage-3 A/B (overnight, unattended): does value-guided LATE closing-plan
# DAgger improve play? Two DAgger corpora identical except late (turn>=32) plans:
#   c32 = VS_MAX_TURN 32 (production: no late plans)   [CONTROL]
#   c60 = VS_MAX_TURN 60 (late closing plans included) [TREATMENT]
# Both warm-started from the SHIPPED policy migrated to v4 (m28_winners_v4), so
# strength is representative and the only difference is the late plans. Batteried
# on the deck-out beds (rocket/grim clones) + mirror vs m28. Measurement only —
# NO ship. Decode: 0 = side-a (our arm) WIN.
set -uo pipefail
cd "$(dirname "$0")/.."
GAMES="${1:-500}"
EPOCHS="${2:-6}"
VNET="${3:-checkpoints/osv3o_setupval1.pt}"   # M18 prod value net: 0.92 t32+
INIT=checkpoints/m28_winners_v4.pt
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
M28="model:checkpoints/m28_winners.pt:clone54618168"

step() { echo "=== $* $(date +%H:%M:%S) ==="; }

step "[1/5] collect c32 ($GAMES games, cap 32)"
rm -rf data/m33_dagger_c32
uv run python -m rl.plan_iter collect --mode expert --games "$GAMES" --workers 8 \
  --value-ckpt "$VNET" --vs-max-turn 32 --shard-size 200 --out data/m33_dagger_c32 || exit 1

step "[2/5] collect c60 ($GAMES games, cap 60)"
rm -rf data/m33_dagger_c60
uv run python -m rl.plan_iter collect --mode expert --games "$GAMES" --workers 8 \
  --value-ckpt "$VNET" --vs-max-turn 60 --shard-size 200 --out data/m33_dagger_c60 || exit 1

step "[3/5] train c32 (warm-start migrated m28)"
uv run python -m rl.plan_iter train --data data/m33_dagger_c32 --name m33_pi_c32 \
  --init "$INIT" --epochs "$EPOCHS" || exit 1

step "[4/5] train c60 (warm-start migrated m28)"
uv run python -m rl.plan_iter train --data data/m33_dagger_c60 --name m33_pi_c60 \
  --init "$INIT" --epochs "$EPOCHS" || exit 1

step "[5/5] battery both arms on deck-out beds"
for arm in c32 c60; do
  A="model:checkpoints/m33_pi_${arm}.pt:clone54618168"
  for s in 1 2; do
    uv run python -m rl.matchrunner play --a "$A" --b "$ROCKET" -n 200 --workers 8 \
      --seed "$s" --checkpoint "runs/m33_${arm}_rocket_s${s}.jsonl"
    uv run python -m rl.matchrunner play --a "$A" --b "$GRIM" -n 200 --workers 8 \
      --seed "$s" --checkpoint "runs/m33_${arm}_grim_s${s}.jsonl"
  done
  uv run python -m rl.matchrunner play --a "$A" --b "$M28" -n 200 --workers 8 \
    --seed 1 --checkpoint "runs/m33_${arm}_vsM28_s1.jsonl"
done

step "DECODE"
uv run python - <<'PY'
import glob, json
def wr(pat):
    res=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            d=json.loads(line)
            if "results" in d: res+=d["results"]
    if not res: return None
    w,dr=res.count(0),res.count(2)
    return (w+0.5*dr)/len(res), len(res)
print("bed          c32              c60")
for bed in ("rocket","grim","vsM28"):
    a=wr(f"runs/m33_c32_{bed}_s*.jsonl"); b=wr(f"runs/m33_c60_{bed}_s*.jsonl")
    fa=f"{a[0]:.3f} (n={a[1]})" if a else "pending"
    fb=f"{b[0]:.3f} (n={b[1]})" if b else "pending"
    print(f"{bed:12s} {fa:16s} {fb}")
print("late plans help closing => c60 > c32 on rocket/grim")
PY
step "M33 Stage-3 A/B DONE"
