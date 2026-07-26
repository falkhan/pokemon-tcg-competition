#!/usr/bin/env bash
# M33 deck-tech early-dig probe (Rec 5): same pilot (m28_winners), 3 decks —
# base clone54618168, V1 (draw depth), V2 (setup speed) — batteried on the
# deck-out beds. Caveat: m28 is trained ON clone54618168, so V1/V2 are ZERO-SHOT
# deck transfer (M24: deck+pilot are a unit). => only a POSITIVE resolves (a
# variant that beats base DESPITE the transfer penalty); flat/worse is ambiguous.
# Decode: 0 = side-a (our arm) WIN.
set -uo pipefail
cd "$(dirname "$0")/.."
GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
MIRROR="model:checkpoints/m28_winners.pt:clone54618168"
arm_deck() { case "$1" in base) echo clone54618168;; v1) echo alakazam_v1;; v2) echo alakazam_v2;; esac; }

for arm in base v1 v2; do
  A="model:checkpoints/m28_winners.pt:$(arm_deck "$arm")"
  for s in 1 2; do
    uv run python -m rl.matchrunner play --a "$A" --b "$ROCKET" -n 200 --workers 8 --seed "$s" --checkpoint "runs/dt_${arm}_rocket_s${s}.jsonl"
    uv run python -m rl.matchrunner play --a "$A" --b "$GRIM"   -n 200 --workers 8 --seed "$s" --checkpoint "runs/dt_${arm}_grim_s${s}.jsonl"
    uv run python -m rl.matchrunner play --a "$A" --b "$MIRROR" -n 200 --workers 8 --seed "$s" --checkpoint "runs/dt_${arm}_mirror_s${s}.jsonl"
  done
done

echo "=== DECK-TECH DECODE ==="
uv run python - <<'PY'
import glob, json
def wr(pat):
    r=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            d=json.loads(line)
            if "results" in d: r+=d["results"]
    return ((r.count(0)+0.5*r.count(2))/len(r), len(r)) if r else (None, 0)
print(f"{'bed':8s} {'base':17s} {'v1 (draw)':17s} {'v2 (speed)':17s}")
for bed in ("rocket","grim","mirror"):
    cells=[]
    for arm in ("base","v1","v2"):
        p,n = wr(f"runs/dt_{arm}_{bed}_s*.jsonl")
        cells.append(f"{p:.3f}(n={n})" if p is not None else "pending")
    print(f"{bed:8s} {cells[0]:17s} {cells[1]:17s} {cells[2]:17s}")
print("deck helps => v1/v2 > base on rocket/grim despite the zero-shot transfer penalty")
PY
echo "=== DECK-TECH DONE $(date) ==="
