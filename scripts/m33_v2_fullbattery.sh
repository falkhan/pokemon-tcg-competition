#!/usr/bin/env bash
# Complete the V2 deck-tech battery: check for regressions on the beds a deck
# change could hurt (lucario 0.72, archaludon, kyogre floor). base vs V2 only.
set -uo pipefail
cd "$(dirname "$0")/.."
arm_deck() { case "$1" in base) echo clone54618168;; v2) echo alakazam_v2;; esac; }
for arm in base v2; do
  A="model:checkpoints/m28_winners.pt:$(arm_deck "$arm")"
  for s in 1 2; do
    uv run python -m rl.matchrunner play --a "$A" --b rule:lucario        -n 200 --workers 8 --seed "$s" --checkpoint "runs/dt_${arm}_lucario_s${s}.jsonl"
    uv run python -m rl.matchrunner play --a "$A" --b generic:archaludon  -n 200 --workers 8 --seed "$s" --checkpoint "runs/dt_${arm}_arch_s${s}.jsonl"
  done
  uv run python -m rl.matchrunner play --a "$A" --b random:kyogre -n 200 --workers 8 --seed 1 --checkpoint "runs/dt_${arm}_kyo_s1.jsonl"
done
echo "=== V2 FULL DECODE ==="
uv run python - <<'PY'
import glob, json
def wr(pat):
    r=[]
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            d=json.loads(line)
            if "results" in d: r+=d["results"]
    return ((r.count(0)+0.5*r.count(2))/len(r), len(r)) if r else (None,0)
print(f"{'bed':10s} {'base':16s} {'V2':16s}  delta")
for bed in ("rocket","grim","mirror","lucario","arch","kyo"):
    pb,nb=wr(f"runs/dt_base_{bed}_s*.jsonl"); pv,nv=wr(f"runs/dt_v2_{bed}_s*.jsonl")
    if pb is None or pv is None: print(f"{bed:10s} pending"); continue
    print(f"{bed:10s} {pb:.3f} (n={nb})    {pv:.3f} (n={nv})   {pv-pb:+.3f}")
PY
echo "=== V2 FULL BATTERY DONE $(date) ==="
