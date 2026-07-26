#!/usr/bin/env bash
# Strength smoke gate (Rec 3, 2026-07-26): a trained / fine-tuned model MUST be
# non-inferior IN PLAY to its init before anything downstream runs. Catches the
# "val_acc looked fine, play cratered" trap that burned M18-plan4, M23, M29,
# M32-B, and M33 — five compute sinks. Plays new-vs-init on the deck (a mirror,
# so ~0.5 if unchanged); if the new model is RESOLVED-WORSE (wr < 0.5 - 1.96*SE)
# it exits 1 so a driver's `... && next_step` halts here.
#
# Usage: strength_gate.sh <new_ckpt> <init_ckpt> <deck> [n=200] [seed=1] [bar]
# Chain it: train ... && strength_gate.sh new init deck && battery ...
set -uo pipefail
cd "$(dirname "$0")/.."
NEW="${1:?new ckpt}"; INIT="${2:?init ckpt}"; DECK="${3:?deck}"
N="${4:-200}"; SEED="${5:-1}"; BAR="${6:-}"
out=$(uv run python -m rl.matchrunner play --a "model:${NEW}:${DECK}" \
      --b "model:${INIT}:${DECK}" -n "$N" --workers 8 --seed "$SEED" 2>&1 | tail -1)
echo "strength-gate (new vs init on ${DECK}): $out"
python3 - "$out" "$N" "$BAR" <<'PY'
import sys, re, math
out, n, bar = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = re.search(r"wr=([0-9.]+)", out)
if not m:
    print("STRENGTH GATE: ERROR — no wr in matchrunner output"); sys.exit(2)
wr = float(m.group(1))
bar = float(bar) if bar else 0.5 - 1.96 * math.sqrt(0.25 / n)   # non-inferior to 0.5
if wr < bar:
    print(f"STRENGTH GATE: FAIL  wr={wr:.3f} < bar={bar:.3f} "
          f"— new is resolved-WORSE than init; STOP.")
    sys.exit(1)
print(f"STRENGTH GATE: PASS  wr={wr:.3f} >= bar={bar:.3f}")
PY
