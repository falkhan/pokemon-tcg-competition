#!/usr/bin/env bash
# M39 P3 — cheap SCREEN before the full panel gate.
#
# Six policy arms x 72 panel cells is ~100 minutes of battery. Two of the
# corpus-B arms carry a visible dose risk before a single game is played
# (init val_acc 0.935 on their own corpus, falling to ~0.80 after training —
# a 13% behavioural rewrite, far past the ~1-epoch dose law), so a 4-cell
# screen at n=400 that kills the obviously-broken arms is worth 4 minutes.
#
# The screen is a KILL filter only. It resolves ~10pp at n=400 (G-12), so it
# may never promote an arm or rank the survivors — that is the panel gate's
# job. Beds are one draw per family deliberately: a screen does not get the
# G-13 treatment because it is not producing a number anyone will quote.
set -u
cd "$(dirname "$0")/.."

declare -A BED=(
  [tuned]="rule:tuned:lucario"
  [m28]="model:checkpoints/m28_winners.pt:clone54618168"
  [wall_d1]="model:checkpoints/m38_bc_wall.pt:greattusk_wall"
  [grim_d1]="model:checkpoints/m39_bc_grim.pt:grim_live"
)
for arm in "$@"; do
  ck="checkpoints/m39_${arm}.pt"
  [ -f "$ck" ] || { echo "[screen] SKIP $arm"; continue; }
  for bed in tuned m28 wall_d1 grim_d1; do
    f="runs/m39_scr_${arm}_${bed}_s1.jsonl"
    [ -f "$f" ] && continue
    uv run python -m rl.matchrunner play \
      --a "model-conserve:$ck:alakazam_v2_h4" --b "${BED[$bed]}" \
      -n 400 --workers 6 --seed 1 --checkpoint "$f"
  done
done
echo "=== M39 P3 SCREEN DONE $(date) ==="
