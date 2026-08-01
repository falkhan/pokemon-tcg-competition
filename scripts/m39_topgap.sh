#!/usr/bin/env bash
# M39 P0.8 — THE INSTRUMENT gap read (docs/M39-plan.md P0.8, docs/M39.md).
#
# The question: how far below the 900+ band are we, actually? Every bed we
# owned before this one was a 700-850 clone, so "0-6 above 800" was our
# entire knowledge of the target band — a 6-game, zero-win sample equally
# consistent with "5pp short" and "hopeless". Those imply different
# milestones, so we measure it before spending the milestone.
#
# Bed: m39_bc_top = BC clone of 329 seats on our own list (9294d9d8) at
# leaderboard score >= 900, piloting that same list (decks/clone54618168.csv).
# val_acc 0.682 — the wall clone that proved faithful was 0.676.
#
# Candidates (the three baselines every M39 arm is judged against):
#   champion   m28_winners plain          the M37 net
#   cont3      m38_w9294_cont3 plain      the M38 net, no rules
#   shipcfg    cont3 + gacfr3             what is live right now (55146658)
#
# Pre-registered readings (plan P0.8) on the pooled champion/cont3 gap:
#   <=10pp   the 800+ band is reachable by closing known loss mass
#   10-25pp  imitation gets us ~850; the 1000 attempt needs P3-B to carry it
#   >25pp    the corpus cannot teach the target band -> M40 is self-play
#
# Usage: bash scripts/m39_topgap.sh [candidates...]   (default: all three)
set -u
cd "$(dirname "$0")/.."

declare -A CAND=(
  [champion]="model:checkpoints/m28_winners.pt:alakazam_v2_h4"
  [cont3]="model:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
  [shipcfg]="modelt-gacfr3:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4"
)
TOP_BED="model:checkpoints/m39_bc_top.pt:clone54618168"

CANDS=("$@"); [ ${#CANDS[@]} -eq 0 ] && CANDS=(champion cont3 shipcfg)

for cand in "${CANDS[@]}"; do
  for seed in 1 2; do
    echo "[m39 topgap] $cand vs m39_bc_top seed $seed  $(date)"
    uv run python -m rl.matchrunner play --a "${CAND[$cand]}" \
      --b "$TOP_BED" -n 400 --workers 8 --seed "$seed" \
      --checkpoint "runs/m39_topgap_${cand}_s${seed}.jsonl"
  done
done
echo "=== M39 TOPGAP DONE $(date) ==="
