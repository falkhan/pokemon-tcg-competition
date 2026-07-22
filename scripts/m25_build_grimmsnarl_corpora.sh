#!/usr/bin/env bash
# M25 Phase 1.1: replay-BC corpora for the Grimmsnarl-mill teacher (deck 3121746f,
# the 1268-score cluster; our worst live matchup AND the field's #2 archetype).
# Subs are three arms of one team playing the identical deck:
#   54861775 (171 eps, max 1268, wr .626)  54863653 (255 eps, 1196, wr .620)
#   54861685 (126 eps, 1153, wr .516 — excluded from per-sub builds, in pooled only)
# M24 law: single-teacher clones beat pooled — per-sub corpora first, pooled as backup.
# teacher_score is stored per row, so corpora re-filter at train time.
set -euo pipefail
cd "$(dirname "$0")/.."
for sub in 54861775 54863653; do
  echo "=== build per-teacher grimmsnarl corpus ${sub} $(date) ==="
  uv run python -m rl.replay_bc build --only-subs "${sub}" --min-score 600 \
    --out "data/bc_m25_grimmsnarl_${sub}"
done
echo "=== build pooled 3-sub grimmsnarl corpus $(date) ==="
uv run python -m rl.replay_bc build --only-subs 54861775 54863653 54861685 \
  --min-score 600 --out data/bc_m25_grimmsnarl_pooled
echo "=== M25 GRIMMSNARL CORPORA DONE $(date) ==="
