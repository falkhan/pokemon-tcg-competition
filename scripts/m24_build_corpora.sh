#!/usr/bin/env bash
# M24 Phase 0.2: rebuild replay-BC corpora after the top-10 snowball.
#   - per-teacher corpora for every sub that landed >=150 episodes (M10 law:
#     single consistent policies clone better than pooled mixtures)
#   - the strong-pilots-on-OUR-deck corpus via --deck-hash (our lucario 60,
#     hash 20dcd313..., is the 0.899-weight meta deck)
# teacher_score is stored per row, so corpora are built broad and can be
# re-filtered at train time without rebuilding.
set -euo pipefail
cd "$(dirname "$0")/.."
for sub in 54618168 54773249 54834745 54840044 54827443 54826859; do
  echo "=== build per-teacher corpus ${sub} $(date) ==="
  uv run python -m rl.replay_bc build --only-subs "${sub}" --min-score 600 \
    --out "data/bc_m24_${sub}"
done
echo "=== build our-deck corpus (hash 20dcd313, min-score 900) $(date) ==="
uv run python -m rl.replay_bc build --deck-hash 20dcd3130bc0 --min-score 900 \
  --out data/bc_m24_ourdeck
echo "=== M24 CORPORA DONE $(date) ==="
