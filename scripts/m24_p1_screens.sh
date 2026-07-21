#!/usr/bin/env bash
# M24 Phase 1 screens: both C-clone variants x both decks (the A/B grid),
# n=200 seed 1 vs rule:lucario. Decode 0=WIN. Full battery follows on winners.
set -euo pipefail
cd "$(dirname "$0")/.."
CLONE_DECK="data/kaggle/clone_54618168_deck.csv"
for ckpt in m24_bc_54618168 m24_bc_pool9294; do
  for deck in clone ours; do
    if [ "${deck}" = "clone" ]; then D="${CLONE_DECK}"; else D="lucario"; fi
    tag="m24_scr_${ckpt#m24_bc_}_${deck}"
    echo "=== ${tag} $(date) ==="
    uv run python -m rl.matchrunner play \
      --a "model:checkpoints/${ckpt}.pt:${D}" --b rule:lucario \
      -n 200 --workers 8 --seed 1 \
      --checkpoint "runs/${tag}.jsonl"
  done
done
echo "=== M24 P1 SCREENS DONE $(date) ==="
