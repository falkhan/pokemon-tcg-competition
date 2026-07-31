#!/usr/bin/env bash
# M38 G3 — the two retrain arms on the gen-1 clean corpus (docs/M38-plan.md
# decision 4). Corpus: data/m38_gen1_{a,b} (4000 games, 337,698 labels,
# corrected teacher, semantic bar, value_solve off).
#
#   m38_ft      : FINE-TUNE from m28_winners (pre-registered expected winner).
#                 Champion is a v3-era net; init = migrate_v3_to_v4(m28_winners)
#                 saved as checkpoints/m38_ft_init.pt — the M21 warm-start
#                 invariant (new columns zero-init, exactly the old net at
#                 init). Recipe = the champion's (epochs 10, lr 1e-4,
#                 SUPPORTER:5 STADIUM:5) so corpus+init are the only variables.
#   m38_scratch : same data + kind weights, fresh v4 net (width-driven from
#                 the shards), lr 3e-4 (the fresh-net default), epochs 10.
#
# Usage: bash scripts/m38_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=(data/m38_gen1_a data/m38_gen1_b)
KIND=(--card-kind-weight SUPPORTER:5 --card-kind-weight STADIUM:5)

if [ ! -f checkpoints/m38_ft_init.pt ]; then
  echo "=== migrate m28_winners -> v4 init $(date) ==="
  uv run python -c "
import sys, torch; sys.path.insert(0, '.')
from rl.plan_iter import migrate_v3_to_v4
m = migrate_v3_to_v4(torch.load('checkpoints/m28_winners.pt', map_location='cpu'))
torch.save(m.state_dict(), 'checkpoints/m38_ft_init.pt')
print('saved checkpoints/m38_ft_init.pt')"
fi

echo "=== train m38_ft $(date) ==="
uv run python -m rl.plan_iter train --data "${DATA[@]}" --name m38_ft \
  --init checkpoints/m38_ft_init.pt --epochs 10 --lr 1e-4 \
  "${KIND[@]}" 2>&1 | tee runs/m38_ft.log

echo "=== train m38_scratch $(date) ==="
uv run python -m rl.plan_iter train --data "${DATA[@]}" --name m38_scratch \
  --epochs 10 --lr 3e-4 "${KIND[@]}" 2>&1 | tee runs/m38_scratch.log

echo "=== M38 TRAIN DONE $(date) ==="
