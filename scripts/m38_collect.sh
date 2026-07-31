#!/usr/bin/env bash
# M38 G3 — gen-1 CLEAN collect (docs/M38-plan.md, decisions 1 + Q2).
#
# Teacher seat: the corrected solver teacher (semantic bar default) piloting
# alakazam_v2_h4 (data/m38_population.json). value_solve OFF (decision 1) —
# no --value-ckpt. Opponent rotation = the live 700-800 mix (post-mortem
# n=63 table), strongest available pilot per archetype:
#   lucario x13 -> rule:tuned  |  wall x9 -> solver:greattusk_wall
#   mirror  x8 -> self (both seats recorded)
#   archaludon x8 -> solver:archaludon
#   grim x8 -> m25 BC clone    |  dragapult x4 -> rule:dragapult
#   rocket x4 -> m30 BC clone
# Only the teacher (alakazam) seat is recorded vs external opponents (M14) —
# the corpus is exactly "the ship deck facing the live field".
#
# TWO FRESH DIRS (collect has NO resume; a crash loses one half, not all):
#   data/m38_gen1_a (seed 11) + data/m38_gen1_b (seed 12), 2000 games each.
# Champion-corpus floor is ~13k rows; this yields ~130k. Workers 8 (CAP).
#
# Usage: bash scripts/m38_collect.sh [games_per_half]
set -u
cd "$(dirname "$0")/.."
N="${1:-2000}"

GRIM="model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv"
ROCKET="model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv"
OPPONENTS=()
for i in $(seq 13); do OPPONENTS+=("rule:tuned:lucario"); done
for i in $(seq 9);  do OPPONENTS+=("solver:greattusk_wall"); done
for i in $(seq 8);  do OPPONENTS+=("self"); done
for i in $(seq 8);  do OPPONENTS+=("solver:archaludon"); done
for i in $(seq 8);  do OPPONENTS+=("$GRIM"); done
for i in $(seq 4);  do OPPONENTS+=("rule:dragapult"); done
for i in $(seq 4);  do OPPONENTS+=("$ROCKET"); done

for half in a b; do
  out="data/m38_gen1_${half}"
  seed=$([ "$half" = a ] && echo 11 || echo 12)
  if ls "$out"/*.npz >/dev/null 2>&1; then
    echo "[m38 G3] $out already holds shards — NO resume; skipping (move it to redo)"
    continue
  fi
  echo "[m38 G3] collect half $half -> $out (n=$N, seed $seed)"
  uv run python -m rl.plan_iter collect --mode expert --games "$N" \
    --decks data/m38_population.json --out "$out" \
    --workers 8 --seed "$seed" --shard-size 100 \
    --opponents "${OPPONENTS[@]}"
done
echo "DONE"
