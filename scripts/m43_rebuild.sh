#!/usr/bin/env bash
# M43 R2-R4 — rebuild the gate-bed + lane-start artifact chain on a fresh box
# (docs/M43-plan.md 2026-08-12 amendment; recipes from docs/M38.md:349-352,
# docs/M39.md:551-553,669-671, docs/M41b.md:483-486,644-658 and the panel law
# in scripts/m39_build_panels.sh — d1/d2/d3 = 10/9/8 epochs, init m28_winners,
# lr 3e-4, no seed = the G-13 lottery draw).
#
# WIDTH LAW (M41b append-and-slice): the current encoder emits width-143
# option vectors, the historical beds are width-100 nets warm-started from
# m28_winners. Every corpus is built at the native width and SLICED to 100
# for bed training (byte-exact per scripts/m42_column_safety.py); the two
# w143 corpora (champion, ogerpon) stay wide for the lane retrains.
#
# Idempotent: existing corpora/checkpoints are skipped — delete to redraw.
# Run AFTER `rl.kaggle_ingest refresh` + `harvest` (fresh parquet), from repo
# root: bash scripts/m43_rebuild.sh
set -eu
cd "$(dirname "$0")/.."

DEV="${M43_DEVICE:-auto}"      # --device for trains; auto = cuda if available

hermes_ping () { hermes send -t telegram "[m43-rebuild] $1" 2>/dev/null || true; }

build_corpus () {   # out  hash  min-score  extra-flags...
  local out="$1" hash="$2" band="$3"; shift 3
  # shards are the done-marker — a failed build can leave deck_registry.json
  # behind with zero shards (the dragapult sync-gap did exactly that)
  if ls "$out"/*.npz >/dev/null 2>&1; then
    echo "=== $out exists — skipping build"; return
  fi
  rm -rf "$out"
  echo "=== build $out (hash $hash, min-score $band) $(date)"
  uv run python -m rl.replay_bc build --deck-hash "$hash" \
    --min-score "$band" --hand-aware --out "$out" "$@" \
    2>&1 | tee "runs/m43_build_$(basename "$out").log" | tail -3
  if ! ls "$out"/*.npz >/dev/null 2>&1; then
    echo "=== FATAL: $out built ZERO shards — check hash/band against opp_decks"
    hermes_ping "R2 FAILED: $out built zero shards"
    exit 1
  fi
}

slice_corpus () {   # src dst
  if ls "$2"/*.npz >/dev/null 2>&1; then
    echo "=== $2 exists — skipping slice"; return
  fi
  uv run python scripts/m41b_slice_corpus.py "$1" "$2" --width 100
}

train_bed () {      # name  data-dir  epochs
  local ck="checkpoints/$1.pt"
  if [ -f "$ck" ]; then echo "=== $ck exists — skipping"; return; fi
  echo "=== train $ck ($3 epochs) $(date)"
  uv run python -m rl.plan_iter train --data "$2" --name "$1" \
    --init checkpoints/m28_winners.pt --epochs "$3" --lr 3e-4 \
    --device "$DEV" 2>&1 | tee "runs/m43_bed_$1.log" | tail -2
}

# ---------------------------------------------------------------- R2 corpora
hermes_ping "R2 corpora: building"

# champion (M28 recipe: 800+, winners-only) — E0 reads data/bc_m38_w9294
build_corpus data/bc_m38_w9294      9294d9d8 800 --winners-only
slice_corpus data/bc_m38_w9294      data/bc_m38_w9294_w100
# wall (M38: 600+, hash d3e4d16c; decks/greattusk_wall.csv == that hash)
build_corpus data/bc_m38_wall_w143  d3e4d16c 600
slice_corpus data/bc_m38_wall_w143  data/bc_m38_wall
# grim band bed (M39: 3121746f at 700+; the historical 700-850 band was the
# era's leaderboard cap, not a filter — today's corpus also contains 1000+
# seats, diaried as a known band-blur vs topgrim)
build_corpus data/bc_m39_grim_w143  3121746f 700
slice_corpus data/bc_m39_grim_w143  data/bc_m39_grim
# topgrim (M39 P0.9: same list at 1000+)
build_corpus data/bc_m39_topgrim_w143 3121746f 1000
slice_corpus data/bc_m39_topgrim_w143 data/bc_m39_topgrim
# archaludon — dominant hash at the 700 band (recipe underspecified in M39;
# reconstruction pre-registered in the amendment)
ARCH_HASH="$(uv run python scripts/export_opp_deck.py --family archaludon \
  --min-score 700 2>/dev/null | grep -oP 'hash \K[0-9a-f]{8}' | head -1)"
echo "=== archaludon dominant hash: $ARCH_HASH"
build_corpus data/bc_m39_arch_w143  "$ARCH_HASH" 700
slice_corpus data/bc_m39_arch_w143  data/bc_m39_arch_a
# loss families (m40_build_lossfam_beds.sh recipe: 900+, ALL seats)
build_corpus data/bc_m40_dragapult_w143 3631d393 900
slice_corpus data/bc_m40_dragapult_w143 data/bc_m40_dragapult
build_corpus data/bc_m40_garchomp_w143  c7b3253f 900
slice_corpus data/bc_m40_garchomp_w143  data/bc_m40_garchomp
build_corpus data/bc_m40_rocket_w143    59e27a5e 900
slice_corpus data/bc_m40_rocket_w143    data/bc_m40_rocket
# ogerpon live corpus (M41b: 900+, hash 356c16bd, NOT winners-only) — stays
# WIDE; Lane B's B3 retrain and the --init-wide width sniff both need w143
build_corpus data/bc_m43_oger_w143  356c16bd 900

# ------------------------------------------------------------- R3 gate beds
hermes_ping "R3 beds: training 21 draws"

train_bed m38_bc_wall        data/bc_m38_wall     10
train_bed m39_bed_wall_d2    data/bc_m38_wall      9
train_bed m39_bed_wall_d3    data/bc_m38_wall      8
train_bed m39_bc_grim        data/bc_m39_grim     10
train_bed m39_bc_grim_b      data/bc_m39_grim      9
train_bed m39_bed_grim_d3    data/bc_m39_grim      8
train_bed m39_bc_topgrim     data/bc_m39_topgrim  10
train_bed m40_bed_topgrim_d2 data/bc_m39_topgrim   9
train_bed m40_bed_topgrim_d3 data/bc_m39_topgrim   8
train_bed m39_bc_archaludon  data/bc_m39_arch_a   10
train_bed m39_bed_arch_d2    data/bc_m39_arch_a    9
train_bed m39_bed_arch_d3    data/bc_m39_arch_a    8
for fam in dragapult garchomp rocket; do
  train_bed "m40_bed_${fam}_d1" "data/bc_m40_${fam}" 10
  train_bed "m40_bed_${fam}_d2" "data/bc_m40_${fam}"  9
  train_bed "m40_bed_${fam}_d3" "data/bc_m40_${fam}"  8
done

# ------------------------------------------------- R4 lane start (champion)
hermes_ping "R4 lane starts: cont3 -> wide"

if [ ! -f checkpoints/m38_w9294_cont3.pt ]; then
  echo "=== train cont3 (champion recipe: 3 ep, lr 1e-4, S:5 St:5) $(date)"
  uv run python -m rl.plan_iter train --data data/bc_m38_w9294_w100 \
    --name m38_w9294_cont3 --init checkpoints/m28_winners.pt \
    --epochs 3 --lr 1e-4 \
    --card-kind-weight SUPPORTER:5 --card-kind-weight STADIUM:5 \
    --device "$DEV" 2>&1 | tee runs/m43_cont3_rebuild.log | tail -4
else
  echo "=== cont3 exists — skipping"
fi

if [ ! -f checkpoints/m41b_wide_prod.pt ]; then
  echo "=== train m41b_wide_prod (M41b: 14 ep, seed 3, --init-wide cont3) $(date)"
  uv run python -m rl.plan_iter train --data data/bc_m38_w9294 \
    --name m41b_wide_prod --init-wide checkpoints/m38_w9294_cont3.pt \
    --epochs 14 --seed 3 --lr 3e-4 --outcome-weight 0.25 \
    --device "$DEV" 2>&1 | tee runs/m43_wide_prod_rebuild.log | tail -4
else
  echo "=== m41b_wide_prod exists — skipping"
fi

hermes_ping "R2-R4 DONE: corpora + 21 beds + cont3 + wide_prod rebuilt"
echo "=== M43 REBUILD DONE $(date)"
echo "next: R5 bestresp corpora (scripts/m39_collect_bestresp.py --bed {wall,grim,archaludon})"
echo "      then R4 anchor: matchrunner wide_prod vs m39_bc_grim n=800"
