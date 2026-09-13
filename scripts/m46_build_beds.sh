#!/usr/bin/env bash
# M46 A1+A2 — band-faithful clone beds from 1100+/1000/900 teacher corpora
# (docs/M46-plan.md Track A; template scripts/m43_rebuild.sh, panel law
# scripts/m39_build_panels.sh — d1/d2/d3 = 10/9/8 epochs, init m28_winners,
# lr 3e-4, no seed = the G-13 lottery draw).
#
# Families and bands (docs/M46-plan.md evidence pin 1):
#   grim1100    3121746f @1100 winners-only   (781 seats / 461 winners list-scope)
#   mirror1100  9294d9d8 @1100 winners-only   (936 / 532)
#   drag1100    f2b4039a + 7c605fb2 + 3631d393 @1100 (pooled ~330; multi-dir train)
#   stall1000   e234578d @1000                (first stall bed ever; 136 list-exact)
#   wall900     6f87e9d4 + e510e4d4 @900      (172 / 72; thin — accept or snowball)
#
# WIDTH LAW (M41b append-and-slice): corpora build at native width 143 and are
# SLICED to 100 for bed training (m28_winners is a w100 net).
#
# Soft-fail per corpus: a thin family logging zero shards pings Hermes and the
# run continues — the band panel's decode refuses missing cells anyway, so a
# hole is caught downstream, and one dead list must not kill the night.
# Idempotent: existing corpora/checkpoints are skipped — delete to redraw.
# Run from repo root: bash scripts/m46_build_beds.sh
set -u
cd "$(dirname "$0")/.."

DEV="${M46_DEVICE:-auto}"
FAILED=()

hermes_ping () { hermes send -t telegram "[m46-beds] $1" 2>/dev/null || true; }

export_deck () {    # hash  (deck_hash pins the exact list, so the vote is
                    # trivial — default --min-score 700 keeps seat counts high)
  local hash="$1"
  if ls data/kaggle/*_"$hash"_deck.csv >/dev/null 2>&1; then
    echo "=== deck $hash exists — skipping export"; return
  fi
  echo "=== export deck $hash"
  if ! uv run python scripts/export_opp_deck.py --hash "$hash" \
      2>&1 | tee -a runs/m46_deck_exports.log | tail -2; then
    echo "=== EXPORT FAILED: $hash"
    FAILED+=("deck:$hash"); hermes_ping "A1 deck export FAILED: $hash"
  fi
}

build_corpus () {   # out  hash  min-score  extra-flags...
  local out="$1" hash="$2" band="$3"; shift 3
  # shards are the done-marker — a failed build can leave deck_registry.json
  # behind with zero shards (the m43 dragapult sync-gap did exactly that)
  if ls "$out"/*.npz >/dev/null 2>&1; then
    echo "=== $out exists — skipping build"; return
  fi
  rm -rf "$out"
  echo "=== build $out (hash $hash, min-score $band) $(date)"
  uv run python -m rl.replay_bc build --deck-hash "$hash" \
    --min-score "$band" --hand-aware --out "$out" "$@" \
    2>&1 | tee "runs/m46_build_$(basename "$out").log" | tail -3
  if ! ls "$out"/*.npz >/dev/null 2>&1; then
    echo "=== SOFT-FAIL: $out built ZERO shards (thin band or dead hash)"
    FAILED+=("corpus:$out"); hermes_ping "A1 SOFT-FAIL: $out zero shards"
  fi
}

slice_corpus () {   # src dst
  ls "$1"/*.npz >/dev/null 2>&1 || return 0   # nothing to slice (soft-failed)
  if ls "$2"/*.npz >/dev/null 2>&1; then
    echo "=== $2 exists — skipping slice"; return
  fi
  uv run python scripts/m41b_slice_corpus.py "$1" "$2" --width 100
}

train_bed () {      # name  epochs  data-dir...
  local name="$1" ep="$2"; shift 2
  local ck="checkpoints/$name.pt" dirs=()
  for d in "$@"; do ls "$d"/*.npz >/dev/null 2>&1 && dirs+=("$d"); done
  if [ "${#dirs[@]}" -eq 0 ]; then
    echo "=== SKIP $ck: no corpus dirs with shards"
    FAILED+=("bed:$name"); return
  fi
  if [ -f "$ck" ]; then echo "=== $ck exists — skipping"; return; fi
  echo "=== train $ck ($ep epochs, data: ${dirs[*]}) $(date)"
  uv run python -m rl.plan_iter train --data "${dirs[@]}" --name "$name" \
    --init checkpoints/m28_winners.pt --epochs "$ep" --lr 3e-4 \
    --device "$DEV" 2>&1 | tee "runs/m46_bed_$name.log" | tail -2
}

# ---------------------------------------------------------------- A1 corpora
hermes_ping "A1 start: 5 deck exports + 7 corpus builds"

export_deck e234578d   # stall (dunsparce/fan-rotom)
export_deck 6f87e9d4   # wall (kanga)
export_deck e510e4d4   # wall (crustle)
export_deck f2b4039a   # dragapult primary
export_deck 7c605fb2   # dragapult secondary

build_corpus data/bc_m46_grim1100_w143   3121746f 1100 --winners-only
slice_corpus data/bc_m46_grim1100_w143   data/bc_m46_grim1100
build_corpus data/bc_m46_mirror1100_w143 9294d9d8 1100 --winners-only
slice_corpus data/bc_m46_mirror1100_w143 data/bc_m46_mirror1100
build_corpus data/bc_m46_drag_a_w143     f2b4039a 1100
slice_corpus data/bc_m46_drag_a_w143     data/bc_m46_drag_a
build_corpus data/bc_m46_drag_b_w143     7c605fb2 1100
slice_corpus data/bc_m46_drag_b_w143     data/bc_m46_drag_b
build_corpus data/bc_m46_drag_c_w143     3631d393 1100
slice_corpus data/bc_m46_drag_c_w143     data/bc_m46_drag_c
build_corpus data/bc_m46_stall1000_w143  e234578d 1000
slice_corpus data/bc_m46_stall1000_w143  data/bc_m46_stall1000
build_corpus data/bc_m46_wall_a_w143     6f87e9d4 900
slice_corpus data/bc_m46_wall_a_w143     data/bc_m46_wall_a
build_corpus data/bc_m46_wall_b_w143     e510e4d4 900
slice_corpus data/bc_m46_wall_b_w143     data/bc_m46_wall_b

hermes_ping "A1 corpora done (failures: ${#FAILED[@]}) — starting A2 draws"

# ------------------------------------------------- A2 G-13 draws (15 beds)
for d in 1:10 2:9 3:8; do
  ep="${d#*:}"; n="${d%%:*}"
  train_bed "m46_bed_grim_d$n"   "$ep" data/bc_m46_grim1100
  train_bed "m46_bed_mirror_d$n" "$ep" data/bc_m46_mirror1100
  train_bed "m46_bed_drag_d$n"   "$ep" data/bc_m46_drag_a data/bc_m46_drag_b data/bc_m46_drag_c
  train_bed "m46_bed_stall_d$n"  "$ep" data/bc_m46_stall1000
  train_bed "m46_bed_wall_d$n"   "$ep" data/bc_m46_wall_a data/bc_m46_wall_b
done

# ------------------------------------------------------------------ summary
echo "=== M46 BEDS DONE $(date)"
echo "=== corpora yields (shards / rows via deck_registry):"
for c in grim1100 mirror1100 drag_a drag_b drag_c stall1000 wall_a wall_b; do
  n=$(ls "data/bc_m46_${c}_w143"/*.npz 2>/dev/null | wc -l)
  echo "  $c: $n shards"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "=== FAILURES: ${FAILED[*]}"
  hermes_ping "A1+A2 DONE WITH FAILURES: ${FAILED[*]}"
  exit 1
fi
hermes_ping "A1+A2 DONE clean: 8 corpora + 15 m46 beds"
