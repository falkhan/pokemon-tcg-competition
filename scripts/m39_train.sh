#!/usr/bin/env bash
# M39 P3 — the policy lane. Low-dose fine-tunes on the cont3 lineage.
#
# Vehicle (the campaign's one repeatedly-positive training result): warm-start
# the SHIPPED net, spend ~1 epoch on a small corpus of strong same-deck live
# seats, take the best-val checkpoint. M38 established the dose law in both
# directions on the same day — 10 epochs regressed, ~1 epoch gained — so the
# arms below run 3 epochs and let best-val select, exactly as cont3 did
# (cont3's best-val landed on epoch 1).
#
# Recipe is held fixed at the champion's (init cont3, lr 1e-4, SUPPORTER:5
# STADIUM:5) so the CORPUS is the only variable between arms. That is the
# whole point of running A and B in parallel: they fail independently, and
# which one wins says which constraint was binding.
#
#   m39_vsloss      corpus A, alpha=0     harvested, winner rows only
#                                         (alpha=0 reproduces --winners-only)
#   m39_vsloss_a25  corpus A, alpha=0.25  do the kept LOSER rows carry signal?
#   m39_bestresp    corpus B, alpha=0     manufactured: our own wins against
#                                         the clone beds (offline best-response)
#   m39_retain      winner + champion shards, multi-dir --data: does data
#                   mixing break the 1-epoch dose law? Runs LAST, on whichever
#                   corpus wins, because it mixes INTO the winner.
#
# Usage: bash scripts/m39_train.sh [arm ...]      (default: the corpus-A arms)
set -u
cd "$(dirname "$0")/.."

INIT=checkpoints/m38_w9294_cont3.pt
# Corpus B is collected one directory per bed (three loss families x two panel
# draws), so every arm that uses it takes all six via the multi-dir --data
# path — the same path the retention arm uses to mix champion shards in.
BR=(data/bc_m39_br_wall data/bc_m39_br_grim data/bc_m39_br_archaludon
    data/bc_m39_br_wall_d2 data/bc_m39_br_grim_d2 data/bc_m39_br_arch_d2)
KIND=(--card-kind-weight SUPPORTER:5 --card-kind-weight STADIUM:5)
COMMON=(--init "$INIT" --epochs 3 --lr 1e-4 "${KIND[@]}")

train_arm () {   # name  extra-args...
  local name="$1"; shift
  if [ -f "checkpoints/${name}.pt" ]; then
    echo "[train] ${name} exists - skip"
    return
  fi
  echo "=== train ${name}  $(date) ==="
  uv run python -m rl.plan_iter train --name "$name" "${COMMON[@]}" "$@" \
    2>&1 | tee "runs/${name}.log"
}

ARMS=("${@:-vsloss vsloss_a25}")
for arm in ${ARMS[@]}; do
  case "$arm" in
    vsloss)
      train_arm m39_vsloss --data data/bc_m39_vsloss --outcome-weight 0 ;;
    vsloss_a25)
      train_arm m39_vsloss_a25 --data data/bc_m39_vsloss --outcome-weight 0.25 ;;
    bestresp)
      train_arm m39_bestresp --data "${BR[@]}" --outcome-weight 0 ;;
    bestresp_lo)
      # Corpus B is ~7x corpus A's winner rows, so "3 epochs, best-val" is a
      # much larger DOSE here even at the same epoch count -- and best-val
      # selects for agreement with corpus B, not for strength (the M38
      # note). This arm pins the single-pass net so the dose law has a cell
      # rather than a hope.
      COMMON=(--init "$INIT" --epochs 1 --lr 1e-4 "${KIND[@]}")
      train_arm m39_bestresp_lo --data "${BR[@]}" --outcome-weight 0 ;;
    retain_a)
      train_arm m39_retain_a --data data/bc_m39_vsloss data/bc_m38_w9294 \
        --outcome-weight 0 ;;
    retain_b)
      train_arm m39_retain_b --data "${BR[@]}" data/bc_m38_w9294 \
        --outcome-weight 0 ;;
    both)
      train_arm m39_both --data data/bc_m39_vsloss "${BR[@]}" \
        --outcome-weight 0 ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
done
echo "=== M39 TRAIN DONE $(date) ==="
