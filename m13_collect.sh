#!/usr/bin/env bash
# M13 setup-state safari — self-play collection with a live bar.
#
#   ./m13_collect.sh [--profile laptop|box] [--games N] [--target N]
#                    [--workers N] [--seed N] [--out DIR]
#
# --target N collects only what's MISSING to reach N total games in the
# dataset (counts existing shards first) — the set-and-forget mode for
# chunked days: ./m13_collect.sh --target 10000, rerun until it says done.
#
# Profiles set the defaults; every flag overrides them individually:
#   laptop (default): workers = cores/4 (min 2), nice -19 — your Citrix VDI
#                     and everything else keep scheduling priority.
#   box:              workers = 3*cores/4 (min 4), nice -10 — the stationary
#                     machine, still polite to concurrent jobs.
# Batches STACK (shards append; seed bumps per batch by default), so chunked
# runs across a day are fine. Ctrl-C stops cleanly; finished shards kept.
set -u

BOLD=$(tput bold 2>/dev/null || true); DIM=$(tput dim 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true); YEL=$(tput setaf 3 2>/dev/null || true)
CYN=$(tput setaf 6 2>/dev/null || true); RST=$(tput sgr0 2>/dev/null || true)

CORES=$(nproc 2>/dev/null || echo 8)
PY=.venv/bin/python
PROFILE=laptop
GAMES=10000
TARGET=""
WORKERS=""
SEED=""
OUT=data/setupval
SHARD_SIZE=100   # flush every 100 games/worker: Ctrl-C loses at most the game in flight (worker flushes on interrupt) or <100 on a hard kill

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2 ;;
    --games)   GAMES=$2;   shift 2 ;;
    --target)  TARGET=$2;  shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    --seed)    SEED=$2;    shift 2 ;;
    --out)     OUT=$2;     shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (see --help)"; exit 2 ;;
  esac
done

case "$PROFILE" in
  laptop) DEF_WORKERS=$(( CORES / 4 > 2 ? CORES / 4 : 2 ));  NICENESS=19 ;;
  box)    DEF_WORKERS=$(( CORES * 3 / 4 > 4 ? CORES * 3 / 4 : 4 )); NICENESS=10 ;;
  *) echo "unknown profile: $PROFILE (laptop|box)"; exit 2 ;;
esac
WORKERS=${WORKERS:-$DEF_WORKERS}
# Auto-seed: one more than the number of existing shard files, so every rerun
# is a FRESH batch (unique deck-sampling stream) that appends to the dataset.
if [ -z "$SEED" ]; then
  SEED=$(( $(ls "$OUT"/shard_* 2>/dev/null | wc -l) + 1 ))
fi
EXISTING=$(ls "$OUT"/shard_* 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ]; then
  HAVE=$("$PY" - <<PYEOF 2>/dev/null || echo "0 0"
import numpy as np, glob
g = s = 0
for f in glob.glob("$OUT/shard_*"):
    ids = np.load(f)["game_ids"]
    g += len(set(ids.tolist())); s += len(ids)
print(g, s)
PYEOF
)
  HAVE_GAMES=${HAVE% *}; HAVE_STATES=${HAVE#* }
else
  HAVE_GAMES=0; HAVE_STATES=0
fi
if [ -n "$TARGET" ]; then
  GAMES=$(( TARGET - HAVE_GAMES ))
  if [ "$GAMES" -le 0 ]; then
    echo ""
    echo "${BOLD}🏆 Pokédex complete!${RST} Dataset already holds ${HAVE_GAMES} games (target ${TARGET})."
    echo "${DIM}   Nothing to collect — go train: see --help's next-step command.${RST}"
    exit 0
  fi
fi
LOG=runs/m13_collect_s${SEED}.log

FLAVOR=(
  "Pikachu is charging up the value head…"
  "Lucario senses a bar-clearing aura…"
  "Solrock is waiting for Lunatone (properly, this time)…"
  "Riolu is doing push-ups on the bench…"
  "Professor's notes: matched pairs make the best rivals…"
  "Makuhita keeps slapping the training dummy…"
  "A wild SETUP STATE appeared!"
)
BALL=("◐" "◓" "◑" "◒")

echo ""
echo "${BOLD}${RED}⚡ M13 SETUP-STATE SAFARI ⚡${RST}"
echo "${DIM}   ${PROFILE} profile · ${GAMES} games · ${WORKERS}/${CORES} workers at nice -${NICENESS} · seed ${SEED} → ${OUT}${RST}"
echo "${DIM}   Dataset so far: ${HAVE_GAMES} games / ${HAVE_STATES} states in ${EXISTING} shard(s) — this batch stacks on top.${RST}"
[ -n "$TARGET" ] && echo "${DIM}   Target ${TARGET}: collecting the missing ${GAMES}.${RST}"
echo ""

mkdir -p runs "$OUT"
rm -f "$OUT"/.progress_w*

nice -n "$NICENESS" "$PY" -m rl.setup_value collect --games "$GAMES" --out "$OUT" \
      --workers "$WORKERS" --seed "$SEED" --shard-size "$SHARD_SIZE" >"$LOG" 2>&1 &
PID=$!
trap 'echo ""; echo "${YEL}✋ Safari pausing — letting each worker finish its current game…${RST}"; touch "$OUT/.stop"; wait $PID 2>/dev/null; rm -f "$OUT"/.progress_w* "$OUT/.stop" 2>/dev/null; echo "${YEL}   All completed games are shard-safe in ${OUT} — rerun me anytime, batches stack.${RST}"; exit 130' INT

START=$(date +%s)
FRAME=0
while kill -0 "$PID" 2>/dev/null; do
  DONE=0
  for f in "$OUT"/.progress_w*; do
    [ -f "$f" ] && DONE=$((DONE + $(cat "$f" 2>/dev/null || echo 0)))
  done
  [ "$DONE" -gt "$GAMES" ] && DONE=$GAMES
  PCT=$((DONE * 100 / GAMES))
  FILLED=$((PCT * 30 / 100))
  BAR=""
  for ((i = 0; i < 30; i++)); do
    if [ "$i" -lt "$FILLED" ]; then BAR="${BAR}■"; else BAR="${BAR}·"; fi
  done
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$DONE" -gt 0 ] && [ "$ELAPSED" -gt 5 ]; then
    ETA=$(( ELAPSED * (GAMES - DONE) / DONE ))
    ETA_STR=$(printf "%dh %02dm" $((ETA / 3600)) $(((ETA % 3600) / 60)))
  else
    ETA_STR="…"
  fi
  SPIN=${BALL[$((FRAME % 4))]}
  MSG=${FLAVOR[$(((FRAME / 8) % ${#FLAVOR[@]}))]}
  printf "\r${RED}%s${RST} ${BOLD}[%s]${RST} %5d/%d ${CYN}(%d%%)${RST} · ETA %s  ${DIM}%s${RST}\033[K" \
         "$SPIN" "$BAR" "$DONE" "$GAMES" "$PCT" "$ETA_STR" "$MSG"
  FRAME=$((FRAME + 1))
  sleep 2
done
wait "$PID"; RC=$?
rm -f "$OUT"/.progress_w*
echo ""
if [ "$RC" -eq 0 ]; then
  STATES=$(grep -o "[0-9]* setup states" "$LOG" | tail -1 || true)
  echo "${BOLD}🎉 Gotta catch 'em all — ${GAMES} games in the box!${RST}  ${DIM}(${STATES:-see $LOG})${RST}"
  echo ""
  echo "${BOLD}Next: evolve your value model →${RST}"
  echo "  ${CYN}${PY} -m rl.setup_value train --data ${OUT} --name osv3_setupval2 \\"
  echo "      --init-v3 checkpoints/osv3_plan0c.pt --epochs 6${RST}"
else
  echo "${YEL}💥 The safari fainted (exit ${RC}) — check ${LOG}${RST}"
fi
