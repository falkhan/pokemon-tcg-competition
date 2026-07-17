#!/usr/bin/env bash
# M13 setup-state safari — VDI-friendly self-play collection with a live bar.
#
#   ./m13_collect.sh [games] [workers] [seed]
#
# Defaults: 10000 games, 6 workers (leaves ~10 cores for your Citrix VDI),
# seed 1 (batch 1 used seed 0). Shards append into data/setupval — batches
# stack safely. Ctrl-C stops the safari cleanly (finished shards are kept).
set -u

GAMES=${1:-10000}
WORKERS=${2:-6}
SEED=${3:-1}
OUT=data/setupval
LOG=runs/m13_collect_s${SEED}.log
PY=.venv/bin/python

BOLD=$(tput bold 2>/dev/null || true); DIM=$(tput dim 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true); YEL=$(tput setaf 3 2>/dev/null || true)
CYN=$(tput setaf 6 2>/dev/null || true); RST=$(tput sgr0 2>/dev/null || true)

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
echo "${DIM}   ${GAMES} games · ${WORKERS} workers (VDI-friendly) · seed ${SEED} → ${OUT}${RST}"
echo ""

mkdir -p runs "$OUT"
rm -f "$OUT"/.progress_w*

"$PY" -m rl.setup_value collect --games "$GAMES" --out "$OUT" \
      --workers "$WORKERS" --seed "$SEED" >"$LOG" 2>&1 &
PID=$!
trap 'echo ""; echo "${YEL}✋ Safari interrupted — finished shards are safe in ${OUT}${RST}"; kill $PID 2>/dev/null; exit 130' INT

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
