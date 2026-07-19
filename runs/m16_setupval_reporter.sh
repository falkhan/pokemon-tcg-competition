#!/usr/bin/env bash
# Watches runs/m16_setupval_train.log and pings Telegram on each new
# epoch / completion / crash line. Stops when the training proc exits.
LOG=/home/falkhan/Documents/python_projects/pokemon-tcg-competition/runs/m16_setupval_train.log
LAST=0
while true; do
  if [ -f "$LOG" ]; then
    CUR=$(wc -c < "$LOG")
    if [ "$CUR" -gt "$LAST" ]; then
      NEW=$(tail -c +$((LAST+1)) "$LOG")
      ECHO=$(printf '%s\n' "$NEW" | grep -E '^epoch |best held-out|Traceback|Error|KeyError|ValueError|warm-start')
      if [ -n "$ECHO" ]; then
        printf '[M16 value retrain] %s\n' "$ECHO" | hermes send -t telegram -f - 2>/dev/null
      fi
      LAST=$CUR
    fi
  fi
  if ! pgrep -f 'rl.setup_value' >/dev/null; then
    sleep 2
    CUR=$(wc -c < "$LOG")
    if [ "$CUR" -gt "$LAST" ]; then
      NEW=$(tail -c +$((LAST+1)) "$LOG")
      printf '[M16 value retrain - FINAL] %s\n' "$NEW" | hermes send -t telegram -f - 2>/dev/null
    fi
    break
  fi
  sleep 20
done
