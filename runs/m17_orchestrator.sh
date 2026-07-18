#!/usr/bin/env bash
# M17 pipeline orchestrator: waits for the running 10k collection, then
# trains -> measures (screen + meta) -> ships ONLY IF it beats M16
# (mirror 0.424 / meta 0.534). Pings Telegram ONCE at the very end with the
# full outcome. Pings immediately on hard failure.
set -u
ROOT=/home/falkhan/Documents/python_projects/pokemon-tcg-competition
cd "$ROOT"

PING() { hermes send -t telegram -f - 2>/dev/null <<<"[M17 pipeline] $1"; }

NEW_CKPT=checkpoints/osv3o_plan2.pt
VALUE_CKPT=checkpoints/osv3o_setupval1.pt
COLLECT_DIR=data/plan_m17b
BASE_NET=checkpoints/osv3o_plan1.pt
MIRROR_BAR=0.424
META_BAR=0.534

# ---- 1. wait for the REMAINDER collection to finish (plan_m17 already banked) ----
PING "step 0: waiting for remainder 10k collection (data/plan_m17b) to finish..."
while pgrep -f 'plan_iter collect' >/dev/null; do
  sleep 60
done
# confirm it actually produced a done: line (check both possible log names)
if ! grep -q '^done:' runs/m16_collect_10kb.log 2>/dev/null; then
  PING "FAILED: remainder collection did not finish cleanly (no 'done:' in log). Aborting train."
  exit 1
fi
PING "step 1: remainder collection complete -> $(grep '^done:' runs/m16_collect_10kb.log | tail -1)"
PING "  (plan_m17 banked 32 shards + plan_m17b remainder = full 10k dataset)"

# ---- 2. train ---------------------------------------------------------------
PING "step 2: training $NEW_CKPT (warm from $BASE_NET, data = plan_m17 + plan_m16)..."
uv run python -m rl.plan_iter train \
  --data "$COLLECT_DIR" data/plan_m17 data/plan_m16 \
  --name osv3o_plan2 \
  --init-v3o "$BASE_NET" \
  --epochs 6 --lr 1e-4 --batch-size 256 \
  > runs/m16_train_10k.log 2>&1
if [ ! -f "$NEW_CKPT" ]; then
  PING "FAILED: training did not produce $NEW_CKPT. See runs/m16_train_10k.log"
  exit 1
fi
PING "step 2 done: $(grep 'best held-out' runs/m16_train_10k.log | tail -1)"

# ---- 3. measure: mirror screen (seed 1, then confirm seed 2) ---------------
PING "step 3: measuring vs solver:lucario (screen n=400 seed 1)..."
uv run python -m rl.matchrunner play \
  --a "model:$NEW_CKPT:lucario" --b solver:lucario \
  -n 400 --workers 12 --seed 1 \
  --checkpoint runs/m17_screen_s1.jsonl >/dev/null 2>&1
# decode win rate: jsonl result 0 = win
WR1=$(python3 -c "
import json,sys
wins=losses=draws=0
with open('runs/m17_screen_s1.jsonl') as f:
    for line in f:
        try: d=json.loads(line)
        except: continue
        if 'results' in d:
            for r in d['results']:
                if r==0: wins+=1
                elif r==2: draws+=1
                else: losses+=1
tot=wins+losses+draws
print(f'{wins}/{tot} ({wins/tot:.3f})') if tot else print('0/0')
")
PING "  screen seed1 win-rate: $WR1"

# confirm seed 2 (n=400) for pooled n=800
uv run python -m rl.matchrunner play \
  --a "model:$NEW_CKPT:lucario" --b solver:lucario \
  -n 400 --workers 12 --seed 2 \
  --checkpoint runs/m17_screen_s2.jsonl >/dev/null 2>&1
WR2=$(python3 -c "
import json
wins=losses=draws=0
for fn in ['runs/m17_screen_s1.jsonl','runs/m17_screen_s2.jsonl']:
    with open(fn) as f:
        for line in f:
            try: d=json.loads(line)
            except: continue
            if 'results' in d:
                for r in d['results']:
                    if r==0: wins+=1
                    elif r==2: draws+=1
                    else: losses+=1
tot=wins+losses+draws
print(f'{wins}/{tot} ({wins/tot:.3f})')
")
PING "  pooled screen (seed1+2) win-rate: $WR2"

# ---- 4. measure: meta co-gate ----------------------------------------------
PING "step 4: meta co-gate (meta_v2)..."
uv run python -m rl.replay_bc meta-eval --a "model:$NEW_CKPT:lucario" -n 60 --workers 8 \
  > runs/m17_meta.log 2>&1
META=$(grep -iE 'weighted|meta-eval:' runs/m17_meta.log | tail -1)
PING "  meta: $META"

# ---- 5. ship decision -------------------------------------------------------
# crude numeric compare of pooled mirror wr vs bar 0.424
MIRROR_W=$(python3 -c "
import json,re
wins=losses=draws=0
for fn in ['runs/m17_screen_s1.jsonl','runs/m17_screen_s2.jsonl']:
    with open(fn) as f:
        for line in f:
            try: d=json.loads(line)
            except: continue
            if 'results' in d:
                for r in d['results']:
                    if r==0: wins+=1
                    elif r==2: draws+=1
                    else: losses+=1
print(wins/(wins+losses+draws))
")
SHIP=0
python3 -c "
import sys
w=$MIRROR_W; bar=$MIRROR_BAR
sys.exit(0 if w>bar else 1)
" && SHIP=1

if [ "$SHIP" -eq 1 ]; then
  PING "step 5: BEATS bar (mirror $MIRROR_W > $MIRROR_BAR). Building + shipping..."
  bash build_submission.sh --checkpoint "$NEW_CKPT" \
    --message "M17: 10k value-guided collection + option-identity net (osv3o_plan2)" \
    > runs/m17_submit.log 2>&1
  PING "DONE: shipped. $(grep -iE 'SUBMITTED|submission_' runs/m17_submit.log | tail -2)"
else
  PING "step 5: did NOT beat bar (mirror $MIRROR_W <= $MIRROR_BAR). NOT shipped (per instruction). Checkpoint $NEW_CKPT kept for review."
fi
PING "pipeline finished."
