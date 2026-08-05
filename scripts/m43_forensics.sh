#!/usr/bin/env bash
# M43 Phase 0 — forensics on the M41b ships (docs/M43-plan.md).
#
# Runs on the box that holds the M38+ artifacts and Kaggle credentials.
# G-10: run only once BOTH subs have >= 45 live games; a thinner read is not
# a verdict. E0 has no accrual dependency and runs regardless (day 1).
#
# Usage: bash scripts/m43_forensics.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SUB_ALAKAZAM=55265099   # m41b_wide_prod : alakazam_v2_h4
SUB_OGERPON=55265105    # m41_ogerpon (re-ship) : decks/ogerpon.csv
WIDE_CKPT=checkpoints/m41b_wide_prod.pt

notify() { hermes send -t telegram "[m43-p0] $1" >/dev/null 2>&1 || true; }

notify "Phase 0 forensics starting (subs $SUB_ALAKAZAM/$SUB_OGERPON)"

# 0.3 — E0 on the wide net's own value head (pre-registered: >=0.62 -> it
# serves as GAE critic AND Phi; <0.58 -> warm-start from m39_retain_b;
# between -> GAE yes, Phi no. See docs/M43-plan.md Phase 0.)
echo "=== E0: $WIDE_CKPT value head ==="
uv run python scripts/m40_e0_value.py --checkpoint "$WIDE_CKPT" \
  | tee "runs/m43_p0_e0.log"
notify "E0 on m41b_wide_prod done — see runs/m43_p0_e0.log (bar 0.62 / kill 0.58)"

# 0.1 — harvest live episodes + our agent's per-decision NN logs
echo "=== ingest: refresh both subs ==="
uv run python -m rl.kaggle_ingest refresh --subs "$SUB_ALAKAZAM" "$SUB_OGERPON" \
  | tee "runs/m43_p0_refresh.log"

# 0.2 — W/L by opponent archetype, per sub (the family split the
# pre-registered consequences key on)
for sub in "$SUB_ALAKAZAM" "$SUB_OGERPON"; do
  echo "=== forensics: sub $sub ==="
  uv run python -m rl.kaggle_ingest forensics --sub "$sub" \
    | tee "runs/m43_p0_forensics_${sub}.log"
done

notify "Phase 0 harvest + family split done. Next: postmortem the losses (rl.postmortem <episode-id>), then Piotr's replay review of >=3 losses/sub."
echo "Phase 0 mechanical steps complete."
echo "Manual follow-ups (see docs/M43-plan.md):"
echo "  - uv run python -m rl.postmortem <episode-id>   # per-loss anatomy"
echo "  - replay review with Piotr (>=3 losses per sub)"
