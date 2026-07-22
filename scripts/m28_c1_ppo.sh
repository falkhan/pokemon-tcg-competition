#!/usr/bin/env bash
# M28 Track C1: PPO from the CLONE base.
#
# M20/M21 ran PPO from a ~0.48 mirror base and M22c-RL's out-of-loop never
# moved. m27_both sits at 0.6375 plain / 0.6863 with O1 — untested from a base
# this strong. KL-anchored to the frozen start (M20 leg B's fix for the
# peak-then-decay drift).
#
# Opponent = rule:lucario, the Pokemon Company sample agent. NOT rule:dragapult:
# that is the sealed out-of-loop floor and training against it retires the only
# clean instrument we own, permanently (docs/DECISIONS.md 2026-07-20).
#
# Pre-registered kill bars (docs/M28-plan.md):
#   - out-of-loop `rule:dragapult` drops below the 0.3738 pin by more than MDE
#   - mirror collapses in the M21 Gate-A shape (18-26pp — well above MDE)
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m rl.ppo \
  --start m27_both.pt \
  --learn-deck clone54618168 \
  --opponents rule:lucario \
  --eval-deck clone54618168 \
  --kl-coef 0.1 \
  --iterations 6 \
  --games-per-iter 300 \
  --workers 8 \
  --eval-every 2 \
  --eval-games 200 \
  --tag m28c
echo "=== M28 C1 PPO DONE $(date) ==="
