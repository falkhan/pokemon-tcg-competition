#!/usr/bin/env bash
# M39 P2 — the anti-deck-out package, gated on the G-13 panel roster.
#
# Every arm is the LIVE Ship A config (`conserve`) plus exactly one new rule,
# so each cell is a single variable against the agent currently on the ladder
# rather than against a hypothetical baseline (G-7's spirit inside one gate).
#
#   r2   conserve + racemode2   P2a: the m37 wall-blanket / pressure-margin
#                               split, never shipped live, now carrying the
#                               P2a id additions (Mega Kangaskhan ex on the
#                               wall side, Fan Rotom on the pressure side)
#   rm4  conserve + racemode4   P2b: demote OUR measured burn sources in a race
#   pkg  conserve + both        the ship candidate if either survives
#
# NOT an arm: `raceash`. The mechanism probe fires it 8 times in 20 wall games
# and 4 in 20 grim games (one Sacred Ash in a 60-card list), so an arm for it
# would be underpowered by construction — exactly the G-12 error of testing an
# effect the battery cannot resolve. It stays in the code, tested and inert
# unless named.
#
# Where a delta can come from, measured before the battery ran
# (scripts/m39_race_probe.py, 20 games/bed): the race trigger is engaged in
# 883/883 wall prompts and 142/979 grim prompts, and in ZERO prompts on
# archaludon / mirror / rocket / top. So wall (.136) + grim (.122) = 25.8% of
# the live mix is the entire reachable mass, and every other cell is a
# negative control -- if one of those moves significantly, the inertness claim
# is wrong, not the rule.
#
# SKIP_TOP=1: the 900+ bed runs our own list, carries no trigger id, and is
# not in the live mix -- it can neither move nor gate anything here.
set -u
cd "$(dirname "$0")/.."

# Wait for a same-prefix battery already in flight (they share the 8 workers,
# and the parallelism cap in CLAUDE.md is 8 TOTAL, not 8 per battery).
LOG="${WAIT_FOR:-runs/m39_pan_control.log}"
if [ -f "$LOG" ] && ! grep -q "PANEL GATE" "$LOG"; then
  echo "[p2] waiting for $LOG to finish  $(date)"
  while ! grep -q "PANEL GATE" "$LOG"; do sleep 30; done
fi

NET=checkpoints/m38_w9294_cont3.pt
export SKIP_TOP=1
bash scripts/m39_panel_gate.sh \
  r2  "model-c-r2:$NET:alakazam_v2_h4" \
  rm4 "model-c-rm4:$NET:alakazam_v2_h4" \
  pkg "model-c-pkg:$NET:alakazam_v2_h4"
echo "=== M39 P2 BATTERY DONE $(date) ==="
