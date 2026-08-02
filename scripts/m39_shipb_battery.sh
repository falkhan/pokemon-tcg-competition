#!/usr/bin/env bash
# M39 P4 — the Ship B confirmation matrix on whichever net P3 selects.
#
# G-4 is the reason this exists and is not optional: "any net change re-runs
# the G5-style on/off matrix before ship — rules tuned on an OLD net's
# behaviour can actively harm a new one." Ship A's `conserve` was measured on
# `m38_w9294_cont3`. If Ship B changes the net, `conserve` is an UNMEASURED
# config on the new net, and shipping an unmeasured config on gate evidence is
# the exact M38 error this milestone's finding #1 is about.
#
# Cells, all on the SHIP NET:
#   <net>_plain     no fixes at all      the G5 `plain` arm, byte-for-byte
#   <net>_conserve  the live Ship A rule (already run by the P3 battery under
#                   the arm's own name, so it is not repeated here)
#   <net>_pkg       + whatever P2 rules survived, ONLY if P2 survived
#
# Usage:
#   NET=m39_retain_a bash scripts/m39_shipb_battery.sh            # G-4 only
#   NET=m39_retain_a WITH_PKG=1 bash scripts/m39_shipb_battery.sh # + P2 rules
set -u
cd "$(dirname "$0")/.."

NET="${NET:?set NET=<checkpoint stem under checkpoints/>}"
WITH_PKG="${WITH_PKG:-0}"
CK="checkpoints/${NET}.pt"
[ -f "$CK" ] || { echo "no such checkpoint: $CK" >&2; exit 2; }

LOG="${WAIT_FOR:-runs/m39_p3_battery.log}"
if [ -f "$LOG" ] && ! grep -q "P3 BATTERY DONE" "$LOG"; then
  echo "[shipb] waiting for $LOG  $(date)"
  while ! grep -q "P3 BATTERY DONE" "$LOG"; do sleep 30; done
fi

ARGS=("${NET}_plain" "model:$CK:alakazam_v2_h4")
[ "$WITH_PKG" = "1" ] && ARGS+=("${NET}_pkg" "model-c-pkg:$CK:alakazam_v2_h4")

bash scripts/m39_panel_gate.sh "${ARGS[@]}"
echo "=== M39 SHIP B MATRIX DONE $(date) ==="
