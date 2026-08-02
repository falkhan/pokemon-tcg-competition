#!/usr/bin/env bash
# M40 S5 — the v4 encoder arm, as a SINGLE-VARIABLE replay of cont3.
#
# What makes this arm worth running is what it holds fixed. `m38_w9294_cont3`
# is the campaign's best-evidenced fine-tune (+6.1pp weighted, z=+6.03 over the
# champion): warm-start m28_winners, ~1 epoch on the 800-1058-band same-deck
# winner harvest, best-val selects. This arm reruns that EXACT recipe — same
# init lineage, same corpus seats, same epochs, same lr, same kind weights —
# with one difference:
#
#     the corpus is encoded with encode_ctx_v4 instead of encode_state_v3.
#
# So `m40_v4_cont3` vs `m38_w9294_cont3` isolates the ENCODER and nothing else.
# That is the question M40 S5 asks and the one Piotr prioritised: the live net
# has been blind to OppMemory (last-4 opponent played ids, last attacker,
# energy-attach target) and to per-bench-slot damage projection since M24, not
# by decision but because rl/replay_bc.py never read obs.logs.
#
# Prerequisites, both already satisfied:
#   - the fidelity probe PASSED at 100.0000% over 3902 prompts
#     (scripts/m40_v4_fidelity.py) — the pre-registered gate for re-encoding;
#   - checkpoints/m38_ft_init.pt = migrate_v3_to_v4(m28_winners), the M21
#     warm-start invariant (new columns zero-init, bit-identical at init), and
#     the SAME init object M38 used, so the lineage is shared with cont3's.
#
# Corpus B (data/bc_m39_br_*) is deliberately NOT in this arm. It came from
# live collection at v3 (scripts/m39_collect_bestresp.py), so it cannot be
# re-encoded — only re-collected, at 3600 engine games plus an arm
# redefinition. That is a second-pass decision, taken only if this arm clears.
#
# Usage: bash scripts/m40_v4_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS=data/bc_m38_w9294_v4
INIT=checkpoints/m38_ft_init.pt
KIND=(--card-kind-weight SUPPORTER:5 --card-kind-weight STADIUM:5)

[ -d "$CORPUS" ] || { echo "no $CORPUS — run: uv run python -m rl.replay_bc build \
--out $CORPUS --min-score 800 --winners-only --deck-hash 9294d9d8 --v4" >&2; exit 2; }
[ -f "$INIT" ]   || { echo "no $INIT (migrate_v3_to_v4(m28_winners))" >&2; exit 2; }

echo "=== train m40_v4_cont3 $(date) ==="
uv run python -m rl.plan_iter train --data "$CORPUS" --name m40_v4_cont3 \
  --init "$INIT" --epochs 3 --lr 1e-4 "${KIND[@]}" 2>&1 \
  | tee runs/m40_v4_cont3.log

echo "=== M40 S5 TRAIN DONE $(date) ==="
echo "Gate it against the v3 twin, which is the whole point:"
echo "  PREFIX=m40_v4 bash scripts/m40_panel_gate.sh \\"
echo "    v4cont3 'model-c-pkg:checkpoints/m40_v4_cont3.pt:alakazam_v2_h4' \\"
echo "    v3cont3 'model-c-pkg:checkpoints/m38_w9294_cont3.pt:alakazam_v2_h4'"
echo "  uv run python scripts/m40_decide.py --arm v4cont3 --control v3cont3 --prefix m40_v4"
