# M15 plan: hand-aware encoding — the net finally sees what it's holding

Status: **started 2026-07-17 (late), branch `feature/m11`.** Diary: [M15.md](M15.md).

## Context (from the M14 observation rounds, docs/M14.md)

Three defect classes measured in live replays, addressed together here:
1. **Hand-blindness** (user hypothesis, encoder-audit confirmed): the hand is
   a 36-dim summed feature pool + count; card-id EMBEDDINGS cover only the 12
   board slots. The value can't see "gust + Gong + energy in hand" when
   scoring a setup; the policy can't see what it holds when sequencing.
2. **Taught energy waste** (51-63% of live attaches saturated): fixed at the
   teacher (`score_attach` card-fact gating, committed 27cc495) — labels must
   be REGENERATED, old expert shards would re-teach the waste.
3. **Wrong-attacker positioning** (30-50% of attacks under board-best in long
   losses): audit target on the teacher's promote/retreat scoring — logged,
   not in this round's scope.

## Design

- **`N_HAND_IDS = 8`**; new `encode_state_v3(state, deck)` = same 1297
  numeric + **20 state ids** (12 board + up to 8 hand card ids, sorted,
  0-padded). `encode_state_v2`/`N_STATE_IDS` UNTOUCHED — zero blast radius
  on existing checkpoints, bundles, parity checks.
- **`OptionScorerV3(n_state_ids=...)`**: parameterized id count (default =
  legacy 12-behavior). Hand-aware nets have state_enc.0 width 1580+128=1708;
  loaders sniff the width. Migration `load_v3_into_v3h`: copy verbatim,
  hand-embedding columns zero-init ⇒ hand-aware(zero-hand) ≡ old net
  (the proven warm-start invariant, third use).
- **Dataset shims**: old shards' state_ids (12) zero-padded to 20 at load —
  same semantics as the plan-column shim (missing info = zeros).
- **Value head**: keeps `osv3_setupval2` this round (12-id, still drives
  setup-plan commits at data-gen). The hand-aware VALUE needs a fresh safari
  batch (collector switched to v3 encoder so future `m13_collect.sh` runs
  carry hand ids) — next session's overnight job.
- **Ship**: user's standing observation-run policy — technical gates
  enforced, win-rate gate informational.

## Pipeline (tonight)

1. Re-pin: `osv3_plan0c` vs the POST-attach-fix `solver:lucario` n=400 (the
   honest reference for everything after; the old 0.415 is against a weaker
   opponent).
2. Collect `data/plan_m15`: 800 games, mixed opponents (self/buddy/rule),
   fixed teacher, **v3 encoder** (labels carry hand ids), runaway cap active.
3. Train `osv3h_plan1.pt`: hand-aware net via `load_v3_into_v3h(osv3_plan0c)`,
   data = plan_m15 (+ old dirs via the pad shim), 5 epochs 1e-4.
4. Ship + measure + monitor row + diaries.

## Files

`rl/encoders.py` (+N_HAND_IDS, encode_state_v3), `rl/policy.py` (V3 param),
`rl/plan_iter.py` (v3 encoder, pad shim, migration, train flag),
`rl/setup_value.py` (v3 encoder for future safaris + pad shim),
`rl/matchrunner.py` (width sniff), `submission/main.py` + `tcg/shipping.py`
(hand-aware bundle), `tcg/network.py` twin, tests.
