# M31 — bench economy as a deterministic rule (final plan)

Branch `feature/m31` from `feature/m30`. Final, code-verified plan (2026-07-23).
Reviewed against `docs/m30-post-mortem.md` and the code; three code-verification
sweeps confirmed every predicate/line reference below. Piotr's four scope decisions
(recorded in "Decisions") are pre-registration, not suggestions; the weights-freeze
and O8-scope choices were re-confirmed with Piotr at the start of M31.

## Context

Why this milestone. `54929991` is live at **818.2** — the campaign record (prev
54903635 @735/720, 54897966 @734, M29 54914673 @665; peak 845 @game 9). The M30 gac
ship (`m28_winners` + O1 telepath + O4 deckguard + O5 ash + O6 conserve on
`clone54618168`) works and loses honestly: avg opponent score in losses 826 vs 698 in
wins, median opponent 778, only 1/26 upset. We lose at the frontier, not the pool
floor. Reaching 1000 means winning the 800–860 band where 9 of 10 losses live.

The n=26 live post-mortem (`docs/m30-post-mortem.md`, directional) names the next
defect: **bench economy**. 4/10 losses spent a mid-game turn (t>3) at bench 0; zero
wins did; two were outright bench-outs. The mechanism is not "refuses to bench" — at
bench 0 with a basic in hand the model benched it every time. The defect is upstream:
it does not **dig for board** when thin. Structural support, re-verified this session
against `decks/clone54618168.csv`: the deck holds only **9 basics in 60** — Abra
(741)×4, Dunsparce (305)×3, Fezandipiti ex (140)×1, Shaymin (343)×1; everything else
is evolutions (Kadabra 742, Alakazam 743, Dudunsparce 66) or trainers/energy. Drawing
a second basic is genuinely hard; the list carries 4× Buddy-Buddy Poffin (1086) for it.

Governing law. The only intervention class with a live win is the deterministic
override rule (O1 → 720; gac → 818). Every weights change since M26 was
neutral-or-worse (M29 −65 ELO; M30 arm D killed offline on exactly the mirror-loss
pattern). M31 stays in the winning class.

## Decisions (Piotr, confirmed 2026-07-23)

1. **Deck frozen** at `clone54618168`; **base weights frozen** at `m28_winners`.
   Rules-only milestone. Rationale re-confirmed with Piotr this session: no weights
   candidate has ever cleared the resolved-better bar the M29 lesson demands, and the
   818 record *is* this base — a weights change would ship downgrade-risk with no
   offline evidence it helps. Telepath-attach (~72%, −13pp vs 54903635) and Hilda
   hoarding are handled as a **P0.4 measurement axis** + the **O8 rule**, not a retrain.
2. **O7 = Poffin-only.** The 4-rung bench ladder in the earlier draft collapses to one
   rung; rationale in "Why the ladder collapsed."
3. **O1b telefull dropped; O9 ashguard is not an arm** — docstring fix + regression
   test only. Telepath is a *measurement* item (P0.4), not a rule.
4. **O8 (Hilda drawfloor) screens in parallel with O7** (confirmed with Piotr: "screen
   both, ship ≤1"), not deferred. Ship at most one new rule (M30 screened 4). O8
   carried only if P0.3 prices Hilda deck-cheap AND it clears its own bar.

### Why the ladder collapsed to one rung (all code-verified this session)

- **Telepath rung is dead code.** O1 (`rl/plan.py:308-316`, keys `TELEPATH_ID=19`)
  promotes a legal Telepath ATTACH whenever the bench has *space* (`sum(occupied) <
  benchMax`, lines 310-311) — there is no *minimum* bench-count threshold. At bench ≤1
  the bench always has space, so a "bench ≤1 telepath" rung cannot change an order O1
  did not already set.
- **Basic-PLAY rung has no evidence and a counter-example.** Declines at bench 0 are 0
  live; the rung could only fire at bench exactly 1. Loss 87677432 is a game where the
  model *did* bench its only available basic (Fez ex) at t3, had it gusted for 2
  prizes, and lost — forcing that play makes the exhibit loss worse.
- **Poké Pad (1152) excluded.** Priced at 1.0 deck cards/use (`docs/M30.md:328`) — a
  draw-1 signature, not a bench-filler — and the most-declined END item live. A
  deck-burning draw item inside a bench rule repeats the mechanism that killed O3 tempo.
- **O9's cost is structurally zero.** The precedence bug is real and confirmed: ash
  promotes on `deckCount<=10` regardless of `ranked[0]` (`rl/plan.py:380-385`),
  contradicting the docstring at `:359-360` (true for O3 tempo, which *is* `ranked[0]`-
  gated at `:386-390`; false for O5 ash). But the turn's manual attach is still unused
  after Sacred Ash resolves, so the next MAIN prompt re-offers the ATTACH and O1
  re-promotes. Collision ceiling ≈ 11 (live ash plays) vs 110 telepath prompts.

## P0 — pin the unknowns (no rule code until these land)

**P0.1 — re-read live at larger n.** `uv run python -m rl.kaggle_ingest refresh --subs
54929991` then `uv run python scripts/live_postmortem.py 54929991`. Every post-mortem
number is n=26. As of the 2026-07-23 review the cache had NOT grown (still 26 games, 10
losses). If refresh yields no new episodes, say so and proceed on the n=26 read
explicitly labelled directional — do not re-run and treat the same data as confirmation.
**Pre-registered milestone kill/pivot:** if at n≥50 the loss-vs-win gap in
bench-0-after-t3 closes, or Poffin declines at bench ≤1 fall below ~0.5/game, the bench
premise is dead — pivot to the supporter axis (O8) rather than proceed on a stale read.

**P0.2 — Dunsparce (305) ACTIVE ability semantics.** Open question: never used by anyone
across 120 cached episodes, offered repeatedly at bench 0 in bench-out loss 87676383 and
declined. Id 305 (basic Dunsparce) is **not referenced anywhere in `rl/plan.py`** —
genuinely unpinned; note it differs from id 66 (Dudunsparce, `DUDUNSPARCE_IDS` at
`rl/plan.py:331`). New probe `scripts/ability_probe.py` reuses the `BehaviorTap` wrapper
(`scripts/offline_behavior.py:39`) / `DrainTap` (`scripts/deck_drain.py:55`): always
take the Dunsparce ABILITY when offered, diff (deckCount, hand, bench, active, discard)
at the **next MAIN prompt** — selection sub-prompts resolve first or the naive read shows
delta 0. If it benches basics it is the in-kit answer and supersedes or joins O7.

**P0.3 — Poffin (1086) and Hilda (1225) semantics, same probe.** Poffin: what it puts
where and its bench-space interaction (deck cost measured 1.5/use, `docs/M30.md:328`).
Hilda: the number that matters is **deck cost per use** — Hilda is absent from the M30
drain table, genuinely unmeasured; the O6 lesson is any draw effect is priced in cards
before it is forced. **This gates whether O8 is carried past screens.**

**P0.4 — telepath as measurement, not a rule.** Two cheap reads, recorded as a
weights-selection axis for a future milestone:
- Recompute attach rate **per (ep, turn)**, not per prompt. `live_postmortem.py:163-166`
  counts per prompt (confirmed); Telepath re-enters the denominator on every later MAIN
  prompt of a turn where the model acted first (~5.0 MAIN prompts/turn), so a config
  that plays more cards/turn scores lower with identical behaviour.
- Compare bench-size histograms between 54903635 and 54929991 (live baseline: bench 0
  4.9% · 1 8.3% · 2 20.4% · 3 25.3% · 4 24.9% · 5 16.1%).

**P0.5 — bed selection by defect reproduction.** Before any screen, baseline
bench-0-after-t3 games, reason-3 (BENCHED) loss share, bench ≤1 prompt share and
Poffin-at-bench≤1 play rate on all four beds (`rule:lucario`, rocket clone, grim clone,
`rule:dragapult`). Pick the primary bed by which one actually reproduces the bench
defect (M30 lesson: grim games rarely reached deck ≤6/≤10, so guard/ash engagement was
unreadable there; the rocket clone earned its place by reproducing the drain mode).

**P0.6 — offline bench-economy counters.** Extend `scripts/offline_behavior.py`.
**Reuse win (agent-verified):** the counters largely already exist in
`scripts/live_postmortem.py` and should be *ported*, not written fresh —
`bench_prompt_hist`, `min_bench_after_t3`, `missed_poffin_low`, and
`supporter_turns`/`all_turns` utilisation. Genuinely new: a **Hilda per-offer rate**
(absent from both files). Add: bench ≤1 prompt share; games with bench 0 after t3; loss
reason 3 = BENCHED (already read at `scripts/deck_drain.py:149-152`, RESULT `type==23`,
`reason` 2=deck-out / 3=benched); Poffin/basic play rate at bench ≤1; supporter-turn
utilisation + Hilda per-offer for O8. Then baseline the exact live gac config on all
beds. All M31 behaviour pins are **relative** to that measured baseline — live
absolutes do not transfer (`docs/M30.md:56`). Instrument caveat (confirmed):
`offline_behavior.py` seeds Python's `random`, not the engine RNG — identical `--seed`
yields different games, so high-variance axes need n=120, two seeds pooled.

## P1 — rule implementation (no games)

Same override law as O1/O4/O5/O6: deterministic, card-fact/id-pinned, reorders the
model's ranked list, OFF unless a fix name is passed.

- **O7 `poffinfloor`** — at bench ≤ 1 AND own `deckCount` ≥ D, promote a legal
  Buddy-Buddy Poffin (1086) PLAY. Narrower than the dead O3 tempo: fires on a *state*
  (bench ≤1 — 13.2% of live prompts), not "any END," and is deck-gated because M30 P3
  showed forced item plays convert into deck-outs vs grind decks. D comes from P0.3/P0.6.
- **O8 `drawfloor`** (Hilda 1225) — predicate implemented now, screened in parallel,
  carried only if P0.3 prices Hilda deck-cheap. Honest caveat carried into the screen: a
  forced draw supporter is exactly what O6 conserve suppresses, and the M27 weights
  route is known-bad (−13.4pp mirror). The supporter defect is real and five milestones
  old (M25 Hilda 0.57% vs teacher 2.7%; M27 supporters 3.8% vs 7.5%; live 14.2%
  turn-level, 18 stranded in losses). End-of-game hand size is confounded (wins end with
  18.4 cards *and* run 14.7 turns; faster conversion produces both) — the **per-offer
  rate is the only clean read**.
- **O9** — not an arm. Correct the `apply_play_overrides` docstring (`rl/plan.py:359-360`)
  and add `test_ash_does_not_displace_o1_telepath` pinning the actual precedence.

Wiring — three call sites, unchanged pattern from M30:
- `rl/plan.py` (+ byte-identical twin `submission/rl/plan.py`, enforced by
  `test_bundle_twins`, `tests/test_attach_override.py:223`): O7 constants + predicate,
  O8 predicate, O9 docstring correction. Both edits must land identically or the twin
  test fails.
- `submission/main.py`: default string at `:107`, call sites `:220-223`
  (`apply_attach_overrides` then `apply_play_overrides`, both fed `_ATTACH_FIXES`).
  An approved ship changes the **default string only**, never the predicate.
- `rl/matchrunner.py`: add `modelt-gacb` (gac + poffinfloor) and `modelt-gacd` (gac +
  drawfloor) to `_MODEL_FIX_KINDS` (`:64-80`); the `fn3`/`fn4` closures apply overrides
  at `:506-511`/`:437-442` unchanged. Adding the kinds auto-extends
  `test_matchrunner_spec_kinds_parse`.
- `tests/test_attach_override.py` (style: `_obs` helper at `:42-50`, SimpleNamespace
  obs): one test per rule firing, one per guard (bench threshold, deck threshold,
  non-MAIN no-op), one per composition with the existing four fixes, plus the O9
  regression test. Full `uv run pytest` green (currently ~640 collected) before any games.

## P2 — screens (kill rules that don't move their own counter)

Baseline + O7 + O8, each solo on the gac base, on the P0.5-selected primary bed plus
`rule:lucario`, n=120 pooled over two seeds via the extended `offline_behavior.py`,
compared to the P0.6 gac baseline. Power-aware criteria, fixed now (bench-0-after-t3 and
reason-3 are rare — 4 and 2 in 26 live games — so on a bed that does not pressure the
bench they read ~0 either way):

- **O7 primary (well-powered):** bench ≤1 prompt share DOWN (13.2% live ⇒ ~130 events at
  n=120) **and** Poffin-at-bench≤1 play rate UP (mechanical engagement, readable at n=60).
- **O7 guards:** deck-out (reason 2) and bench-out (reason 3) loss share not up.
- **O7 advisory** unless the bed yields ≥10 events: bench-0-after-t3, reason-3 share.
- **O8 primary:** supporter-turn utilisation UP and Hilda per-offer rate UP. **Guard:**
  deck-out share not up (this is the conserve-interaction guard — O8 must not reintroduce
  the deck-out mode O6 suppresses).
- **"Not degraded" = z > −1.96 on the pooled n=120 read, not a point comparison** — fixed
  here, before the numbers, to avoid the M30 telepath judgment call.

Any rule that does not move its own primary counter is dropped and diaried (kills are
results; M30 killed 3 of 4 this way). At most one new rule ships; a composed arm is built
only from survivors.

## P3 — battery with pre-registered bars

Reuse the M30 harness (`scripts/m30_p4_battery.sh`, `scripts/m30_p4_decide.py`) as
`scripts/m31_battery.sh` / `scripts/m31_decide.py`. Floors first, n=800 mirror last,
only for arms still alive. Re-pinned to the **LIVE gac config** (not 54903635): mirror
**0.6750** n=800 · dragapult **0.3750** n=400 · kyogre **≥0.95** · rocket **0.5675** n=400
(advisory) · grim **0.6687** (advisory). Known ratchet property (recorded, accepted):
re-pinning to the incumbent's newest estimate carries ~2.5% failure odds per gate for a
genuinely equal arm; the incumbent is the correct comparator.

**Bar — rule arms, weights frozen → non-inferiority:**
- mirror z > −1.96 vs 0.6750 AND dragapult z > −1.96 vs 0.3750 AND kyogre ≥0.95;
- **plus** the behavioural claim, offline, relative to the P0.6 gac baseline: the rule's
  own primary counter strictly improved, and neither deck-out (reason 2) nor bench-out
  (reason 3) loss share degraded.
- Tie-break among qualifiers: best primary-counter improvement, then the rocket advisory.
  Dragapult is **not** a tie-breaker (M29's trap axis).
- If no arm clears: evidence to Piotr, no ship, no exceptions.

## P4 — QC + STOP (hard rule)

Read `verify-before-consequential-actions.md` first. Then `./build_submission.sh
--checkpoint m28_winners.pt --deck clone54618168` → deck.csv md5 **8e8cf124** ✓ · 21/21
bundled weight arrays byte-match `m28_winners.pt` ✓ · direct-import assert of the new
`PKM_ATTACH_FIXES` default string ✓ (`submission/main.py:107`). QC = 3 games of the
actual bundle via `tcg.evaluation.play_games("submission/main.py", <opponent>, 3,
replay_prefix="m31_qc_<arm>")`. For a bench rule the opponent must actually pressure the
bench (gust-carrying) — name and justify the pick in the diary (M30 substituted rocket
for grim for exactly this reason). Then **STOP** for Piotr's manual replay review and
explicit go. No submit without it.

## Files to touch

- `rl/plan.py` + `submission/rl/plan.py` (twin) — O7 constants + predicate, O8 predicate,
  O9 docstring correction
- `submission/main.py` (`:107` default string, `:220-223` call sites)
- `rl/matchrunner.py` (`:64-80` spec kinds; `fn3`/`fn4` closures unchanged)
- `tests/test_attach_override.py` — firing / guard / composition tests +
  `test_ash_does_not_displace_o1_telepath`
- `scripts/offline_behavior.py` — port bench counters from `live_postmortem.py`, add
  reason-3 split, supporter-utilisation + Hilda per-offer counters
- new: `scripts/ability_probe.py`, `scripts/m31_battery.sh`, `scripts/m31_decide.py`
- `docs/M31.md` (incremental diary), `docs/M31-plan.md` (this plan), `docs/MILESTONES.md`

## Verification

1. `uv run pytest tests/test_attach_override.py -v`, then the full suite (currently
   green) — before any game runs. Twin byte-identity (`test_bundle_twins`) and spec-kind
   parse (`test_matchrunner_spec_kinds_parse`) must pass.
2. P0 semantics diaried with probe evidence; the gac offline baseline measured and pinned
   as the relative reference; the P0.5 bed choice justified by measured defect reproduction.
3. P2 screens show each surviving rule moves its own **primary** counter vs that baseline;
   every kill diaried at the moment it lands.
4. P3 decoded by `scripts/m31_decide.py` with the bars encoded in the script — the script
   prints the verdict (decode 0 = WIN).
5. P4 QC replays in `replays/m31_qc_*` for Piotr's review; ship only on his go.

## Sequence, cost & notifications

Branch `feature/m31` from `feature/m30` → P0 probes + baselines (~1h, mostly compute) →
P1 code + tests (no games) → P2 screens (~1.5–2.5h, n=120 × 3 configs, `--workers 8` MAX)
→ P3 battery (~2–3h, floors-first) → P4 QC + STOP. **Hermes Telegram** (`hermes send -t
telegram`) update at every phase transition, hourly heartbeat on any run >30 min,
immediate notification on any kill-gate or failure. Diary each experimental observation
incrementally as produced — kills as valuable as passes.

## What this milestone is deliberately NOT

- **Not a weights change** — base frozen at `m28_winners` (confirmed with Piotr this
  session). The ~72% telepath rate is weights-borne; P0.4 first establishes whether the
  number is even real per-turn before any future weights decision.
- **Not a deck-list change** — frozen at `clone54618168`. The 9-basics-in-60 read is a
  structural finding for a later milestone; changing the list breaks the BC corpus
  fidelity assumption and M4 precedent is 0/30.
- **Not a reaction to the grim 0–3 / rocket 0–2 live cells** — n=2–3 is far below
  resolvable (`floor-gate-sample-size`); let the A/B accumulate to n≥8–10 per family.

**Honest expectation.** Rule milestones have been worth tens of ELO, not hundreds —
818 → 1000 will not come from one rule. The path that has worked four times running:
keep the base frozen, convert each measured defect into a deterministic rule, gate it on
a pre-registered bar, let the live A/B resolve it. Bench economy is the next in that
queue — one rule with ~23 measured firing opportunities per 26 games, not a ladder.
