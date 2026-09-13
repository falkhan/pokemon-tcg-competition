# M46 plan — band-faithful offline opponents + pivotal-mistake guardrails

r1 2026-08-14. Piotr resolved both open forks same day (session Q&A):
**Q1 = alakazam line only** (O parked, lucario dead per m45 A4);
**Q2 = the band-weighted panel DECIDES** guard probes and the ship gate,
tuned/iono anchors demoted to advisory (registered instrument change).
Basis:
`docs/m44-live-postmortem.md` (the 800-ceiling forensics). Hard deadline:
comp ends 2026-08-16 — every step below is scoped to that window.

r2 2026-08-14 (pre-execution review, plan-vs-tree + RL/TCG practice check):
corrected evidence pins (grim seat scope, parquet source, mcts status),
rewrote A3's escalation around the M40 closure, added the A0 decoder work
item + named bars + null-panel kill calibration + panel acceptance test,
fixed B2-B4's armed-attack primitive, specified guard precedence and `core`
expansion, and added the late-ship rating math to the ship decision.

Objective: attack the two root causes the post-mortem measured —
(A) our offline opponents don't represent the 780+ band that decides ELO
(pooled wr there 0.30; grim 39% share / mirror 23% / wall+stall+dragapult
~14% at wr≈0), and (B) the pilots commit deterministic pivotal mistakes
(Run-Away-Draw self-benchout; END/RETREAT while an attack is armed) that
no measurement change fixes.

## Evidence pins (verified against the tree 2026-08-14)

1. Teacher bands for the killer families ALREADY sit in the ingest.
   (r2: per-seat bands live in `opp_decks.parquet` (14,984 rows), not
   episodes.parquet (10,022 rows = episode level); only 7,565 episodes
   have cached raw logs a `build` can read, so corpus yields may land
   lower — verify at build time. No snowball needed except optionally
   for wall.)
   | family | list | seats @1100+ (winners) | current bed |
   |---|---|---|---|
   | grim | `3121746f` | 781 (461) list-scope (r2; 1079 archetype-wide) | topgrim @1000 (all seats) |
   | mirror | `9294d9d8` | 936 (532) | m39_bc_top @900 (r2: all seats, NOT winners; and no mirror clone bed is in the gate roster today — the mirror cell is m28_winners on our own deck, so mirror1100 fills a HOLE) |
   | dragapult | `f2b4039a`+`7c605fb2`+`3631d393` | ~330 (150+) pooled | one list @900, val_acc 0.56 |
   | stall | `e234578d` (dunsparce/fan-rotom) | 62 (31); 136 @1000 list-exact (r2; 188 loose family sweep) | **NO BED EXISTS** |
   | wall | `6f87e9d4`+`e510e4d4` (kanga+crustle) | 28 (13); 172 (72) @900 (r2) | d3e4d16c @600 (!) |
2. `replay_bc build` supports `--deck-hash --min-score --winners-only
   --hand-aware`; multi-dir training is the established pooling pattern.
3. Search-boosted pilots exist in matchrunner: `solved:<ckpt>:<deck>
   [:max_nodes[:deadline_ms]]` (M40 S3 per-spec budgets). r2 corrections:
   800:400 is the only production-proven budget, and the deadline caps
   loop ITERATIONS, not wall time (M40) — bigger node budgets are
   wall-clock-unbounded. `mcts:` is EXCLUDED: it is a closed kill, not a
   ladder (M8.4 flat; M28 −11.0pp z=−3.17 with explicit do-not-reopen
   conditions). And M40 already ran solver-composite calibration at
   800:400 — every family failed its bar ("the composite wrap adds
   ~nothing against our pilot", docs/M40.md:1538-1615); the re-opening
   justification for A3 is the NEW 1100+ winners hand-aware corpora, not
   search.
4. Fix framework: `apply_play_overrides(obs, ranked, fixes)` sees full
   state; deckguard/conserve are the demote template, ash/benchfloor the
   promote template; `rl/combat.py` `_best_damage(..., scaling=True)`
   gives armed-attack detection (r2: NOT `_charged_best` — tuple return,
   affordability-blind, 0 for Alakazam without scaling; see Track B);
   probe tokens via `_MODEL_FIX_KINDS`
   (m45 `model-pz-bf`/`-gv` precedent).
5. Live ground truth per family×band vs our exact bundles exists (the
   post-mortem table) — usable as a bed-calibration target.
6. IMPORTANT context from M45 A3: benchfloor/gustveto did NOT CLEAR the
   adoption bar on tuned/iono anchors (r2 precision: −2.9pp/−3.2pp,
   INSIDE M45's own pre-registered ±4.9pp bar, and measured on the
   since-killed lucario d3 lineage — weakly informative in either
   direction). The band-panel re-probe is justified by the post-mortem,
   not by those probes (registered instrument change, Piotr sign-off =
   Q2, resolved in header).

## Track A — band-faithful opponent program

- **A0 panel + decision rule.** New gate roster weighted by the measured
  780+ mix: grim .39, mirror .23, wall .06, lucario .06 (bed exists:
  m45_bc_lucario_d3), stall .05, dragapult .04, remainder split over
  garchomp/archaludon/rocket (existing beds, uncalibrated — they carry the
  recorded-bias tag like any uncalibrated bed). Cells at n=800 × 2 seeds
  (VALIDATION Tier-3; compute is NOT a constraint — a 25-bed n=400 pass
  measures ~10 min at 8 workers, so the full Tier-3 panel is ~40 min).
  Pre-registered decision rule for ANY ship: pooled band-weighted Δ ≥
  +2pp = pass, Δ ≤ 0 = kill (m43_laneA bars precedent), AND no single
  family at z ≤ −2 (bar validated by the A0t null panel below). The
  weights derive from n=83 live games (grim's .39 is ~32 games) — the
  decode bootstraps the weights and reports whether the verdict is
  weight-stable. A QC sweep by a ≥20%-share family blocks a ship
  (post-mortem lesson 7.2). ENDOGENEITY note (r2): this panel is
  all-endogenous clones, exactly what the M22 endogeneity law and the
  2026-07-20 dragapult evaluation-only decision warn about; A3
  calibration-to-live-band-wr is the mitigation, and if calibration fails
  on grim AND mirror the panel is DEMOTED TO ADVISORY (M26 precedent) —
  anchors + guards-only fallback, escalate to Piotr.
- **A0t decoder + spec tooling (tonight, ~2-3h — the real overnight
  item, r2).** No existing decoder runs the A0 rule: gate_spec.py pools
  UNWEIGHTED and hash-refuses post-hoc spec edits; m40_decide.py has the
  right statistics (share-weighted Δ with correct SE) but is frozen,
  roster-hardwired, weights by whole-population mix, and reads a
  different cell layout than gate_spec writes; data/m39_live_mix.json is
  missing on this box; per-family kill logic exists nowhere; `stall`
  must be added to BED_FAMILY in BOTH m40_decide.py and m39_live_mix.py
  (known-drift pair). Build the band decoder + spec and hash BEFORE any
  panel cell runs. Then CALIBRATE the family-kill bar: one
  control-vs-control null panel (~10-20 min) — z ≤ −2 over ~9 families
  fires ≈19% of the time under a true null (the M26 uncalibrated-kill
  failure); if the null panel fires any family, re-set the bar from the
  observed null spread before decoding any probe.
- **A0v acceptance test (before any probe read is believed, r2).** The
  candidate A/B m41b_wide_prod vs ppo_best_m44_K_r1 has a KNOWN live
  answer (810 vs 735). The panel must rank m41b above K; if it does not,
  the instrument failed validation → guards-only fallback, escalate to
  Piotr. This is the same A/B the ship path needs anyway — it costs
  nothing extra.
- **A1 corpora (tonight, CPU, ~1h, parallel).** r2 mechanics: first
  export the 5 missing deck CSVs via `scripts/export_opp_deck.py --hash
  <8hex>` (e234578d, 6f87e9d4, e510e4d4, f2b4039a, 7c605fb2); then SEVEN
  `replay_bc build` invocations, not five (`--deck-hash` is
  single-valued), all with IDENTICAL encoding flags (mixed-width shards
  are refused by `--init-wide` pooling):
  `grim1100` = 3121746f @1100 winners hand-aware (r2: ~461 seats, not
  631 — thinner than budgeted but workable; lucario shipped on 602);
  `mirror1100` = 9294d9d8 @1100 winners hand-aware;
  `drag1100` = f2b4039a + 7c605fb2 + 3631d393 @1100 (three builds,
  multi-dir train);
  `stall1000` = e234578d @1000 hand-aware (first stall bed);
  `wall900` = 6f87e9d4 + e510e4d4 @900 (two builds, multi-dir train;
  thin — accept or top up by targeted snowball of those lists' top subs
  if time allows).
- **A2 clones (tonight GPU, ~15 min TOTAL — r2: measured 20-45 s/draw
  on the 3070, M39/M43/M45 records; NOT overnight — the night goes to
  A0t + Track B/C code):** G-13 3-draw protocol (10/9/8 ep, init
  m28_winners). Draw selection by STRENGTH (A3 calibration read), not
  val_acc — the val_acc→strength gap is a standing registered lesson.
- **A3 bed calibration (the new instrument, 08-15 AM):** a bed is
  CALIBRATED when clone-vs-our-live-bundle (n=400, both seats) reproduces
  the live band wr within CI: grim bed should beat the K bundle ≈0.69,
  mirror bed ≈0.63, stall ≈1.0→use ≥0.8 floor, dragapult ≈1.0→≥0.8,
  wall ≈0.8+. r2: the calibration lever is the 1100+ winners hand-aware
  corpora THEMSELVES — M40 already ran solver-composite calibration and
  every family failed its bar; those corpora are the re-opening
  justification. A raw clone that falls short gets ONE escalation rung:
  `solved:<ckpt>:<deck>:800:400` (the only proven budget; larger node
  counts are wall-clock-unbounded and the M40 record says search adds
  ~nothing against our pilot). A solver-assisted pass is tagged
  "calibrated-via-solver", distinct from full calibration. Still-short
  beds keep their measured gap as a RECORDED BIAS on every gate read
  (never silently treated as full-strength).
- **A4 (stretch / post-comp):** calibrated clones into the PPO opponent
  pool; planzero-consistent collection+training (close the 58.3%
  non-zero-plan train/serve skew). Not on the pre-deadline critical path.

## Track B — guardrail fixes

All demote/promote reorders inside `apply_play_overrides` (fix-string
class, ship-selectable, replay-invisible when off). Piotr's amendment
2026-08-14: these are built as DECK-AGNOSTIC rules wherever possible
(Track C below carries the generic design); B1-B4 are the
alakazam-probed instances of those generic implementations. Names +
designs:

- **B1 `dudguard0` — the self-benchout veto.** Demote every Dudunsparce
  ABILITY below the whole rest of the ranking when our bench-alive == 0.
  Run Away Draw shuffles the LAST Pokémon away = instant loss; strict
  domination, no counterexample exists. Adoption bar: unit tests + a
  no-regression floor (fires in ~2-4% of games; 3 live autolosses found).
- **Shared "armed" primitive (r2 — B2/B3/B4 all use it):** armed :=
  `combat._best_damage(active, opp_active, scaling=True) > 0`
  (affordability-aware, this-turn). NOT `_charged_best` as r1 wrote — it
  returns a (damage, cost) TUPLE, ignores affordability by design, and
  reads 0 for our own Alakazam without `scaling=True` (printed damage 0),
  so the r1 spec would never fire for the deck's main attacker. Any
  combat.py touch mirrors into tcg/combat.py (contract at rl/combat.py
  header).
- **B2 `attackfloor` — no END while an attack is armed.** If `ranked[0]`
  is END and an ARMED attack option is on the menu, promote the model's
  best-ranked ATTACK. (r2: the damage>0 armed gate removes r1's accepted
  0-damage/immune-wall edge case by construction; the Spiky counter-chip
  edge remains — the probe decides. Template: the existing `tempo` fix,
  same gated-on-END promote shape.)
- **B3 `retreatguard` — no retreat that disarms an armed attacker.** If
  `ranked[0]` is RETREAT while the active is ARMED, demote RETREAT below
  the best ATTACK. The damage>0 gate lets genuine wall-escape retreats
  through (mirror t5 evidence: ep92684808). r2 note: this also forbids
  the defensive sac-retreat (pull a damaged armed attacker out of KO
  range, sacrifice a promo) — vs 360-one-shot grim the enforced
  stay-and-trade is usually right and retreat cost often disarms anyway,
  but the mirror family cell is where a kill would show; its per-family
  n must have power (A0's n=800 × 2 seeds).
- **B4 `bosscombo` — no gust without a follow-up.** Demote Boss's Orders
  PLAY when no ARMED attack exists this turn (complements `gustveto`,
  which handles the opp-match-point case). Live Boss play rate is
  0.4-2.7% at offer; the one observed use had no attack behind it.
  r2 dependency: B4 forbids gust-to-strand, the classic anti-stall tempo
  play, and our live stall wr is 0.00 — B4 adoption ADDITIONALLY requires
  the A1 stall cell present in the panel and non-killing, not just
  pooled Δ ≥ 0.
- **B5 string restore (K line).** A/B the current
  `conserve,racemode2,racemode4,planzero` vs
  `+ash,ashguard,tempo,deckguard,hammer` — the Sacred-Ash-timing
  (13-42 vs the incumbent's 4-12) and END-with-item (33% vs 18%)
  regressions look string-inherited.

**Probe protocol:** new `_MODEL_FIX_KINDS` tokens `model-pz-dg0`, `-af`,
`-rg`, `-bc` (+ the B5 composite, recorded as a composite — not
single-variable, accepted under deadline), single-variable on the
candidate nets, n=800 × 2 seeds/cell (r2: Tier-3; compute trivial).
PRIMARY read = the A0 band-weighted panel; tuned/iono anchors run as
ADVISORY only (Q2). Pre-registered bars: B1 no-regression; B2-B5
pooled-weighted Δ ≥ 0 AND no family kill z ≤ −2 (bar as calibrated by
the A0t null panel). Probes decode only AFTER A0v passes. All probe
results diaried at production time, kills included.

## Track C — generic rules layer (deck-agnostic, Piotr's amendment)

Goal: instill game-rule knowledge and best practices into EVERY pilot,
current and future, not per-deck patches. Constraint discovered during
planning: the engine is a native `libcg.so` and NO card has
ability/effect text in our features — generic rules therefore build on
(a) option types + structured card features (is_ex, is_basic, costs,
retreat_cost — all deck-agnostic), (b) `rl/combat.py` damage/readiness
math, and (c) card-fact tables MINED EMPIRICALLY from the 10k-episode
replay corpus.

- **C3 first (enabler): `scripts/build_rule_facts.py` → `data/
  rule_facts.json`.** One log-walking pass over the raw episodes mining:
  1. *self-removal abilities* — ABILITY use followed by the user's own
     Pokémon leaving play (discovers Run-Away-Draw-class effects across
     the whole meta, not just Dudunsparce);
  2. *zero-damage pairs* — (attacker card, defender card) where ATTACK
     events dealt 0 across ≥N observations (discovers Crustle-class
     immunities without effect text);
  3. *per-card deck cost* — net own-deck change attributable to each
     optional play/ability (generalizes the m30/m39 hand-measured burn
     table: Enriching 4.0/attach etc.);
  4. *gust-class cards* — PLAY followed by an opponent forced switch
     (generalizes GUST_IDS beyond Boss's Orders; SWITCH=8/CHANGE=9 log
     events).
  Committed artifact, regenerated on meta refresh; tests assert the
  seed facts are discovered (Dudunsparce self-removal, Crustle-vs-ex
  zero damage, Boss's Orders gust).
  r2 implementation notes (verified vs cg/api.py): there is NO ABILITY
  LogType and no abilityId on Option — ability use is mined from each
  seat's DECISION stream (chosen ABILITY option at a MAIN prompt,
  resolved area+index → board card id; precedent
  rl/postmortem.py describe_option) plus board-state diffs; raw episodes
  carry both seats' streams, so teacher seats are minable. A 0-damage
  attack likely emits NO HP_CHANGE event — "ATTACK with no defender
  HP_CHANGE" counts as 0. Zero-damage pairs use ALL-zero semantics: one
  positive damage observation vetoes the pair (conditional protections
  otherwise poison the table); per-card deck cost uses the WORST-CASE
  observed cost, not the mean. Reuse scripts/m37_pm_burn.py's
  deckCount-delta attribution and rl/postmortem.py primitives — do not
  write the walker fresh (1.5h only holds with reuse). B1-B5 do NOT
  block on C3 — only C1 lastmon/deckzero and the C2 generalizations do.
  Side task: the post-mortem's narrative tool
  (scratchpad/game_narrative.py) was never committed — rewrite it on
  rl/postmortem.py and commit it this time (standing save-down rule).
- **C1 Tier-0 universal vetoes (game-rule level; adoption bar = tests +
  no-regression floor, they are strict-domination class):**
  - `lastmon` — demote any option in the self-removal registry when it
    would leave us with zero Pokémon in play (B1 `dudguard0` = its
    Dudunsparce instance).
  - `benchzero` — never END at bench==0 while a basic PLAY is legal
    (promote the model's best-ranked basic). Distinct from the M45-killed
    `benchfloor` (≤1): the ==0 case is donk/bench-out protection and
    near-strict domination (2 live flash-losses + 3 self-kills found).
  - `deckzero` — demote any optional play whose mined deck-cost would
    take our own deck to 0 (a turn-start deck-out) below attacks/END.
- **C2 Tier-1 best-practice reorders (probe-gated on the panel, Q2):**
  B2 `attackfloor`, B3 `retreatguard`, B4 `bosscombo` (gust-class
  registry, not Boss-specific) — already generic by construction; plus
  - `futileattack` — demote repeating an attack the zero-damage table
    (or this game's own observed 0) says does nothing to the current
    defender, whenever a damaging attack or gust-class play exists
    (the O-vs-Crustle 17-turn tunnel fix, generalized);
  - `deadattach` — generalize O19 `deadenergy` (combat.energy_is_dead
    is already deck-agnostic): demote attaching energy a target cannot
    use (note: deadenergy lives in apply_ATTACH_overrides, not the play
    path).
- **Guard precedence (r2 — materially matters):** apply_play_overrides
  picks at most ONE promotion, first-match-wins, in a fixed chain (ash →
  gustsnipe → benchfloor → poffinfloor → drawfloor → tempo → hammer).
  New promote guards slot deliberately: `benchzero` BEFORE `attackfloor`
  — playing a basic is non-turn-ending, attacking ENDS the turn, so at
  bench==0 with an armed attack the basic must win the promotion (the
  attack can still happen later the same turn). General rule: promotions
  of non-turn-ending options outrank promotions of turn-ending ones;
  demotes compose after (they already fire only when ranked[0] is itself
  demoted — correct veto semantics for B1/B4). Unit tests must cover the
  STACKED cases (bench==0 + armed + ranked[0]==END, etc.).
- **Packaging: a `core` meta-token** that expands to the ADOPTED generic
  guards — explicit in every ship string (no silent always-on defaults:
  preserves ship_verify's fix-set equality check and single-variable
  probe discipline). r2 constraint: `core` NEVER appears literally in the
  bundle string or in a `_MODEL_FIX_KINDS` frozenset — ship_verify check
  3 fails on any name without a `*_FIX_*` constant, and check 3b is
  literal set equality between the bundle string and the gate arm's
  frozenset. Expansion happens when COMPOSING the `--fixes` string, and
  the matchrunner token stores the same expanded set; both sides carry
  the expanded names, keeping 3b meaningful. Rides along on any future deck's ship; each ship's
  own probe/gate still validates it there. Post-comp: apply `core` in
  PPO collection pools too, so training and serving share the rules
  (pairs with A4's planzero-consistency work).

## Ship path (08-15, if bars pass)

Candidate net A/B on the panel: `m41b_wide_prod` vs `ppo_best_m44_K_r1` —
this IS the A0v acceptance test (the live 735-vs-810 read is the known
answer; a panel that inverts it fails validation). Winner + adopted guard
string → gate vs the 810 incumbent as control under the A0 decision rule
→ export → `ship_verify` → `qc_battery` + bespoke stall leg (new: we've
never QC'd vs stall and its live wr is 0.00) → **FULL STOP for Piotr's
replay review + explicit go + deck confirmation** → ship + MODELS dict
entry in the ship commit.

r2 ship-path parameters (named now so the ritual can't drift):
- `build_submission.sh --deck alakazam_v2_h4 --fixes <expanded string>`.
- `ship_verify --deck alakazam_v2_h4 --gate-arm <adopted token> --corpus`
  = a BC corpus (m44 ritual precedent) — NOT the PPO shards: check 6a
  demands ≤1% non-zero-plan rows for a planzero bundle and the PPO shards
  carry 58.3% (the M44 registered exception; don't re-trip it).
- `qc_battery --prev dist/submission_neural_20260813_145528.tar.gz`
  (the K alakazam tarball) — PINNED, never the `sorted(dist)[-2]` default
  (any re-export silently turns the mirror leg into a self-mirror).
- qc_battery runs scripts/ci_gate.py first and aborts on failure — run it
  early, not at 22:00.
- The stall leg is a separate `play_games` call (qc_battery has no flag)
  vs a bed built the SAME DAY — label its read "uncalibrated opponent".

r2 late-ship rating math (decision input for Piotr): a sub shipped 08-15
evening accrues ~35-60 rated games by comp close (~15-18 in hour 1, then
~1.3/h measured) = the n≈50/±93-ELO catastrophe-detection row of the
VALIDATION dwell table. Recent subs peaked 950-984 in hour 1 then decayed
for 20h — the number visible at close is an early-burst artifact, not a
settled rating; and 55485260 STOPPED being matched after 25 games when
its score sank. Declared dwell-n (G-10): 40 games, catastrophe screen
only — no strength verdict at that n. OPEN QUESTION for Piotr before
ship: how is final standing computed (best sub or latest sub)? Best-sub →
a late ship is free option value; latest-sub → shipping something that
reads low at close is a real risk. Related option (Piotr's call): bank
the guards-only ship (B1+B5 on m41b) EARLY on 08-15 — it depends on
NOTHING in Track A (B1's bar is unit tests + no-regression floor) and
buys convergence time, at the cost of a second QC + review cycle.

Fallback: guards-only minimal ship (B1 + B5 on the m41b net) — still
gated, still QC'd, never skipped. r2 HARD TRIGGER (pre-registered, no
sunk-cost continuation): if the candidate gate has not STARTED by 16:00
on 08-15, the guards-only ship becomes the milestone regardless of
pipeline state.

## Timeline / cost (workers ≤ 8 everywhere; pausable state on >1h runs)

| when | what | wall |
|---|---|---|
| tonight | A1: 5 deck-CSV exports + 7 corpus builds (CPU, parallel) | ~1h |
| tonight | A2 clone draws (GPU; measured 20-45 s/draw — r2, was "~6h") | ~15 min |
| tonight | A0t: band decoder + spec hash + control-vs-control null panel | ~2-3h |
| tonight | C3 rule-facts miner (~1.5-2h, reuse m37_pm_burn/postmortem) → C1/C2+B guard code + stacked tests | ~3h |
| 08-15 AM | A2 floors + A3 calibration battery; A0v acceptance A/B; B probes (n=800×2s) | ~3h |
| 08-15 PM | gate vs incumbent + export + ship_verify + QC (+ stall leg) | ~2.5h |
| 08-15 16:00 | HARD fallback trigger: candidate gate not started → guards-only ship path | — |
| 08-15 eve | Piotr review window → ship (dwell-n 40, catastrophe screen only) | — |

Hermes updates at every stage transition + hourly heartbeat on the
overnight runs; failures/kills notified immediately.

## Kill criteria / escalations

- A0v FAIL (panel ranks K above m41b, contradicting live) → instrument
  invalid, panel demoted to advisory → guards-only fallback + escalate
  to Piotr (r2).
- A0t null panel fires a family kill → the z ≤ −2 bar is uncalibrated
  (M26) → re-set from the null spread before decoding anything (r2).
- A3 calibration unreachable at the 800:400 solver rung → bed ships with
  a recorded bias; if grim AND mirror both miss by >15pp, the panel is
  not trustworthy → demote to advisory (M26 precedent) and escalate to
  Piotr before any ship decision.
- Any guard probing negative on the PANEL → not adopted (anchors alone
  can neither adopt nor kill this milestone — Q2).
- Gate kill under the A0 rule → NO SHIP without explicit override
  (M44's K precedent now has its live answer: the kill was right).
- 16:00 08-15 hard trigger (ship path) → guards-only ship becomes the
  milestone (r2).
- Piotr forks Q1/Q2: RESOLVED (header) — alakazam only; panel decides,
  anchor disagreements diaried as findings, never deciding. OPEN r2
  question for Piotr: final-standing computation (best sub vs latest
  sub) + whether to bank the guards-only ship early (ship path).
