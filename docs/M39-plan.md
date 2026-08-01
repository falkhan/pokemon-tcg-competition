# M39 plan — breach 800: fix what we measure, then fix what loses

Drafted 2026-08-01 from the [M38 post-mortem](m38-post-mortem.md), a critical
review of the [M38 diary](M38.md) and [BACKLOG](BACKLOG.md). Status: DRAFT,
awaiting Piotr's review (decision points at the end).

## Objective

Implied ELO **≥ 800, converged (n ≥ 50)** on a live sub. The campaign has
never held it: M35 peaked at 817 on a hot pool, M37 settled at ~719, M38
regressed to 659. Across M37+M38 we are **0-6 in the 800+ band** and
6-8 (0.43) in the 700-band — the ceiling is not variance.

**The arithmetic of 800.** Implied ELO tracks roughly avg_opp + ~7 per WR
percentage point above 50%. At avg_opp ≈ 700 that means **WR ≈ 0.62–0.65
sustained**. The M38 loss anatomy prices the levers:

| lever | live evidence (M38, 54 games) | plausible gain |
|---|---|---|
| wall+grim+stall block | 2-13 (~13% WR, ~28% of games) | → 50% ≈ +11pp WR ≈ +75 ELO |
| deck-races from winning positions | 5 losses (18% of all losses) | subset of above + mirror/garchomp |
| 700-band sweeps (43% of losses) | policy-quality ceiling | the remaining ~+40–60 ELO |

**Conclusion the plan is built on: the rules/econ lane alone caps out around
~740–760. Breaching 800 requires BOTH the anti-deck-out package AND a
policy-quality gain in the losing matchups.** Two lanes, shipped separately
so each is measured live (M38's single-shot ship left the lanes confounded
until the post-mortem untangled them).

## Critical-review findings that shape this plan

From auditing the M38 diary and BACKLOG against the post-mortem — these are
the *process* defects; the post-mortem already ranks the *product* defects.

1. **A pre-registered decision rule fired and was overridden.** G5 measured
   the rule stack at −7pp on the wall bed; decision-3's trigger ("strip only
   if actively harmful") was explicitly MET in the diary — and gacfr3
   shipped anyway on minimal-diff grounds. Live wall went 1-5. Guardrail G-1
   below makes triggers binding.
2. **The pooled gate was honest but mis-weighted, and the data to weight it
   existed at gate time.** The +6.1pp was carried by iono/dragapult
   (+14/+17pp, ~7% of live games); re-weighted by the M37 live mix (already
   in the M37 post-mortem) the delta was ≈0. This was knowable before
   submit. Guardrail G-3: the weighted pool IS the gate number.
3. **Watch items were picked by offline z-score, not live share.** Rocket
   (−6pp offline, 0 live games) was THE watch item; wall/grim/stall (28% of
   live games) had none. Guardrail G-2.
4. **QC predicted nothing.** 11W-1L against a roster covering zero of the
   top-3 live loss families. The multi-deck QC rule was followed to the
   letter and still measured the wrong thing.
5. **Bed *construction* age was exempt from the drift law.** The grim clone
   was re-pinned (numbers refreshed) but the clone itself is an M26-era
   600-band artifact; live grim pilots at 738–838 flipped 6-2 → 1-6. Third
   bed-fidelity transfer failure (garchomp M35, wall-solver M37, grim M38).
   Guardrail G-6: beds used in ship gates must be ≤1 meta old.
6. **The M37 ranked recommendations lost to the attractive lane.** M38 spent
   its cycle on the teacher-fix (H1) — which delivered instruments and two
   clean kills but zero live ELO — while rec #2 (racemode4) was never built
   and rec #4 (AWR) never launched. M39 schedules directly from the
   loss-mass ranking.
7. **Post-mortem rec #4 (cross-family fine-tune) contains a vehicle error
   this plan corrects.** BC imitates the demonstrator's *seat*: fine-tuning
   our alakazam policy on grim/wall winner seats teaches piloting *their*
   decks — action distributions we never face — the same
   demonstration-doesn't-transfer failure as the teacher distill, dressed in
   live data. The transferable corpus is **same-deck alakazam winner seats
   filtered to the losing matchups** (how strong alakazam pilots beat grim/
   wall/stall). Cross-family seats are harvested for **beds only** (P0),
   never for labels (P3).
8. **BACKLOG re-scopes** (strike/move on milestone start): "Rule-stack
   strip" → P1. "Kanga-only wall blindspot" → P2a, generalized. "racemode4"
   → P2b. "Universal margin raceconserve" → subsumed by P2a (post-mortem
   concurs). "Band gate refresh" → P0 ride-along. **Grim entry on the
   stop-investing list: REVOKED** (1-6 live is new evidence; the stop-list
   logic itself worked as designed). **"Gen-2 collect + value-net retrain"
   needs re-scoping, not just deferral**: M38's central negative result —
   solver-teacher labels cannot beat the winners lineage at any dose — cuts
   the value of any further solver-teacher corpus. If a value-net retrain
   returns, it trains on the harvest corpus. W_COUNTER stays parked (E0b).

## Deck decision (2026-08-01, discussed with Piotr): STAY on alakazam — with a falsification test

Question raised: should M39 swap decks, and would the planned pilot work
transfer if we ever do?

**Call: stay.** Grounds:

1. **The deck is not the ceiling.** The harvest shows other pilots flying
   the exact same list (hash 9294d9d8) to 947 / 977 / **1058**, and
   alakazam+kadabra is the top-scoring family in the meta recon (903+). We
   are targeting 800 in a vehicle that demonstrably cruises past 1000 in
   other hands — the gap is the pilot.
2. **Our best asset is deck-bound.** The 800–1058-band winners corpus (231
   seats) is the richest demonstration pool we've found for ANY deck (the
   wall corpus tops out ~830). Imitation's ceiling is the demonstrator
   band; alakazam has the highest one available.
3. **Swap cost vs stay cost is asymmetric and shrinking.** A swap resets
   the corpus, the m28→cont3 lineage, the mirror 7-2 (~17% of live games),
   and the rules stack. Meanwhile M38's harvest+clone pipeline cut the
   cost of a future swap from weeks to days — staying now is low-regret.

**The honest counterargument, kept live:** the ~2.5 cards/turn engine burn
is deck-intrinsic; the wall/stall deck-race weakness comes bundled with the
deck. The 900+ pilots manage it — but "manage or dodge" is an open question
the P3 corpus build answers for free:

- **Falsification test (rides on P3, no extra cost):** the vs-loss corpus
  census — how many 800+ alakazam winner seats exist AGAINST
  grim/wall/stall? Plenty → the matchups are winnable with better lines,
  stay-decision confirmed. Nearly none → top pilots lose or dodge them too;
  the econ package (P2) is the only lever, and a deck hedge becomes an M40
  agenda item with data behind it.
- **Optional hedge (P5, default OUT of scope — Piotr's call):**
  clone-scouting. The `m38_bc_wall` recipe (138 seats → a pilot that beats
  our champion 2:1 in its matchup) can BC-clone the best wall or dragapult
  seats and run them on the weighted roster — a day of compute for an
  actual number on "what a swapped-deck pilot would be worth." Parked in
  BACKLOG unless Piotr pulls it in.

### Transferability of the M39 work (if a swap ever comes)

- **Fully deck-agnostic** — all of P0 (harvest snowball, clone-bed
  construction, weighted gate, QC roster, bed-era guardrail), the campaign
  laws (dose ~1 epoch, winners-lineage-not-teacher-distill, re-measure
  rules per net), and the P3 *vehicle* (low-dose fine-tune on same-deck
  winner seats works for any deck with harvestable winners).
- **Transferable mechanism, deck-specific parameters** — the P2 rules
  lane. Trigger sets are keyed to OPPONENT ids (meta-side: Crustle, Tusk,
  Trevenant, Fan Rotom) and transfer to any deck we pilot; the demote
  lists are OUR card ids (Enriching Energy, Poké Pad, Dawn, Sacred Ash…)
  and would be re-instantiated per deck. Roughly 70% pattern / 30% this
  engine. A lean-engine deck might not need racemode4 at all.
- **Not transferable** — checkpoints and corpora (the learned policy is
  alakazam lines; a new deck means fresh BC from its winner seats, not a
  fine-tune of cont3). And one thing that follows us deck-independently in
  the wrong way: the 43% sweep mass (supporter under-play, slow setup) is
  a property of BC imitation quality, not of this deck — swapping does not
  escape the P4 design question.

## Process guardrails (standing, binding from M39 on)

- **G-1 Binding triggers.** A pre-registered decision trigger that fires
  flips the default. Overriding requires a diary entry titled `OVERRIDE`
  with rationale, written *before* export.
- **G-2 Watch items by live share.** Ranked by (live opponent share ×
  offline effect), never offline z alone. Every family ≥10% of the last
  live sample gets a watch item or an explicit waiver.
- **G-3 Weighted gate.** Ship gates pool per-bed WR weighted by the live
  opponent mix (config committed as `data/m39_live_mix.json`, from the
  pooled M37+M38 samples: lucario ~.19, mirror ~.17, grim ~.13, wall ~.11,
  archaludon ~.09, dragapult ~.07, stall ~.04, iono ~.02, rest ~.18). The
  weighted number is THE gate; unweighted reported for continuity. A gate
  whose bed roster covers <80% of live-mix mass is invalid.
- **G-4 Re-measure rules against every new net.** Any net change re-runs the
  G5-style on/off matrix before ship — the M38 law: rules tuned on an old
  net's behavior can actively harm a new one.
- **G-5 Pool-drift control.** The previous sub stays in the monitor as the
  in-window comparator; a new ship is judged against the parent's
  same-window ELO, not its frozen number.
- **G-6 Bed era check.** Every bed in a ship gate must be built from the
  current or previous meta's live seats, or be re-cloned first.

## Phases

### P0 — Measurement integrity (BLOCKING: no gate, no ship until done)

The post-mortem's P0, plus the guardrail infrastructure. Everything M39
measures depends on this; it goes first and alone.

1. **Harvest round 3** (snowball from the M38 sample's sub ids): grim/marnie
   winner seats at 738–838 (CoCoSh, yujinki, Hamachi…), fan-rotom and
   hop-stall seats, 800+ mirror seats, wall top-ups. Also chase the
   903–1058 mirror band (P3 feedstock).
2. **New beds**: `m39_bc_grim` (700–850 seats — retires the M26 relic, with
   an epoch marker in MILESTONES.md), `m39_bc_stall` (fan-rotom +
   trevenant seats; if seat volume is thin, a combined stall bed is
   acceptable v1). Keep `m38_bc_wall` (validated live-faithful).
3. **Weighted gate harness**: `data/m39_live_mix.json` + weighted pooling in
   the gate script (extend `scripts/m38_gate.sh` → `scripts/m39_gate.sh`).
4. **Fresh pins** on the full roster: M38 ship config (cont3+gacfr3),
   cont3 plain, plain champion. These are the baselines every M39 arm is
   judged against.
5. **QC battery fix**: add wall, grim, and stall legs (3 games each) to
   `scripts/qc_battery.py` alongside tuned/iono/dragapult + previous-ship
   mirror — QC must cover the top-3 live loss families forever after.
6. Ride-along: re-freeze `band_decode.py` on the current 700-band sample.

Exit criteria: new beds pinned with 2-seed n=800 champion baselines; QC
battery runs green mechanically; weighted-gate harness smoke-tested.

### P1 — Strip ship (cheap, single-variable, pre-funded by G5)

Candidate: **`m38_w9294_cont3` PLAIN** — the exact G5 "plain" config
(wall .323 vs shipped stack's .254, no significant loss anywhere measured).

- First: a one-page **per-rule strip ledger** defining "plain" precisely —
  which O-rules remain (deckguard/conserve M30 is validated ✅-holds and is
  NOT part of the gacfr3 economy stack; it stays). Match the G5 arm
  byte-for-byte.
- Gate on the P0 roster, weighted (G-3). Requirement: no weighted-pool
  regression vs the M38 ship config pin; wall/stall cells ≥ the plain pins.
- Export → `prize_semantics_probe` → expanded QC → **STOP for Piotr's
  replay review and explicit go** → submit, monitor row in the ship commit.
- Watch items (per G-2): wall+stall WR (expect improvement — this is the
  live test of G5's −7pp), mirror (does losing the stack cost the 7-2?),
  grim (no rules-lane change expected yet).

This ship doubles as the live falsification test of the G5 measurement and
as a fresh pool-drift control for Ship B.

### P2 — Anti-deck-out package (rules lane, on real beds)

Targets the 29% deck-race loss mass, 5 of them thrown from winning
positions. Two independent, separately-measured changes:

- **P2a — stall/pressure coverage.** Add Fan Rotom and Hop's
  Phantump/Trevenant (+ Kangaskhan variants, the BACKLOG's original
  blindspot) to `_RACEMODE_PRESSURE_IDS`; ship the margin-gated pressure
  branch — `racemode2`'s split logic already exists at
  [plan.py:559](../rl/plan.py:559), never shipped live. Trigger sets get a
  regression test enumerating covered opponent ids so the next new stall
  variant is a one-line diff.
- **P2b — racemode4 burn-source demote.** Demote the *real* burn (M37
  audit, twice confirmed): Enriching Energy (−4/attach), Poké Pad, surplus
  Dawn/Hilda once set up, Sacred Ash before deck ≤ 2. Margin/state-gated,
  NOT blanket — the engine is needed when not in a deck race. Ceiling ~5–6
  cards/game; 5 of 8 M38 deck-race losses were within a ≤9-card margin.

Measurement: single-variable cells on `m38_bc_wall` + `m39_bc_stall` +
mirror (mirror deck-outs exist too), n=800 2-seed. Kill gate: a variant
ships only if wall+stall improves ≥5pp with no weighted-pool harm (G-3) —
and per G-4, the surviving package re-runs the on/off matrix on whatever
net P3 produces.

### P3 — Targeted fine-tune (policy lane, corrected corpus recipe)

The vehicle that worked (low-dose fine-tune on live winner seats, dose law
~1 epoch) aimed at the corpus flaw that kept M38's gains in-family — with
finding #7's correction applied.

- **Corpus `bc_m39_w9294_vsloss`**: same-deck (9294d9d8 + 6934f4)
  **winner seats filtered to games against grim/wall/stall/dragapult
  opponents** — demonstrations of strong alakazam pilots beating what beats
  us, in our own action space. Snowball until ≥100 seats; if thin, blend
  with the general 800+ corpus and upweight the vs-loss rows.
- Arms (all on the cont3 lineage, champion recipe, best-val≈epoch-1):
  1. `m39_vsloss` — the filtered corpus, low dose.
  2. `m39_vsloss_awr` — same corpus, `--outcome-weight` (the M37-validated
     AWR hook, never launched; near-zero incremental cost as an arm).
  3. optional `m39_mirror903` — 903–1058-band mirror depth.
- Gate: weighted pool on the P0 roster vs the cont3-plain pin. **The
  grim/wall/stall/dragapult cells are the point — a pooled win carried by
  in-family cells does not ship** (the M38 lesson, now mechanically
  enforced by G-3).
- Kill condition: no arm moves the loss-matchup cells ≥3pp pooled → the
  lane is dead for imitation and the P4 design question fires early.

### P4 — Ship B + the sweep-mass design decision

- Ship B candidate: best P3 net × surviving P2 rules, G5-style on/off
  matrix (G-4), weighted gate, expanded QC, Piotr review, submit.
- **Success check against the objective**: converged implied ELO vs the
  P1 ship (G-5 in-window comparison). 800+ band results are now visible in
  QC-adjacent terms via the weighted gate's high-band beds.
- **If the 43% sweep mass hasn't moved** (P3 kill or live flatness), the
  scoping question goes to Piotr with the fresh evidence, candidates in
  post-mortem order: opponent-conditional fine-tune arms (per-family heads
  or opponent-deck features), vs accepting the BC ceiling and pushing the
  econ/rules lane. This is a design fork, not a default — standing rule:
  prompt Piotr.

## Planned code changes (file-by-file)

Grounded against the current tree 2026-08-01; verified line refs may drift
once P0 lands. Twin rule applies throughout: any `rl/*` change regenerates
`submission/rl/*` via the export and rides with the twin-parity tests.

### P0 — measurement integrity

- **`data/m39_live_mix.json`** (NEW): committed live-mix weights (pooled
  M37+M38 samples) + a `coverage` field the gate script checks against the
  80% rule (G-3).
- **`scripts/m39_gate.sh`** (NEW, from `m38_gate.sh`): bed roster swaps in
  `m39_bc_grim` / `m39_bc_stall`, drops the M26 grim relic.
- **`scripts/m39_decide.py`** (NEW, follows the `m3X_decide.py`
  convention): weighted pooling over per-bed cells using the live-mix
  json; prints weighted AND unweighted pools + per-cell z vs control;
  refuses to emit a verdict if roster coverage < 80% of mix mass.
- **`scripts/m39_build_beds.sh`** (NEW, from the m25/m38 clone recipes):
  harvest-round-3 snowball (`kaggle_ingest` refresh on the M38-sample sub
  ids), then `replay_bc build` with `--only-deck-hash` for the grim
  700–850 seats and the fan-rotom/trevenant stall seats (`--min-score`
  per family), then clone training → `checkpoints/m39_bc_grim.pt`,
  `checkpoints/m39_bc_stall.pt`. No `rl/` code change expected — existing
  seat-side filters suffice for beds.
- **`scripts/qc_battery.py`**: add a `MODEL_AGENTS` dict
  (`{name: (checkpoint, deck_csv)}`) with wall/grim/stall legs beside the
  existing `SAMPLE_AGENTS` + prev-ship mirror; spawn model opponents via
  the `model:<ckpt>:<deck>` spec already used by the gate batteries. QC
  covers the top-3 live loss families permanently.
- **`scripts/band_decode.py`**: re-freeze weights on the current 700-band
  sample (data refresh; constants updated in place).
- **`MILESTONES.md`**: epoch marker retiring `m25_bc_grim_54861775.pt` as
  a gate bed (G-6).

### P1 — strip ship

- **No `rl/plan.py` logic change.** The ship config drops the gacfr3 fix
  string to match the G5 "plain" arm byte-for-byte; deckguard/M30 conserve
  is not part of that stack and stays. The per-rule strip ledger lands as
  a table in this doc before export.
- **Ride-along hardening** (BACKLOG, M37 audit): unknown names in
  `_ATTACH_FIXES` / play-fix strings currently silently ignored — fail
  loudly (`ValueError`) on unrecognized names, + test. We are editing fix
  configs this milestone; this is the moment the typo-guard pays.
- Export via `tcg.shipping` as usual; `prize_semantics_probe` standing
  regression; monitor row in the ship commit.

### P2a — stall/pressure coverage

- **`rl/plan.py`**: add Fan Rotom + Hop's Phantump/Trevenant + Kangaskhan
  variant ids (from the harvested decklists) to `_RACEMODE_STALL_IDS` /
  `_RACEMODE_PRESSURE_IDS` ([plan.py:382](../rl/plan.py:382),
  [plan.py:401](../rl/plan.py:401)). Ship config moves `racemode3` →
  `racemode2` ([plan.py:559](../rl/plan.py:559), exists, never shipped
  live): blanket conserve vs `_RACEMODE_WALL_IDS`, margin-gated vs the
  pressure set. Verify the wall-side behavior of r2 ≡ r3 on the wall bed
  before relying on it (single cell, cheap).
- **`tests/`**: NEW coverage regression test enumerating every opponent id
  the racemode triggers cover, so the next stall variant is a one-line
  diff caught by review, not a live 0-2.

### P2b — racemode4 burn demote

- **`rl/plan.py`**: NEW `PLAY_FIX_RACEMODE4 = "racemode4"` +
  `_RACEMODE4_BURN_IDS` — the measured burn sources (Enriching Energy,
  Poké Pad, surplus Dawn/Hilda once setup-complete, Sacred Ash) demoted
  ONLY under the racemode2 trigger conditions (opponent wall/pressure set
  + deck-margin gates) — state-gated, not blanket: the engine is needed
  when not in a deck race. Sacred Ash additionally gated to deck ≤ 2 (it
  was played at deck 0/2 in M37 losses). Target ~5–6 cards/game saved.
- **`tests/`**: fixture tests per demote condition (in-race vs
  out-of-race, setup-complete vs not).

### P3 — targeted fine-tune

- **`rl/replay_bc.py`**: NEW `--opp-deck-hash` filter on `build()` —
  restrict to seats whose OPPONENT's decklist hash matches (multi-valued:
  grim/wall/stall/dragapult hashes). `build()` currently filters only the
  imitated seat ([replay_bc.py:329](../rl/replay_bc.py:329)); this is the
  one real `rl/` addition of the lane. Pairs with the existing
  `--only-deck-hash 9294d9d8 --winners-only --min-score 800 --hand-aware`.
- **`scripts/m39_vsloss_census.py`** (NEW, small): counts 800+ winner
  seats by opponent family — doubles as the deck-decision falsification
  data. Runs before corpus build.
- **`scripts/m39_train.sh`** (NEW, from `m38_train.sh`): arms `m39_vsloss`
  (low-dose per the dose law), `m39_vsloss_awr` (same corpus,
  `--outcome-weight` — the M37-validated hook, no code change), optional
  `m39_mirror903`. Thin-corpus fallback: blend with the general 800+
  corpus via the multi-dir train path (as m38 gen-1 a+b).
- **`scripts/m39_gate.sh`**: arm cells vs the cont3-plain pin, weighted
  verdict via `m39_decide.py`.

### P4 — ship B

- No new code: G5-style on/off matrix as extra `m39_gate.sh` cells,
  export via `tcg.shipping`, expanded QC, monitor row
  (`notebooks/model_monitor.ipynb` MODELS dict) in the ship commit.

### P5 (optional, default OUT) — clone-scout

- No new code if pulled in: `scripts/m39_build_beds.sh` recipe pointed at
  the best wall/dragapult seats + a `m39_gate.sh` roster run.

## Ship cadence

Two submits: **Ship A (P1 strip)** early — cheap, single-variable, live
value regardless of outcome; **Ship B (P2×P3)** as the milestone headline.
Both behind the full pre-ship battery: export parity, twin regeneration,
`prize_semantics_probe`, expanded QC, Piotr's manual replay review and
explicit go (no exceptions), monitor-row-in-ship-commit.

## What would kill this milestone (pre-registered)

- P0 harvest can't produce faithful grim/stall beds (insufficient seats at
  band) → fall back to best-available clones, but G-3's 80%-coverage rule
  still applies with `m38_bc_wall` + refreshed grim at whatever band exists.
- P1 strip regresses the weighted pool → gacfr3 interactions are
  net-positive after all; keep the stack, P2 proceeds anyway (its rules are
  new, not re-tunes).
- P3 all-arms kill → imitation exhausted at this corpus quality; P4 fork
  fires early.
- Both ships flat live at converged n → the 800 objective moves to a
  training-objective milestone (the P4 fork becomes M40's plan).

## Decision points for Piotr (before P0 launch)

1. **Strip scope** (P1): full gacfr3-economy removal matching the G5 plain
   arm, deckguard/M30 conserve retained — confirm, or name rules to keep.
2. **Two-submit cadence** — confirm Ship A goes out before the P2/P3 work
   completes (it is also our pool-drift control).
3. **Deck**: family decision MADE 2026-08-01 (stay on alakazam, see the
   deck-decision section); the *variant* still needs the standing per-ship
   confirmation — alakazam_v2_h4 assumed for both ships.
4. **P3 corpus correction** (finding #7): same-deck vs-loss filter instead
   of the post-mortem's cross-family fine-tune — endorse, or run a small
   cross-family arm anyway as a falsification cell.
5. **P5 clone-scout scope**: default is OUT of M39 (parked in BACKLOG);
   pull it in only if you want the swapped-deck number this milestone. The
   P3 census may make it moot either way.
