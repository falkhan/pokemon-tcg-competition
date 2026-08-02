# M40 plan — leave the band, with the lanes now decided

**Status: DECIDED v3, 2026-08-02.** v1 (bank +24) was killed by Piotr's steer
and the slot correction; v2 (the stub) laid out S1–S6 with open decisions;
v3 records Piotr's calls (§6), reframes §1 per the critical review, and fixes
the three workplan gaps the code verification found (S5 fidelity walker, S6
three-site fix, retention re-encode). The critical review's claim-by-claim
verification is summarized in §7.

**Campaign target: implied ELO ≥ 1000, converged (n ≥ 150, per G-10).**
Unchanged (stub decision 4 stands) — but see §1: the target's feasibility
rests on a hypothesis M40 tests rather than assumes.

---

## 1. Why we are stuck — TWO hypotheses, not one

The campaign's scores: M30/M35 **817**, M37 **736**, M38 **665**, M39 Ship A
**780**. Five milestones inside one ~150-point band (M36's 609 sits below
it). Two facts frame the question:

| | |
|---|---|
| our WR vs `m39_bc_top` — a BC clone of **900–1058-band** pilots of our own list | **0.532** (panel of 3 draws, n=3600) — *above parity* |
| our live record vs **800+** opponents, M37+M38 | **0-6** (recomputed at n=94; the M37 post-mortem's own n=63 read was 1-3 — band labels drift with the ladder, so hold the exact count loosely) |

**v2 called compression "the only story that fits both numbers". That was
over-claimed.** Two hypotheses fit, and they point at different work:

- **H-A (compression):** BC clones retain only a fraction of their
  demonstrator's edge, so every bed we own is a ~700-band opponent wearing a
  700–1000 label, and five milestones optimized against the band we are
  already in. Evidence: D1's head-to-head put the 1000-band grim clone over
  the 750-band clone at 0.623 (z=+12.04) ≈ 86 ELO of separation from
  demonstrators ~250 apart — roughly a third to a half retained.
  **Weakness: that factor comes from ONE draw-pair in a regime G-13 says has
  ~±6pp construction lottery per draw** — the honest range on the retention
  factor is wide (X5 measures it properly).
- **H-B (matchup structure):** `m39_bc_top` is a MIRROR clone — top pilots of
  our own list. The live 800+ losses were grim 738/766/838, dragapult 853,
  archaludon — **non-mirror families**. Parity with 900-band mirror pilots
  plus 0-6 vs 800+ non-mirror pilots is consistent with no compression at
  all. The M39 diary flags exactly this limitation, twice.

Both hypotheses agree on one thing: the offline apparatus cannot currently
measure the thing the target needs, and **M40's first job is to build the
instrument that discriminates them** (S3: composite beds + a non-mirror
high-band ceiling panel + the compression error bar). If H-A holds, offline
numbers have been systematically optimistic and the live ladder (S4) is the
only honest instrument. If H-B holds, the fix is bed *coverage* at altitude,
which the harvest can already build (grim: 562 seats at 1000+).

What survives from v2 unconditionally: Ship A's +115 came from *removing*
rules, and removals transfer because they do not depend on out-playing
anyone. The structural lanes below (S5, S6) are removals-or-faculties of the
same opponent-independent class.

## 2. The lanes, as decided

### ~~S1 — within-turn combo solver on the neural line~~ — REJECTED (Piotr, 2026-08-02)

**The stop-invest line stands. The solver stays a label/analysis instrument,
not a pilot component.** Recorded so the next reader does not re-derive the
lane from the same screen:

- The screen (+23pp, `solved:` vs `model-conserve` on `m38_bc_wall`, n=30)
  had three defects: n=30 resolves ~35pp (G-12); the bed was **d1, the
  weakest of the three wall draws** (panel mean 0.227 vs d1's 0.296); and the
  control carried Ship A's fix string, not Ship B's — the racemode package
  already takes +11.2pp on the same wall panel, and the two mechanisms
  attack the same loss mass, so the marginal gain was never measured.
- M22c's own record: the solver ate 57.8% of prompts to change 0.5% of
  actions for +3.5pp; ARCHITECTURE.md forecloses non-lethal tiers on five
  below-null measurements; `turn_solver.py` has never been bundle-ready
  (not in the REQUIRED check or purity tier), and the observed 117.9 ms max
  move against `SOLVED_DEADLINE_S=0.05` shows the deadline does not
  actually cap latency.

**Instrument use is NOT rejected** — S3's `solved:` composites are exactly
the "label/analysis instrument" role the stop-invest line preserves.

### S5 — adopt the v4 encoder (opponent memory + bench threat) — HEADLINE

Unchanged in substance from v2: the live net is v3 and therefore blind to
`OppMemory` (last-4 opponent played ids, last attacker, energy-attach
target) and to `_slot_extras`' per-bench-slot damage projection; every
harvest corpus since M24 is v4-blind because `rl/replay_bc.py` never reads
`obs.logs`; 98.1% of cached observations carry non-empty logs, so this is a
re-encode, not a collection campaign. Vehicle: `migrate_v3_to_v4`
warm-start (bit-identical at init, tested) + low-dose fine-tune **with
retention** (M39's law: retention is the default, not an arm).

**Three corrections to v2's workplan, from the code verification:**

1. **The fidelity probe fails as written.** `iter_replay_decisions`
   silently skips prompt classes (INACTIVE seats, blank obs, empty menus,
   context overflow, deck-return) that `OppMemory.observe()` must see
   exactly once each; prefix-dedupe only cancels *re-delivered* events, so
   events landing in a skipped prompt are dropped permanently. **First S5
   deliverable is a full-prompt log walker** (new iterator that visits
   every own prompt, not just labelled decisions); only then is the probe
   meaningful — reconstruct memory features offline for games we also have
   live and assert they match. Probe fails after the walker → the lane
   dies (an encoder that lies is worse than one that is blind).
2. **Retention needs the champion shards in v4 too.** Mixing a v4 corpus
   with v3 champion shards will not load. The champion corpus is from
   cached replays with logs, so it re-encodes — but the job is roughly
   double what v2 priced, and the fidelity surface includes it.
3. **The precedent is weaker than v2 implied.** v4 shipped in M21, but its
   live value was never isolated (the M21 ships were the ~600-band "first
   neural >0.50 mirror" era), and M23's breakthrough clone beat the
   v4/plan-PPO lineage *without* v4. The lane's value rests on mechanism
   plausibility — bench-threat blindness ↔ the 43% slow-setup sweep mass —
   not on recovering a proven win. Priced accordingly.

Cosmetic fix from verification: the energy-attach target is a 7-way one-hot
inside the `N_MEM` block, not a 5th id slot.

**E3 (archetype classifier) stays unscoped until S5 is priced** — S5
delivers the same information end-to-end with tested code.

### S6 — plan-head train/serve mismatch — CHEAPEST LANE, decided end-state

Unchanged evidence: `plan_head`/`plan_enc` byte-identical since
`m28_winners`; 100% of training rows since M24 at plan=0
([plan_iter.py:574](../rl/plan_iter.py:574)); the pilot serves a non-zero
plan every own MAIN prompt ([main.py:218](../submission/main.py:218)). The
conditioned branch is out-of-distribution by construction. Screen (n=200,
single-process): +6.0/+5.0/+4.5pp on wall/m28/top, −2.0 grim — direction
consistent, below per-cell MDE.

**Correction from verification: the fix is three sites, not one line** —
the bundle (`submission/main.py`) plus the same logic duplicated in both
matchrunner eval twins (v3 and v4 pilots), or one new fix-kind in
`_MODEL_FIX_KINDS`. The fix-kind route is preferred: it makes plan-zeroing
a config token, testable offline and shippable without code divergence.

**Pre-registered outcomes (Piotr, 2026-08-02) — the v2 contradiction is
resolved:**

- battery positive (weighted gain, z≥2) → serve plan=0, single-lane
  shippable immediately;
- **battery null → serve plan=0 ANYWAY, as pure risk-removal** (serve then
  matches the training distribution exactly), bundled with the next net
  ship rather than spending its own ship;
- battery negative (zeroing measurably hurts) → the conditioned branch is
  somehow load-bearing; keep as-shipped and open a diary entry, because
  that result would be genuinely surprising and worth a mechanism probe.

Battery design: full G-13 panel roster, n=1200/draw + 900+ ceiling panel,
run on **both** `cont3` and `retain_b` (the floor net — a result that holds
only on one lineage is not a serving decision).

### S2 — self-play iteration — FULL LANE IN M40 (Piotr, 2026-08-02)

Piotr's call: run the whole lane this milestone, in parallel with the
structural lanes, accepting the split focus. Order inside the lane is
unchanged (each step gates the next):

1. **E0 — value net on the harvest corpus** (BACKLOG sweep #6). Trained
   incl. loser rows (the α=0.25 lesson — loser rows carry signal).
   **Pre-registered exit: it must out-rank the outcome proxy on the loss
   families, or the branch dies on day two.**
2. **Exploration design that isn't uniform noise.** M39's `bestresp`
   produced a corpus the net already agreed with 93.5% — pre-registered
   kill at collection time: **init val_acc > 0.90 kills the corpus before
   training**, not after gating. Policy-temperature or ε-over-top-k
   sampling on MAIN prompts; sweep the rate until agreement lands
   materially below 0.90.
3. **Opponent diversity beyond three clones per family** (the Showdown
   opponent-distribution overfit warning): collection pool spans panel
   draws × families, plus `solved:` composites once S3 builds them — the
   one place solver strength enters training data legitimately (as an
   opponent, not a teacher).
4. Retention mixing and α=0.25 are defaults for every fine-tune in this
   lane, not arms.

Honest cost, restated: this is the expensive lane and it may return nothing
inside one milestone; the burden it must clear is a corpus that moves the
**900+ ceiling panel**, which five imitation arms could not. Online PPO
stays parked (327M env steps).

### S3 — instruments above the clone ceiling + the compression error bar

Prerequisite for reading S2 (and §1) honestly. Three deliverables:

1. **Solver composites as opponent beds**: `solved:<bed_ckpt>:<bed_deck>`
   works today for arbitrary checkpoints; one small code change needed —
   the solver budget is module-global (`SOLVED_DEADLINE_S`/`MAX_NODES`
   shared by every spec in a run), so per-spec budget override lands
   first. Verify composite > plain clone head-to-head (panel draws, not
   single); then re-run the ceiling gap vs a composite panel.
2. **A non-mirror high-band ceiling bed** — the H-B test. Grim has 562
   seats at 1000+; build the 3-draw panel and add it to the ceiling
   roster. If we are at parity with top mirror clones but well under
   against top grim clones, H-B explains the live record without
   compression.
3. **X5 — compression error bar**: panel×panel head-to-head (3 topgrim
   draws × 3 grim-750 draws, n=400/pairing = 3600) → a CI on the 0.623 →
   a CI on the retention factor. §1's keystone number currently has none.

Pre-registered readings (unchanged from v2): gap re-opens vs composites →
H-A confirmed, every historical bed absolute gets an asterisk; gap holds →
H-B or honest beds, and the 0-6 needs a different explanation.

### S4 — the ladder as the high-band instrument

The slot correction (§3) makes the ladder usable: 5 slots/day, ~210 to
deadline, and **read bandwidth** (games/day toward a declared n) is the
real constraint. Decided sequence:

1. **X0 — accrual audit FIRST** (an hour, from the listings cache):
   games/day per sub, and whether accrual *splits* when several subs are
   live (M37/M38 overlap window is the existing evidence; Ship A/B overlap
   is accruing now). Sets M40's experiment count (~3 vs ~15 reads) and
   how many parallel reads Phase-L may run.
2. **FLOOR SHIPS NOW (Piotr, 2026-08-02): `m39_retain_b` × package** —
   vs Ship B this changes only the net, so it is a G-7-compliant
   single-lane ship; old subs keep accruing, so its n=150 read runs in
   parallel with Ship B's. Offline basis: +3.47pp weighted, z=+7.44, G-4
   matrix already paid. Deck: `alakazam_v2_h4` (per standing rule,
   re-confirm at export). Standard pre-ship battery: `ship_verify`, QC
   battery (`scripts/qc_battery.py`), monitor row in the ship commit,
   Piotr's replay review + explicit go. **G-10 declared read-n: 150.**
3. **Replication ship** — re-submit an existing tarball unchanged (Ship A
   or B, pick after X0): the campaign's first live replicate; measures
   ladder read noise and tests H-A's offline-vs-live-inflation prediction
   directly. Interpret under pool drift (G-5) — the ladder is
   nonstationary, so a delta vs the original is not automatically lottery
   noise.
4. Subsequent ships (S6 if positive, S5 if it gates through, S2 products)
   each single-lane with declared read-n. **G-10 discipline is mandatory:
   cheap slots + continuous reading is a multiple-comparisons machine.**

## 3. Budget (correction carried from v2, unchanged)

| resource | quantity | binding? |
|---|---|---|
| submissions | 5/day × 42 days ≈ 210 | no |
| calendar | 42 days to 2026-09-13 | yes |
| **decision-grade live reads** | ~3 / ~8 / ~15 at 10 / 30 / 55 games/day (n=150 per G-10) | **YES — measured by X0, not assumed** |

Reserve-slot concept deleted; failed swings cost days, not slots; live
replication affordable for the first time; G-7 survives on attribution
grounds; G-10 becomes more important, not less.

## 4. Pre-registered kills (v3, decided)

- **X0 shows accrual splits hard across live subs** → parallel reads are an
  illusion; revert to sequential ships, floor read takes priority.
- **S5 walker-corrected fidelity probe fails** → the replay log window
  differs from live; kill the re-encode. (A probe failure *before* the
  walker exists is unattributable and kills nothing.)
- **S5 fine-tune (v4, retention, α=0.25) does not beat the floor on the
  weighted panel** → v4 observability is not worth a net ship this
  milestone; record and hand to M41 with the corpus already built.
- **S6 battery null** → serve plan=0 anyway (risk-removal, pre-registered
  above), bundled with the next net ship. **S6 battery negative** → keep
  as-shipped + mechanism probe.
- **E0 cannot out-rank the outcome proxy on the loss families** → S2's
  advantage-weighted branch dies; the lane falls back to
  exploration+retention filtered self-imitation only.
- **S2 collection init val_acc > 0.90** → the exploration design failed
  again; kill at collection, before training.
- **No S2 arm moves the 900+ ceiling panel** (now incl. the non-mirror
  bed) → imitation AND offline self-play are exhausted on this engine
  budget; M41 pivots to the ladder as the only instrument (S4-only
  cadence) and the target conversation reopens with data.
- **Nothing beats the floor by decision day** → the floor is already live
  (it ships now); this kill converts to "no further net ship this
  milestone", which costs nothing.

## 5. Standing guardrails

G-1…G-13 carry over unchanged. **G-14 is now scoped, not proposed**: a
`ship_verify` check asserting serve/train input parity — encoder version
(`enc_ver` buffer or input width), plan-vector statistics served vs
training-corpus statistics. Both of this milestone's silent regressions
(S5, S6) are the same missing check; verification confirmed every existing
gate (parity, deck, twin, fix-name, weights) passes an S6-style mismatch.
Lands with the first M40 ship.

## 6. Decisions — ALL ANSWERED (Piotr, 2026-08-02)

1. **S1 solver lane: NO — stop-invest stands.** Solver remains an
   instrument (S3 composites, S2 opponent pools); never a pilot component.
2. **S6: battery runs; on null, serve plan=0 anyway** (risk-removal,
   bundled with next net ship); on positive, single-lane shippable.
3. **Floor: ship now** as single-lane-vs-Ship-B, parallel n=150 read.
4. **S2: full lane in M40**, in parallel with structural lanes; E0 exit
   criterion and the collection kill still gate each internal step.
5. **Target: 1000 stays**, with §1 reframed — the number the campaign can
   *defend* by decision day depends on S3's verdict on H-A vs H-B.
6. **Iterative cadence: yes, if X0 supports it** — every ship still
   declares read-n in advance (G-10).
7. S5 was not re-asked in the decision round: it proceeds per the stub's
   standing recommendation (**yes, gated on the fidelity probe**), now with
   the walker prerequisite. Flag to Piotr if this should be pulled.
8. **Nothing net-related finalises before Ship B's n=150 read** — its wall
   cell carries the falsifiable mechanism prediction (opponent deck-outs
   ~30% live), which is also the cleanest H-A/H-B evidence available for
   free. The floor ship (decision 3) is the explicit exception, made
   knowingly: it is the same rules package on a stronger net, and its read
   runs in parallel rather than replacing Ship B's.

## 7. What the critical review verified (2026-08-02, claim-by-claim)

Kept short; the full report is in the session record.

- S1 code comments, `solved:` spec generality, greedy-argmax bundle: ✅
  (module-global solver budget caveat; stale `main.py:185` pointer in the
  turn_solver docstring — fix in passing).
- S5 dims (`V4_EXTRA_DIM=341`, `N_MEM=51`, 12×5 slot extras), zero
  `logs` refs in `replay_bc.py`, `migrate_v3_to_v4` tested + exercised: ✅.
  `iter_replay_decisions` prompt-skipping: **the fidelity probe as drafted
  in v2 would have failed for the wrong reason** — hence the walker.
- S6 `BCDatasetV3` zeros, non-zero serve path, no existing zeroing flag,
  no input-distribution check anywhere in `ship_verify`/`tcg.shipping`: ✅.
- Panel infra defaults to exactly the pre-registered battery spec
  (n=1200/draw, 3 families + ceiling panel), with no slack: ✅.
- Score-history, slot arithmetic, read-bandwidth table: ✅. The 0-6 datum:
  unstable across recomputations (1-3 at n=63 → 0-6 at n=94); n≤9 either
  way; G-9 bars gating on it.

---

## BACKLOG disposition (v3)

| item | M40 disposition |
|---|---|
| Within-turn solver on the neural line | **REJECTED (Piotr 2026-08-02) — stop-invest unchanged.** Instrument use only (S3/S2 opponents) |
| Value-net retrain / gen-2 (sweep #6) | **E0 — first step of S2, runs now** |
| Offline best-response (sweep #2) | **S2 step 2–3**, exploration redesign + diversity, collection kill pre-registered |
| Advantage-filtered BC (sweep #1) | inside S2 once E0 passes; α=0.25 is default |
| Retention (sweep #3) | default on every fine-tune |
| **Plan-head mismatch (S6)** | battery + pre-registered serve-plan=0 end-state |
| **Encoder v4 adoption (S5)** | headline; walker → probe → re-encode (incl. champion shards) → retention fine-tune |
| Opponent-deck inference (sweep #4) | unscoped until S5 priced |
| ROIDA loser replays (sweep #5) | blocked on E0 |
| Online PPO (sweep #7) | parked — 327M env steps |
| Architecture inductive bias (sweep #8) | parked — needs an S2-scale corpus |
| Heavy test-time search / DT / LLM self-improvement / deep equilibrium | stop-invest, **now including S1 by Piotr's decision** |
| `rl/`↔`tcg/` duplication, W_COUNTER, `setup_plans_late` | housekeeping; competes against calendar only |
| **S4 — ladder-as-instrument** | active: X0 → floor ship → replication ship → declared-dwell cadence |
