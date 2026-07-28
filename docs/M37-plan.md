# M37 plan — beating the 600 band (anti-stall race mode + band-weighted gate)

Status: **draft v1** (2026-07-28) — built from docs/m36-post-mortem.md, its
recommendations re-verified against the codebase this session, supporting
experiments run (see Execution log). Awaiting Piotr's arm pick.

Input: sub 55030954 implied ELO 609 — BUSTED PLACEMENT, not a regression.
The agent seeds at ~600 and the 600 band is 40% stall/grim — exactly our
worst matchups. Leaderboard selection stays 55011605/54929991 (both ~790–817).
**M37's objective is a config that beats the 600-band mix**, so a fresh sub
can climb out instead of farming/being farmed at 0.500.

## Environment note (this machine, 2026-07-28)

This is the Windows checkout; the campaign's Linux box holds artifacts that
were never synced. Verified state:

- `checkpoints/m28_winners.pt` was MISSING — **reconstructed byte-faithfully**
  from `submission/policy_weights.npz` (save_npz is a plain state_dict dump);
  matchrunner smoke-verified. Rocket/grim BC-clone checkpoints
  (`m30_bc_rocket_54834745.pt`, `m25_bc_grim_54861775.pt`) and the M35/M36
  ship tarballs in `dist/` are still missing → the rocket/grim clone beds and
  the QC battery's prev-ship mirror leg are UNAVAILABLE until synced or
  rebuilt. (A solver-piloted live-list grim bed was tried as a substitute
  and FAILED the strawman law — see P0 results.)
- `data/kaggle/raw/` re-fetched for sub 55030954 (53 eps); `meta_v3/` not
  synced (only v1/v2) — re-freeze if a meta-gate run needs it.
- `hermes` CLI not installed here — no Telegram status updates this session.
- Throughput: ~1.1 s/game at `--workers 8` (24-core box) for normal beds;
  stall beds run longer (~4× — the games go 30+ turns).
- pm tooling (`live_postmortem.py` path-ported, `pm_extra.py`, `pm_probe2.py`)
  reproduces the post-mortem EXACTLY here (ELO 609, identical ledger) —
  but pm_extra/pm_probe2 are still UNTRACKED; commit with this milestone.

## Post-mortem recommendations — codebase verification

| rec | verdict after code check |
|---|---|
| 1a. new stall beds from live replays | **PARTIALLY SUPPORTED — the wall-CSV recipe transfers to hop but NOT to garchomp/grim.** Lists extracted + legal (`decks/hops_stall.csv`, `decks/cynthia_garchomp.csv`, `decks/grim_live.csv`). Strawman results: `solver:hops_stall` **VALID** (wr 0.40–0.47, 61% of losses true deck-outs = the live mode); `solver:cynthia_garchomp` **INVALID at every cheap rung** (solver 0.665 / solver2 0.710 / solver-dev 0.615 vs live 0-2 — the pilot can't play heal-tank discipline); `solver:grim_live` **INVALID** (0.790 vs live 0.375 — evolution/Munkidori piloting). Garchomp/grim beds need **BC clones** (the M25/M30 recipe); clone data verified viable locally: cynthia family 512 seat-rows (dominant 323), hop 262 (232), grim 378 (+1339 munkidori-variant). |
| 1b. archetype-detected race mode | Feasible in the allowed deterministic-obs class. `rl/determinize.infer_deck` (containment scoring, top-1 ≥0.9 by t2, 98% probe) is the detection asset; the M36 raceconserve margin-gated design (mirror_race_probe.py, PARKED) is the conserve half. New work: an O12 `racemode` fix keyed on opponent-board archetype facts (Trevenant/Snorlax/Crustle/Garchomp line on board) — see W1. |
| 1c. ship gacfv rider | `PLAY_FIX_GUSTVETO` (O11) implemented, tested, measured non-inferior on all 7 M36 beds; `submission/main.py` `_ATTACH_FIXES` still gacf. Flipping = 1-line default change + export. Still the cheapest rules lever. |
| 2. band-weighted ship bar | **SUPPORTED + IMPLEMENTED.** Band composition computed from the 55030954 cache (n=52): mirror 12, grim 8, lucario 8, wall 5, starmie 4, other 4, rocket 3, hop 2, garchomp 2, dragapult 2, arch 1, solrock 1. **`scripts/band_decode.py` (new, frozen weights) implements the gate**; local beds cover 38/52 = 73%. Baseline composite for the ship config = **0.6378 ± 0.030 — an UPPER BOUND**: the grimlive (0.79) and garchomp (0.665) cells sit on strawman-invalid beds; live band WR was 0.500. The 0.64→0.50 gap IS the bed-fidelity gap, concentrated exactly where the clones are missing. Gate becomes trustworthy once the two clone beds replace those cells. |
| 3. grim tech slot | Harness exists (`scripts/m36_decktech.sh` pattern + `rl/deck_search.py` neighborhood search). Grim = Darkness→Psychic 2× weakness; needs a Tier-0/Tier-1 pass. BLOCKED on a strawman-valid grim bed (`solver:grim_live` measured 0.79 vs live 0.375 — invalid); bed = M25 grim clone (sync) or fresh clone (P0b). |
| 4a. AWR-style weighted BC | **Cheaper than the post-mortem implies**: `rl/bc.py` already has per-example CE weights (`--weighting none|score|winner`, `WeightedDataset`, M10) and shards carry `teacher_score` + `seat_won` per decision. An AWR arm = rebuild corpus with `winners_only=False` + one new weighting scheme in `_example_weights` (~15 lines) + retrain. |
| 4b. matchup-specialist mixture | Routing asset exists (determinize containment); bundle would need a 2nd `policy_weights_wall.npz` + router in main.py (build_submission change). Medium. The no-transfer law is not violated (generalist frozen). |
| 4c. expert iteration on the race | **Downgraded** — `rl/turn_solver.py` is WITHIN-TURN only (no opponent model, MAX_DEPTH 8); the deck-out race is a multi-turn counting problem it cannot see. The cheap rung is the deterministic race arithmetic as an O-rule (cards-drawable vs prizes-needed is computable from obs), not solver-guided trajectories. |
| 4d. auxiliary race-value head | Training-arm scale; park behind the O-rule rung. |
| 4e. latent `.prize` feature bug | **Confirmed still present**: `_make_plan`'s `wins` (rl/plan.py:109) tests `target_prize >= len(op.prize)` and `concedes` (:116) tests vs `len(me.prize)` — inverted vs the M36-verified semantics (a player's `.prize` = prizes THEY still need; comment at :114 is wrong). Queued for any training arm (feature-distribution shift forbids a rules-milestone fix). |
| 5. stop-investing list | Honored: no brick work, no telepath work, dragapult clone stays parked (1-1 live). |

---

## W1. Anti-stall race mode — O12 `racemode` (the big arm, 50% of loss mass)

Design (the allowed deterministic-obs-fact class, M36 raceconserve generalized):

- **Trigger — archetype facts on the opponent's BOARD** (no hidden info):
  any of {Hop's Trevenant/Phantump/Snorlax, Crustle/Dwebble, Cynthia's
  Garchomp line + heal tools, Great Tusk} visible ⇒ opponent is a
  stall/heal-tank line. (Board-fact test, same class as benchfloor's
  card facts; determinize-style containment NOT needed for v1 — explicit id
  sets are simpler and testable.)
- **Action — switch objective from tempo to card-economy racing**, we already
  win deck-out races (4 live wins by opp-deck-0; dud 0/71, fez 0/48 at
  deck≤6): (a) extend deckguard/conserve demotes to fire from game start
  (not deck≤N) while the stall trigger holds; (b) the raceconserve margin
  gate (my_deck < opp_deck − 5, my_deck ≤ 25, my_deck > 6) for optional
  draws; (c) KEEP attacking — Powerful Hand still lands on unattached
  targets; hammer-4 covers special-energy walls.
- **Bars (pre-register in scripts/m37_decide.py before any A/B):** strictly
  better (z > +1.96) on ≥1 of the strawman-VALID stall beds {hop, wall};
  non-inferior (z > −1.96) on mirror/luc/arch (+ clone beds as P0b lands);
  kyogre ≥ 0.95 floor. Band composite reported alongside (decision aid,
  not a bar). Diag check: racemode wins vs hop should show MORE won
  deck-out races (opp deck 0), the mechanism claim.
- Ship shape stays single-variable: rules-only arm on the h4 deck
  (m37 rules vs 55030954) or rules+gacfv composed — decide with Piotr.

## W2. Band-weighted ship gate (fixes the band trap) — STANDING infra

`scripts/band_decode.py` (frozen n=52 weights) is the gate INPUT; ship bar =
per-bed pre-registered bars (as always) PLUS band composite ≥ the current
config's baseline (measured in P0 this session — see Execution log). The
[[no-validated-live-predictor]] discipline: composite is pre-registered,
weighting adds no power, per-bed n governs.

## W3. Grim slot (5 live losses, 2× weakness, 15% of the band)

One Tier-0/Tier-1 deck-tech pass with the M33/M34 harness — but ONLY on a
strawman-valid grim bed: `solver:grim_live` measured 0.79 vs live 0.375 this
session, so any deck-tech "gain" on it would be noise on an invalid
instrument. Bed = the M25 grim clone (sync `m25_bc_grim_54861775.pt` from
the Linux box) or a fresh clone from the pinned live subs (P0b). Candidates:
break the Darkness weakness math (spread tools, a non-Psychic secondary
attacker slot, or anti-Munkidori tech). Same bars as h4: strictly better on
the grim bed, non-inferior elsewhere, single-variable. Only if W1 leaves
appetite.

## W4. AWR arm (training, conditional/stretch)

Rebuild the live corpus (winners_only=False, min-score unchanged) → new
`_example_weights` scheme `advantage` (CE weights must stay ≥ 0): w = 1 +
α·(2·seat_won − 1), α≈0.5 (winners 1.5×, losers 0.5×), optionally scaled by
final prize margin. Warm-start from m28_winners; full non-inferiority
battery + band composite. Only if W1/W3 fail to move the band composite, or
as the next milestone's arm. Preconditions: none technical — the weighting
infra (`WeightedDataset`, shard `seat_won`/`teacher_score`) is verified
present; the change is ~15 lines in `_example_weights` + a corpus rebuild.

## Explicitly NOT this milestone

- Wall-specialist mixture policy (4b) — behind W1's cheap rung.
- Race-value head (4d) — behind W1.
- `.prize` feature fix — training arms only, never a rules milestone.
- Dragapult clone — parked (1-1 live watch).
- Brick/telepath — closed.
- Resubmitting an old config to re-roll placement — Piotr's call, not ours.

## Proposed M37 shape

- **P0a (DONE this session):** stall lists extracted + strawman-checked (hop
  bed VALID; garchomp/grim cheap-pilot ladder exhausted); band weights
  frozen; band-composite baseline 0.6378-upper-bound measured.
- **P0b — clone beds (the strawman answer for garchomp/grim):** targeted
  refresh of the pinned opponent subs (Execution log) → `replay_bc build
  --only-subs` per teacher (M24 law: single-teacher beats pooled) → BC
  train → strawman-check each clone (must LOSE like live: garchomp
  deck-outs, grim weakness bleed). Grim shortcut if the Linux box syncs
  `m25_bc_grim_54861775.pt` first. Fidelity risk is real (one historical
  grim clone failed at 0.565); the rocket-clone precedent (M30) is the
  groove.
- **P1 — O12 `racemode`** implementation + tests (rl/plan.py + twin sync;
  trigger id sets pinned below); battery {gacf control, gacf+racemode} ×
  {hop, wall, mirror, luc, arch, kyo + clone beds as they land} n=200×2
  seeds; pre-registered bars in scripts/m37_decide.py BEFORE any A/B:
  strictly better z>+1.96 on ≥1 of {hop, wall}; non-inferior everywhere
  else; kyogre ≥0.95; band composite reported alongside.
- **P2 (iff P1 clears):** compose with gacfv (rules delta vs 55030954 =
  racemode+gustveto or racemode alone — Piotr's single-variable call).
- **P3 (appetite-gated):** W3 grim deck-tech pass on the grim clone bed.
- **P4 (stretch / next milestone):** W4 AWR arm.
- **Ship gate:** per-bed bars + band composite ≥ 0.6378 baseline on the SAME
  bed set (or the clone-corrected baseline once P0b lands) + multi-deck QC
  battery (scripts/qc_battery.py — needs the M36 tarball synced from the
  Linux box for the prev-ship mirror leg) + Piotr's replay review + explicit
  go. No submit without it.

## Open questions for Piotr

1. Arm priority: O12 racemode (50% loss mass, medium build) first, or the
   cheap gacfv flip alone as a fast M37.0 ship?
2. Racemode trigger breadth: stall-family id sets only (v1), or include
   grim (their Munkidori/heal engine also outlasts us)?
3. Sync request: rocket/grim clone checkpoints + M35/M36 ship tarballs from
   the Linux box (restores 2 proven beds + the QC mirror leg immediately),
   or build fresh clones here from the pinned live subs (P0b — also yields
   the garchomp bed, which never existed anywhere)?
4. Grim deck-tech (W3) in-milestone or deferred?
5. AWR appetite: if racemode clears its bars, is W4 the M38 headline?

---

## Execution log — milestone phases (2026-07-28, feature/m37)

- **KICKOFF — Piotr's five calls (in-session):** (1) racemode arm, (2) grim
  IN the trigger, (3) sync from his laptop (file list in the session plan;
  all gitignored — needs `git add -f` or scp), (4) grim deck-tech deferred
  to M38, (5) AWR greenlit (pre-work now, M38 headline). Role change
  recorded: Claude implements + runs all phases; Piotr orchestrates,
  reviews replays, gives the ship go.
- **P0 CODE DONE — O12 racemode + racemoder implemented.** rl/plan.py:
  `PLAY_FIX_RACEMODE`/`PLAY_FIX_RACEMODER`, `_RACEMODE_STALL_IDS` (15 ids),
  `_RACEMODE_GRIM_IDS = {646,647,648}` (separate, droppable per bar B3;
  Munkidori/Froslass deliberately excluded — splashable), `_opp_board_ids`
  helper, demote-block accumulation (margin variant = the m36 raceconserve
  gate: behind by >5, 6 < deck <= 25). matchrunner: `modelt-gacfr` /
  `modelt-gacfrr`. Twin re-synced byte-identical. Tests: `_obs` extended
  with opp board/deck kwargs + 4 new O12 cases (fires on stall AND grim
  boards incl. Fez body; margin guards incl. strict-< edge; blanket
  variant; gacf composition incl. ash-promote precedence + deck<=6
  deckguard overlap). Suite 656 passed (4 pre-existing Windows artifacts).
- **P0 AWR HOOK DONE — `apply_outcome_weights` + `--outcome-weight ALPHA`**
  in rl/plan_iter.py (multiplicative CE-weight edit on `results < 1` rows,
  the channel apply_class_weights uses; alpha=0 == winners-only). Unit test
  green. Discovery credit: BCDatasetV3 already loads `results` — no dataset
  change needed.
- **P0 PRE-REGISTRATION — scripts/m37_decide.py committed BEFORE any
  battery number.** Bars B1-B7 (ship delta >= +3pp pooled trigger beds, no
  single-bed regression > 5pp, variant pick, grim drop rule, mirror
  inertness, paired band gate, QC, AWR done-bar). scripts/m37_battery.sh
  runs trigger beds only (O12 provably inert elsewhere) + the inert smoke.
- **P0 FIRE PROBE (scripts/racemode_fire_probe.py, new) — O12 is
  mechanically LIVE on the hop bed.** n=12 seed 5: blanket `gacfrr` 532
  MAIN prompts / 520 trigger-true / **299 fires** (the draw ability is the
  model's top pick on 56% of trigger states — the self-drain mechanism in
  person), W-L 8-4. Margin `gacfr`: 175 margin-open / **63 fires**, 4-8.
  W-L at n=12 is noise; the point is fires >> 0 for both variants, so a
  battery null would be a real verdict, not a trigger bug.

- **B4 AMENDED (pre-decode, 2026-07-28 — the M8/M9 fix-before-decode
  groove): workers-8 matchrunner is NOT run-reproducible**, so B4's
  "identical W/L" criterion was the wrong instrument. Evidence: gacfr
  mirror seed 1 n=100 gave 55-45 then 51-49 on rerun (identical config);
  mirror deck ∩ trigger ids = ∅; fire probe on mirror (single-process,
  n=30): **trigger_true 0 / o12_fires 0 over 1313 MAIN prompts** — O12 is
  inert where it must be; the divergence is engine-RNG × mp.Pool
  scheduling noise under load. B4 now = probe-authoritative (0/0) + a
  ±0.14 sanity band on the workers-8 smoke. Corollary recorded: NO
  seed-matched per-game A/B claims at workers 8, statistical n only (the
  campaign's standing practice anyway).
- **B4 PASS:** probe 0/0 (authoritative); sanity band gacfr 0.550 vs gacf
  0.520, |Δ| 0.030 ≤ 0.14 ✓.
- **P1 (garchomp clone) data landed:** refresh +158 eps (episodes.parquet
  5381 rows) + harvest. The byte-identical live garchomp list = 160
  seat-rows across 3 subs (54964845×111 @683, 55042761×48 @655,
  54974938×1) — BELOW the 197-689 working-clone band; the wider cynthia
  family adds variant lists (dominant gabite+roselia 323 rows, possibly a
  different build). Proceeding with the 2-sub 159-row corpus first (M24
  single-teacher law; 54964845 alone = 111 games ≈ the wall clone's 304-game
  scale in decisions), fidelity screen decides if it's enough.

- **P1 garchomp clone TRAINED:** `data/bc_m37_garchomp` (157 eps → 159
  teacher seats, 10,910 decisions, 3 shards) → `m37_bc_garchomp.pt`
  (fresh V3-as-BC, 10 epochs, val_acc **0.626** — above the working rocket
  clone's 0.587; the failed grim clone was 0.565). Deck CSV =
  `data/kaggle/garchomp_m37_deck.csv` (the byte-identical live list — all
  159 teacher seats play it). Fidelity screen + strawman running.
- **B7 AWR done-bar PASS:** 1-epoch smoke on the garchomp corpus with
  `--outcome-weight 0.25` → "non-win x0.25: 4939 rows (45.3%)" (inside the
  30-50% band; teachers won ~52%), training completes. The full corpus
  rebuild stays blocked on the laptop sync (deck_registry.json params +
  raw eps or re-fetch); code path is validated.

- **P1 garchomp clone bed ACCEPTED (with caveat).** Screen vs rule:lucario
  0.470 (bar ≥0.45 ✓). Strawman vs ship config: 0.600 n=100 / 0.500 diag
  seed — NOT the outright live 0-2, but the diag (n=30) reproduces BOTH
  live mechanisms: 4/15 losses true deck-outs (incl. a prizes-1/2 deck-out
  agonizer — the live ep88562454 signature) + 9/15 heal-tank prize grinds;
  2 bench-out bricks. Far better fidelity than solver 0.665 (which showed
  ~no deck-out losses). Caveat recorded: the bed UNDERSTATES the live
  threat (n=2 live may also overstate it); treat the garchomp cell as
  directional, hop+wall as the primary B1 evidence. Battery leg launched.

- **P3 BATTERY DECODE (m37_decide.py, bars as pre-registered): NEITHER v1
  ARM CLEARS B1.** Per-bed pooled n=600/arm vs same-battery gacf control:
  `gacfrr` (blanket) wall **+14.8pp z=+5.20** but hop **−11.9pp z=−4.19**,
  garchomp **−17.5pp z=−6.17** → KILLED (regressions). `gacfr` (margin)
  hop −1.6 (z−0.55), wall +4.2 (z+1.46), garchomp +1.2 (z+0.41) — no
  regression anywhere, pooled **+1.25pp < +3pp bar → FAIL**. B2 pick (moot
  for ship): gacfr. B4 PASS (probe-authoritative + sanity band).
  **Mechanism read: the race lever is REAL and large exactly where the
  opponent applies no prize pressure (crustle/tusk wall: blanket conserve
  = won deck-out races), and NEGATIVE where the stall line attacks
  (hop's Trevenant/Snorlax and garchomp DO take prizes — refusing draws
  all game starves our own setup).** The v1 trigger conflated the two.
- **P3b — racemode2 (v2 arm) DESIGNED + PRE-REGISTERED BEFORE its battery:**
  split the trigger: `_RACEMODE_WALL_IDS` {58,344,532,345,533,607}
  (Great Tusk/Dwebble/Crustle/Terrakion — the no-pressure farm family) →
  BLANKET demote; `_RACEMODE_PRESSURE_IDS` (hop + garchomp lines +
  grim) → the m36 MARGIN gate. One fix name `racemode2`, arm
  `modelt-gacfr2`. v1 arms/constants stay untouched (measured artifacts).
  Bars: same B1 shape (pooled trigger beds ≥ +3pp, no bed regression
  > 5pp, per-bed noninf z>−1.96); FRESH gacf controls rerun in the same
  battery (drift law honored literally even though the code delta is
  provably additive for gacf). Prediction stated up front: wall ≈ +15pp,
  hop/garchomp ≈ margin-arm's ~0/+1pp → pooled ≈ +5pp.

- **P3b INTERIM + PRE-REGISTERED CONFIRM AMENDMENT (fixed BEFORE the
  confirm runs, M36 P2 groove).** v2 fresh-control legs: wall gacfr2
  **0.565 vs control 0.408 (+15.7pp)** — prediction confirmed; garchomp
  gacfr2 0.550 vs control 0.607 (−5.7pp, z −1.99 — nominally over both
  kill lines). Scrutiny: (a) the garchomp deck contains ZERO
  `_RACEMODE_WALL_IDS` and six pressure ids (verified) → on this bed
  racemode2 ≡ racemode (margin-only), and that identical behavior measured
  **+1.2pp** on the SAME bed in v1; (b) the bed is high-variance under
  worker scheduling — the two batteries' IDENTICAL control configs differ
  +4.5pp (0.5617 v1 → 0.6067 v2; seed 1 reproduced 0.540 exactly, seeds
  2/3 swung). AMENDMENT (pre-registered now): garchomp CONFIRM at fresh
  seeds 4+5, BOTH arms, n=200/seed; the garchomp B1 verdict uses pooled
  v2-prefix n=1000/arm (seeds 1-5); same kill thresholds (delta < −5pp or
  z < −1.96 kills). The equivalence argument (a) is recorded as
  supporting context, not a bar substitute.

- **P3b v2 FULL DECODE (m37_decide --v2, bars as pre-registered): gacfr2
  DOES NOT CLEAR.** hop −0.5pp (z −0.17, noninf ✓), wall **+15.7pp
  z=+5.50** ✓, garchomp 5-seed pooled n=1000 −4.7pp **z=−2.13 →
  KILL(noninf)**; pooled +2.0pp < +3pp. The margin-gated pressure branch
  is dead weight at best: hop ~0, garchomp mildly negative (all-evidence
  pooled across both batteries ≈ −2.5pp z≈−1.4).
- **P3c — O12d `racemode3` (wall-family blanket ONLY) = the milestone's
  surviving arm.** Implemented + tests (38 green) + twin synced.
  Evidence chain: (a) on the wall bed the pressure branch cannot fire
  (wall deck ∩ PRESSURE_IDS = ∅) → gacfr2's wall runs ARE gacfr3
  measurements: **0.565/0.550/0.580 pooled 0.5650 n=600 vs fresh control
  0.4083, +15.7pp z=+5.50**; seed-1 rerun as literal gacfr3: 0.565 ✓.
  (b) hop/garchomp/mirror/grim decks contain no wall ids → provably
  inert; empirically confirmed: fire probe **0 fires over 414 hop + 309
  garchomp prompts**. (c) Therefore no bed can regress; pooled
  trigger-bed delta = +4.3pp ≥ +3pp; B5 band composite +2.1pp on covered
  mass (all from the wall cell, weight 5/52). **Caveat for the ship
  decision: racemode3 contains NO grim coverage — the battery evidence
  reversed the kickoff call #2 (conserve measured neutral-to-negative on
  every pressure-family bed). Piotr's arm pick required.**

- **SYNC LANDED + VERIFIED (Piotr, 2026-07-28):** m25 grim + m30 rocket
  clone checkpoints + deck CSVs + M36 tarball (md5 1dcb568d ✓) +
  deck_registry.json. Pin smokes on this box: grim bed 0.650 n=100 (M36
  pin 0.6425 ✓), rocket 0.610 (pin 0.6375 ✓). racemode3 provably inert vs
  both (no wall ids in either list).
- **EXPORT + FROM-TARBALL VERIFY DONE (gacfr3 ship state):**
  submission/main.py default flipped to
  `telepath,deckguard,ash,conserve,benchfloor,racemode3` (the invariant:
  ship changes the default string, never the predicate). build pipeline
  run manually (the .ps1 wrapper trips PowerShell 5.1 NativeCommandError
  on a benign engine warning): export + gate ("gate game OK: rewards
  [1,-1]") + tar → **dist/submission_neural_20260728_223734.tar.gz**
  (5.4 MB). Verified FROM the tarball: deck md5 ad014c58 == alakazam_v2_h4
  (60 cards, hammer ×4 / Hilda ×3), fixes string = gacfr3, rl/plan.py
  carries PLAY_FIX_RACEMODE3, all required files present. NOT submitted.
- **QC BATTERY DONE (m37_qc_gacfr3): 3-0 tuned, 3-0 iono, 3-0 dragapult,
  0-3 prevship → sweep INVESTIGATED and closed as mirror variance.**
  Evidence chain: (1) the two bundles' policy_weights.npz are
  BIT-IDENTICAL (21/21 arrays) and card_features match — the npz→pt→npz
  checkpoint reconstruction is exact; (2) the only code delta is
  racemode3, which cannot fire in mirror (no wall ids in either list; 0
  fires/1313 probed prompts); (3) play_games is slot-fair (verified,
  swap_slots default); (4) loss replays are full-length games (169-176
  actions, no crash/timeout signature); (5) two 12-game extensions: 4-8
  then **6-6** — pooled bundle-vs-bundle 10W-17L over 27 = 0.370,
  P(≤10|27, p=0.5) ≈ 12% — an unlucky draw against a provably-identical
  agent. QC replays in replays/m37_qc_gacfr3_*.html + index.html.
  **STOPPED for Piotr's replay review + explicit go. NOT submitted.**

## Execution log (2026-07-28, planning session)

- **ENV — m28_winners.pt reconstructed** from submission npz (21 arrays,
  plan_enc present, no enc_ver → v3o family); matchrunner smoke 0-4 at n=4
  vs solver:wall (consistent with the 0.42 pin at tiny n). Saved to
  checkpoints/m28_winners.pt.
- **P0.1 — cache refresh + stall extraction DONE.** 53 eps re-fetched.
  hop_stall: 2 variants Δ1 card, both beat us (0-1 each); v0 (3 Trevenant,
  1 Boss) → decks/hops_stall.csv. garchomp: 1 identical list across both
  teams (0-2) → decks/cynthia_garchomp.csv. grim: 4 variants, dominant v0
  at 5 eps (1-4 vs us) → decks/grim_live.csv. All legal.
- **P0.2 — post-mortem reproduced on this box**: pm_extra 55030954 → ELO
  609, ledger identical to docs/m36-post-mortem.md. Band weights frozen
  into scripts/band_decode.py (n=52).
- **P0.3 LAUNCHED — m37_strawman.sh seed 1**: ship config (m28_winners +
  gacf + alakazam_v2_h4) vs 7 beds (hop, garchomp, grimlive, wall, mirror,
  luc, arch) n=200. Early hop read at n=16: 0.500 — strawman validity will
  hinge on the LOSS-MODE diag (live hop losses were deck-outs), not WR
  (live n=2 carries no WR information).
- **P0.3 INTERIM — reference beds REPRODUCE the M36 pins on this box**
  (reconstructed-checkpoint validity): luc 0.6875 n=200 (m36 pin 0.6925),
  arch 0.9700 n=200 (pin 0.9450). Cross-machine measurement path is sound.
- **P0.3 INTERIM — hop bed: wr 0.4700 n=200 with the SOLVER pilot** — below
  0.5 against a cheap pilot (cf. wall 0.36 in M36); consistent with live 0-2.
  Loss-mode diag n=30 running. garchomp bed: solver rung too WEAK (~0.67
  interim, live was 0-2 deck-outs) — strawman rung escalation launched
  (solver2, solver-dev n=100 each), M36 pilot-ladder groove.
- **P0.3 HOP LOSS-MODE DIAG (n=30, seed 7, workers=1): STRAWMAN PASSES.**
  12W-18L (wr 0.400). Losses: **11/18 = 61% true deck-outs** (our deck=0,
  prizes-left 1-4 — the live farm mode exactly), 6 prize-losses (solver
  pilots the hop list semi-aggressively — Trevenant/Snorlax do attack),
  1 bench-out brick (moves=12). Wins include 1 won deck-out RACE (opp deck
  0 at moves=105) — the M36 economy machinery shows up here too. VERDICT:
  `solver:decks/hops_stall.csv` is a VALID bed for measuring anti-stall
  fixes (wr 0.40-0.47, dominant loss mode = deck-out).
- **P0.3 COMPLETE — full seed-1 strawman ledger (n=200/bed):** hop **0.4700**
  ✓valid · garchomp 0.6650 ✗ (rungs solver2 0.710, solver-dev 0.615 — ladder
  EXHAUSTED, live 0-2 not reproduced by any cheap pilot) · grimlive 0.7900 ✗
  (live 0.375) · wall **0.3950** ✓ (m36 4-seed pin 0.4213, z≈−0.75) · mirror
  0.6000 (pin 0.5350, z≈+1.84 — seed-1-high, 2-seed batteries as usual) ·
  luc 0.6875 ✓pin · arch 0.9700 ✓pin. **Conclusion: opponent PILOTING quality
  is the bottleneck for garchomp/grim beds — the real 600-band stall
  opponents are trained agents; only the BC-clone recipe reproduces them
  (M30 rocket precedent). Hop is the exception: its farm mode is mechanical
  (mill + wall) and survives a cheap pilot.**
- **Clone-viability + provenance pinned (opp_decks.parquet, 10270 rows):**
  cynthia family 512 seat-rows (dominant `cynthia_s_gabite+cynthia_s_roselia`
  323); hop family 262 (`hop_s_phantump+dunsparce` 232); grim
  `marnie_s_grimmsnarl_ex+marnie_s_impidimp` 378 (+munkidori-variant 1339).
  Working-clone band was 197–689. Live opponent subs for targeted refresh:
  hop 54559989 (nun), 53989103 (bit_shinon); garchomp 54964845 (Jim
  Eastburn), 55042761 (hideMonkey); grim 54806089/55045468/55048039/
  55055132/55055255.
- **P0.4 — 600-band composite BASELINE (band_decode.py): 0.6378 ± 0.030**
  over 38/52 covered — UPPER BOUND (grimlive+garchomp cells strawman-invalid;
  live band WR 0.500). The honest gate needs the clone beds in those cells.
- **ENV — test suite on this box: 651/655 pass.** 4 Windows-only artifacts
  (test_meta_eval path-separator asserts ×3, test_network tempfile lock ×1)
  — flagged for a separate cleanup task, not milestone-blocking.
- **P0.5 — gacfv rider pre-cleared on the new hop bed: 0.4775 n=200 vs gacf
  0.4700 n=200 (z≈+0.15, non-inferior)** — as predicted for a rare-trigger
  rule (hop games seldom reach opp match point). The M36 all-beds
  non-inferiority claim now extends to the one new valid bed; gacfv stays
  the cheapest ship-ready rules lever.
- **O12 trigger id sets pinned** (cards_features.csv): Great Tusk 58,
  Dwebble {344,532}, Crustle {345,533}, Terrakion 607, Hop's Phantump 878 /
  Trevenant 879 / Snorlax 304, Cynthia's Gible 379 / Gabite 380 / Garchomp
  ex 381 / Roselia 341 / Roserade 342 / Spiritomb 387. NB duplicate-printing
  ids for Dwebble/Crustle — use BOTH in the set.
