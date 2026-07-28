# M36 plan — attacking the M35 loss modes

Status: **refined v2** (2026-07-27) — stub reviewed against docs/m35-post-mortem.md
+ codebase re-verified this session (see Appendix B additions), awaiting Piotr's
arm pick. Input: sub 55011605, implied ELO 817, tied-best. Loss mass to attack:
**stall walls 30% · close-race 39% (incl. the gust misfire) · dragapult 1-3 ·
mirror deck-out edge**. Bricks are CLOSED (draw variance).

Refinement deltas vs the stub: W2 code-shape correction (new PLAY-demote branch,
not an existing slot); E-section added (levers the stub missed: meta-weighted
gate via the dormant `rl/league.py`, automated deck search, wall-targeted
fine-tune, damage-formula probe promoted from open question to P0.5); data
premises verified (wall subs 0 rows in opp_decks.parquet, dragapult clone
54827443 = 197 rows).

---

## W1. Stall walls (0-6, 30% of losses) — the big mass

**What the forensics + codebase say:**

- **It is ONE deck.** All 4 farm opponents (subs 54832838 / 54291312 / 55006462 /
  54334435, scores 683–814) played a byte-identical 60: 4× Great Tusk (140HP,
  max_dmg 160) / 4× Crustle (150HP) / 4× Dwebble / 1× Terrakion; **all 8 energies
  SPECIAL** (4× Mist, 4× Rock Fighting); 4× Boss's Orders (snipes our 50HP Abras),
  4× Xerosic, 4× Explorer's Guidance, heal (2× Lisia's Appeal, Jumbo Ice Cream),
  Neutralization Zone. Full list in Appendix A — ready to save as
  `decks/greattusk_wall.csv`.
- **Type math is FOR us, not against us** (unlike grim): Great Tusk weak=5
  (Psychic ×2 — our attack type), Terrakion weak Grass, Crustle weak Fire.
  The matchup is a TEMPO/ECONOMY loss (t30–39, we took 1–3 prizes), not a
  weakness bleed — in principle winnable with our deck.
- **Enhanced Hammer is full-coverage denial here**: 8/8 of their energies are
  special. M33's v2 deck-tech cut hammer 4→3. With a wall bed, "hammer-4 revert"
  (swap-out candidate: Hilda, 4.2% play rate — watch supporter count) is a
  cheap, directly-motivated deck A/B.
- **Bed path exists**: `generic:<deck.csv>` / `solver:<deck.csv>` matchrunner
  specs; a new bed is one `bed_spec` line in an m36_battery.sh clone of
  scripts/m35_battery.sh. **Gate: the M26/M30 strawman law** — the bed must
  reproduce the live loss mode (m35 config LOSES with deck-outs/low prize-take)
  before any fix is measured on it. M32-B warning applies (`generic:archaludon`
  sat 0.86 vs live 0.47) — but wall decks are low-skill (sit + tank + gust).
  **Pilot ladder if generic: is too weak** (refinement — more rungs than the
  stub listed): `generic:` → `generic2:`/`generic2a/b:` → `solver:` →
  `solver2:` variants → `solver-dev:` (all take a deck CSV,
  rl/matchrunner.py:52-59,124-126) → BC clone as last resort.
- **Clone fallback is thin**: verified this session — all 4 farm subs have **0
  rows** in opp_decks.parquet (stale harvest). P0: `kaggle_ingest refresh --subs
  <4 subs>` then **`kaggle_ingest harvest`** (rewrites opp_decks.parquet from the
  raw cache — this is the missing second step) → answers both clone-viability
  and "do they farm everyone or just Alakazam?". Historical wall subs top out at
  54 seat-rows single-sub — below clone viability (archaludon died at 62;
  working clones had 197–689).

**Candidate fixes, in order of measurability once the bed exists:** hammer-4
deck-tech → prize-close deck-tech → E2 neighborhood search → E3 fine-tune
(weights levers were measured-spent only vs the OLD meta — see E3).

## W2. Gust misfire at opp-prize≤1 — 5/48 Boss plays, ALL 5 in losses

- **Code-shape CORRECTION (refinement):** the stub said "DEMOTE family slot
  confirmed". Verified: `apply_play_overrides`' demote branch fires **only when
  `ranked[0]` is an ABILITY** (rl/plan.py:465-471) — deckguard/conserve demote
  board abilities. A gust veto demotes a **PLAY**, so it needs a new sibling
  branch: `opp prizes ≤ 1` and top pick is PLAY with card id in `GUST_IDS`
  (already pinned = {1182}, rl/plan.py:25) → demote all such PLAYs. Same family
  and precedence story, ~10 lines + a `modelt-gacfv` entry in
  `_MODEL_FIX_KINDS` (rl/matchrunner.py:64) + fix-string constant. Still the
  allowed deterministic-obs-fact class.
- **Design constraint verified**: `_attack_damage` returns 0 for variable
  attacks (`dmg <= 0` gate, rl/plan.py:55; `_ATK[338]/[1072] = (0, …)`) — a
  "veto unless target KO-able" gate is blind to Alakazam, our main attacker.
  Start with the **blanket demote** (any tempo error at opp-pz≤1 is fatal
  anyway); add a KO-gate for non-variable attackers only if beds object — or
  after E4 unblinds the damage table.
- **Ship path is the M31/M35 groove**: rule in rl/plan.py (vendored by
  tcg/shipping.py), default string in submission/main.py `_ATTACH_FIXES`,
  letter arm `gacfv` vs `gacf`, n=400/bed non-inferiority + engagement counter
  (forced-state probe à la scripts/ability_probe.py — the trigger is too rare
  offline for a strength claim; final read = live A/B).

## W3. Dragapult 1-3 live (was 3-3)

- **A dragapult BC clone IS viable** (unlike archaludon): sub 54827443
  (LumenLiquidity, @1194) verified at **197 seat-rows** in opp_decks; a refresh
  can grow it; recipe = the m25/m30 clone scripts. Risk: the grim clone failed
  fidelity (0.565 → useless pilot); the rocket clone worked. Medium cost,
  coin-flip.
- n=4 live is not a resolved regression. **Recommendation: WATCH** — build the
  clone only if the next ~40 live games keep dragapult under water.

## W4. Mirror deck-out edge (mirror = 22% of the meta now)

- obs at MAIN exposes both `deckCount`s → a deck-RACE-aware conserve (fire the
  economy demotes EARLIER when behind on the deck race in long games) is codable
  in the allowed class. But: composition law (gacbf died), and the mirror bed
  resolves only 7pp at n=800 — behavior counters, not strength, would gate it.
  **Park unless Piotr prioritizes**; needs a concrete trigger design first.

## Chronic / watch (do NOT spend on)

- **Bricks: CLOSED** — both residuals were zero-basic hands.
- **Telepath 70.7% per-prompt** vs 85–86% prior: run the per-TURN probe first
  (M31 P0.4 artifact law) before any alarm — P0 item.
- Hilda 4.2% / supporter-turns 59%: weights-borne (6 dead attempts); only
  remaining lever is deck-teching the slot (see W1 hammer swap).
- Poké Pad END-declines: partially ADAPTIVE conservation (M30 P3); leave.

---

## E. Refinement pass — levers the stub missed

### E1. Meta-weighted ship gate / local leaderboard (small→medium, mostly EXISTING code)

The [[no-validated-live-predictor]] problem is structural: our beds measure the
~29% of the meta we already beat; the mass we lose to is unmeasured. Two tiers:

- **E1a — meta-weighted portfolio decode (SMALL, this milestone).** One script
  aggregating the m36 battery JSONLs into a composite WR weighted by live meta
  shares (n=55: mirror 12, archaludon 11, lucario 8, wall family 6, grim 5,
  dragapult 4, other 9). Adding the wall bed makes the portfolio cover ~75% of
  the live field — the first offline number with a defensible claim to predict
  live movement. Freeze weights via `kaggle_ingest meta` (meta_v3 snapshot) so
  the gate is pre-registered, not post-hoc. Caveat ([[measurement-power-
  discipline]]): weighting adds NO statistical power — per-bed n still governs;
  the composite is a decision aid, not a new significance bar.
- **E1b — revive `rl/league.py` (MEDIUM, standing infra).** Verified: a full
  persistent OpenSkill (PlackettLuce) leaderboard over (deck, pilot) entries
  already exists — anchors, anchor round-robin, standings, gates, promote,
  CLI (rl/league.py:59-69,228-248,465-474,514-600; tests in
  tests/test_league.py). It is pre-M18-stale: the 8-anchor field (lucario/iono/
  dragapult experts, generic, bc_v1, random) contains NONE of the current meta.
  Work: modern anchor set = the 6 m35 beds + wall bed; check `Entry` pilot
  specs accept `modelt-gacf:`/`model:` clone specs (likely small adaptation).
  Payoff: every future candidate gets one ordinal + persistent history instead
  of 24 pairwise runs read by hand.

Recommendation: **E1a now** (it rides the battery we run anyway), E1b as
standing infra when a milestone has slack.

### E2. Automated deck search around alakazam_v2 (EXISTING code, compute-medium)

Verified: `rl/deck_search.py` has `validate_deck`, `mutate`, `hill_climb`,
`field_hill_climb` (fitness vs a FIELD of specs) and `rate_population` (random-
pairing OpenSkill tournament, :179-197); `rl/deck_build.py` emits legal-by-
construction decks (65 already in decks/gen/). Use: neighborhood search ≤2
swaps from alakazam_v2 with the wall bed IN the fitness field — automates the
anti-wall tech hunt beyond the hand-picked hammer-4. Constraints: search runs
under generic/solver pilots (fidelity gap — M34 lesson: Tier-0/1 gains did NOT
convert in-game), so any winner still faces the full m28_winners battery +
strawman-checked wall bed; workers ≤8. This is the cheap middle ground between
"hand-pick one swap" and "design a new deck".

### E3. Wall-targeted fine-tune — weights are only "spent" vs the OLD meta (high cost, P3/stretch)

Every weights-lever kill (6 dead supporter attempts, M32-B PPO, M33 value arc)
was measured against the pre-wall opponent pool. The wall archetype is a NEW
opponent distribution; RL best practice says a meta shift is answered by
training on the shifted distribution — rules and deck-tech are patches. The
groove exists: `plan_iter collect` vs wall/stall beds (+ current beds for
retention) → warm-start fine-tune of m28_winners → full non-inferiority battery
(catastrophic-forgetting guard). Preconditions: (a) wall bed passes strawman,
(b) hammer-4/rules fail to move wall WR, (c) a milestone's worth of appetite.
Collection constraints: workers ≤8, fresh `--out` dirs (no resume).

### E4. Alakazam damage-formula probe (small, high option value) — PROMOTED to P0.5

`_attack_damage` models our main attacker as 0 damage (verified, rl/plan.py:55)
— plan features, turn solver, and any KO-gate are blind to Alakazam's output.
Probe recipe exists: scripts/ability_probe.py's forced-state pattern — force
attacks 338/1072 across controlled board states, regress damage on (energies,
hand size, bench, …). If the formula is simple, wire it into rl/combat's `_ATK`
handling and re-measure solver-adjacent beds. Payoff compounds into W2's
KO-gate, prize-close planning vs walls, and any future value work.

---

## Proposed M36 shape (pending Piotr)

- **P0 probes (cheap, ~1 session):**
  1. Save `decks/greattusk_wall.csv` (Appendix A) → strawman check: m35 config
     vs `generic:`/`solver:` wall, n=200×2 seeds — does it lose by deck-out?
  2. `kaggle_ingest refresh --subs 54832838 54291312 55006462 54334435` **then
     `kaggle_ingest harvest`** → field-wide W/L + clone-data count.
  3. Per-turn telepath probe on the 55011605 cache.
  4. Mirror-loss forensics (the sereinless deck-out race) for a W4 trigger design.
  5. Freeze meta_v3 weights (`kaggle_ingest meta`) for the E1a portfolio decode.
- **P0.5 (E4):** Alakazam damage-formula probe (forced-state, one session).
- **P1 (rule arm):** O11 `gustveto` — new PLAY-demote branch (W2 correction) →
  m36 battery {gacf, gacfv} on all beds n=400 + engagement probe.
- **P2 (iff wall bed passes strawman):** hammer-4 / deck-tech A/B on wall bed +
  full non-inferiority battery; E2 neighborhood search feeds candidates if the
  hand-picked swap disappoints.
- **P2.5 (E1a):** meta-weighted portfolio decode over the m36 battery outputs —
  report alongside, not instead of, per-bed bars.
- **P3 (conditional/stretch):** E3 wall fine-tune only if P2 fails to move wall
  WR; dragapult clone only if live keeps it under water; E1b league revival
  when slack.
- Ship shape stays single-variable vs 55011605 (rules-only, or deck-only).

## Open questions for Piotr

1. Arm priority: wall bed + deck-tech (30% loss mass) vs gust veto (cheap,
   small edge) — or both in one milestone (M35 precedent: 4-arm matrix)?
2. Gust veto: blanket demote now, or wait for E4 to enable the KO-gated design?
3. Dragapult clone: build now (~1 day) or watch 40 more games?
4. E1: portfolio decode only (E1a), or also invest in the league.py revival
   (E1b) as standing pre-ship infrastructure?
5. E3 appetite: if hammer-4 doesn't move the wall matchup, is a wall-targeted
   fine-tune milestone (M37) on the table?

---

## Execution log (2026-07-27, feature/m36)

- **P0.1 DONE — `decks/greattusk_wall.csv` saved.** Extracted from the opponent
  seat's first 60-length action in eps 88328736/88329266/88329805/88339280
  (seats from episodes.parquet — the raw blobs carry empty per-seat info; all 4
  rows confirmed Team Pierogachu losses). All 4 multisets byte-identical AND
  exactly match Appendix A. `deck_search.validate_deck` = legal. Scratch script:
  scratchpad extract_wall_deck.py.
- **P0.1b LAUNCHED** — scripts/m36_strawman.sh: modelt-gacf m28_winners +
  alakazam_v2 vs {generic,solver}:greattusk_wall, n=200 × seeds {1,2}, workers 8.
- **P0.2 LAUNCHED** — `kaggle_ingest refresh --subs 55011605 --opp-subs
  <4 wall subs>` (wall subs are OPPONENT subs → the snowball channel, not
  --subs) then `harvest`.
- **P0.3 DONE — telepath alarm DISMISSED (per-turn artifact, as the M31 law
  predicted).** scripts/telepath_turn_probe.py (new, reusable): M35 per-prompt
  70.7% (164/232 — reproduces live_postmortem exactly) but per-TURN **87.2%**
  (164/188 offered turns; other-energy 2.7%, no-attach 10.1%). Priors: M31
  per-turn 85.0% (no-attach 9.6%), M33 per-turn 99.2% (n=132, the outlier).
  M35 ≈ M31; the per-prompt drop is denominator dilution (benchfloor adds
  pre-attach prompts). The ~10% no-attach residual predates gacf. No action.

- **P0.1b INTERIM (seed 1, generic pilot): STRAWMAN REPRODUCES — wr 0.310**
  (62W-138L n=200) for the m35 ship config vs `generic:greattusk_wall`. Even the
  CHEAP pilot beats us with this list (live 0-4; generic:archaludon had sat at
  0.86 — this bed does not have that problem). Loss-mode check pending.
- **P0.4 DONE — mirror race probed (scripts/mirror_race_probe.py, new), W4
  trigger designed, recommendation PARK.** 12 mirror games 8W-4L, ONE deck-out
  loss (ep88344662: decked t18 with opp still at 18 cards — self-drain ~1
  extra card/turn, not a long-game race). Naive "demote draw-abilities when
  behind on race" fires 14× incl. 7 in WINS (t3-t6 setup digs — demoting those
  costs setup; composition risk). Margin-gated design isolates the loss:
  **raceconserve = demote dud/fez at MAIN iff my_deck < opp_deck − 5 AND
  my_deck ≤ 25 AND my_deck > 6** → fires 3× in the deck-out loss (t4/t10/t12,
  margins 6/7/15), ~1 borderline fire in a win (ep88332444 t5, deck 25 v 33).
  Value ≈ 1 loss per 12 mirror games ≈ 2% of games — below the mirror bed's
  7pp resolution. PARKED unless Piotr prioritizes; design is ready if so.

- **P0.1b DONE — STRAWMAN FULL PASS on BOTH axes; wall bed is VALID.**
  scripts/m36_strawman.sh n=200×2 seeds: `generic:` wall 0.310/0.395 (pooled
  0.3525), `solver:` wall 0.345/0.315 (pooled 0.330) — the m35 ship config
  loses ~2:1 (live 0-4 is consistent with wr 0.33 at n=4). No M32-B
  generic-sits-at-0.86 problem here. Loss-mode diag (n=40, seed 3, solver
  wall): **28/28 losses are DECK-OUTS** (our deck=0; prizes-left median ~3;
  opp took ≤1 prize in 25/28) — the live mode reproduced exactly. Our wins:
  6 by taking all 6 prizes, 7 by out-decking THEM. → P2 hammer-4 A/B is GO
  on `solver:greattusk_wall` (solver pilot, the stronger + loss-mode-valid rung).

- **E4 DONE — Powerful Hand pinned (scripts/damage_probe.py, new), and it
  REWRITES the wall thesis.** Engine card text + forced-state probes (wall
  n=16+12+24, lucario control n=12): **attack 1072 = 20 × hand_size damage
  counters, weakness/resistance NEVER applied** (20.0/card exactly vs
  psychic-weak Tusk), **blocked to ZERO by the wall's special energies on the
  target** (Mist/Rock-Fighting-attached: 70/76 then 25/27 events = 0 dmg at
  hands up to 21); basic/fighting energies do NOT block (320 dmg through
  [6,6] vs lucario bed); stadium irrelevant (lands through Neutralization
  Zone when target naked; blanked under our Nighttime Mine when attached).
  Kadabra 1071/Abra 1070 follow the damage table WITH weakness ×2 (existing
  `_attack_damage` model correct for them). **Consequences:** (a) the plan's
  "type math is FOR us" premise is FALSE — counters never double, 140HP Tusk
  needs hand≥7; (b) the wall's farm mechanism = energy-attached mons are
  IMMUNE to our main attacker; (c) **Enhanced Hammer is the mechanical
  UNLOCK** (strip special → 20×hand lands) — hammer-4 revert (P2) upgraded
  from tempo story to causal story; (d) a gustveto KO-gate is now computable
  deterministically from obs (20×hand + "opp-active-special-energy → 0") if
  the blanket demote regresses on beds.

- **P0.2 DONE — refresh (500 eps, parquet 5717 rows) + harvest fixed the
  stale opp_decks.** `crustle+great_tusk` now **609 seat-rows / 304 games**
  (avg score 777) — far ABOVE clone viability (working clones 197–689), so a
  wall BC clone is a real P3 option. Field-wide forensics (4 wall subs): they
  are NOT dominant — 0.45/0.49/0.58/0.47 overall — they are an
  **anti-Alakazam farm**: pooled 75W-46L (0.62) vs the alakazam archetype,
  while LOSING to grim 15-44 (0.25) and rocket 4-17. Mirror-heavy meta (22%
  alakazam) is what feeds them. Dragapult 54827443 unchanged at 197 rows.
  Live cache 55→57: both new games LOSSES — **dragapult now 1-4 (WATCH
  continues)**, +1 rocket deck-out.
- **P0.5 DONE** — meta_v3 frozen (8 decks incl. wall d3e4d16c) for the E1a
  pre-registered weights.
- **NEW FINDING — prize-array semantics pinned (and `_make_plan`'s comment is
  INVERTED).** Empirics (diag end-states: prizes-left hits 0 exactly at our
  wins and 3-5 at wall losses matching live; forced kyogre probe: a 3-prize
  jump on OUR array while mowing exes with Powerful Hand): **a player's
  `.prize` = the prizes THEY still need.** rl/plan._make_plan's comment says
  the opposite, and its `wins`/`concedes` features compare against the wrong
  array — a LATENT FEATURE BUG the frozen net was TRAINED with. DO NOT fix in
  a rules milestone (feature-distribution shift on a frozen net); queue for
  the next training arm. O11 uses the verified semantics (`len(op.prize)<=1`).
- **P1 DONE — O11 gustveto implemented.** rl/plan.py: `PLAY_FIX_GUSTVETO`
  PLAY-demote branch (blanket, no KO-gate, per plan W2); rl/matchrunner.py:
  `modelt-gacfv`; tests: 3 new cases incl. gacf composition + promote
  precedence (suite 633+22 green); submission/rl/plan.py twin synced
  (test_bundle_twins); smoke 6-0 vs random. NOTE: submission/main.py
  `_ATTACH_FIXES` deliberately UNTOUCHED — flips only on a ship decision.
- **P1b LAUNCHED** — scripts/m36_battery.sh {gacf CONTROL fresh, gacfv} × 7
  beds (m35 six + solver wall) × n=200 × 2 seeds = 5600 games, heartbeat on.
  Fresh controls everywhere: no cross-battery jsonl reuse (drift law).

- **PRE-REGISTERED BARS (fixed 2026-07-27 before any A/B number, in
  scripts/m36_decide.py):** P1 `gacfv` vs same-battery gacf control:
  non-inferiority z > −1.96 on EVERY bed + kyogre ≥ 0.95; NO offline strength
  claim required (engagement probe + live A/B carry it); any regression
  kills. P2 `h4` (+1 hammer/−1 Hilda, deck single-variable on gacf rules):
  wall STRICTLY better z > +1.96 (E4 mechanism predicts a real move) +
  non-inferiority on the other 6 + kyogre ≥ 0.95. Ship shape single-variable
  vs 55011605; Piotr picks the arm. Support scripts ready:
  scripts/m36_decktech.sh (h4 runner), scripts/portfolio_decode.py (E1a,
  frozen n=57 weights, coverage 77%), scripts/gustveto_engagement.py
  (running vs arch bed n=60), decks/alakazam_v2_h4.csv (validated legal).
- m35 gacf portfolio decode (E1a smoke, no wall bed): 0.7139 ± 0.021 over
  37/57 live-game mass — inflated by the missing wall cell; the m36 number
  with wall≈0.33·w7 is the honest field read.

- **P1 engagement probe DONE (scripts/gustveto_engagement.py, new):** vs arch
  bed n=60 → ZERO trigger states (we win 54-6; opp never reaches match
  point). vs MIRROR bed n=60 seed 42 → 23 opp≤1 prompts with gust legal,
  **4 would-veto events (top-pick gust at opp match point) — ALL 4 in
  LOSSES, 0 in wins** = the live 5/5-in-losses signature reproduced offline;
  engagement ≈ 1 per 15 mirror games. CAVEAT flagged, not quoted as a
  measurement: the probe series itself ran 0.32 (19W-41L) vs the 0.53 m35
  mirror pin (z≈−3 at n=60) — probe scripts are NOT the authoritative path
  ([[verify-before-consequential-actions]]); the battery's fresh mirror
  n=400×2 arms adjudicates.

- **P1b BATTERY DECODE (28/28 runs, scripts/m36_decide.py): `gacfv` CLEARS
  the pre-registered bar on ALL 7 beds.** Per-bed pooled n=400 vs
  same-battery gacf control: wall 0.3550 v 0.3650 (z −0.29), rocket 0.5950 v
  0.6375 (z −1.24, widest — advisory watch), grim 0.6625 v 0.6425 (+0.59),
  luc 0.6925 v 0.6987 (−0.19), mirror 0.5150 v 0.5350 (−0.57), arch 0.9450 v
  0.9450 (0.00), kyo 0.9800 (floor ≥0.95 ✓). No offline strength claim
  (by design — engagement ≈ 1/15 mirror games, all-in-losses signature);
  the ship case = zero regression + live A/B. gacf mirror 0.5350 replicates
  the m35 pin → the engagement probe's 0.32 was probe-path noise, resolved.
- **E1a PORTFOLIO (frozen n=57 weights, 77% coverage): gacf 0.6548 ± 0.019,
  gacfv 0.6479 ± 0.019** — indistinguishable, as a rare-trigger rule should
  be. The honest field number is ~0.65 (m35's wall-blind decode read 0.714);
  the wall cell (0.36 × w7) costs ~6pp of covered-field WR — exactly the P2
  target mass.
- **P2 LAUNCHED** — scripts/m36_decktech.sh w+full: h4 (+1 hammer/−1 Hilda)
  on gacf rules, wall decisive (strict z>+1.96) then 6-bed non-inferiority.

- **P2 DECODE (14/14 runs): h4 wall +6.25pp (0.4275 v 0.3650) z=+1.81 —
  UNDER the strict +1.96 bar; all 6 other beds non-inferior** (luc +3.4pp
  z+1.06, grim +2.1, arch +1.0, kyo 0.99, rocket −0.8; mirror −2.25pp
  z−0.64, seed-split 0.555/0.470). Direction + size match the E4 mechanism
  prediction; the bar verdict as pre-registered is FAIL.
- **PRE-REGISTERED AMENDMENT (fixed 2026-07-27 BEFORE launching the confirm,
  M8/M9 screen→confirm groove):** wall CONFIRM at FRESH seeds 3+4, BOTH arms
  (gacf control + h4), n=200/seed. Decision rule: h4 passes iff pooled
  z>+1.96 over all 4 seeds AND the fresh-seed delta alone is positive.
  If it stays under: h4 = no-ship on the strict bar, gacfv remains the clean
  arm, wall lever escalates to E2 neighborhood search / E3 fine-tune.

- **P2 CONFIRM (fresh seeds 3+4, amended rule): h4 PASSES.** All-4-seed
  pooled: h4 **0.4213 n=800 vs gacf 0.3613 n=800, +6.00pp, z=+2.46** (>+1.96
  ✓); fresh-seed delta alone +5.75pp (positive ✓). Per-seed h4
  0.430/0.425/0.415/0.415 — no seed-staleness. Mechanism diag (n=40): losses
  still 24/24 deck-outs, but wins 12→16/40 incl. a NEW win line — 4 early
  BENCH-OUT wins at moves 27-32 (hammer denial + pressure before the wall
  develops), absent from the control diag.
- **h4 portfolio 0.6685 ± 0.018** vs gacf 0.6548 / gacfv 0.6479 (decision
  aid only — deltas inside CI).
- **MILESTONE GATES COMPLETE — both arms clear; arm pick is Piotr's:**
  (1) `gacfv` rules arm — non-inferior everywhere, fires ~1/15 mirror games
  all-in-losses, live A/B carries the claim; (2) `h4` deck arm — wall
  +6.0pp resolved z=+2.46 + non-inferior on the other 6 beds. Ship shape is
  single-variable vs 55011605 — ONE of the two. Next after the pick:
  build_submission export (+ md5/deck verify per [[ship-deck-verify]]) → 3-game
  bundle QC → Piotr's replay review → submit go.

- **ARM PICK (Piotr, in-session, 2026-07-27): h4 deck arm.** gustveto parks
  as a measured, ship-ready rules arm for a future milestone.
- **EXPORT + IDENTITY VERIFY DONE:** `./build_submission.sh --checkpoint
  checkpoints/m28_winners.pt --deck alakazam_v2_h4` (bare NAME — a csv path
  trips the root-deck guard, which correctly hard-errored on the first
  attempt). Bundle `dist/submission_neural_20260727_151253.tar.gz` (5279 KB),
  gate game OK. Verified FROM THE TARBALL: deck.csv md5 `02d22dca` ==
  decks/alakazam_v2_h4.csv (≠ v2 `7b1c123b`), 60 cards, hammer ×4 / Hilda ×3;
  bundle main.py fixes = `telepath,deckguard,ash,conserve,benchfloor` (gacf,
  unchanged — deck is the single variable); submission/deck.csv same md5.
  NOT submitted (no --message).
- **QC DONE: 3W-0L vs sample-agent-tuned** (rewards [1,-1]×3), replays
  `replays/m36_qc_h4_tuned_00{0,1,2}.html`. Stopped for Piotr's review.

## SHIPPED — sub 55030954 (2026-07-27)

**SHIPPED: sub 55030954 = m28_winners + gacf + alakazam_v2_h4**, Piotr's
explicit "Looks good, SUBMIT" after replay review. Bundle
`dist/submission_neural_20260727_151253.tar.gz` (md5 1dcb568d). Pre-submit
verified from the tarball: deck md5 `02d22dca` == alakazam_v2_h4 (≠ v2
`7b1c123b`), 60 cards, hammer ×4 / Hilda ×3; fixes string = gacf unchanged.
Status PENDING at submit. **Clean live A/B vs 55011605 (settled 789.6) — the
ONLY change is the deck slot (+1 Enhanced Hammer / −1 Hilda), so the live
delta isolates the hammer-4 wall tech.**

**Post-ship (2026-07-27, Piotr feedback):** single-opponent QC is NOT proper
QC — mandate changed to a multi-deck battery (scripts/qc_battery.py: 3 games
each vs tuned/iono/dragapult + the PREVIOUS ship tarball as the mirror leg;
CLAUDE.md rule updated, [[qc-multi-deck-battery]] memory saved). First live
games of 55030954: 1W (mirror, t19) / 1L (crustle/tusk wall DECK-OUT t38 —
but contested: prizes 4/3 vs M35's 6-left shutouts; n=1, consistent with the
offline 0.42 claim; transfer verdict at ~n=40).

**Watch next session (live A/B 55030954 vs 789.6):** wall-family W/L (offline
0.36→0.42, the target claim — does +6pp transfer?); bench-out wins appearing
(the new win line); mirror share/W-L (h4 mirror seed-split 0.555/0.470 was
the softest cell); dragapult watch continues (1-4 at n=57 → clone decision at
~40 more games); supporter-turn rate (Hilda 4→3 removes dead offers — expect
per-prompt supporter rate UP, absolute supporter turns possibly down);
gustveto stays parked+measured for a future rules milestone.

## Appendix A — the wall list (identical across all 4 farm opponents)

```
4x Great Tusk (58)      4x Buddy-Buddy Poffin (1086)   4x Boss's Orders (1182)
4x Crustle (345)        4x Pokégear 3.0 (1122)         4x Explorer's Guidance (1185)
4x Dwebble (344)        4x Switch (1123)               4x Xerosic's Machinations (1197)
1x Terrakion (607)      4x Fighting Gong (1142)        2x Colress's Tenacity (1194)
4x Mist Energy (11)     4x Poké Pad (1152)             2x Lisia's Appeal (1204)
4x Rock Fighting (20)   1x Ultra Ball (1121)           1x Jumbo Ice Cream (1147)
                        1x Neutralization Zone (1247)
```

Extraction: deck step of eps 88328736 / 88329266 / 88329805 / 88339280 (the
opponent seat's first 60-length action), `data/kaggle/raw/episode_<ep>.json.gz`.

## Appendix B — tool/pipeline facts checked (stub session + refinement session)

- `rl/matchrunner.py` bed specs: `generic:` / `random:` / `solver:` /
  `solver-dev:` / `generic2*`/`solver2*` all take deck CSVs (parser :119-142,
  resolve_deck :98-109); `rule:` / `model:` / `ext:` / `rank:` / `vsolver:` /
  `solved:` / `mcts:` also exist. No `clone:` spec (it's an arm label in
  m28 scripts).
- Rule arms: `_MODEL_FIX_KINDS` (rl/matchrunner.py:64-95) — `gacfv` = one new
  entry + fix string.
- W2 correction verified: demote branch is ABILITY-gated (rl/plan.py:465-471);
  gust veto = new PLAY-demote branch; `GUST_IDS={1182}` pinned at rl/plan.py:25.
- `_attack_damage` blind to variable attacks: `dmg <= 0 → return 0`
  (rl/plan.py:55); handles weakness ×2 / resistance −30, affordability,
  CONDITIONAL_ATTACKS.
- Data premises verified 2026-07-27: opp_decks.parquet (3091 subs) has 0 rows
  for all 4 wall subs; 197 rows for dragapult 54827443. `kaggle_ingest harvest`
  is the step that rewrites opp_decks.parquet after a refresh (impl :650-689);
  `meta` freezes a meta_vN snapshot; `forensics --sub` gives W/L by archetype.
- `rl/league.py` = dormant local leaderboard (OpenSkill/PlackettLuce, anchors,
  round-robin, standings, gate/promote CLI, tests) — pre-M18 anchor field,
  unused since; `rl/rank.py` is NOT a leaderboard (within-turn action ranker,
  M12).
- `rl/deck_search.py`: validate_deck, hill_climb, field_hill_climb,
  rate_population (OpenSkill tournament); `rl/deck_build.py`: legal deck
  generator (65 decks in decks/gen/). alakazam_v2 itself was hand-authored
  (M33), not script-emitted.
- QC path: `tcg/evaluation.play_games` (and `rl/eval.play_games` +
  `json_prefix`) run kaggle_environments with file agents — the pre-ship QC
  harness; matchrunner specs do NOT apply there.
- scripts/m35_battery.sh `bed_spec()` is the plug point for m36; m35 run
  JSONLs are all present in runs/ as baseline reference.
