# M34 plan — cut the opening-brick blowouts (consistency deck-tech + an aggro instrument)

Direction chosen by Piotr after the M33 post-mortem (docs/m33-post-mortem.md):
**consistency deck-tech + aggro bed.** M33 (alakazam_v2) cured the slow death
(deck-outs 34.6%→12% of losses, implied ELO 714→743) but the loss mass moved to the
front of the game: **6/17 losses (35%) are zero-prize opening BRICKS vs aggro**
(bench 0-1, deck still 37-42 deep at t6-12 → dead hand, Alakazam line never online,
swept before we set up). This milestone attacks that specific failure.

## The instrument problem (must be solved FIRST — it's the enabling gate)

The two matchups now driving losses — fast aggro and the grim weakness — are **both
unmeasured offline** ([[no-validated-live-predictor]]). For aggro specifically:

- **A starmie clone is INFEASIBLE**: best single sub = 4 seat-rows in opp_decks
  (grim clone needed 689; archaludon was declined at 335).
- **A fresh lucario clone would be thin/mid-tier**: top sub 142 rows @ score 708 —
  weaker than the live 750+ aggro that actually swept us.
- `rule:lucario` (the company Mega-Lucario sample agent) is the strongest available
  aggro proxy — but we already beat it ~0.70 offline, so its pooled WR **drowns the
  brick subset**. Pooled WR vs one opponent is the wrong gate here.

**Resolution — a two-tier, WR-independent brick instrument (the real deliverable of
Stage 1):**

- **Tier 0 — opening-hand brick simulator (instant, combinatorial, no engine).**
  Draw N opening hands from a deck list; classify a "brick opening" (e.g. no Abra AND
  no basic-search/Poffin out to develop the line by turn ~2, accounting for mulligan
  + first draws). Pure combinatorics → ranks candidate lists in milliseconds. Cheap
  filter before any game collection. Baseline the shipped v2 first.
- **Tier 1 — in-game brick rate by turn (engine, the validating gate).** Over a
  collection, measure the **brick signature directly**: share of games with bench ≤1
  at turn ≥4 and/or deck-not-dug (deck > threshold) at turn ~6 — the live sweep
  fingerprint — plus a setup-by-turn curve (turns-to-2-bench, turns-to-Alakazam).
  This is WR-independent, so it isn't drowned by the 0.70 pooled number. Cross-check
  with WR vs `rule:lucario` on the fast-loss subset.

If Tier 0 and Tier 1 disagree (opening-brick down but in-game brick flat), the brick
is a pilot problem not a deck problem → pivot to the bench-floor rule idea, don't
ship a deck.

## Program (each stage a cheap go/no-go; pilot = m28_winners UNCHANGED throughout)

**Stage 1 — build + baseline the brick instrument (LOW effort; the enabling gate).**
- 1a. Tier-0 opening-hand simulator (`scripts/brick_sim.py`), baseline on
  `decks/alakazam_v2.csv`. Report P(brick opening) with a CI.
- 1b. Tier-1 in-game brick metric: extend `scripts/offline_behavior.py` (or a new
  `scripts/brick_probe.py`) to emit bench≤1-at-t≥4 share, deck-at-t6 distribution,
  turns-to-2-bench / turns-to-Alakazam. Baseline on a fresh collection of the shipped
  bundle vs `rule:lucario`, n≥200, `--workers 8`.
- **GATE:** the in-game brick share must be **non-trivial and correlate with losses**
  (bricks over-represented among losses vs wins). If bricks are ~equal in wins and
  losses, the deck-tech thesis is wrong → STOP and reconsider (rules/other lever).

**Stage 2 — generate + rank consistency candidates (LOW→MED effort).**
- Enumerate concrete swaps that raise opening consistency while holding the M33
  deck-out gains and the 60-card frame. Candidate levers (each single-variable):
  - draw/dig up: Hilda is played only 5.4% live — is it dead weight? consider a
    higher-impact draw supporter or an extra Poké Pad;
  - the disruption↔consistency trade: Enhanced Hammer (3) / Nighttime Mine (1) /
    Xerosic (3) are non-develop cards — swapping one for a consistency card is the
    natural test (note v2 already cut a Hammer + a Mine; this continues that axis);
  - evolution-line ratio (4-4-4 + 4 Rare Candy is heavy — a 4-3-4 or freeing a slot);
  - basic-search / bench establishment (Poffin maxed at 4 — is there another out?).
- Rank all candidates on Tier 0 first (instant), take the top 2-3 to Tier 1.
- **GATE:** a candidate must lower BOTH Tier-0 opening-brick AND Tier-1 in-game brick
  share vs v2, materially (pre-register a threshold), with no new dead weight.

**Stage 3 — full battery, hold the M33 gains, QC, ship decision (the payoff).**
- Full battery n=400/bed on the survivor(s): the deck-out beds **rocket + grim**
  clones (must stay ≥ v2 — do not regress the M33 win), **mirror / lucario /
  archaludon / kyogre floor** (non-inferior), and the new **brick share** as a
  first-class reported metric.
- Zero-shot deck transfer applies (pilot is m28-trained on the base list; transfer
  penalty priced in per the M24 law — true effect likely larger with a fit pilot).
  `scripts/strength_gate.sh` is NOT the relevant guard (no retrain); the battery
  non-inferiority + brick metric are.
- Mandatory pre-ship QC ritual: `play_games` the actual bundle vs an aggro opponent
  (rule:lucario) AND a stall opponent (rocket clone), save replays, STOP for Piotr's
  manual review + explicit go before any Kaggle submit.

## Honest risks
- **The aggro bed is a proxy, not the live matchup** — rule:lucario is weaker/cleaner
  than the live 750+ pilots that swept us. The Tier-0/Tier-1 brick metrics (deck
  properties, WR-independent) are the trustworthy gates; WR vs rule:lucario is only a
  cross-check. This is exactly the grim lesson: don't trust a single clone's WR delta.
- **Deck-tech may be near its ceiling** — v2 already spent the obvious setup-speed
  swaps. If Stage 2 finds no candidate that lowers brick share without regressing a
  bed, the answer is "v2 is the deck; the frontier is the pilot/prize-race" → escalate
  to the solver-relabel bigger bet (post-mortem option 5), not a marginal deck ship.
- **Variance floor** — some brick rate is intrinsic to any evolution-heavy deck; the
  goal is to lower it, not eliminate it. Pre-register "material" so we don't ship noise.

## Do NOT re-propose (measured-spent — see post-mortem + campaign memory)
- Pilot rules for the supporter/gust holes (5 milestones; imitation can't fix gust).
- PPO fine-tune / self-play DAgger for out-of-loop strength (no-transfer ×3 + M33 S3).
- V1-style draw-depth deck-tech (cut Xerosic/Shaymin) — was a KILL (rocket −8pp,
  mirror −14pp); the disruption cards earn their slots against the field.

## Stage-2 battery RESULT (2026-07-26) — one safe survivor, none resolved-better

Full WR battery (`scripts/m34_battery.sh`, pilot=m28_winners, deck the only variable,
pooled n=400/cell, 2-prop z vs same-session v2 baseline; `runs/m34_battery_full.log`):

| bed | v2 base | mine (−Mine +Ultra) | ace (−Enrich +Master) | agg (−2 disrupt +2 Ball) |
|---|---|---|---|---|
| **rocket** | 0.588 | 0.632 (+4.5, z+1.3) | 0.637 (+5.0, z+1.5) | **0.450 (−13.8, z−3.9)** |
| **grim** | 0.723 | 0.720 (−0.3, z−0.1) | **0.650 (−7.3, z−2.2)** | 0.667 (−5.5, z−1.7) |
| lucario | 0.718 | 0.682 (−3.5) | 0.680 (−3.7) | 0.718 (0.0) |
| mirror | 0.463 | 0.480 (+1.8) | 0.530 (+6.8, z+1.9) | 0.440 (−2.3) |
| arch | 0.940 | 0.935 | 0.963 | 0.920 |
| kyo | 0.970 | 0.963 | 0.963 | 0.978 |

- **agg KILLED** — rocket 0.450 (z−3.9). The V1 lesson reproduced: cutting 2
  disruption cards craters the stall matchup.
- **ace FAILS the hold gate** — improves rocket (+5.0) and mirror (+6.8) but
  **resolvedly regresses grim (−7.3, z−2.2)**, a deck-out bed we must hold. Adding the
  clean (no-discard) search REQUIRES cutting the deck's lone ACE-spec (Enriching
  Energy), and losing that energy hurts the Darkness matchup.
- **mine = the only survivor** — holds both deck-out beds (rocket +4.5, grim flat),
  non-inferior elsewhere. But NOT resolved-better anywhere (all directional).
- Bonus finding: adding a search item *improves* the rocket/stall matchup (+4.5–5.0)
  — faster setup closes the grind before deck-out, not just anti-aggro. And **what you
  cut matters**: stadium (clean) < ACE energy (hurts grim) < 2× disruption (craters rocket).

## Tier-1 in-game brick RESULT (2026-07-26) — the Tier-0 gain does NOT convert

`scripts/offline_behavior.py` (already emits the brick signature: `min_bench_after_t3`,
bench≤1 share, missed-basic-bench), v2 vs mine, both vs rule:lucario, pooled 2 seeds
n=400:

| metric | v2 | mine |
|---|---|---|
| WR vs rule:lucario | 0.685 | 0.682 |
| **bench≤1 prompt share** | **13.4%** | **15.3%** (worse) |
| bench-0-after-t3 losses | 18/126 | 27/127 (worse) |
| avg turns | 11.1 | 10.9 |

- **The Tier-0 opening-brick gain (−2.1pp) does NOT convert in-game.** mine's in-game
  bench establishment is flat-to-slightly-WORSE across 2 seeds. Likely cause: **Ultra
  Ball's 2-card discard** (Tier-0 counts it as a free searcher; in-game it costs cards)
  + the zero-shot pilot not leveraging it. Lesson: the combinatorial opening-brick
  metric OVERSTATES card-cost search; the in-game metric is the truth.
- **rule:lucario is too weak a proxy** — for v2 only 18/126 losses (14%) are bench-0
  and WR is 0.685; it does NOT reproduce the live 750+-pilot brick-blowout (35% of live
  losses). We still lack a faithful aggro bed, and where we CAN measure, mine doesn't help.
- **The redirect signal:** `basic PLAY at bench≤1` is only **~39%** (v2 201/510) — the
  pilot DECLINES a playable basic ~60% of the time at low bench. Combined with hard-brick
  being just 1.2% (Tier-0), the brick residual is more a **pilot bench/dig problem than a
  deck problem** — pointing at a bench-floor RULE (post-mortem option B), not deck-tech.

## Stage-2 VERDICT: no deck-tech variant is a resolved improvement to ship.
agg killed, ace regresses grim, mine is safe-but-neutral (mechanism doesn't convert).
The cheap gate did its job in ~2h. Decision point for Piotr — options in the response.

## Mechanics
- Branch `feature/m34` (M33 shipped bundle + docs not yet committed — sort the commit
  point with Piotr per CLAUDE.md "commit only shipped milestones").
- Hermes status updates at each stage transition; heartbeat on any >30min collection.
- MILESTONES.md row on ship.
