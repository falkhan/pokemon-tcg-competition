# M31 post-mortem — sub 54935640 (gacb), 54 live games

_Written 2026-07-24 after refreshing the cache to 54 episodes (`rl.kaggle_ingest
refresh --subs 54935640`, +40 new since the n=14 read in `docs/M32.md`). Ship
config: `m28_winners` + O1 telepath + O4 deckguard + O5 ash + O6 conserve + **O7
poffinfloor** on `clone54618168`. Tools: `scripts/live_postmortem.py`, plus a new
probe promoted to **`scripts/live_deck_race.py`** (live deck-race + drain
attribution + archetype W/L, the replay-side counterpart to the offline
`deck_drain.py`)._

_Power caveat up front (`measurement-power-discipline`): 54 games, 26 losses.
The M31-vs-M30 headline is **directional, not resolved**. The findings in
"Deck-out" and "Matchup structure" are pooled over M29+M30+M31 (n=142) and are
the parts that carry weight._

## Headline: M31 settled ~100 points below M30, and the n=14 alarm was real

| sub | milestone | n | W–L | WR | last-10 avg score | avg opp score | implied ELO |
|---|---|---|---|---|---|---|---|
| 54935640 | **M31 gacb** | 54 | 28–26 | 0.519 | **719** | 697 | **710** |
| 54929991 | M30 gac | 44 | 25–19 | 0.568 | 812 | 764 | 811 |
| 54914673 | M29 pooled-winners | 44 | 24–20 | 0.545 | 662 | 628 | 660 |
| 54903635 | M28f | 49 | 23–26 | 0.469 | 731 | 731 | 710 |

The 908 peak at game 4 is the placement ramp off the 600 seed, not strength; the
sub settled at ~707–719. M31 won a *lower* share of games than M30 against
*weaker* opposition (697 vs 764 avg opponent), which is why the opponent-adjusted
implied ELO gap (710 vs 811) is larger than the raw WR gap.

**Be honest about the statistics:** 0.519 vs 0.568 is z ≈ 0.49 — nowhere near
significant. Two subs at n≈50 cannot resolve a 5pp WR difference (`floor-gate-sample-size`,
`measurement-power-discipline`). What is fair to say: M31 showed no gain, sits
~100 points lower on the live metric, and the early n=14 read (6W–8L) was not a
fluke that reverted — it converged to "flat-to-down", not "up".

## O7 poffinfloor: the mechanism worked live, the score did not follow

The rule engaged exactly as designed — this is not a wiring failure:

| probe | M30 (gac) | M31 (gacb) |
|---|---|---|
| Poffin declined at bench ≤ 1 | 51 | **7** |
| prompts at bench = 1 | 7.3% | **4.2%** |
| prompts at bench = 5 | 16.9% | **24.9%** |
| Poffin plays per game | 1.55 | 1.54 |

O7 did not *add* Poffin plays (1.55 → 1.54 per game); it **relocated** them to
low-bench spots. So the "O7 burns extra deck and worsens deck-out" hypothesis is
**not supported** — deck cost was neutral. We bought a wider bench and got no
score for it.

One genuine regression to note: `missed bench (bench==0, basic playable,
declined)` went **0 → 12 prompts**. O7 promotes Poffin at bench ≤ 1, and at
bench 0 it appears to displace a directly-playable basic. Small n, but it is a
new failure spot introduced by the rule.

## Deck-out is the top loss axis — and we have been treating the wrong cause

Deck-out share of losses is persistent and slightly rising: **M29 6/20 (30%) →
M30 6/19 (31.6%) → M31 9/26 (34.6%)**. Four milestones of economy rules have not
moved it.

First, a wiring check, because the raw probe output looks alarming:
`dud ability at deck<=6: used 0/98` and `fez ability at deck<=6: used 0/26`.
**This is correct behaviour, not a bug** — O4 `deckguard` (`rl/plan.py:335`,
`:440`) demotes Dudunsparce draw at `deckCount <= 6` and O6 `conserve`
(`:344`) does the same for Fezandipiti. The rules are firing at ~100% fidelity.

So the rules work and deck-out is still #1. The reason is that the diagnosis was
wrong. Pooled over M29+M30+M31:

**1. Deck-out games are LONG, and our burn rate in them is the LOWEST.**

| | wins | other losses | deck-outs |
|---|---|---|---|
| avg turns (M31) | 14.9 | 18.0 | **25.6** |
| avg turns (M30) | 14.2 | 17.2 | **30.0** |
| avg turns (M29) | 12.7 | 21.9 | **37.7** |
| our burn/turn (M31) | 2.63 | 2.36 | **2.00** |

**2. Controlling the confound.** Longer games mechanically dilute per-turn burn,
so per-turn averages alone prove nothing. Median deck remaining at a **fixed
turn** (pooled, n=142):

| turn | win | loss-other | **deck-out** |
|---|---|---|---|
| 4 | 30.0 | 33.0 | **37.0** |
| 8 | 16.0 | 20.0 | **22.0** |
| 10 | 14.5 | 18.0 | **18.0** |
| 12 | 12.0 | 19.0 | **12.0** |
| 16 | 10.0 | 13.0 | **5.0** |

In the games we deck out we are **not** burning faster — through turn 10 we hold
*more* deck than in games we win (22 vs 16 at T8). The curves only cross around
T13. We do not over-draw ourselves to death; **we start slow, fail to close, and
the game outruns the deck.**

**3. Win rate collapses with game length.**

| turns | n | W–L | WR | 95% CI | deck-outs |
|---|---|---|---|---|---|
| 0–10 | 34 | 22–12 | 0.65 | [0.49, 0.81] | 0 |
| 11–15 | 48 | 31–17 | 0.65 | [0.51, 0.78] | 2 |
| 16–20 | 31 | 13–18 | 0.42 | [0.25, 0.59] | 5 |
| 21–25 | 11 | 7–4 | 0.64 | [0.35, 0.92] | 1 |
| **26+** | 18 | 4–14 | **0.22** | [0.03, 0.41] | **13** |

Collapsed: **≤15 turns 53W–29L (0.65, n=82) vs ≥16 turns 24W–36L (0.40, n=60)**,
z ≈ 2.9, p ≈ 0.003. This is the best-powered finding in the whole read.

**4. Prize evidence agrees.** In M31's 9 deck-out losses we were level or behind
on prizes in **8 of 9**, and **3 of 9 were hard stalls with all 6 prizes still
unclaimed** after 24–33 turns (vs team_rocket, cynthia_gabite, crustle/tusk).
M30 had 0 hard stalls, M29 had 2. We are not losing a card-economy race while
ahead on board — we are failing to break through at all and then running out.

**Conclusion: the low-deck economy rules (deckguard/conserve/ash) fire at
`deckCount ≤ 6`, roughly ten turns after the divergence that decides the game.
They cannot fix deck-out, and the fixed-turn curve says the intuition behind them
is backwards — the losing profile is under-digging early, not over-drawing late.**

## Matchup structure — and an instrument problem in the offline battery

Pooled M29+M30+M31 (n=142), archetype from the opponent's deck step:

| archetype | n | meta share | W–L | WR | 95% CI | deck-outs |
|---|---|---|---|---|---|---|
| **lucario** | 32 | 22.5% | 23–9 | **0.72** | [0.56, 0.87] | 1 |
| **alakazam (mirror)** | 30 | 21.1% | 14–16 | **0.47** | [0.29, 0.65] | 5 |
| **archaludon** | 17 | 12.0% | 8–9 | **0.47** | [0.23, 0.71] | 0 |
| crustle/tusk | 13 | 9.2% | 7–6 | 0.54 | [0.27, 0.81] | 5 |
| **team_rocket** | 10 | 7.0% | 1–9 | **0.10** | [0.00, 0.29] | 6 |
| marnie/froslass (grim) | 9 | 6.3% | 3–6 | 0.33 | — | 0 |
| **dragapult** | 9 | 6.3% | 7–2 | **0.78** | [0.51, 1.00] | 0 |
| hop | 6 | 4.2% | 2–4 | 0.33 | — | 3 |

Deck-outs concentrate in the stall/wall/mill half: rocket 6, mirror 5,
crustle 5, hop 3 = 19 of 21.

**The instrument problem.** Cross-referencing these against the beds that gate
our ships:

- `decks/lucario.csv` — the battery's primary bed — is a **Mega Lucario ex** list
  (Hariyama / Lunatone / Makuhita / Mega Lucario ex / Riolu / Solrock). Live WR
  **0.72**: our second-best matchup.
- `rule:dragapult` — M32-B's **sealed** primary transfer gate — is live WR
  **0.78**: our **best** matchup, with ~0.22 of headroom.

So the two instruments that decide what we ship measure the two archetypes we
already beat, together **28.8%** of the meta. Meanwhile **46.4%** of the meta —
mirror 21.1% @ 0.47, archaludon 12.0% @ 0.47, rocket 7.0% @ 0.10, grim 6.3% @
0.33 — sits at or below coin-flip, and of those only rocket and grim appear
anywhere in the battery. **`archaludon` is 12% of the meta at 0.47 and appears in
no bed and no training pool at all.**

This is a concrete, mechanical explanation for `no-validated-live-predictor`: the
offline battery is largely blind to the half of the meta where our losses live.

Secondary deck-tech note: of 30 live Alakazam opponents, only **8 run our exact
list**; the most common variant (×7) is +Nighttime Mine −Enhanced Hammer, and
several cut Xerosic's Machinations / Shaymin for extra Dudunsparce, Night
Stretcher and Rare Candy — i.e. the field's Alakazam builds trend toward *more*
draw and recovery. Recorded as an observation only; it does not by itself
overturn M32-P0's "our list is the highest-scoring variant" finding.

## What this means for M32

M32-B (PPO vs the decks we lose to) was **cancelled at iter 0 — no iteration
completed and nothing was observed**, so re-specifying its instrument now is
legitimate pre-registration, not post-hoc gate shopping.

1. **Keep the training pool as designed** (rocket 0.35 + grim 0.35 + mirror
   0.30). It targets 0.10 and 0.33 matchups — it is well chosen.
2. **Replace the sealed transfer gate.** `rule:dragapult` at 0.78 live cannot
   resolve an improvement. The honest out-of-loop gauge is **archaludon** — 12%
   of the meta, 0.47 WR, genuinely absent from every pool and bed. Needs a clone
   built from `opp_decks.parquet` (same path as the rocket/grim clones).
3. **Add mirror and archaludon to the reporting battery** regardless of gate
   choice, and stop treating `rule:lucario` as the headline number.
4. **The O-rule family looks spent.** Four milestones of low-deck economy rules
   have not moved deck-out, and the fixed-turn curve says they target a symptom
   that appears ~10 turns after the game is decided. If deck-out is to be
   attacked directly, the lever is **setup speed at turns 4–10**, not
   conservation at `deckCount ≤ 6`.

## Reusable artifact

`scripts/live_deck_race.py <sub_id> [<sub_id> ...]` — live deck-race forensics:
per-outcome turn/burn profile, deck-out losses with prize state and hard-stall
flagging, W/L + deck-out counts by opponent archetype, and drain attribution.
Run it on every future ship alongside `scripts/live_postmortem.py`.

_Caveat on the drain-attribution table: it charges the deck delta between two
consecutive MAIN prompts to the action chosen at the earlier one, so
turn-ending actions (ATTACK/END/EVOLVE) absorb the following turn-start draw and
any opponent-phase mill. Use it for the draw-card rows (Dawn, Poké Pad,
Dudunsparce, Fezandipiti, Poffin), not for the turn-enders._
