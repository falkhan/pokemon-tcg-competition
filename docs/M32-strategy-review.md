# M32 strategy review — where the ceiling actually is (2026-07-25)

Written after M32-B killed (3rd PPO no-transfer). Piotr's call: step back and
review before spending more compute. This is an honest accounting, not a plan to
execute — the next experiment is Piotr's to pick from the options at the end.

## 1. The reframe: it is the PILOT in even matchups, not the deck, not the type wall

New analysis this session (read-only over the 851-game cache, card-type from
`data/cards_features.csv`; our alakazam is Psychic, weakness = Darkness, confirmed):

| bucket | share of games | our WR | share of all losses |
|---|---|---|---|
| **Darkness (2× vs us — hard type wall)** | **7.2%** | 0.279 | **9.5%** |
| non-weakness (everything else) | 92.8% | 0.466 | 90.5% |

- **The type wall is small.** Winning *every* Darkness game would move overall WR
  only ~0.45 → 0.51. rocket + grim (our 0.10 / 0.33 nightmares) are a real 2×
  disadvantage no pilot overcomes — but they are ~10% of the loss mass, not the
  campaign.
- **~90% of our losses are on decks with no type edge over us**, at WR 0.466 —
  a skill/closing deficit.
- **The smoking gun: we do not beat the mirror.** Psychic-type opponents (the
  mirror + near-mirrors), full history: 162 games @ **0.41**; the current pilot's
  tighter alakazam-mirror read is **0.47** (n=30, M31 post-mortem). Same deck,
  same everything — symmetric — and we sit at/below 0.50. We are an **at-or-below
  field-median alakazam pilot.** To climb the ladder we must be consistently
  >0.5 vs the field on the even matchups (mirror 0.47, archaludon 0.47, crustle
  0.54); we are ~coin-flip there. That gap is the whole ~710→1251 story.

**Implication for archetype-switching (option A, dragapult):** it escapes only
the ~10% type wall and inherits the *same* BC-pilot ceiling that leaves us
median-minus. It is not justified by the type data.

## 2. The measured-laws ledger — what is conclusively spent

Every one of these was resolved with pre-registered gates; do not re-propose:

- **Deck (within Alakazam):** meta-optimal — our list is byte-identical to the
  highest-scoring variant (M32 P0). Deck *choice* is not the lever.
- **BC pilot fidelity:** ceiling ~0.66 to the teacher (M24/25/26); more epochs,
  score/winner-weighting, corpus pooling all null-or-negative. Architecture
  exonerated (V3 = V2 fidelity, M23 E1).
- **PPO:** no out-of-loop transfer — **3× now** (M22 0.17–0.22 vs dragapult;
  M23 flat at 30-iter convergence; M32-B *degraded* lucario −8.8pp / archaludon
  −4.8pp while overfitting the one rocket matchup). It optimizes the training
  distribution and costs everything else.
- **Self-play signal:** the RL data carries ~zero/negative supporter→win
  correlation (M23 S2) — the loop cannot learn the behaviors that win because its
  own data does not encode them.
- **O-rules (deterministic overrides):** 5 milestones (M26→M31), no live gain;
  the closing-speed post-mortem showed they target a symptom ~10 turns after the
  game is decided.

## 3. Root-cause hypothesis for the median-minus pilot

Two threads, both consistent with the data:

1. **A 66%-fidelity clone has no edge over the field.** We reproduce one strong
   pilot's move ~2/3 of the time and substitute *something else* the other 1/3 —
   and that 1/3 can be incoherent, dragging a copy of a 1251 pilot down to ~791.
   BC cannot close the last third (measured ceiling), and nothing we have adds
   coherent skill on top (PPO degrades, rules are blunt).
2. **We lose LONG games and our value function is blind there.** Best-powered
   live finding (p≈0.003): WR 0.65 ≤15 turns vs 0.40 ≥16; we under-dig turns
   4–10 and fail to close. The setup value head is only trustworthy to ~turn 32
   (`rl/matchrunner.py` vsolver `VS_MAX_TURN=32`), i.e. useless exactly in the
   grind where we lose — so neither search nor the pilot can navigate the close.

## 4. Genuinely-unexplored levers, honest priors, cheap first tests

| lever | what it attacks | prior | cheap first probe |
|---|---|---|---|
| **Deck-tech for early-dig** (draw-heavier variant: +Dudunsparce/Night Stretcher/Rare Candy, −Xerosic/Shaymin — the field's own trend) | the closing/under-dig 90% AND slightly the type wall | **MED, cheapest** | build 1–2 variants, battery the *current* pilot on them vs the live-faithful beds; no retrain needed |
| **Better long-game value function** | the root cause of the closing failure (helps search + setup value + pilot) | MED, hard | re-measure value-head calibration past turn 20 on recent corpora; is it fixable with more grind data? |
| **Dragapult archetype (A)** | ~10% type wall + "fresh ground" | LOW (inherits BC ceiling; reframe says type wall is small) | feasibility probe: is there a strong *cloneable* dragapult sub, and what is its fidelity ceiling? |
| **Accept the ceiling** | — | honest baseline | none; hold M30-class config, stop spending |

## 5. Recommendation

The reframe kills the strongest case for dragapult (the type wall is only ~10%)
and confirms the bottleneck is pilot skill in even matchups, where **every tool
we have for adding skill is measured-capped.** That is a hard place to be, and it
should be said plainly: we may be at or near the achievable ceiling for a
BC-cloned pilot, and further RL spend has a low expected return.

Given that, the highest expected-value *cheap* next step is the **deck-tech
early-dig probe** — it targets the best-powered live weakness (under-digging), it
is the perennially-deferred "biggest untouched lever," and critically it does
**not** require beating the BC-fidelity ceiling or the PPO no-transfer wall: it
changes the *inputs* to the existing pilot rather than trying to make the pilot
smarter. It is a few hours of no-retrain battery work with a clean live A/B, and
it can fail fast.

Dragapult (A) stays available but lower-EV; the long-game value function is the
most *fundamental* fix but the least certain and most expensive. Piotr to choose.
