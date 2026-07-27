# M30 post-mortem — sub 54929991 (gac), first 26 live games

_Written 2026-07-23 evening from the first 26 cached episodes (refresh
2026-07-23 ~17:00). Ship config: `m28_winners` + O1 telepath + O4 deckguard +
O5 ash + O6 conserve on `clone54618168`. Analysis: `rl.kaggle_ingest
forensics`, `scripts/sub_behavior.py`, `scripts/loss_forensics.py`, plus a new
consolidated probe promoted to `scripts/live_postmortem.py` (loss-reason
classification from final board state, bench economy, turn-level trainer
rates, O-rule engagement)._

_Sample-size caveat up front: 26 games, 10 losses (and episode 87671454 is
the standard self-validation game — the sub plays itself, so the ladder
record is 15W–10L). Everything below is a directional read, not a resolved
one (`measurement-power-discipline`)._

## Headline: highest-reaching sub ever, and the losses are honest

Score trajectory (after each game, episode order):

```
600W 670W 773W 704L 737W 810W 764L 816W 845W 822L 788L 762L 737L
767W 746L 763W 743L 762W 780W 794W 779L 792W 796W 785L 809W 800W
```

Peak **845** at game 9 — above 54903635's 735@44 and any prior sub — sitting
~800 after 26. The loss profile is the healthiest we have shipped: average
opponent score in losses **826** vs **698** in wins, and only one loss
(87676917, opp 809 vs our 822) was even mildly an upset. M29's signature
(1W–4L open, losses to 339-rated opponents) is gone; this sub loses at the
frontier, not to the pool floor.

## Loss anatomy: 6 prizes, 2 BENCH-OUTS, 2 deck-outs

Classified from the final observed board (kaggle replays carry no RESULT
log; `deck_drain.py`'s reason codes are offline-only):

| mode | eps | note |
|---|---|---|
| prizes | 87673172, 87674780, 87676917, 87679086, 87682536, 87684257 | opponent at 1–2 prizes left at our last decision |
| **bench-out** | **87676383 (t6), 87677432 (t5)** | active KO'd with empty bench, 8–9 cards in hand |
| deck-out | 87678004 (t41), 87680159 (t27) | **both vs Team Rocket** |

### The bench-outs (Piotr's observation, confirmed — but the mechanism is subtler)

Both are **basics-drought games, not refuse-to-bench games**. Turn-by-turn:

- **87677432** (vs Archaludon, 822): opening hand had ONE basic (Dunsparce).
  T3 we benched Fezandipiti ex — our only other basic all game — and the
  opponent immediately gusted it active and KO'd it for 2 prizes. That left
  a 20 HP Dunsparce alone; hand was all evolutions (Alakazam ×2, Kadabra)
  plus supporters. Dead by t5.
- **87676383** (vs Marnie/Froslass, 953): bench was empty from turn 0 to the
  end. We never drew a single second basic; Hilda (t6) and Dawn didn't find
  one; Poké Pad was offered on 3 consecutive prompts at bench 0 and declined
  each time. Dudunsparce (140 HP) KO'd at t6, game over with 9 in hand.

Across all 26 games: **4/10 losses spent a mid-game turn (t>3) at bench 0;
zero wins did.** Bench ≤1 on 13.2% of all MAIN prompts. Crucially, when
bench was literally 0 and a basic WAS in hand, the model benched it every
time (0 declines) — the defect is upstream: it does not *dig* for board when
thin. Buddy-Buddy Poffin (benches basics from deck) is played on only
**31% of turns it is playable**, including 23 prompt-declines at bench ≤1.

Open question flagged for the next milestone's P0: our Dunsparce (id 305)
has an ACTIVE-area ABILITY that was repeatedly offered at bench 0 in
87676383 and never taken — and **nobody in the field used it once across
the 120 most recent cached episodes**, so its semantics are unmeasured
(custom card pool; the engine card DB is inside libcg). If it is a
bench-filler (call-for-family style), declining it lost us that game.

### The deck-outs: rocket family still unsolved, but closer

Live rocket record **0–2** despite the offline gac read being
resolved-better (0.5675, z=+2.11). Both losses are instructive:

- 87680159: 27 turns, we decked at 0 with the opponent at **1 card** — the
  exact mutual race `conserve` targets, lost by one draw.
- 87678004: 41 turns, decked while holding **25 cards in hand** — the M29
  hoarding signature, still alive in long grinds.

Deck-out share of losses: **2/10 = 20%** (M29: 30%, M26: 19%).

## O-rule engagement live: all four rules verifiably working

| instrument | this sub | comparison |
|---|---|---|
| Dudunsparce draw taken at deck ≤6 | **0/44 offered** | baseline was 86/88 |
| Fezandipiti draw taken at deck ≤6 | **0/22 offered** | 17/60 games pre-O6 |
| Sacred Ash deck-at-play | 6,8,8,9,10,10,10,12,14,23,25 | was mostly 0–2 |
| END with playable item | **7.6%** (5/66) | M29 26.2%, M26 12.0% |
| telepath attach | **71.8%** (79/110) | M26 84.9%, M29 67.7% |
| avg turns | 14.7 | M26 13.6, M29 16.9 |

The one watch-list item that reproduced: **telepath attach is ~72%, still
~13pp below 54903635** — the offline −5pp point-read was real, and it comes
from the weights (m28_winners), not the rules. Attach discipline is a
candidate axis for the next weights decision.

## Trainer-card usage: the user-visible "low trainer usage" quantified

Turn-level rates (a turn counts as an offer if the card was playable at any
MAIN prompt that turn — supporters are once-per-turn so this is the honest
denominator):

| card | kind | turns offered | played | rate |
|---|---|---|---|---|
| **Hilda** | supporter | 127 | 18 | **14.2%** |
| Dawn | supporter | 115 | 34 | 29.6% |
| Poké Pad | item | 102 | 36 | 35.3% |
| Boss's Orders | supporter | 98 | 23 | 23.5% |
| Xerosic's Machinations | supporter | 95 | 26 | 27.4% |
| **Buddy-Buddy Poffin** | item | 93 | 29 | **31.2%** |
| Nighttime Mine | stadium | 42 | 21 | 50.0% |
| Rare Candy | item | 42 | 23 | 54.8% |
| Sacred Ash | item | 27 | 11 | 40.7% |
| Enhanced Hammer | item | 27 | 20 | 74.1% |
| Night Stretcher | item | 26 | 12 | 46.2% |
| Lana's Aid | supporter | 20 | 15 | 75.0% |

A supporter is played on **70.7%** of turns where at least one is available
(116/164) — i.e. roughly every third supporter-turn is wasted, and it is
overwhelmingly **Hilda** (the draw supporter) being hoarded. End-of-game
hands confirm it: losses end with 12.7 cards in hand on average, wins with
**18.4**; trainers stranded in lost games: Enhanced Hammer 21, **Hilda 18**,
Rare Candy 10, Xerosic 8, Boss's Orders 7. This is the M27 supporter
under-play defect (model 3.8% vs teacher 7.5% per-offer) persisting through
m28_winners — the O-rules never touched supporters.

## Matchup ledger (26 games)

| opponent family | W–L | notes |
|---|---|---|
| Alakazam mirror | 2–3 | (+1 self-validation W); mirror parity claim offline was 0.675 |
| Archaludon/Cinderace | 3–1 | loss was bench-out 87677432 |
| Mega Lucario (both variants) | 3–1 | |
| **Marnie/Froslass (grim family)** | **0–3** | one bench-out + two prize losses; offline advisory was 0.669 |
| **Team Rocket** | **0–2** | both deck-outs; offline read 0.5675 resolved-better |
| Kangaskhan/Crustle | 2–0 | was 0–2 in M29 |
| Dragapult | 1–0 | |
| others (Kyogre, Starmie, Arboliva, Hop's) | 4–0 | |

The two 0-fer families are exactly the offline advisory axes; at n=2–3 per
cell nothing is resolved, but grim 0–3 with a bench-out is the axis to watch
as episodes accumulate.

## What this feeds into M31

1. **Bench economy is the new named defect.** Cheapest lever candidates, in
   the P3 "one precise intervention" spirit: (a) an O7 bench-floor rule —
   promote Poffin/Poké Pad/basic plays when bench ≤ some floor early-game
   (NOT a blanket tempo-force; that died in P3) — and (b) pin the Dunsparce
   (305) ability semantics first; it may already be the in-kit answer.
   Measure the offline frequency of "bench 0 + dig available" states before
   building anything.
2. **Hilda hoarding** (14.2%) is the sharpest single number behind both the
   trainer-usage complaint and the hoarded-hand loss signature; a supporter
   under-play fix has been deferred since M27 and keeps showing up.
3. **Rocket/grim live cells** — do nothing yet; let the A/B accumulate to
   n≥8–10 per family before reacting (floor-gate-sample-size).
4. **Telepath attach ~72%** — carry as a weights-selection axis, not a rule.
5. Re-run this exact probe (`scripts/live_postmortem.py 54929991`) once
   ~60+ episodes are cached; all reads above are n=26.
