# M36 post-mortem — sub 55030954 (m28_winners + gacf + alakazam_v2_h4)

Forensic, 2026-07-28. Live cache refreshed (`kaggle_ingest refresh --subs
55030954`): 53 episodes, 1 self-validation game (vs Team Pierogachu) excluded.
Tools: `scripts/live_postmortem.py`, `scripts/pm_extra.py` (implied ELO,
matchup ledger, trajectory), `scripts/pm_probe2.py` (brick anatomy, Boss@opp≤1,
deck verification) — all three now promoted into `scripts/` on main.

## Verdict: BUSTED PLACEMENT — implied ELO 609 (0.500 vs a 608-avg field). The agent did not regress; it got seeded into, and cannot escape, a stall/grim-saturated 600 band.

| sub | milestone | n | W-L | WR | avg_opp | implied ELO |
|---|---|---|---|---|---|---|
| 54929991 | M30 (gac, base deck) | 44 | 25-19 | 0.568 | 764 | **817** |
| 55011605 | M35 (gacf, alakazam_v2) | 55 | 32-23 | 0.582 | 759.5 | **817** |
| 55030954 | **M36 (gacf, alakazam_v2_h4)** | 52 | 26-26 | **0.500** | **608.2** | **609** |

The "clean live A/B vs 789.6" framing in the M36 plan was wrong in one
structural way: **a new sub does not inherit its predecessor's band.** It seeds
at ~600 with placement-K swings of ±100 for the first handful of games. M36's
first 6 ladder games drew 4 stall decks (crustle ×2, hop's trevenant ×2) and
lost all 4 by deck-out → score sank to 481 by game 6. It then ground back to a
~600–650 oscillation (peak 649) and settled at 0.500 — because the 600 band is
where the farm decks live, and the farm decks are exactly our worst matchups.

**Band composition is the whole story:** stall walls + grind + grim/marnie =
21/52 = **40% of M36's opponents** vs ~24% in M35's 759-band sample. Against
the shared strong-meta families the agent is unchanged (mirror 8-4, lucario
5-3, starmie 4-0). The deck/pilot beats the decks that live at 750+; it loses
to the cheap decks that live at 600. The ladder is not transitive, and
climbing out requires beating the 600-band mix specifically.

Sub-to-sub A/B on the hammer-4 tech is therefore **confounded by band** —
only the within-family reads below are usable.

## Loss anatomy (26 ladder losses, prizes-remaining me/opp)

| mode | count | share | detail |
|---|---|---|---|
| **Deck-out** | **9** | **35%** | crustle/tusk wall ×3, hop's trevenant ×2, Cynthia's garchomp ×2, rocket ×1, lucario-solrock ×1. Same mechanism as M35: we survive to t17–38 but cannot take 6 prizes off heal/tank lines. One agonizer: ep88562454 (garchomp) decked out at **prizes 1/1** — one prize from winning. |
| **Bench-out brick** | 4 | 15% | ep88459419, 88466529, 88496613, 88590857 — all probed prompt-by-prompt: **zero basics in hand at essentially every prompt** (t1–t8). Benchfloor engaged (0 declined benches at bench≤1 all sub); these are draw variance off 9 basics, not pilot error. |
| **Zero/low-prize sweep** | 4 | 15% | lucario t7, mirror t15 (hand=2), grim t8 (hand=1), dragapult/chi-yu — opp raced to ≤2 while we took ≤0–1. Two feature hand-starvation. |
| **Contested race lost** | 9 | 35% | me≤5/opp≤3 at end: grim ×3, rocket ×2, mirror ×1, archaludon ×1, hop-snorlax ×1, mixed ×1. Competed to the wire, opp closed first — the era's chronic mode (M35: 39%). |

## Matchup ledger (52 ladder games)

| family | W-L | note |
|---|---|---|
| mirror | 8-4 | 23% of the meta; h4's "softest cell" seed-split fear did NOT materialize |
| grim/marnie | **3-5** | biggest single-family loss count; Darkness→Psychic 2× weakness; 8 games = double M35's share |
| lucario | 5-3 | was 8-0 in M35 — but 2 of the 3 losses are no-basic bricks, 1 a t7 sweep; not a pilot change |
| crustle/kanga wall | **2-3** | was **0-4** in M35 — the h4 target claim (see below) |
| hop stall (trevenant/snorlax) | 0-2 | deck-outs; hammer does not touch this line |
| garchomp | 0-2 | both deck-outs (heal-tank) |
| rocket | 0-3 | 1 deck-out + 2 races lost; offline bed says ~0.56 — n=3, watch |
| lucario-solrock | 0-1 | deck-out |
| starmie | 4-0 | solved |
| dragapult | 1-1 | M35 watch item: was 1-3; no clone build needed yet |
| archaludon | 0-1 | low volume in this band |
| other | 3-1 | |

**New win line confirmed:** 4 wins ended with the **opponent at deck 0** (2×
crustle, 2× mirror) — we won the deck-out race outright. The
deckguard/conserve economy machinery now wins races, not just delays losses.

## M36 watch items closed (pre-registered in M36-plan)

1. **Wall-family W/L (the target claim, offline 0.36→0.42):** crustle 2-3 live
   = 0.40 — **the +6pp transferred almost exactly** on the special-energy wall
   family it was built for. But the broader stall family (hop, garchomp,
   solrock) went 0-5: Enhanced Hammer is a special-energy answer and those
   lines don't run special energy. h4 fixed the crustle cell, not "walls".
2. **Bench-out wins appearing:** yes — the 2 crustle wins are opp-deck-out
   wins (new since M35, where crustle was 0-4 all deck-out losses).
3. **Mirror share/W-L:** 12 games, 8-4 — holds; softest-cell fear cleared.
4. **Dragapult:** 1-1, regression halted; clone decision stays parked.
5. **Supporter-turn rate:** 59.4% of turns (M35 59.1%, flat). Hilda per-prompt
   4.2%→6.5% — dead-copy removal moved the rate the predicted direction.

## Recent-change validation ledger

| change | verdict | evidence |
|---|---|---|
| **h4: +1 Enhanced Hammer (M36)** | ✅ validated, narrow | crustle 0-4→2-3; EH top play-rate 34.3% (61 plays); offline +6pp reproduced live |
| **h4: −1 Hilda (M36)** | ✅ validated harmless | Hilda still near-dead at 6.5% per-prompt; supporter-turn rate flat |
| **benchfloor / gacf (M35)** | ✅ validated | 0 declined benches at bench≤1 in 52 games; all 4 bench-out losses had no basic in hand (unactionable) |
| **deckguard/conserve (M30)** | ✅ validated, now offensive | dud 0/71, fez 0/48 at deck≤6; 4 wins by winning the deck-out race |
| **ash rule (M30)** | ✅ holds | Sacred Ash mostly played at deck≤15 |
| **alakazam_v2 base (M33)** | ⚠️ band-exposed | Night Stretcher 24.2%, Rare Candy 24.2% still earn slots — but 9 basics → 4 brick losses (7.7% of games), and the deck has no answer to non-special-energy walls or grim weakness. The deck is tuned for the 750 meta it no longer faces. |
| **gacf gust (no veto shipped)** | ⚠️ residual | 2 Boss plays at opp-prize≤1, both in losses (M35: 5/5) — smaller but alive; **gacfv (O11 gustveto) is measured + ship-ready and stays the cheapest rules lever** |
| **telepath attach** | ✅ resolved non-issue | per-prompt 73.3% but the M36-plan probe pinned per-TURN ~87% as the correct metric |

**No shipped change shows a live regression signature.** The 817→609 delta is
seeding + band composition, not the hammer tech. Corollary: **leaderboard
selection should point at 55011605/54929991** (both settled ~790–817) — do not
burn a resubmit on reshipping, per Piotr, until we have a config that beats the
600-band mix.

## Chronic under-plays (unchanged all campaign — the BC ceiling)

- **Poké Pad 8.9%** (967 offers / 86 plays); 25 of 27 END-with-playable-item
  turns were Poké Pad declines.
- **Boss's Orders 9.5%** — the gust hole; **Dawn 10.0%, Xerosic 11.8%**.
- Supporters 10.4% per-prompt; only 59.4% of turns play a supporter.

## Where the next lever is (initial M37 recommendations)

Ranked by loss mass × tractability:

1. **Anti-stall prize-close / race-mode (50% of losses: 9 deck-outs + several
   contested races).** The hammer unlock worked but only covers crustle.
   Concrete moves: (a) extract hop's-trevenant/snorlax and Cynthia's-garchomp
   stall lists from these 53 replays → 2 new offline beds (same recipe as
   `decks/greattusk_wall.csv`); (b) the mirror race-conserve rule designed and
   PARKED in M36 generalizes: **archetype-detected race mode** — when the
   opponent's board says stall (Trevenant/Snorlax/Crustle/heal-tanks), switch
   the objective from tempo to card-economy racing (we already proved we can
   win deck-out races); (c) ship gacfv (gustveto) as the cheap rules rider.
2. **Meta-weighted ship bar (fixes the band trap).** Freeze a `meta_v3`
   snapshot weighted by the **600-band composition** (40% stall/grim), not the
   750 meta. A new sub must fight through the 600 band first — the composite
   WR against band-weighted beds becomes the first defensible live predictor
   ([[no-validated-live-predictor]]) and the M37 ship gate.
3. **Grim/marnie (5 losses, 2× weakness):** band-inflated to 15% of games.
   A deck slot that breaks the Darkness matchup (tech card or line change) is
   worth one Tier-0/Tier-1 pass with the M33/M34 deck-tech harness.
4. **RL practices worth importing (beyond the current BC + O-rules stack):**
   - **Advantage-weighted BC / offline RL (AWR-style):** winners-only BC (M28's
     one resolved positive) throws away the loss signal. Weight replay
     decisions by outcome-derived advantage instead of hard-filtering — same
     pipeline, softer filter, learns "what losers did wrong". Low-risk arm on
     the existing corpus (5,294 live episodes cached).
   - **Matchup-specialist mixture (league/exploiter pattern, AlphaStar):** the
     no-transfer law (3× reproduced) applies to fine-tuning the *generalist*.
     Train a **wall-specialist policy on wall beds only**, route to it at
     runtime on archetype detection, keep m28_winners frozen for everything
     else. The generalist is untouched, so the law is not violated.
   - **Expert iteration on the race sub-problem:** the deck-out race is a
     counting problem — `turn_solver` territory. Generate solver-guided
     race-mode trajectories on the wall beds and distill into the pilot as BC
     data (sidesteps the M24 self-play DAgger wall and the PPO transfer law).
   - **Auxiliary race-value head:** a small head predicting the deck-out-race
     winner (cards-drawable vs prizes-needed) as an input to conserve/race
     rules — replaces hand-tuned deck≥N thresholds with a learned signal.
   - **Latent feature bug arm:** the M36 E4 finding (`.prize` array semantics,
     `_make_plan` comment inverted) is still queued — a cheap training arm to
     check whether fixing the feature moves any bed.
5. **Stop-investing list:** brick residual (4 losses are pure draw variance,
   benchfloor at ceiling; M34 showed deck-tech neutral), telepath alarm
   (resolved), dragapult clone (1-1, parked).
