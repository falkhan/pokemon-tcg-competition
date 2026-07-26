# M33 post-mortem — sub 54997669 (m28_winners + alakazam_v2 deck-tech)

Rapid forensic, 2026-07-26. Live cache refreshed to n=42 (`kaggle_ingest refresh
--subs 54997669`). Tools: `scripts/live_postmortem.py 54997669`, implied-ELO solve.

## Verdict: a modest, REAL gain over M31 — but still well below M30. Not the win we wanted.

| sub | milestone | n | W-L | WR | avg_opp | **implied ELO** |
|---|---|---|---|---|---|---|
| 54929991 | M30 (gac rules, base deck) | 44 | 25-19 | 0.568 | 764 | **817** |
| 54935640 | M31 (gacb rules, base deck) | 54 | 28-26 | 0.519 | 697 | **714** |
| 54997669 | **M33 (gacb rules, alakazam_v2)** | 42 | 25-17 | 0.595 | 671 | **743** |

Implied ELO already opponent-adjusts, so the field being weak (671) is priced in.
M33 recovered **~29 ELO** of the ~103 M30→M31 lost — directionally up, n=42 so not
resolved — but sits **~74 ELO below M30**, the highest-reaching sub. Config note:
M33 = M31's rule set (`telepath,deckguard,ash,conserve,poffinfloor`) + the v2 deck.
Deck swaps confirmed in the bundle: +1 Rare Candy (3→4), −1 Enhanced Hammer (4→3),
+1 Night Stretcher (1→2), −1 Nighttime Mine (2→1).

## The deck-tech did exactly what it was aimed at — and the loss mass MOVED.

The setup-speed thesis (M31 finding: WR 0.65 ≤15t vs 0.40 ≥16t, p≈0.003) was to
close faster and stop decking out. Live, it worked **on that axis**:

- **Deck-outs collapsed: M31 34.6% of losses → M33 12% (2/17).** Both remaining
  deck-outs are the classic walls (lucario grind t15, crustle/tusk t33).

But the mass relocated to the **front of the game**, which the deck-tech did NOT
address:

### Loss anatomy (all 17 losses, prizes-remaining me/opp):

| mode | count | share |
|---|---|---|
| **Zero-prize blowout (me6 — took 0 of 6 prizes)** | **6** | **35%** |
| Prize-race loss, competed (me3–5) | 7 | 41% |
| Close race (me1–2) | 2 | 12% |
| Deck-out | 2 | 12% |

- **10 of 17 losses the opponent got to opp≤1 prize** — we get run to the wire on
  the PRIZE RACE and lose it. The era's failure is no longer "lose slow" (deck-out)
  but "lose the race."
- **The 6 zero-prize blowouts are opening BRICKS vs aggro:** e.g. ep88244141 (starmie,
  t6, deck 42/42, hand 2, **bench 0**), ep88244722 (lucario, t10, deck 37, **bench 0**,
  hand 3), ep88246344 (grim, t12, **bench 0**). Deck ~37–42 that deep in = we drew/
  played almost nothing → dead opening hand, no Abra→Kadabra→Alakazam line, swept
  before we set up. The Alakazam evolution line is fragile and the deck-tech traded
  disruption (Enhanced Hammer, Nighttime Mine) for setup speed — good vs stall, worse
  vs aggro that punishes a brick.

### The structural bleed is UNCHANGED: grim/marnie **0-3 live**.
Darkness→Psychic 2× weakness (`tcg/constants.py WEAKNESS_MULTIPLIER=2`, memory
[[meta-gate-is-mirror-gate]] family). The offline grim-clone said **+7.7pp** for v2
— it did NOT transfer because the offline clone is type-NEUTRAL (Munkidori/Marnie)
and never models the live weakness. Classic [[no-validated-live-predictor]]: our beds
measure a different matchup than the live one. The v2 offline "grim gain" was largely
illusory; the real live gain came from the deck-out drop, not from grim.

### Matchup table (live, n≈35 classified):
lucario **10-4 (0.71)** · dragapult 3-3 (0.50) · **grim 0-3 (0.00)** · archaludon
2-1 · crustle family (kanga/tusk) 1-2 (walls/deck-outs) · **starmie 0-1** (new fast
aggro, t6 blowout) · mirror 2-0 · hop/kyogre/annihilape 1-0 each.

## Cards: what's played, what isn't

**The deck-tech cards are being used (the swaps landed):**
- Rare Candy 20.2% (183 off / 37 played) — skips Kadabra, high use ✓
- Night Stretcher 18.2% (253 / 46) — piece recovery, high use ✓
- Enhanced Hammer 28.6% (still used at 3 copies), Lana's Aid 35.1% (heal, high value)

**The standing under-play holes (unchanged all campaign):**
- **Boss's Orders 9.4%** (448 off / 42 played) — the chronic gust hole. In the 6
  zero-prize sweeps a timed Boss to snipe a benched attacker is exactly the missed
  swing. Measured-spent by imitation (M21: solver commits 0 gust plans) — not fixable
  by BC or a rule.
- **Supporters 9.5% overall** — the M23 S2 finding (winners play FEWER early
  supporters; the corpus carries ~0 supporter→win signal). Hilda 5.4% (near-dead; the
  killed O8 drawfloor card).
- **Telepath attach 85.1%** — fine, ≈ M31's 86.2% per-turn.
- **Missed bench: 9 prompts at bench==0 with a basic playable and declined** — the
  concrete setup-failure pilot defect that maps onto the zero-prize sweeps.
- deckguard/conserve verified firing (dud 0/42, fez 0/34 demoted at deck≤6 = designed).

## What changed vs the M31 premise

M31 said "the lever is setup speed at turns 4–10." The deck-tech pulled that lever
and it **paid** (deck-outs −2.5×, ELO +29). But it revealed the next wall: we now
lose the **prize race**, dominated by (a) opening bricks vs aggro and (b) the grim
weakness. Neither is a closing-speed problem; both are front-of-game / matchup problems.

## Options to improve (ranked by EV given campaign history)

1. **More consistency deck-tech to cut the 6 brick-blowouts (#1 loss mode, 35%).**
   Deck-tech is the ONLY lever showing live traction. The sweeps are opening bricks
   (bench 0-1, deck ~37-42 deep). Candidates: raise opening consistency (Poffin/Poké
   Pad ratio, a search line for Abra, revisit the Enhanced Hammer/Nighttime Mine cuts
   that removed our aggro-disruption). **Measurement risk:** our offline beds are
   stall clones (rocket/grim) — they do NOT reproduce "brick vs aggro," so we'd need
   an aggro bed (lucario/starmie clone) to gate this honestly.
2. **Cheap rules-revert probe: gac (M30) vs gacb (M33) on the v2 deck.** M31's added
   `poffinfloor` was neutral-to-negative live (M31 post-mortem: relocated Poffin
   plays, bought nothing). M30's gac reached 817. "M30 rules + v2 deck" is UNTESTED
   and plausibly beats the shipped gacb+v2. Low effort, low ceiling, offline-gateable.
3. **grim/marnie counter — but build a weakness-faithful bed FIRST.** 0-3 live, 6% of
   meta, the most reliable loss. Any tech (a non-Psychic attacker, or accept the
   matchup) is unvalidatable until we have a grim clone that carries the Darkness
   weakness. Medium effort, uncertain payoff.
4. **Rollback insurance: re-ship M30 54929991 (817).** M33 (743) is below M30. If the
   goal is max leaderboard now, M30 is highest-reaching — but its rules ran on the
   BASE deck, so option 2 (M30 rules + v2 deck) likely dominates a straight rollback.
5. **The bigger bet: solver-relabel human replay states (ei-mode on-policy DAgger,
   strength-gated).** Addresses the "fidelity ≠ strength" wall broadly — would help
   the prize race in general, not one matchup. Highest ceiling, highest effort,
   feasibility designed (Rec 4 note) but unproven. The standing complementary bet.

## Reusable / notes
- `scripts/live_postmortem.py <sub>` + implied-ELO solve = the standard read; run both
  every ship.
- Standing instrument gap: no aggro bed and no weakness-faithful grim bed — the two
  matchups now driving losses are both UNMEASURED offline. Building either is
  prerequisite to validating options 1 and 3. See [[no-validated-live-predictor]].
