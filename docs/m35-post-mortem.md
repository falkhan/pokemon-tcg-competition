# M35 post-mortem — sub 55011605 (m28_winners + gacf rules + alakazam_v2)

Forensic, 2026-07-27. Live cache refreshed to n=55 (`kaggle_ingest refresh
--subs 55011605`). Tools: `scripts/live_postmortem.py 55011605`, implied-ELO solve,
targeted replay probes (brick hands, Boss-at-opp-pz≤1).

## Verdict: BEST SUB OF THE CAMPAIGN (tied) — implied ELO 817, back to M30's level.

| sub | milestone | n | W-L | WR | avg_opp | **implied ELO** |
|---|---|---|---|---|---|---|
| 54929991 | M30 (gac rules, base deck) | 44 | 25-19 | 0.568 | 764 | **817** |
| 54935640 | M31 (gacb rules, base deck) | 54 | 28-26 | 0.519 | 697 | **714** |
| 54997669 | M33 (gacb rules, alakazam_v2) | 42 | 25-17 | 0.595 | 671 | **743** |
| 55011605 | **M35 (gacf rules, alakazam_v2)** | 55 | 32-23 | 0.582 | **759.5** | **817** |

Clean A/B vs M33 (only gacb→gacf changed): **+74 implied ELO**. The field M35
faced was ~89 points stronger than M33's (759.5 vs 671) at the same raw WR —
the opponent-adjusted read is the real one. Live updated_score currently 797.2
and climbing (M33 settled 736.7, M30 759.4) — same directional story, though
raw score is time-confounded ([[no-validated-live-predictor]]).

## Benchfloor TRANSFERRED — the offline→live prediction landed almost exactly.

- **basic-play-at-bench≤1: only 2 declined prompts in 55 games** (M33-era pilot
  declined ~60%; offline predicted 98% compliance ✓).
- **bench-0-after-t3 loss share: 3/23 = 13%** (offline predicted 27%→15% ✓).
- **Zero-prize bench-0 bricks: 2/23 = 8.7% of losses** (M33: 6/17 = 35%).
- **The 2 residual bricks are DRAW VARIANCE, not pilot error**: probed both
  (ep88325608 t2, ep88336649 t3) — hand contained **zero basics at every
  prompt**. Nothing to bench; no rule or pilot change can act. The brick lever
  is now genuinely SPENT (M34 said deck-tech spent; now pilot confirmed too).

## Loss anatomy (all 23 losses, prizes-remaining me/opp):

| mode | count | share | root cause |
|---|---|---|---|
| **Deck-out** | **7** | **30%** | Stall walls: 4× crustle/tusk/terrakion (t30–39, we still needed 3–5 prizes), 1× hop's trevenant stall, 1× garchomp t25, 1× mirror. Not "one turn short" — prize-close inability vs high-HP walls; deckguard/conserve delays the clock but doesn't win the race. |
| **Zero-prize blowout (me6)** | 7 | 30% | 2 = unfixable no-basic bricks (above); 4 = aggro sweeps where opp raced to ≤1 prize (2× grim, 2× archaludon); 1 = dragapult (opp4). |
| **Close race lost (me1–5, opp≤2)** | 9 | 39% | Competed to the wire, opp closed first. 3× mirror, 2× dragapult, 2× archaludon, 1× grim, 1× starmie. |

**14 of 23 losses the opponent reached opp≤1** — the prize-race remains the era's
failure mode (M33: 10/17). Deck-out share rebounded 12%→30% but that is meta
shift, not regression: the crustle/tusk/terrakion wall list appeared 4× (4
different usernames, byte-similar lists — likely one agent family farming the
ladder) vs ~1 game in M33's sample.

## NEW: the gust-snipe no-veto case FIRED — 5/5 in losses.

The M35 watch item ("add opp-prize≤1 veto if seen"). Probed all 48 Boss's Orders
plays: **5 were made with opponent at 1 prize left — all 5 games LOST**
(ep88337695, 88339291, 88340879, 88345210, 88346273). Confounded (opp at 1 prize
= usually already losing), but the pattern is exactly the predicted hazard:
burning the turn's supporter on a gust that doesn't convert while the opponent
is at match point. M36 candidate: opp-prize≤1 gust veto **unless the gust target
is KO-able this turn** — deterministic card-fact, allowed override class.

## Matchup table (live, all 55 classified):

| matchup | W-L | note |
|---|---|---|
| mega lucario | **8-0** | solved |
| alakazam MIRROR | 8-4 | meta now copies our deck — 12 games (M33: 2) |
| archaludon | 6-5 | highest-volume; even |
| ursaluna / latias / other | 4-0 | |
| starmie/froslass | 2-1 | M33's 0-1 scare didn't materialize |
| garchomp | 1-1 | loss was a t25 deck-out |
| grim/marnie | **2-3** | first live grim WINS of the campaign (M33 0-3) — but still weakness-driven losses (2 shutouts) |
| dragapult | **1-3** | regressed from 3-3 |
| crustle/tusk/terrakion stall | **0-4** | ALL deck-outs — the new farm |
| kanga/hop stalls | 0-2 | same wall family |

Weak axes: **walls (0-6 vs the stall family)** and **dragapult (1-3)**. Grim
improved but is structurally weak (Darkness→Psychic 2×).

## Cards: under/overplayed (prompt-level offer→play)

**Underplayed (chronic, unchanged all campaign):**
- **Hilda 4.2%** (711 offers / 30 plays) — near-dead slot (M33 5.4%). Deck-tech-out candidate.
- **Boss's Orders 9.2%** (520/48) — the chronic gust hole (M33 9.4%); AND the 5 misfires above.
- **Poké Pad 9.8%** (984/96) — most-offered, least-played item; 30 of 31 END-with-playable-item turns were Poké Pad declines.
- Supporters overall 9.8%; only **59.1% of turns play a supporter**.

**Highest-rate (working as designed):**
- Enhanced Hammer 40.0%, Lana's Aid 31.5%, Night Stretcher 17.9%, Rare Candy 17.7% — the M33 deck-tech cards still earn their slots.

**O-rule engagement:** dud 0/41, fez 0/37 at deck≤6 (deckguard ✓); Sacred Ash
mostly played at deck≤15 (ash rule ✓); poffin declined at bench≤1 44× (expected —
poffinfloor removed in gacf, bricks fell anyway → M35's drop-poffinfloor call
vindicated). **Watch: telepath attach 70.7%** (164/232) vs 85–86% in M31/M33 —
possibly per-prompt dilution from benchfloor reordering, worth one probe if it
persists.

## Where the next lever is

1. **Anti-wall / prize-close vs stall (0-6, 7 deck-outs)** — biggest single loss
   mass (30%). We survive to t30+ but can't take 6 prizes off 180HP walls.
   Needs a stall-archetype offline bed (the crustle/tusk/terrakion list is
   extractable from these replays) before any fix is measurable.
2. **Gust veto at opp-prize≤1** — cheap, deterministic, evidence above; ships as
   a rule-letter A/B like M31/M35.
3. **Dragapult (1-3) regression** — n small; watch, don't build yet.
4. **Brick residual is noise-floor** — 2 no-basic hands in 55 games. STOP
   investing here.
5. Mirror is now 22% of the meta — mirror-specific edge (who decks out first —
   sereinless beat us at it) may be worth a probe.
