# M37 post-mortem — sub 55065484 (m28_winners + gacfr3 + alakazam_v2_h4)

Forensic, 2026-07-29. Live cache refreshed (`kaggle_ingest refresh --subs
55065484`): 64 episodes, 1 self-validation game excluded → 63 ladder games.
Tools: `scripts/live_postmortem.py`, `scripts/pm_extra.py`, plus two new
M37-specific probes promoted into `scripts/`: `m37_pm_probe.py` (racemode3
live engagement, adjudication anatomy, Boss@opp≤1, WR-by-band) and
`m37_pm_burn.py` (per-source deck-burn attribution in wall games).

## Verdict: band escape SUCCEEDED — implied ELO 736 (vs M36's 609). racemode3 fired flawlessly live but was strategically insufficient: the wall family went 3-6 anyway, all losses deck-outs. New equilibrium ~740, set by the 700–800 band mix.

| sub | milestone | n | W-L | WR | avg_opp | implied ELO |
|---|---|---|---|---|---|---|
| 54929991 | M30 (gac, base deck) | 44 | 25-19 | 0.568 | 764 | 817 |
| 55011605 | M35 (gacf, alakazam_v2) | 55 | 32-23 | 0.582 | 759.5 | 817 |
| 55030954 | M36 (gacf, alakazam_v2_h4) | 52 | 26-26 | 0.500 | 608.2 | 609 |
| 55065484 | **M37 (gacfr3, alakazam_v2_h4)** | 63 | 33-30 | **0.524** | **715.8** | **736** |

Trajectory: placement opened 1-3 (dip to 492 by game 4 — same placement-K
exposure as M36), then a clean climb through the 600 band to a peak of 774
by game ~42. The tail is the story: **7W-14L after the peak** against
740–790 opposition. WR by opponent band: 600–699 **8-6**, 700–799 **18-19
(0.49)**, 800+ **1-3**. The agent now beats the 600 band (M37's objective —
achieved) and draws the 700 band; the 750–800 mix is the new wall.

## racemode3 live verdict (the shipped single variable)

**Mechanically perfect, strategically insufficient.**

- Engagement (m37_pm_probe): in all 9 wall-family games the trigger was true
  on essentially every MAIN prompt (e.g. 48/48, 50/50), and the agent used
  the demoted Dudunsparce/Fezandipiti draws **0 times across ~410 offers**.
  The rule did live exactly what it was measured to do offline. Inertness
  also confirmed: hop 0 games (band moved past hop), garchomp 1-1 (was
  0-2) — no contradiction with the pre-registered "unchanged" prediction.
- Outcome: **wall 3-6 (M36: 2-3)**. All six losses are deck-outs — including
  two the classifier called "other" because we ended at deck=1: the next
  mandatory draw kills us. One agonizer, ep88694896: **we led 4-0 on prizes
  at t35 and lost by deck-out** with the opponent holding 21 cards.
- Deck races lost by 3–21 cards (median ~12). By t9 we were already 15–25
  cards behind in every loss.
- Root cause (m37_pm_burn): dud/fez were never the main leak against
  disciplined pilots. Our burn ≈ **2.5–2.7 cards per own turn vs the
  opponent's ~1.4–1.5**. The remaining burn is the engine itself:
  Alakazam/Kadabra evolution triggers (avg 2–3 deck cards per evolve,
  attribution-window measure), Dawn 2.4/play, Rare Candy 3.0,
  **Enriching Energy 4.0 per attach** (5 attaches in wall games = 20 cards),
  Poké Pad 1.0 ×19 plays, Hilda 2.0 ×8. Sacred Ash was played at deck 0 and
  2 in two losses (recycle value at the last possible moment).
- **Bed-fidelity lesson (same as garchomp/grim, now for walls):** the
  offline `solver:wall` bed showed +15.7pp z+5.50 because the solver pilot
  over-digs its own deck; live 700–800 wall pilots conserve (~1.5/turn) and
  farm us regardless of prize state. The live wall opponents sat at
  697–836 ELO — **walls are not a 600-band artifact**; escaping the band
  did not escape the walls. Any future anti-wall measurement needs a wall
  BC clone (M30 rocket recipe; this milestone's garchomp-clone groove).

## Loss anatomy (30 ladder losses)

| mode | count | share | detail |
|---|---|---|---|
| **Deck-out** | **10** | **33%** | wall ×6 (incl. 2 next-draw), **mirror ×3**, rocket ×1. New: the mirror deck-outs — we burned 46 cards by t14 in ep88689207 (lost race 0v7), lost ep88722903 **by a single card** (0v1, prizes 2/3), ep88820444 0v8. The mirror is partly a deck race between two hyper-digging Alakazam engines, and we lose it. The M36 margin-gated raceconserve (designed for exactly this, PARKED) never shipped — racemode3's trigger has no mirror coverage. |
| **Swept** (we took ≤1–2 prizes) | 9 | 30% | dragapult ×3 (two ended at hand=2 — hand starvation), lucario ×2, grim, garchomp, kyogre, mirror. Fast aggro closes before our setup lands — policy-quality territory, not rules. |
| **Contested race lost** (me≤4/opp≤2) | 9 | 30% | lucario ×2, archaludon ×3, rocket ×2, mirror, grim. The era's chronic mode continues (M35 39%, M36 35%). |
| **Bench-out brick** | 2 | 7% | t2/t3, no basics — draw variance, benchfloor unactionable (0 declined benches at bench=0 all sub). |

## Matchup ledger (63 ladder games, M36 in parens)

| family | W-L | note |
|---|---|---|
| lucario | 8-5 (5-3) | biggest volume at this band; 2 sweeps + 2 close losses |
| crustle/kanga wall | **3-6** (2-3) | racemode3 target — see verdict above |
| mirror | **3-5** (8-4) | 3 of 5 losses are deck-race deck-outs; also QC prevship leg was 0-3. Two independent below-0.5 samples — watch, with a concrete candidate fix (margin raceconserve) |
| archaludon | 4-4 (0-1) | new volume at 750+; 3 close losses + 1 brick |
| grim/marnie | **6-2** (3-5) | self-resolved with ZERO code change — band/pilot shift. Deck-tech target deprioritized |
| dragapult | 1-3 (1-1) | all 3 losses sweeps, 2 with hand=2 endings |
| rocket | 1-3 (0-3) | 1 deck-out + 2 close; offline bed says 0.61 — persistent gap, low n |
| garchomp | 1-1 (0-2) | improved; racemode3 inert here as predicted |
| starmie | 2-0 (4-0) | still solved |
| kyogre/other | 3-1 | one sweep loss |
| lucario-solrock | 1-0 (0-1) | |

## Watch items closed (pre-registered at ship)

1. **Wall-family W/L (offline 0.408→0.565, the transfer claim): FAILED live**
   — 3-6, worse than M36's 2-3 (small n, but directionally opposite). The
   rule fired perfectly; the bed lied about the opponent. First transfer
   failure of the campaign's offline→live claims, and it's a *pilot-fidelity*
   failure, not a rule failure.
2. **Band escape: YES** — 609 → 736, the milestone objective delivered by
   the 600–699 record (8-6) and early climb.
3. **hop/garchomp unchanged (inertness):** confirmed — hop absent at this
   altitude, garchomp 1-1.
4. **Grim cell:** 6-2 without intervention. M38 grim deck-tech now low
   priority.
5. **gustveto:** residual shrank on its own — only 2 Boss plays at opp≤1
   (1W-1L; M35 was 5/5 in losses). Stays parked; not worth a submit slot.

## Recent-change validation ledger

| change | verdict | evidence |
|---|---|---|
| racemode3 (M37) | ✅ mechanical / ❌ strategic | 0 demoted-draw uses over ~410 offers; wall still 3-6, all deck-outs; leak is elsewhere (see burn audit) |
| h4 hammer (M36) | ⚠️ neutral here | EH play rate 38% (top card); walls at this band don't fold to it; mirror slipped 8-4→3-5 (EH is dead in mirror — dead-card hypothesis unproven, n small) |
| benchfloor (M35) | ✅ holds | 0 declined benches at bench=0; both bricks unactionable |
| deckguard/conserve (M30) | ✅ holds | dud 0/153, fez 0/23 at deck≤6 |
| chronic under-plays | unchanged | Poké Pad 9.7%, supporters 10.2%/prompt, 58.4% supporter-turns — the BC ceiling |

## Where the next lever is (M38 recommendations, ranked by loss mass × tractability)

1. **Wall BC clone bed FIRST (P0, blocking).** `solver:wall` is now
   strawman-invalid for economy fixes — it beat itself by over-digging.
   Clone from the live wall subs seen at 689–836 this sample (the
   garchomp-clone recipe from this milestone; check seat-row counts vs the
   197–689 working band). No anti-wall measurement without it.
2. **racemode4 — widen the wall-trigger demote to the real burn sources:**
   Enriching Energy attaches (−4/use), Poké Pad, surplus Hilda/Dawn once
   the board is built; plus wall-aware Sacred Ash timing (don't sit on
   recycle value until deck≤2). Estimated ceiling ~5–6 cards/game saved —
   flips the ≤9-card-margin races (≈3 of the 6 losses), not the blowouts.
   Measure on the clone bed (1), same pre-registered-bars discipline.
3. **Universal margin raceconserve (revive the parked M36 design) for the
   mirror deck race.** 3 mirror deck-out losses (one by a single card);
   burn hits 3–6 cards/turn in mirror digs. Compose: racemode3 blanket on
   walls + margin gate (behind >5, 6<deck≤25) on ANY opponent. Uniquely,
   this is measurable TODAY on a high-fidelity bed — the mirror bed is
   model-vs-model, no solver strawman problem.
4. **AWR training arm (greenlit, the M38 headline).** The 9 sweep losses +
   9 contested races (60% of loss mass) are policy-quality, not rules —
   the O-rule seam is thinning (racemode3 was mechanically perfect and
   moved nothing live). Corpus rebuild unblocked (deck_registry synced);
   `--outcome-weight` hook validated (B7). Ride-along: the `.prize`
   feature fix (training arms only).
5. **Band gate refresh:** re-freeze `band_decode.py` weights on this n=63
   700–800 sample (lucario 13, wall 9, mirror 8, archaludon 8, grim 8,
   dragapult 4, rocket 4, ...) — the 600-band weights are stale one
   milestone after they were built.
6. **Stop-investing list:** grim deck-tech (6-2 live), gustveto (2 residual
   plays), hop beds (absent at this altitude), brick residual (2 losses,
   unactionable), starmie (solved).
