# M29 post-mortem — sub 54914673 (m29_pooled_winners), first 44 live games

_Written 2026-07-23 morning, from the first 44 cached episodes (refresh
2026-07-23 ~09:00). Analysis scripts: `rl.kaggle_ingest forensics`,
`scripts/sub_behavior.py`, plus a new loss-forensics probe (scratchpad,
worth promoting to `scripts/` if reused)._

## Headline: the drop is real, and it is the weights

Raw W/L looks fine (24W–20L, 0.55) — but that record was earned against a
much weaker pool. Same-game-count comparison, all three subs at exactly 44
games:

| sub | score @44 games | W–L @44 | avg opponent score faced |
|---|---|---|---|
| 54897966 (M25) | 724.1 | 22–22 | 713.5 |
| 54903635 (M26, O1 rule on the M25 weights) | 734.7 | 22–22 | 732.3 |
| **54914673 (M29)** | **665.3** | 24–20 | **628.1** |

0.50 against a ~730 pool implies ~730 strength; 0.55 against a ~628 pool
implies ~665. **The M29 weights are playing roughly 60–70 rating points below
the previous two ships.** This is not a matchmaking artifact: the sub opened
1W–4L (including a loss to a 339-rated opponent), sank to 419, and has been
grinding back through the weak pool ever since — 3 of its 20 losses are
upsets to opponents rated >20 below it.

## The offline gain was real — it just doesn't carry the ladder

The dragapult floor gain that justified the ship (+10.75pp, z=+3.13) shows up
live: **drakloak+dreepy 3W–1L**. The losses live elsewhere, on archetypes the
battery never measures:

| opponent archetype | W–L |
|---|---|
| alakazam+kadabra (same-deck mirror!) | 1–3 |
| team_rocket_* (arbok / murkrow / spidops) | 0–3 |
| mega_kangaskhan_ex+crustle | 0–2 |
| mega_lucario_ex+riolu | 1–2 |
| ceruledge / cynthia / hydreigon / solrock (one-offs) | 0–4 |

The live same-deck mirror at 1–3 directly contradicts the offline mirror read
(0.676 n=800, flat vs pin) — consistent with the standing
`no-validated-live-predictor` finding. The weak-pool meta (stall/grind decks:
crustle, kangaskhan, team rocket) is simply outside the four-floor battery.

## Item/tempo regression — Piotr's replay observation, quantified

The new weights are measurably more passive than 54903635, same O1 rule,
weights the only variable:

| instrument | 54897966 | 54903635 | **54914673** |
|---|---|---|---|
| END chosen with a playable ITEM in hand | 16.4% | 12.0% | **26.2%** of END turns |
| telepath attach rate (same O1 override) | — | 84.9% | **67.7%** |
| avg turns per game | — | 13.6 | **16.9** |
| deck cards consumed per our-turn | — | 5.18 | **4.47** |
| deck-out share of losses | — | 5/26 (19%) | **6/20 (30%)** |

The declined items are the tempo cards: **Poké Pad (36 END-turns), Buddy-Buddy
Poffin (16)**. Boss's Orders and Xerosic usage actually went UP (45 and 44
plays vs 15 and 20) — the supporter under-play defect from M27 did not get
worse; the regression is specifically item tempo and attach discipline.

Case study, ep 87567226 (57-turn loss vs "pokemon master"): attacked on 27 of
29 own turns yet still decked out holding **36 cards in hand** — 1 ATTACH and
12 PLAYs in the whole game. It draws relentlessly (Dudunsparce) and hoards
everything, converting nothing, until the deck runs dry. The diary's
pre-registered watch ("deck-out share") has triggered.

## Root-cause hypothesis: pooling diluted the teacher

`bc_m29_samedeck_winners` pooled 783 wins from **8 teachers with a 600 score
floor** — most of the new mass (334 games) is from 54773249 (score 1182, ~70
below our teacher). Winners-only filtering keeps the games those weaker
teachers won, i.e. disproportionately wins over weak opponents, where sloppy
item tempo goes unpunished. The mixture (val_acc 0.664 vs 0.699) taught a
slower, hoarding style. **This is live evidence that the M24 law —
single-teacher beats pooled — extends to the same-deck winners-only cell**
that M29 treated as new territory.

Process note, recorded plainly: the pre-registered ship bar (mirror
resolved-better) was NOT met, and the judgment-call ship rode on the
dragapult floor. The dragapult gain was real and replicated live — and the
sub still dropped ~65 points. Lesson: a resolved gain on one clean floor is
not a proxy for ladder strength when the regression lives on axes we do not
measure. The bar existed precisely to prevent this ship; next time the letter
of the bar holds, or the arm goes to Piotr as evidence, not a ship.

## What M30 should take from this

1. **Rollback decision for Piotr**: 54903635 (M26 ship: m25_bc_alakazam_v3h + O1) remains
   the strongest live evidence. Decide whether M30 re-ships its bundle or
   trains a fresh single-teacher candidate; do not leave 54914673 as the
   flagship on hope — its pool-adjusted strength is resolved-worse, not noisy
   (the 44-game score gap, 665 vs 735, is far outside the ~±15 same-n
   spread the previous two subs show).
2. **Single-teacher A/B on the refreshed cache**: 54773249 now has 334 cached
   wins alone — train `winners(54773249)` and `winners(54618168, refreshed)`
   as separate single-teacher arms before ever pooling again.
3. **Close the battery's coverage hole**: add a stall/grind floor
   (kangaskhan/crustle-style or team-rocket clone from harvested decks) —
   three archetype families produced 0W–5L and none is measured offline.
4. **Item-tempo instrument as a gate**: END-with-playable-item rate and
   telepath attach rate are now cheap, computable offline on QC games. Pin
   them (≤12–16% and ≥85% respectively, per the two prior ships) as ship
   sanity checks alongside the kyogre floor.
5. The mirror gate mispredicted sign twice in one ship (offline flat, live
   1–3). Do not spend n=800 mirror samples before the dragapult and
   coverage floors are run.
