# M37 post-mortem TLDR — sub 55065484 (gacfr3 + alakazam_v2_h4)

Executive summary of [m37-post-mortem.md](m37-post-mortem.md) (full forensics,
2026-07-29). Analysis commit: 60f985f.

**Sub 55065484: implied ELO 736 (33W-30L over 63 games, avg opponent 716).**
The milestone objective — escape the 600 band — was achieved (8-6 in that
band, climbed to a 774 peak). But the agent settled at ~740 because it goes
18-19 in the 700–799 band and 1-3 above 800.

## What caused the losses (30 total)

1. **Deck-outs, 10 losses (33%)** — and here's the milestone's hard lesson.
   racemode3 worked *perfectly*: in all 9 wall games the trigger was true on
   every prompt and the agent used the demoted Dudunsparce/Fezandipiti draws
   0 times in ~410 offers. Yet the wall family went **3-6, every loss a
   deck-out** — one while leading 4-0 on prizes. The burn audit shows why:
   we burn ~2.6 deck cards per own turn vs the opponent's ~1.5, and the leak
   isn't those two abilities — it's the engine itself (Alakazam/Kadabra
   evolve triggers, Dawn 2.4/play, Rare Candy 3.0, **Enriching Energy 4.0
   per attach**, Poké Pad ×19). The offline `solver:wall` bed showed +15.7pp
   only because the solver pilot over-digs its own deck; live wall pilots
   (who sit at 689–836, not just the 600 band) conserve and farm us. Also
   new: **3 of 5 mirror losses are deck-race deck-outs** — one lost by a
   single card.
2. **Swept, 9 losses (30%)** — dragapult (1-3, two losses ending at hand=2),
   lucario, grim, kyogre closed 5–6 prizes while we took ≤1. Policy quality,
   not rules.
3. **Contested races lost, 9 (30%)** — the era's chronic mode, unchanged.
4. **Bench-out bricks, 2** — unactionable draw variance.

Good news: grim self-resolved to **6-2** with zero code change (was 3-5),
garchomp 1-1, the gust hole shrank to 2 residual plays.

## M38 recommendations (ranked)

1. **Wall BC clone first** — `solver:wall` is now strawman-invalid for
   economy fixes; no anti-wall measurement is trustworthy without a clone
   (the M37 garchomp-clone recipe; live wall subs are in the cache).
2. **racemode4** — widen the wall demote to the real burn: Enriching Energy,
   Poké Pad, surplus Hilda/Dawn, plus Sacred Ash timing (it was played at
   deck 0 and 2 in losses). Flips the close races, not the blowouts.
3. **Universal margin raceconserve** for the mirror deck race — the parked
   M36 design, and it's measurable *today* on the model-vs-model mirror bed
   (no strawman problem).
4. **AWR training arm as the headline** — 60% of loss mass (sweeps +
   contested races) is policy-quality; the O-rule seam is thinning when a
   mechanically perfect rule moves nothing live. Already greenlit, corpus
   rebuild unblocked.
5. **Re-freeze the band gate** on the n=63 700–800 composition — the
   600-band weights went stale in one milestone.
