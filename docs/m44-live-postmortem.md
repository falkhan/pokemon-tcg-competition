# M44/M43 PPO ships — live post-mortem: the 800-ELO ceiling

Written 2026-08-14 (Kaggle ingest refreshed to 10,022 episodes; scores via
kaggle CLI the same morning). Subject: every honest-PPO submission plus its
BC control. Comp ends 2026-08-16.

## 1. Where the ships settled

| sub | what | score | vs its control |
|---|---|---|---|
| 55265099 | M41b alakazam, `m41b_wide_prod` (BC) | **810.0** | the ceiling itself |
| 55485268 | M44 K, `ppo_best_m44_K_r1` (league PPO) | 734.8 | +14 vs seed (noise) |
| 55464234 | M43a, `ppo_best_m43a_base` (first honest PPO) | 720.4 | **−90 vs its BC parent** |
| 55464395 | M43b ogerpon wide (self-play retrain) | 700.1 | +52 vs incumbent |
| 55265105 | M41b ogerpon incumbent (BC) | 648.5 | — |
| 55485260 | M44 O, `ppo_best_m44_O_r1` (league PPO) | **546.3** | −102 vs sibling, gate said +5.51pp |
| 55490338 | "M44 PPO Ogerpon" — NOT one of our ritual ships (filename `submission.tar.gz`, submitted 2026-08-13 19:45, desc "69% vs grimmsnarl, 82% vs alakazam") | 494.1 | flagged for Piotr — manual submit? |

The 800 ceiling is literally `m41b_wide_prod`, a behavior-cloning net.
Every PPO descendant of it sits 75–90 below it live while beating it
offline (+3.72pp z=5.93 on the 25-bed gate). O inverted harder: gate PASS
+5.51pp z=9.31 → live −102 vs its own sibling. Two more offline→live
inversions for [[no-validated-live-predictor]].

## 2. The ceiling mechanism: who lives at 780+ and what we score there

Pooled over all six subs (n=83 games vs 780+ opponents):

| family | share of 780+ band | our wr | same family at <700 |
|---|---|---|---|
| grim/froslass | 38.6% | **0.31** | 0.82 |
| alakazam mirror | 22.9% | **0.37** | 0.74 |
| wall (Crustle/Tusk) | 6.0% | **0.00** | 0.41 |
| lucario(+solrock) | 6.0% | 0.40 | 0.67 |
| stall (Fan Rotom/Hop's) | 4.8% | **0.00** | 0.60 |
| dragapult | 3.6% | **0.00** | 0.50 |

Pooled 780+ wr = **0.30**. Per-sub band reads: K 0.82 / 0.48 / **0-8** at
<700 / 700-780 / 780+; m43a 1.00 / 0.40 / 0.24; the 810 incumbent survives
because it holds **0.41** at 780+. ELO here is just "where your winrate
crosses 0.5" — nothing structural below 780 blocks us; the wall is grim +
mirror + the autoloss trio (wall/stall/dragapult) played by GOOD pilots.
We beat the same decks at 0.74–0.82 when weak pilots hold them, which is
why gates built from our own mid-strength clones keep passing candidates
that stall at ~730.

## 3. Why the instruments passed losers (the methods finding)

1. **The 25-bed pooled delta buries bed-level kills.** Both M44 gates
   CONTAINED the live verdict and averaged it away: K's gate showed grim
   −7.0/topgrim −7.4/wall −11.0 (and QC got 0-3 SWEPT by grim) — pooled
   to a "mild" −2.18. O's gate showed walls −16→−29, tuned −22.5, iono
   −23.8, pooled to **+5.51 PASS** on the back of +23→+40 vs grim/topgrim
   CLONES. Live, O fell into the <700 band where walls/dragapult are 19%
   of games and its gate-known weaknesses executed it; it never met the
   grims it out-scores clones against.
2. **Beds are weak-pilot proxies.** Our grim/dragapult beds are BC clones
   (val_acc 0.56–0.64) of ≥700-score seats. Live 780+ grim/dragapult
   pilots (800–940 scores) play a different game: t3 Rare-Candy
   Grimmsnarl one-shotting for 360, Budew item-lock into Phantom Dive.
   Beating our clones harder (what PPO optimizes and the gate measures)
   does not transfer — measured: grim clones vs live 780+ grim = +23pp
   offline vs 0.31 live.
3. **The league is self-referential.** BT-ELO over 4 of our pairs + 2
   rule anchors selects for in-pool exploitation. K's league 1119.6 and
   its 0-8 record vs live 780+ coexist without contradiction — they are
   different populations. In-leg vs_teacher promotion is the same trap
   one level down (already registered in M44; now live-confirmed: K's
   +3.0pp promotion → +14 ELO vs seed, inside noise).

## 4. Live loss causes (loss-cause telemetry + replay narratives)

Loss mix: prize-race dominates everywhere (K 19/23, O 11/12); deck-out is
the stall-matchup failure mode (K: 2 losses at t=54/61 vs Crustle walls,
end-hand 42 cards; m43b: 3); bench-out appears as the t≤4 flash-loss donk
(thin board, active KO'd) and one self-kill (below). The league's 3g
telemetry predicted "prize-race dominant" correctly but its bench-out
share (O ~20% in-league) did not reproduce live — pool-composition
artifact again.

Decision-level findings from full-game narratives
(`scratchpad/game_narrative.py`, episodes cited):

- **Run-Away-Draw self-benchout (deterministic autoloss).** With a LONE
  Dudunsparce on board the pilot uses Run Away Draw, shuffling its only
  Pokémon into the deck → instant bench-out loss with 6 prizes intact.
  Confirmed 3× (K ep92754565 t3; incumbent 810 eps 90145446, 90178716) =
  ~2–4% of alakazam games, 4–8% of losses. `deckguard` exists for low-deck
  but NO bench==0 guard exists; K's ship string doesn't even carry
  deckguard.
- **Attack-into-immunity tunneling (O vs walls, 0W-4L live).** ep92682986:
  O attacked Crustle with the same ex attack for 17 consecutive turns for
  ZERO damage (Crustle blocks ex damage), took 0 prizes in 19 turns, and
  self-milled deck 44→4 with Bug Catching Set/Pokégear/Energy Retrieval
  spam — played Judge at deck=4. No damage-feedback adaptation exists.
- **Tempo passes in the mirror (K, ep92684808 vs 838-rated).** Same deck
  both sides; we out-damaged per hit (220/240/200 KOs) but got fewer
  attacks: at t5 with a fully-evolved Alakazam we retreated into
  Fezandipiti and ENDED without attacking; at t11 played Boss's Orders
  (a rarity) with no energy to attack behind it — wasted gust. Opponent
  attacked every turn from t2. One passed turn = the mirror.
- **Energy scatter vs grim (K, ep92685585 vs 829-rated).** Against a
  360-per-turn one-shot clock our attachments went Dunsparce → Kadabra →
  Dudunsparce → fresh Abra across t6–t12; one real attack in 12 turns.
  Attach-to-the-attacker discipline does not exist under pressure.
- **Disruption never used.** Boss's Orders play rate at offer: O 0.4%
  (1/247!), K 2.7%, incumbent 1.9%. Judge 4.3%, Tool Scrapper 0/117,
  Nighttime Mine 1.3–1.5%. The pilots hoard: end-of-game hands of 20–42
  cards; supporter played on 40–46% of turns.
- **Dragapult is deck-level unwinnable at top pilot quality (0W-7L
  pooled PPO-family).** ep92868490: Budew item-lock t3–t5, then Phantom
  Dive one-shots every 200-HP attacker while spreading 60; our
  bench-scaling attack collapses to 40 dmg into 320 HP when they play
  thin. No pilot tuning fixes this polarity; the 810 incumbent's window
  happened to contain zero dragapult games (band-composition luck).

## 5. What PPO actually changed vs its BC parent (same deck, live)

| behavior | 810 BC incumbent | 735 K (PPO) |
|---|---|---|
| supporter turns | 46.3% | 39.8% |
| END with playable item | 18.4% | 32.7% |
| Dudunsparce draw at deck≤6 | 38/48 | 11/37 |
| Sacred Ash at deck | 4–12 (defensive) | 13–42 (wasted) |
| Fez draw at deck≤6 | 0/59 | 0/57 |

PPO did not add new mistakes so much as intensify passivity: fewer
supporters, more hoarding, worse recycling timing. Consistent with
optimizing wins against a pool of equally passive opponents — passivity
beats passivity, then loses to live aggression. Two caveats: part of the
ash/END gap is the ship fix string (K carries
conserve,racemode2,racemode4,planzero — no ash/ashguard/tempo, which
earlier ships had), and serve-time planzero zeroes the plan input that
the M44 PPO phase trained WITH (58.3% non-zero plan rows) — we ship a
policy evaluated under an input regime it wasn't trained in.

## 6. Improvement paths, ranked

**Before the deadline (2026-08-16) — cheap, high-confidence:**
1. **Guard the self-kill:** new fix `dudguard0` — veto Run Away Draw (and
   any self-remove ability) at bench==0. ~+2–4pp unconditional on
   alakazam, zero risk, one probe run.
2. **Restore the resource fixes on K's string:** A/B
   `+ash,ashguard,tempo` (and consider deckguard) vs the current string —
   the ash-timing and END-with-item regressions look string-inherited,
   not net-inherited.
3. **Boss-sequencing + immunity guards:** demote ATTACK onto a target our
   last attack damaged for 0 (wall tunneling); demote gust unless an
   armed attack follows. Both are apply_play_overrides-class rules.
4. **Re-weight the decision instrument:** gate PASS must additionally
   require no kill on the 780+-band-weighted subset (grim/topgrim/mirror
   /wall/stall beds); a pooled positive can no longer overrule a
   band-weighted kill. One-line change to the decode, retroactively
   explains every inversion.

**Structural (post-comp or one overnight shot):**
5. **Strong-clone beds and pools.** The m45 lucario harvest proved the
   ladder holds 1100+ teacher bands (479 seats for one list). Harvest
   1000+ grim, dragapult and mirror-alakazam seats, clone them, and put
   them in BOTH the gate roster and the PPO opponent pool. PPO's gradient
   currently never sees the opponents that decide the ladder.
6. **Train/serve consistency:** either train under planzero (zero the
   plan in collect+PPO) or ship with plans on — stop optimizing a
   plan-conditioned policy and serving it plan-blind.
7. **Tempo/planning capability:** greedy per-prompt argmax cannot
   represent "an attack this turn is worth more than perfect card
   order". Options in ascending cost: (a) value-head afterstate veto
   (O17 `vveto` exists, never validated live) applied to END/RETREAT
   when ATTACK is armed; (b) full-turn enumeration at the first MAIN
   prompt scoring afterstates with the critic (turn_solver revival);
   (c) BC from 1100+ seats whose demonstrations already contain correct
   tempo — the mirror evidence says the 838-rated opponent's edge was
   pure sequencing, and that is learnable from data we can harvest.
8. **Deck-level honesty:** alakazam_v2_h4 is polarity-capped vs top
   dragapult and out-clocked by candied Grimmsnarl (360 > our 240). A
   pilot that fixes every leak above still may not clear ~850 on this
   deck. The mega_lucario_ex+solrock clone lineage (d3, floors 0.585/
   0.372) is the best-evidenced alternative seat if a future slot exists.

## 7. Standing-lesson updates

- Two new offline→live inversions logged; the pooled-delta aggregation
  rule is now the identified failure point (bed-level kills existed both
  times). [[no-validated-live-predictor]] extended.
- QC battery sweeps are live-predictive (K's 0-3 grim sweep foretold the
  780+ grim record) — a sweep by a family with ≥20% top-band share
  should be a ship blocker, not a note.
- The 3g loss-cause tags transfer band-conditionally: prize-race
  dominance replicated live; bench-out shares did not (pool artifact).
