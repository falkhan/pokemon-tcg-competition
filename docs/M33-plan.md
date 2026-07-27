# M33 plan — value-guided closing: turn a better long-game value into shipped behavior

Draft plan from the M32 pipeline investigation (2026-07-25). Piotr leaned option 3
(long-game value function). Three parallel code investigators + the M8/M13
history + two self-verifications produced a concrete, de-risked program. Awaiting
Piotr's go before executing Stage 1.

## The diagnosis (verified against current code)

The shipped agent is a **greedy argmax over the policy head**; the value head is
exported but **unused at play time** (`submission/main.py:13-21`), and the
override law bans live value consumption (`plan_iter.py:56`). We lose long games
(WR 0.65 ≤15 turns vs 0.40 ≥16, p≈0.003; deck-out the top loss axis). Root-cause
chain, each link confirmed:

1. **BC only imitates the teacher**, uniform-weight per decision, with no value
   signal in long games — so the greedy policy has no learned "close it out"
   behavior (fidelity ceiling ~0.66; 6 milestones of fidelity work bought zero
   strength → **fidelity ≠ strength**).
2. **The one path that could inject better-than-teacher closing labels is turned
   OFF past turn 32.** `value_solve` (value-guided SETUP-plan DAgger at data-gen)
   returns nothing once `obs.turn >= VS_MAX_TURN=32` (`plan_iter.py:101`) —
   exactly the turns where we deck out.
3. **It's off because the setup value is near-chance past turn ~32** (held-out
   matched-pair acc 0.79–0.80 at t4–15 vs 0.46–0.66 at t32–47, `docs/M13.md`).
   Four concrete, mostly-fixable causes (value investigator, code-grounded):
   - **(a) prize-matched pairs erase the signal in stalls** — pairs match on
     prize counts (`setup_value.py:213`), but long games are long *because
     prizes stop moving*; within a matched late pair almost nothing separable
     remains except deck-out/tempo, which the encoder barely shows.
   - **(b) feature saturation/blindness** — `turn/30` saturates at ~1.0 past
     turn 30; `deckCount/60` compresses the decisive 0–15 range; and the
     setup-value net is `extra_dim=0` (`setup_value.py:277`), so it **never sees
     opponent deck count** (v4-only) — blind to the variable that decides the
     grind. (Recorded symptom: val_acc 0.738 at deck 30+ but 0.601 at deck 7–15.)
   - **(c) stratified-pair sparsity** (computed on `data/setupval_v3`, 138,872
     states): not raw scarcity (16.6% of states are turn>32) but **pairs** —
     57.5% turn<16 vs 23.8% turn≥32, and prize/deck stratification shatters late
     states into ~3.6 pairs/key vs 60–136 early: ~20× fewer per-config examples.
   - **(d) MC-outcome target noise over a long credit horizon** — structural; no
     TD/discounting (the PPO critic path is dead, `value_train.py:1-10`).

## The mechanism that makes this SHIP (verified feasible)

`_teacher_step` (`plan_iter.py:165-189`): when a solver/value line clears the bar
it is committed as a **SETUP plan label** (`derive_plan` → matched candidate);
`value_solve` supplies those plans on non-exact turns (`:180-184`). BC then clones
the plan block into the greedy policy. So: **better late value → value_solve fires
past turn 32 → late-game closing plans enter the corpus → cloned into the shipped
greedy policy.** No search is shipped, so we dodge the two walls that killed MCTS
three times: OOD value on search-reached states and determinization in an
imperfect-information game. Bonus: this **revives the plan block**, which is inert
in the replay-BC lineage (`encoders.py:270-274`) yet encodes exactly the closing
quantities (`lethal / wins / return_ko / concedes / opp_ttk`).

## Program (sequenced, each stage a cheap go/no-go)

**Stage 1 — fix the late-game value (LOW effort; the enabling gate).**
- 1a. Append late-game features to the setup-value trunk via the existing
  append-and-zero-init warm-start pattern (`encoders.py` v4-safe point): opponent
  deckCount, a two-sided low-deck ramp (mirror `DECK_LOW_AT=15`), a
  turns-to-deck-out estimate per side, and a de-saturated turn scalar
  (`min(turn,60)/60`). Rebuild the setup-value net with these (drop the
  `extra_dim=0` restriction so it sees them).
- 1b. Fix `build_pairs` (`setup_value.py:203-227`): in late buckets stop matching
  on prizes (let deck-out/tempo be the discriminator), coarsen/drop `deck_idx`,
  raise the per-cell cap. Keep strict early-bucket matching.
- **GATE:** per-bucket held-out accuracy past turn 32 rises from ~0.46–0.66
  toward the ~0.80 early-bucket bar (harness already reports per-bucket,
  `setup_value.py:284-297`). **If it does not clear, the late value is
  intrinsically unrankable (mechanism (d) dominates) → STOP.** Cheap kill.

**Stage 2 — extend value-guided DAgger to the close (MED effort).**
- Raise `VS_MAX_TURN` past 32 (only after Stage 1 clears).
- Regenerate a data-gen collection with `value_solve` committing SETUP plans on
  turns 32+; **bias collection toward grind/control matchups** (rocket/grim/wall
  clones, not just the mirror) so the closing skill covers where we actually deck
  out. `--workers 8` cap; fresh `--out` dir (no resume).
- **GATE:** `setup_plans` count rises on late turns; spot-check the late plans
  are sane (not degenerate stalls).

**Stage 3 — clone into the shipped policy and measure STRENGTH (the payoff).**
- `plan_iter train` a candidate mixing the new plan corpus with the replay-BC
  corpus (revives the plan head with real closing plans).
- Battery on the live-faithful beds (rocket + grim clones, mirror, archaludon),
  reading **long-game / deck-out outcomes specifically**, not just pooled WR.
  Pre-register the gate on closing (deck-out share ↓, long-game WR ↑) with
  non-inferiority elsewhere. QC per the mandatory ritual. Ship only on a resolved
  strength gain.

## Risks (honest)

- **fidelity ≠ strength** — gate every stage on strength/closing behavior, never
  val-acc. This program changes the *target* (better-than-teacher closing labels),
  so it is NOT the killed fidelity-raising class — but the discipline still holds.
- **Determinization ceiling** — search under hidden info is imperfect; `value_solve`
  is a shallow SETUP commit with an exact lethal/prize tier, so less sensitive
  than deep MCTS, but line quality is still capped by determinization.
- **Late value may be intrinsically unrankable** — mechanism (d). Stage 1's gate
  is precisely the cheap test for this; we stop there if so.
- **Mirror-trained value may not transfer to control matchups** — mitigated by
  Stage 2's grind-biased collection; verify on the rocket/grim beds.

## Complementary bigger bet (hold for after)

BC investigator's #1 lever: **solver-relabel the HUMAN replay states**
(distillation beyond the teacher on the full distribution, not just late game) —
the only lever that can exceed `teacher × fidelity` broadly. Higher effort (new
path bridging `replay_bc` states into the solver). Pursue after the closing
program if Stage 3 ships, since both share the "solver-as-improvement-operator"
machinery.

## Do NOT re-propose (measured-spent)

- Shipping search/MCTS (OOD value + determinization; dead M8, M22 C1, M32 review).
- Naive fidelity-raising: epochs, score/winner-weighting, corpus pooling
  (M24/25/26; fidelity ≠ strength).
- PPO fine-tune for out-of-loop strength (no-transfer ×3: M22/M23/M32-B).
- Live value/override consumption (M8.1/M12/M13 override law).

## Execution log — Stage 1 (2026-07-25)

**Code changes (rl/setup_value.py; shipped encoders.py untouched):**
- `late_game_feats(me, op, turn)` → 5 scalars: opp deckCount/60, my & opp
  deck-out ramps (mirror `DECK_LOW_AT=15`), de-saturated turn `min(turn,60)/60`,
  late-phase flag `turn>=16`. Defined in setup_value.py (NOT encoders.py) so the
  shipped bundle's encoders stay byte-identical.
- Collection appends the 5 feats to `states` (width 1361→1366) and stores
  `my_deck`/`opp_deck` metadata columns. `_load_rows` tolerates their absence
  (old corpora still load).
- `build_pairs(coarse_late, late_cap)`: in late buckets (turn≥16) drops `deck_idx`
  from the match key (keeps prizes matched → no leak) and raises the cap — lifts
  the ~3.6 pairs/key late sparsity.
- `train(late, coarse_pairs)`: `late` builds the net with `extra_dim=LATE_DIM=5`
  and feeds the LATE block; baseline slices it off (`extra_dim=0`) so the A/B
  differs ONLY in the added features + pair construction. From-scratch both
  (warm-start with `--late` is guarded off — state_enc width differs).
- Safety: collect CLI default workers 12→8 (CLAUDE.md deadlock cap).

**Verification:** 607→ existing `tests/test_setup_value.py` pass (2/2). Synthetic
checks: `late_game_feats` ranges correct; `coarse_late` yields more late pairs
(1211 vs 827) with early pairs unchanged (337=337, no leak into early buckets).
End-to-end smoke (10 games): shards carry all columns at width 1366; both train
recipes run (`extra_dim=0 w=1361` vs `extra_dim=5 w=1366`).

**A/B design (clean, same corpus, from scratch):** collect one fresh corpus
`data/setupval_m33`; train `baseline` (extra_dim=0, old pairs) vs `m33` (late +
coarse-pairs); compare held-out matched-pair accuracy PER BUCKET, focus t32+.
GATE: does t32+ climb from ~0.46–0.66 toward the ~0.80 early bar? Yes → Stage 2;
no → STOP (late value intrinsically unrankable).

## Stage 1 GATE RESULT (2026-07-26) — premise falsified; features null; value already good late

Fresh 4000-game corpus `data/setupval_m33` (54,365 states, 15.2% turn≥32). Clean
same-corpus A/B, both from scratch, decoded on ONE common late-heavy val set
(`scripts/m33_stage1_decode.py`, coarse pairs on the seed-0 val games):

| regime | n | baseline | late+coarse | z(new−base) |
|---|---|---|---|---|
| t<16 | 656 | 0.768 | 0.738 | −1.28 |
| t16–31 | 223 | 0.762 | 0.695 | −1.60 |
| t32–51 | 105 | **0.838** [0.76,0.90] | 0.848 [0.77,0.90] | +0.19 |
| t52+ | 91 | 0.670 | 0.637 | −0.47 |
| t32+ all | 196 | **0.760** [0.70,0.81] | 0.750 | −0.23 |

**(1) The Stage-1 fix is a KILL.** Late features + coarse pairs add nothing late
(t32+ z=−0.23) and slightly hurt early/mid (the extra columns are noise the net
overfits; coarse pairing shifts the train mix). Do NOT ship the features.

**(2) The premise is falsified on current data.** The baseline setup value is
NOT near-chance late — t32–51 = 0.838 (CI above the 0.80 early bar), t32+ = 0.760.
The historical "0.46–0.66 past turn 32" (docs/M13.md, the basis for
`VS_MAX_TURN=32`) does not reproduce on a fresh from-scratch net. Likely causes:
the M13 number was warm-started + small-late-n (possibly noise), and/or a
from-scratch trunk specializes better for ranking than a BC-warm-started one.
The base v3 features (my deckCount + prize race + race block + per-slot) already
rank prize-matched late pairs at 0.84 WITHOUT opponent deck count.

**Implication (the redirect):** the late value is not the closing blocker — it is
already usable, so **`VS_MAX_TURN=32` is over-conservative.** Stage 2 (extend the
value_solve DAgger past turn 32 → clone late closing plans into the greedy
policy) is still viable and now BETTER motivated — but it needs NO new features:
just a freshly-retrained baseline setup value (m33_sv_base is one) as the leaf.
The real remaining risk is the OLD wall: whether value_solve's late lines are
actually GOOD under determinization (imperfect-info search), not whether the
value can rank. That is what Stage 2 must test. Checkpoints kept:
`checkpoints/m33_sv_base.pt` (good-late value, no features), `m33_sv_late.pt`
(null arm). The `--late`/`--coarse-pairs` code stays but is unused going forward.

## Stage-2 probe RESULT (2026-07-26) — late-plan generation VIABLE

`plan_iter collect --mode expert --value-ckpt m33_sv_base --vs-max-turn 60`,
300 games, workers 8 (new `--vs-max-turn` flag + `setup_plans_late` counter):
- **242 late (turn≥32) bar-clearing SETUP commits, 0 errors** (2795 total commits,
  4299 value_solve calls). Median solved-line margin 999 ≫ the 200 threshold.
- ⇒ the good-late value net GENERATES late closing plans past turn 32; the
  determinization/OOD wall does NOT block generation. VS_MAX_TURN=32 was leaving
  those on the table.
- Caveat: a commit means the value BELIEVES the line beats stand-pat — not proof
  the plan improves real outcomes (margin-200 was calibrated for the OLD net).
  Whether late plans HELP is the Stage-3 A/B question below.

**Overnight experiment (autonomous, Piotr away): cap-32 vs cap-60 DAgger A/B.**
Isolates the late-plan effect: two DAgger corpora identical except late plans
(cap 32 = none past t32; cap 60 = late plans), warm-started from the shipped
m28_winners so strength is representative, batteried on the deck-out beds
(rocket/grim clones + mirror) reading WR + game length. Late plans help ⇒
cap-60 closes better. Measurement only — NO ship without Piotr's go.

## Value-net cross-check (2026-07-26) — the M18 PRODUCTION value net is EXCELLENT late

Evaluated existing production setup-value nets on the SAME m33 late buckets:
- **osv3o_setupval1 (M18 prod): t32-51 0.87, t52+ 0.99, t32+ 0.92** — even better
  than my fresh m33_sv_base (t32+ 0.76). Definitively confirms the "near-chance
  past turn 32" was a MEASUREMENT ARTIFACT (small-n / stale corpus in M13), not a
  property of the value net. VS_MAX_TURN=32 is drastically over-conservative.
- osv3o's value_solve committed-margin median is ~299 (right at the calibrated
  200 band); m33_sv_base's was ~999 (inflated → mis-calibrated / over-commits).
  osv3o is the net the margin-200 threshold was CALIBRATED for.
⇒ Restarted the Stage-3 A/B on **osv3o_setupval1** (better late + correctly
calibrated), not m33_sv_base. The Stage-1 late-feature work is fully vindicated as
unnecessary — the shipped value machinery already ranks late states at 0.92.

## Stage-3 A/B RESULT (2026-07-26) — NO-GO: self-play DAgger fine-tune destroys strength (M24 wall)

Both arms warm-started from migrated m28, fine-tuned on the DAgger corpus (osv3o
value net), batteried n=400/bed:

| bed | c32 (control) | c60 (late plans) |
|---|---|---|
| rocket | 0.228 | 0.133 |
| grim | 0.235 | 0.101 |
| vs m28 | 0.110 | 0.065 |

- **Both arms COLLAPSED vs m28 (0.065–0.11)** — fine-tuning the replay-BC policy on
  plan_iter SELF-PLAY DAgger catastrophically degraded strength. The M24 law
  ("self-play data underperforms replay-BC") reproduced: train val_acc looked fine
  (0.72) but play strength cratered — **fidelity ≠ strength** yet again.
- **c60 (late plans) was WORSE than c32 on every bed**, not better. So late closing
  plans did NOT help — but the A/B is CONFOUNDED (both models broken by the
  fine-tune), so this is not a clean read of the late-plan effect.
- **Verdict: the value-guided-closing path is a NO-GO as executed.** The blocker was
  never the value quality (Stage 1: it's 0.92 late). It is that the only mechanism
  we have to inject closing behavior — self-play DAgger — is fundamentally
  incompatible with the replay-BC strength source. Injecting plans requires either
  (a) mixing the replay-BC corpus (schema/plan gap) or (b) the BC-agent's
  solver-relabel-HUMAN-states path (keeps the replay distribution). Both are Stage-2
  post-mortem follow-ons for Piotr, not tonight.

## M33 arc summary
S1 premise falsified (value already 0.92 late) · S2 late plans generate fine ·
S3 injecting them via self-play DAgger breaks the policy (M24). Option 3 does not
yield a stronger closing policy by this route. Reusable: --vs-max-turn flag +
setup_plans_late counter (plan_iter), m28_winners_v4.pt, scripts/m33_*.

## CORRECTION (2026-07-26, Piotr) — Stage-3 conclusion OVERSTATED

The verdict "self-play DAgger is fundamentally incompatible with the replay-BC
strength source" is NOT what Stage 3 tested. The protocol (scripts/m33_stage3_ab.sh
→ plan_iter train --init, lr 3e-4 default, no decay, no KL anchor, no corpus mix,
6 epochs over ~25k rows ≈ the whole 28.6k-decision replay corpus) is effectively a
FROM-SCRATCH solver-lineage train off an m28 initialization — a catastrophic-
forgetting protocol that NO data source survives. Stage 3 established only that
*this protocol* destroys a replay-BC policy (which the M24 law already predicted).
The late-plan treatment got no clean read (the A/B admits this).

**RL diagnosis (the frame the report circled but missed):** DAgger assumes the
relabeling expert ≥ the policy. Ours isn't — the solver+value pilot END-TO-END
(~0.43 mirror) is weaker than the human clone (0.59+). DAgger toward a globally
weaker expert drags the policy globally toward it. The solver is only LOCALLY
superior — the late-closing slice where the value now ranks 0.92. So: inject ONLY
the locally-superior slices, keep human labels elsewhere. The margin must gate
solver-VS-TEACHER (overwrite only where the solver's line differs from the human's
AND clears a high bar) — VS_MARGIN=200 gates solver-vs-STAND-PAT, the wrong axis.

**Not dead — mis-tested.** The closing thread stays open; the next test needs a
non-destructive injection protocol (from-scratch on replay-BC + a small late slice,
or margin-gated solver-relabel of HUMAN replay states). Strength-gate everything.

## Deck-tech probe RESULT (2026-07-26, Rec 5) — V2 (setup speed) is a RESOLVED POSITIVE

Same pilot (m28_winners), 3 decks, n=400/bed, vs the deck-out beds:

| bed | base clone | V1 (draw depth) | V2 (setup speed) |
|---|---|---|---|
| rocket | 0.560 | 0.482 | **0.640** (+8.0pp, z≈2.3) |
| grim | 0.662 | 0.676 | **0.739** (+7.7pp, z≈2.4) |
| mirror | 0.496 | 0.355 | 0.487 (held) |

- **V2** (+1 Rare Candy, +1 Night Stretcher, −1 Enhanced Hammer, −1 Nighttime Mine):
  resolved gains on BOTH deck-out beds, mirror held — and DESPITE zero-shot deck
  transfer (m28 trained on the base list). The true effect with a pilot fit to V2
  could be larger (M24: deck+pilot are a unit). First resolved offline gain on the
  live-faithful deck-out beds in the campaign.
- **V1** (draw depth, cut Xerosic/Shaymin): a KILL — rocket −8pp, mirror −14pp.
  Cutting the disruption/tech hurt more than extra draw helped.
- ⇒ the deck-tech lever (skipped when option 3 was picked) is the promising thread.
  m28 + alakazam_v2 (decks/alakazam_v2.csv) is a candidate NOW — no retrain, the
  transfer penalty already priced in. Next: full battery (add lucario/archaludon/
  kyogre floor) to confirm no regressions, then QC + ship decision.

## Rec 4 feasibility (2026-07-26) — solver-relabel of HUMAN replay states is INFEASIBLE
The solver needs `search_begin_input`, a LIVE-engine handle (cg/game.py:15); the
agent's logged observation (what replays store) has it None (cg/api.py:449). So the
solver cannot run on logged replay states, and re-simulating them needs
deterministic replay of all hidden randomness. Feasible equivalent proposed:
on-policy DAgger via plan_iter "ei" mode — the m28 clone rolls out (its induced =
deployment distribution), solver+value relabels ONLY the high-margin
solver-vs-clone slice (cap ~10-20%), train from scratch, strength-gated. Awaiting
Piotr's go; deck-tech V2 now looks higher-EV.

## V2 FULL BATTERY (2026-07-26) — clean: resolved-better on deck-out beds, no regressions

| bed | base | V2 | delta |
|---|---|---|---|
| rocket | 0.560 | 0.640 | +0.080 (z≈2.3) |
| grim | 0.662 | 0.739 | +0.076 (z≈2.4) |
| mirror | 0.496 | 0.487 | −0.009 |
| lucario | 0.695 | 0.708 | +0.013 |
| archaludon | 0.843 | 0.865 | +0.022 |
| kyogre floor | 0.975 | 0.970 | −0.005 |

**m28_winners + alakazam_v2 (decks/alakazam_v2.csv) is a strong ship candidate:**
resolved gains on the two matchups we lose, non-inferior everywhere else, no
retrain (transfer penalty already priced in — true effect likely larger). Swaps
vs base: +1 Rare Candy, +1 Night Stretcher, −1 Enhanced Hammer, −1 Nighttime Mine
(setup speed → closes the grind faster, the p≈0.003 live finding). Next per the
mandatory ritual: build_submission --deck alakazam_v2, md5-verify the tarball deck,
QC 3 games vs a stall opponent, Piotr review + explicit go before submit. Honest
caveat: offline; but the rocket/grim clones are the most live-credible beds built.

## V2 SHIP CANDIDATE — bundle built, deck-verified, QC'd (2026-07-26) — AWAITING PIOTR

- Bundle: `dist/submission_neural_20260726_101347.tar.gz` (m28_winners + alakazam_v2).
  Build gate OK (rewards [1,-1], 176 decisions).
- **Deck verified:** bundle deck.csv md5 `7b1c123b…` == decks/alakazam_v2.csv;
  ≠ base clone `8e8cf124`. 60 cards, Rare Candy 4 / Enhanced Hammer 3 confirmed.
- **QC:** actual bundle vs rocket clone (the deck-out opponent) = **2W–1L**
  (win_rate 0.667, matches offline 0.64), no crashes. Replays:
  `replays/m33_qc_v2_rocket_000..002.html`.
- **STOPPED for Piotr's manual replay review + explicit submit go** (no submit
  without it). Submit when approved:
  `./build_submission.sh --checkpoint m28_winners.pt --deck alakazam_v2 --message "M33: alakazam_v2 setup-speed deck-tech"`

## M33 SHIPPED — sub 54997669 (2026-07-26)

**SHIPPED: sub 54997669 = m28_winners + alakazam_v2** (setup-speed deck-tech),
Piotr's "Let's try shipping". Deck md5-verified 7b1c123b (=alakazam_v2, ≠ base
fossil) at build AND immediately pre-submit; QC 2W–1L vs rocket clone, no crashes.
Tarball dist/submission_neural_20260726_101347.tar.gz. Status PENDING (settling).
Replaces M31 54935640 (688.7); M30 54929991 was 759.4. Clean live A/B: same pilot
(m28_winners), the DECK is the only change vs the base list → the deck-tech effect
is isolated live. Swaps: +1 Rare Candy, +1 Night Stretcher, −1 Enhanced Hammer,
−1 Nighttime Mine.

**Watch next session:** live score of 54997669 vs 688.7 (M31) / 759.4 (M30);
rocket/grim/stall matchup WR (offline +8pp/+7.7pp — does the deck-tech transfer
live where 5 milestones of pilot/rules did not?); deck-out share; game length
(the setup-speed thesis predicts shorter games / fewer deck-outs). Run
scripts/live_postmortem.py + scripts/live_deck_race.py once ~40 games cache.
Commit the shipped milestone when Piotr asks (CLAUDE.md: commit only shipped).
