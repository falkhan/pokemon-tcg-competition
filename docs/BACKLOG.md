# Backlog — deferred items for future milestones

Standing doc (created 2026-07-31, M38 refinement pass, Piotr's decision 3:
minimal-diff ships, everything else tracked here). One line of origin per
item so the context survives; strike or move items into a milestone plan when
they're picked up.

## Deferred from M38 by design

- **Gen-2 collect with a retrained value net.** **PROMOTED to the M40
  headline by M39's ceiling result** (2026-08-02): five policy arms across
  two independently-failing corpora were ALL negative on the 900+ panel, so
  the pre-registered branch "imitation is exhausted for the 1000 target"
  fired. The enabler debt (value net -> real per-decision advantages) is now
  the blocking item, not a nice-to-have. Original entry: M38 gen-1 runs with
  `value_solve` OFF because the M14 setup-plan tier scores lines with an
  old-lineage value net trained on prize-phobic play. Once a clean-corpus BC
  exists: retrain the value head on the clean corpus, then decide (with a
  measurement) whether the setup-plan tier still earns its keep in a gen-2
  collect. Watch item feeding this: M38 R5 (early-game label quality under
  greedy fallback).
- ~~**Rule-stack strip.**~~ **DONE — M39 P1/Ship A (sub 55172160).** The
  strip was gated and then INVERTED by measurement: `conserve`-only beat both
  the full stack (+1.74pp, z=+3.42) and the full strip (+1.03pp, z=+2.03).
  A mechanism probe, not a threshold, is what rescued the surviving rule.
  Original entry: G5 strips only actively-harmful rules from gacfr3;
  redundant-but-harmless ones ride one more milestone. A pure-strip submit is
  a cheap single-variable follow-up ship once the retrained net's live
  behavior is known.
- **W_COUNTER principled retune / 2-ply prize-exchange term.** If E0b shows
  return-KO walk-ins, the minimal fix (constant retune) ships with the
  retrain; the principled version — score net prize exchange (my prizes taken
  − expected return-KO prize value) instead of a flat counter penalty — is a
  separate solver milestone.
- **`setup_plans_late` (M33) re-evaluation.** Late-turn (≥32) setup commits
  are exactly the deck-out-adjacent behavior the old objective rewarded;
  re-justify the tier against a race-closing policy before it returns in
  gen 2.

## From the M37 audit (logged, not scheduled)

- ~~**Silent fix-name ignore**~~ **DONE — M39, as a PRE-SHIP check rather
  than runtime code** (`scripts/ship_verify.py`): raising inside the agent
  would trade a silent-typo risk for a live-crash risk. Extended 2026-08-02
  with a behavioural probe that runs the BUNDLE's own `rl.plan` in a
  subprocess, because "the name resolves" does not prove the rule fires.
  Original entry: an unrecognized name in `_ATTACH_FIXES` is
  silently dropped — nothing would catch a typo in a ship config. Minor
  hardening: fail loudly on unknown names.
- ~~**Kangaskhan-only wall blindspot**~~ **DONE — M39 P2a.** Mega Kangaskhan
  ex (756) added to `_RACEMODE_WALL_IDS` (blanket half): it is in all 10
  cached `kanga` lists and 30 of 65 wall lists. **Honest caveat carried into
  M40: unmeasurable offline** — `kanga` has 19 seats total, far under the
  100-seat bed floor, so this rides on a coverage test and an inertness
  proof (`tests/test_race_bed_inertness.py`), not on a gate cell.
  Original entry: racemode3/4 trigger needs a wall
  Pokémon visible on board (`_RACEMODE_WALL_IDS`); a Kanga-only variant is
  invisible to it. Revisit when a live sample shows one.
- **`rl/` ↔ `tcg/` duplication** (~200 KB, seven pairs, manual parity tests)
  — the unfinished M6 migration. Documented in the audit; needs a dedicated
  housekeeping milestone, not a ride-along.

## From the M37 post-mortem (ranked list, below the M38 cut)

- ~~**Band gate refresh**~~ **DONE — M39 P0 ride-along** (re-frozen on the
  current 700-band sample; the old weights are marked superseded).
  Original entry: re-freeze `band_decode.py` weights on the n=63
  700–800 sample (the 600-band weights are one milestone stale). Cheap;
  candidate ride-along for any M38/M39 ship commit.
- **Stop-investing list** (do not resurrect without new evidence): grim
  deck-tech (~~6-2 live, self-resolved~~ REVOKED by M38 post-mortem — 1-6
  live; **new evidence arrived 2026-08-03 — see "The deck question,
  RE-OPENED by the leaderboard census" above: grim is 49.5% of the top 250
  and we are 2-13 against it**), gustveto (2 residual plays), hop beds
  (absent at altitude), brick residual (2 losses, unactionable), starmie
  (solved).

## The deck question, RE-OPENED by the leaderboard census (2026-08-03)

**STATUS 2026-08-05: resolved in practice by M41/M41b** — option (b) shipped
as a second live arm (`m41_ogerpon` on `decks/ogerpon.csv`, sub 55221491,
re-shipped as 55265105) without abandoning the alakazam line. M41b then
measured the residual as PILOT strength, not deck strength (ogerpon deck
0.940 vs grim under a rule pilot, 0.562 under the alakazam-trained wide
candidate), which is M43 Lane B's target. The analysis below stays as the
record of the decision's evidence base.

Origin: `notebooks/leaderboard_decks.ipynb` / `scripts/leaderboard_decks.py`,
first survey of what the top of the ladder actually plays (top 250, 222
labeled). Full numbers in the M40 diary entry of the same date.

**This section deliberately re-opens a question M40b called closed** ("with the
clone ceiling, the deck A/B and this, the deck-switch question is closed on
three independent measurements"). That verdict stands on its own evidence; the
census adds an axis none of those three measurements covered, so the item goes
to the backlog for a decision rather than being actioned or dropped.

### The observation

We are optimizing against the wrong distribution, and the typing is against us.

| | leaderboard top-250 | our live mix | our live record |
|---|---|---|---|
| **grim** (Marnie's Grimmsnarl ex) | **49.5%** | 9.7% | **2-13 (13.3%)** |
| mirror (ours) | 18.0% | 19.4% | 20-13 (60.6%) |
| wall | 9.5% | 11.0% | 8-9 (47.1%) |
| rocket | 4.1% | 5.2% | 1-7 (12.5%) |
| dragapult | 3.2% | 7.7% | 1-11 (8.3%) |
| lucario | 1.8% | 11.0% | 12-5 (70.6%) |
| starmie | 0.5% | 7.1% | 8-3 (72.7%) |
| **archaludon** | **0.0%** | **13.6%** | 14-7 (66.7%) |
| iono | 0.0% | 0.7% | 0-1 |

**We beat almost exactly the decks that are not up there, and lose to the ones
that are.** Our four best matchups (archaludon, starmie, lucario, mirror) are
2.3% of the top field between them excluding mirror; our four worst (dragapult,
rocket, grim, garchomp) are 59.1%. `archaludon` holds 13.6% of our bed weight
and three dedicated beds (`arch_d1/d2/d3`) and does not appear ONCE in the top
250; `iono` likewise.

**The typing is structural, not incidental** (`data/cards_features.parquet`):

- The whole Marnie's Grimmsnarl line (Impidimp/Morgrem/Grimmsnarl ex, Darkness,
  320 HP on the ex) is **weak to Grass**.
- Our Alakazam (Psychic, 140 HP) is **weak to Darkness**.

So the dominant deck resists us and we fold to it, by printed weakness. The
field has already found the answer: **excluding grim, 42.0% of the top field is
Grass** (47 teams, mean 1011.7 — Teal Mask Ogerpon ex, Dipplin, Crustle, Team
Rocket's Spidops). Fire (which beats Grass) is only 2.7% of the field but
carries the highest mean score of any energy, 1056.0 — the counter-counter is
under-occupied.

### The decision (Piotr, 2026-08-03) — three options, not yet chosen

**(a) Switch to Grimmsnarl.**
*For:* the shell is measured stronger — grim@1000+ beats our archetype **0.70**
head-to-head with a worst matchup of 0.45 (M40 Track A rider). It is half the
ladder. And the corpus asymmetry below is large.
*Against:* **Track A killed it at 0.436 weighted vs the ≥0.55 bar** (z=−51; it
lost even to the iono rule agent). Note precisely what that measured: *our net
piloting a grim clone*, i.e. a pilot result, not a deck result. A switch means
re-basing the entire imitation corpus, not editing `deck.csv`.

**(b) Engineer an anti-grim deck (likely Grass).**
*For:* it is the field's own answer, and deckbuilding is part of the
competition. Infra is half-built and unused: `rl/deck_search.py` already has
`validate_deck` + the mutation primitive (written for a "Phase 1" that never
ran), `decks/gen/` holds 63 machine-generated lists, and `build_meta_field`
freezes opponent snapshots to evaluate against.
*Against:* an invented list has **no demonstrator population to clone**, which
is how every one of our pilots has ever been trained. A *harvested* Grass list
(Ogerpon/Dipplin/Spidops) avoids that; a searched one does not.

**(c) Teach our current deck to beat these decks.**
*For:* zero deck risk, all existing tooling and corpus apply, and it is the
current campaign's path.
*Against:* 2-13 live is the result of already trying, and the offline
instrument cannot currently score the attempt — see the measurement blocker.

### The measurement blocker — applies to ALL THREE options

**We cannot presently measure a fix against real grim.** The panel says we beat
top grim clones **0.684–0.691**; live we are **2-13**. That is not a
contradiction, it is X5: BC retains 0.20 of demonstrator edge, so a "1000+ grim
bed" is really an ~800-band opponent (M40 H-A, confirmed by two independent
comparisons). Every grim bed we own flatters us by roughly the gap we care
about. Whichever option is picked, a grim opponent at the true band is a
prerequisite, or the decision is unfalsifiable offline and costs live slots.

### The corpus asymmetry — the genuinely new argument

Harvestable opponent seats by family (`data/kaggle/opp_decks.parquet`):

| family | seats 900+ | seats 1000+ | **winning seats 1000+** |
|---|---|---|---|
| **grim** | 815 | 577 | **339** |
| garchomp | 414 | 371 | 181 |
| rocket | 412 | 337 | 165 |
| dragapult | 145 | 133 | 78 |
| **mirror (our list)** | 506 | 67 | **40** |

M40's blocking result is "imitation is exhausted for the 1000 target." That was
measured on **our** list — which has only **40 winning 1000+ seats in
existence**. grim has **8.5×** as many. The ceiling may be a property of our
deck's demonstrator population rather than of imitation itself, and no
experiment we have run separates those two. This is the axis the three prior
"closed" measurements did not test.

### Cheapest discriminating probe (recommended before any commitment)

Separate **deck strength** from **pilot strength**, which every measurement so
far has confounded. Pilot each candidate deck with a *deck-agnostic rule pilot*
against the grim beds and against each other:
`uv run python -m rl.matchrunner play --a generic:<deck> --b generic:<deck> -n 800 --workers 8`.
Same pilot on both sides ⇒ the delta is the deck. Candidates: our
`alakazam_v2_h4`, a harvested grim list, and a harvested Grass list exported
with `scripts/export_opp_deck.py --family ogerpon` (79 seats, hash `356c16bd`).

**CORRECTION (2026-08-03, same day).** An earlier draft of this line said the
sample agents are the deck-agnostic pilot. **They are not** —
`sample-agent*/main.py` are hard-coded card-id cascades (`Mega_Lucario_ex =
678`, 15–45 id comparisons each), and `tcg/teachers.py`'s own docstring records
the failure mode ("the Lucario brain played the Kyogre deck through all of
M1"): handed a foreign deck they degrade to a thin generic tail, so the probe
would have measured the *brain's card coverage*, not the deck. The genuinely
deck-agnostic pilot is `rl/generic_pilot.py` (`generic:` / `solver:`), which
scores off the global card DB — only `Carmine` and `Boss's Orders` are
name-keyed, and only under an opted-in `fixes` token.

Note also that `matchrunner` yields **no watchable replays** — it drives
`cg.game` directly, so there is no `visualize` payload. Replay review needs the
`kaggle_environments` path (`rl/eval.py::play_games`, which accepts the
callables `make_pilot` returns). M41 Phase 3 bridges the two.

If the Grass list beats grim under a fixed pilot, (b) is live and cheap to
stage via harvested demonstrators. If nothing beats grim under a fixed pilot,
the deck is genuinely dominant and the question collapses to (a)-with-a-real-
corpus versus (c)-forever.

**Supersedes** the `grim deck-tech` line on the stop-investing list below —
that line says "do not resurrect without new evidence", and this is the new
evidence it was waiting for.

## Research sweep 2026-08-01 — methods beyond the current pipeline (M39+)

Provenance: /deep-research sweep (30 sources searched, 33 fetched, 161
claims extracted). The workflow's adversarial-verify phase crashed twice;
verification was done inline by cross-source consistency + our own campaign
evidence instead — single-source claims are flagged. Ranked by (evidence
quality in comparable settings) × (fit to our constraints: 16k–340k-label
corpora, CPU-only, ~5 games/s engine, dose law).

### High priority — near-term milestone candidates

**M39 RESULTS (2026-08-02) — read this before re-scoping any of #1–#3.** All
three ran as M39 P3 arms on the G-13 panel roster; the outcomes reorder this
list rather than confirm it.

| sweep item | M39 arm | weighted vs live | verdict |
|---|---|---|---|
| #1 advantage-filtered BC | `m39_vsloss_a25` (α=0.25) | −0.82pp | **NOT null**: α=0.25 beats winners-only (α=0) by **+2.28pp** — the discarded loser rows carry real signal, qualifying M28's winners-only law. Still ≤ control on its own. |
| #2 offline best-response | `m39_bestresp` | −0.85pp | did **not** even win its own target beds (−0.4pp there), so the pre-registered exploiter-overfit KILL did not fire — it is simply weak alone. |
| #3 retention by data mixing | `m39_retain_a` / `m39_retain_b` | **+0.80pp / +1.85pp (z=+4.01)** | **the lane's actual result.** Mixing champion shards back in swung corpus A **+3.90pp** and corpus B **+2.70pp**, turning both losing corpora into wins, with no new loss code. |

**Consequence for future scoping: the binding constraint was catastrophic
forgetting, not corpus volume and not demonstrator band** — a third answer the
plan's A/B diagnostic did not enumerate. Retention should be the DEFAULT on
every future fine-tune, not an arm. Two follow-ups M39 did not run: a
retention arm at **α=0.25** (both retention arms used the worse α=0), and
corpus B at a higher exploration rate.

**And the ceiling trigger fired:** no arm moved the 900+ panel (all five
negative), so per the M39 plan's pre-registered branch, **imitation on this
corpus quality is exhausted for the 1000 target** and #6 (value net) plus
self-play become the M40 agenda rather than more harvesting.


1. **Advantage-filtered BC on the harvest corpus (upgrade of the queued AWR
   arm).** Evidence: the strongest domain match in the sweep — a Pokemon
   Showdown agent (arXiv 2504.04395, RLC 2025) reached top-10% of human
   ladder via exactly our staged recipe (BC → offline RL on the same
   replays → self-play fine-tune); offline-RL objectives beat pure BC
   significantly, and the CHOICE of variant (exp-weighted AWR vs binary
   filter vs MaxQ) mattered little. AFBC (arXiv 2110.04698) prefers the
   binary advantage filter `1{A>0}` over exp-weighting (temperature
   sensitivity) and finds ~100k expert samples sufficient — bracketing our
   corpus. Kumar et al. (arXiv 2204.05618) locate the offline-RL-over-BC
   advantage precisely at sparse rewards + noisy data + long horizons =
   our regime.
   *Pros:* `--outcome-weight` hook already validated (M37, never
   launched); near-zero incremental cost as an M39 P3 arm; binary filter
   removes the one hyperparameter.
   *Cons:* the Showdown evidence is at 100–1000× our data scale (38M
   timesteps, 15–200M-param transformers); proper advantage estimates need
   a value net — ours is old-lineage prize-phobic (see gen-2 item), so v1
   must use outcome (seat_won) as the advantage proxy.
2. **Offline best-response via self-play harvest vs the loss-family beds
   ("manufacture the vs-loss corpus").** **PARTIALLY PICKED UP by M43 as
   Lane B** (2026-08-05): self-play rows manufactured for the OGERPON
   corpus gap via `m40_s2_collect`, with the two ingredients the M39 arm
   lacked — retention mixing and α=0.25 — combined for the first time.
   Evidence: BC-init + best-response
   RL vs a FIXED opponent took an exploiter from 42%→90% vs ByteRL (arXiv
   2404.16689); adding synthetic self-play data drove the Showdown agent's
   jump from ~58% to 64–80% win rates. Our translation needs NO new infra:
   `plan_iter collect` games vs `m38_bc_wall`/`m39_bc_grim`/stall beds,
   filter winning seats, low-dose fine-tune per the dose law — an offline
   best-response out of existing tooling. Natural fallback if the M39 P3
   harvested vs-loss corpus runs thin.
   *Pros:* targets exactly the 2-13 wall/grim/stall loss block; corpus
   size is manufacturable on demand; entire pipeline exists.
   *Cons:* exploiter overfit is measured and real (0.90 at 32 decks →
   0.54 at 1024 — arXiv 2404.16689), so gate on the full weighted roster,
   never the target bed alone; the Showdown paper hit opponent-distribution
   overfitting from realistic self-play partners (needed forced diversity);
   we imitate OUR OWN net's winning seats — in-family label quality caps
   the ceiling (M38's lesson about demonstrator strength).
3. **Retention-regularized fine-tuning to break the dose law.** Evidence:
   ICML 2024 Spotlight (arXiv 2402.02868) identifies our exact failure —
   fine-tuning erodes competence on states the fine-tune corpus
   underrepresents ("state coverage gap" + "imperfect cloning gap"), and
   shows retention methods (replay/BC-mixing of pre-training data,
   kickstarting-KL to the frozen parent) let the full transfer happen,
   doubling the NetHack neural SOTA. EWC UNDERPERFORMED BC-based retention
   in both their testbeds — try it last. Caution from StratFormer (arXiv
   2604.25796, single source): a KL anchor can destroy the gains the
   fine-tune was for, cross-entropy anchoring preserved them; and
   kickstarting failed completely where the parent never saw the new
   states — anchor only on states the champion handles well.
   *Pros:* directly attacks the ~1-epoch ceiling (M38's w9294_cont
   10-epoch collapse is textbook state-coverage-gap); simplest arm is pure
   data mixing — champion-corpus rows blended into fine-tune batches, no
   new loss code; would let bigger corpora actually be absorbed.
   *Cons:* new hyperparameter surface (mix ratio / anchor coefficient); no
   card-game evidence; must beat the epoch-1 dose-law baseline that
   already works, on the weighted gate.
4. **Opponent-deck inference feeding the rules layer.** Evidence: a plain
   n-gram model over the opponent's played cards predicted a Hearthstone
   opponent's most-likely card at >95% top-1 by turns 3–5 from only 50k
   replays (Bursztein blog — old + single source, but mechanism is
   trivially replicable); PTCG-Bench (arXiv 2605.29653) found game-history
   context worth ~115 rating points in Pokemon TCG specifically.
   Consumer: the racemode/conserve triggers currently key on VISIBLE board
   ids — a deck classifier fires conserve BEFORE the wall/stall board
   shows, generalizing the trigger-blindspot fix (M38's Fan Rotom 0-2 mode)
   to unseen variants.
   *Pros:* tiny model, CPU-trivial, trains on the existing replay cache;
   the rules layer is a safe consumer (no net retrain); archetype-few meta.
   *Cons:* BRExIt (arXiv 2206.00113) warns opponent-prediction bolted on
   WITHOUT a consumer made agents worse — build the trigger consumer first,
   never an auxiliary head; needs re-training each meta era (cheap);
   early-game misprediction argues for confidence-gated firing.

### Medium — research-grade, needs an enabler first

5. **Use the discarded loser replays (ROIDA-style).** Offline IL that
   splits unlabeled auxiliary data into high/low-quality via a
   positive-unlabeled discriminator, weighted-BC on the good + TD on the
   rest (arXiv 2410.03626, TMLR) — worked from as few as 3–7 expert
   trajectories. Our harvest keeps winner seats only; the losers are ~50%
   of decisions thrown away.
   *Pros:* doubles usable data at zero harvest cost; PU discriminator is
   small and CPU-cheap.
   *Cons:* D4RL/manipulation evidence only; single source; TD component
   needs value infrastructure — sequence after the value-net retrain.
6. **Value-net retrain on the harvest corpus** (merge with the existing
   "gen-2 collect + value-net retrain" item, re-scoped by the M38
   post-mortem away from solver-teacher data). Now also the enabler for
   proper advantages in #1 and #5. **RESOLVED INTO A MEASUREMENT by M40 E0
   + M43 Phase 0** (2026-08-05): every BC net already trains a value head
   (`plan_iter train`'s 0.5·Huber term); E0 is the instrument, and M43's
   0.3 step runs it on `m41b_wide_prod` with pre-registered consequences
   (pass → GAE critic + Φ; kill → `m39_retain_b` warm start). A dedicated
   retrain is only scheduled if both heads fail their consumers.
7. **Online PPO best-response.** The strongest exploiter evidence (#2's
   ByteRL result) used PPO fine-tuning, and M28 left partial infra
   (`scripts/m28_c1_ppo.sh` — audit its state before writing this off).
   *Pros:* highest measured ceiling vs fixed opponents.
   *Cons:* sample scale (the Rummy study, arXiv 2606.21975: 327M env
   steps) is ~2 weeks of our engine flat-out at best; offline #2
   approximates it inside existing tooling — do #2 first.
8. **Architecture inductive bias.** Largest single design win in a
   CPU-scale card-game study (+8–11pp from a game-structure-matched net,
   arXiv 2606.21975, single source).
   *Cons:* requires from-scratch training, and every fresh-train arm we've
   run on ≤340k rows collapsed (M38 scratch/fresh both killed) — parked
   until a self-play-scale corpus exists (#2 could generate one).

9. **AlphaZero-style self-learning over a card pool** (scoped 2026-08-03).
   *The premise is now measured, not assumed.* M40 S3: wrapping the net in
   the solver is worth **+125 ELO [+90, +163]** over the same net unwrapped —
   a policy-improvement operator, which is AZ's entire thesis. M40 X5:
   BC cloning retains only **0.20 [0.07, 0.33]** of demonstrator edge, so
   imitation provably cannot reach the band. Together: the ceiling is real
   and search is the lever that clears it.
   *Blockers, hardest first:*
   (a) **imperfect information** — hidden hand/prizes/deck order. Naive
   determinized MCTS suffers strategy fusion (playing around a card it
   cannot know); the sound fix is belief-state search, which is already
   stop-invest below on compute grounds.
   (b) **the engine is ONE global mutable `Battle` per process** — MCTS needs
   cheap save/restore. `cg.api.search_begin/step/end` is the only hook and
   is scoped to within-turn search; whether it supports re-descent from an
   arbitrary node or only forward DFS decides a 10–50× constant.
   (c) **no batched inference** — one observation per forward call today;
   batching across workers is the single biggest speed lever (~10×).
   (d) **the card pool is not AlphaZero at all.** AZ has one game; a pool is
   a *family* of games indexed by deck pairs plus an outer deckbuilding
   optimization → **PSRO / double oracle**, not AZ. The bed roster is
   already a hand-maintained version of that population.
   *Cost, from measured throughput:* a search-based self-play game costs
   about a composite game (<0.67 games/s at 8 workers) ⇒ ~58k games/day;
   small-game AZ convergence is 1e5–1e6 games ⇒ 2 days–1 month, ×
   determinization count. Consumes a whole campaign; do not start one
   inside a deadline.
   *Already in place:* option-menu policy head (a working AZ policy head),
   value head validated by E0 (0.642 matched-pair, 68% within-game
   variance), `rl/mcts.py` priors+leaf, determinized search API, resumable
   collector, trainer, panel gate.
   *Missing:* batched inference, ISMCTS over prompt nodes, **visit-count
   policy targets** (AZ trains on the visit distribution, not the argmax —
   our collectors train on executed actions), replay buffer + iteration
   loop, population management, and a gate that survives X5.

10. ~~**DISTIL THE SEARCH — the cheap 80% of #9, and the next thing to
   try.**~~ **DEAD — M41b Stage B (2026-08-04).** Both teacher budgets
   INCONCLUSIVE; the effect is matchup-specific (wall +4.3pp twice-
   replicated, grim NEGATIVE at both budgets — and grim is where the live
   loss mass is). Killer arithmetic: at X5's 0.20 retention a +1.06pp
   teacher edge yields ~0.2pp in the student ⇒ ~1,000,000 games/cell to
   detect. Sharper reason it died: the solver's edge is substantially the
   value of features we had already built and never trained on (the M41b
   width retrain captured that directly). Do not re-run without a
   qualitatively better teacher. Original entry:
   Collect a corpus whose labels are the **composite's** actions rather than
   the bare net's, and fine-tune on it. That is AZ's inner loop run once:
   no MCTS, no belief states, no engine work. `scripts/m40_s2_collect.py`
   already does resumable collection against `solved:` arms — the change is
   the arm, not the machinery, so this is hours not weeks.
   *Why it is the right next probe:* it tests the ONE assumption #9 rests on
   — that search output is learnable by our net at our scale. If distilling
   a measured +125 ELO teacher does not move the gate, the full build
   almost certainly would not either, and that is a day spent instead of a
   month.
   *Not in tension with the stop-invest below:* that line forbids SERVING
   search at inference (latency, and the M22c/G5 evidence). This ships a
   plain greedy net that was TRAINED on search output — inference cost
   unchanged.

11. **Energy-fetcher forensics flag, feature gated on its fire rate**
   (from the M41b census discussion, Piotr 2026-08-04). The census killed
   menu-level affordability (the engine never offers an unaffordable
   ATTACK), but the PLANNING version — "active is one energy short, a
   retrieval trainer is in hand, the fetch → attach → attack line exists
   and was not taken" — is unmeasured. Method: measure first. Add the flag
   to the `rl.postmortem` taxonomy (fires when active ≤1 energy short of
   its best attack + a fetcher id in hand + no fetch line taken that turn);
   if its fire rate on real replays is material, the encoder answer is a
   curated fetcher-id set (the `GUST_IDS`/`CONDITIONAL_ATTACKS` pattern)
   plus a typed active-deficit column, riding the width-143 retrain. The
   state already holds the raw ingredients (typed discard pools,
   `energyAttached`, M41b econ/board slots) — only trainer-effect
   semantics are missing, and search composes them without semantics.

12. **Re-measure the T5 dev tier post-epoch (`solver-dev:` arms).** The
   M8.1 kill (pooled 0.492) was measured on the inverted leaf and is
   formally pre-epoch (ARCHITECTURE.md §15 amendment, 2026-08-04). The dev
   tier is the trigger that covers exactly the fetch-to-become-able-to-
   attack scenario (`W_DEV_READY` + `W_DEV_RACE` vs the 900 margin), and
   it ships nowhere today (`dev=False` everywhere). Two sanctioned uses:
   as a LIVE override it needs a fresh post-fix battery before any ship;
   as LABELS it feeds #10's distillation corpus immediately (the
   `score_siblings`/`solve_turn_line(dev=True)` path M12 built).

13. **BC-the-meta -> clone population -> PPO with SHAPED rewards** (Piotr,
   2026-08-04; the "what if the teacher is weak" lane). **PART (c) PICKED
   UP by M43 as Lane A** (2026-08-05, `docs/M43-plan.md`): the
   falsification order below runs verbatim — A-base (outcome reward,
   KL-anchored) vs A-phi (`--shaping value`, Φ = the frozen start's V(s),
   landed in `rl/collector.py`/`rl/ppo.py` with the M43 plan commit);
   event bonuses stay banned. Parts (a)/(b) stay parked as written. Three
   parts with very different standing — the value of this entry is keeping
   them apart.

   **(a) BC across meta decks — population construction, NOT a strength
   lever.** As a strength lever it is foreclosed: imitation is exhausted at
   the 900+ band (M39: five arms, two independent corpora, all negative)
   and a BC clone of a complex hybrid deck can be WEAKER than the rule
   sample agent (M40). As OPPONENTS it is exactly right, and #9(d) already
   frames the pool as PSRO/double-oracle with the bed roster as "a
   hand-maintained version of that population." Scope it as bed-building
   and it is correct and useful; scope it as "more BC will make us
   stronger" and it re-runs a measured dead end.

   **(b) Clones vs clones = the double oracle.** No objection on merit;
   it is #9's population loop, parked on compute (~58k games/day,
   convergence 1e5-1e6 games), not on evidence.

   **(c) The shaped-reward part — where the trap is.** PPO on V3 is live,
   not dead (§15's amendment: the kill was v2-only; M20's PPO produced the
   champion), and `rl/ppo.py` already has `compute_gae` + the clipped
   update. The proposed signals split into two classes and the split is the
   whole point:

   * `board advantage`, `energy efficiency` are STATE functions -> legal
     potentials. Potential-based shaping is policy-invariant ONLY in the
     form `F(s,s') = gamma*Phi(s') - Phi(s)` (Ng/Harada/Russell 1999).
   * `cards taken`, `avoided attacks`, `survived pokemon` are EVENT
     bonuses -> not potentials -> they change the optimal policy and the
     agent farms the proxy. **Concrete hazard, not theoretical: "cards
     taken" pays the agent to draw, and DECK-OUT is one of our measured
     live loss families** — the `conserve`/`racemode`/`W_DECK_LOW`
     machinery exists to fight exactly that. This class must not be added
     without a potential reformulation.

   Precedent for the failure mode is in our own code: `ENTROPY_COEF` was
   cut 10x because the bonus "overpowered the weak advantage signal and
   diffused the policy toward random" (`rl/ppo.py:31`). An auxiliary term
   drowning the real gradient has already happened here once.

   **Two reasons the shaping may be redundant.** We already HAVE a
   validated potential: the E0-passed value head (0.642 matched-pair, 68%
   within-game variance), and GAE already consumes V for precisely this
   credit-assignment job — so `Phi = V(s)` is the principled "board
   advantage" and the honest question is whether a hand-crafted Phi beats
   it. And the credit-assignment gap the proposal intuits is already
   MEASURED: `rl/ppo.py:106 advantage_by_type` (M23 audit) found supporter
   advantages ~0 or noise-drowned while attack advantages dominate.

   **Counterfactual signals are features, not rewards.** "Boss's Orders on
   a Pokemon that could attack us" is expensive and noisy as a reward term
   and cheap as an INPUT — and it half-exists already (O18 `gustsnipe`,
   M41b's matchup slots). Prefer a census over a training run wherever a
   signal can be handed to the net directly.

   *Pre-registered falsification order if this is picked up:* (1)
   `advantage_by_type` on a fresh post-epoch PPO shard to confirm the gap
   still exists; (2) `Phi = V(s)` potential shaping as the control arm —
   free, principled, no new signals; (3) hand-crafted potentials only if
   (2) leaves something on the table, and only genuine state functions;
   (4) never the event bonuses without a potential reformulation. Standing
   caveat: offline gates measure relative deltas on ~750-800 proxies; live
   value decides on the ladder.

- **Deep equilibrium search lane (NFSP/CFR/ReBeL/Student of Games):**
  belief-state enumeration is intractable for collectible card games
  (~10^198 consistent decks in LOCM — arXiv 2404.16689); ReBeL's training
  ran on 90 DGX-1 machines (arXiv 2007.13544); SoG's guarantee is
  compute-scaled (Science Advances 2023). Do not spend a milestone here.
- **Heavy test-time search hybrids:** GO-MCTS needs 25–42 s/turn and
  millions of training games (arXiv 2404.13150); the sound-search
  alternative (IJCAI 2024 look-ahead-on-policy) is validated only on
  Leduc-scale games; PTCG-Bench's authors excluded search-based RL as
  ill-suited to Pokemon TCG. External confirmation of our own G5/M37/M38
  rule-stack-and-solver findings: search-free policy nets beat search at
  CPU budgets (Showdown top-10% search-free; Rummy 580k-param net at
  0.33 ms/action beating a 2.4 s/action searcher). The solver stays a
  label/analysis instrument, not a pilot component.
  **AMENDED 2026-08-03 (M40 S3):** this line is about SERVING search at
  inference and stands unchanged. It does **not** cover search as a
  *training signal* or as an *opponent*. Measured since it was written: the
  solver adds **+125 ELO** to a bed it wraps — 2.6× the entire harvestable
  demonstrator band range (X5) — and after X5 voided bed absolutes,
  composites are the only route to a target-band opponent. So the same code
  is correctly rejected pilot-side and is our strongest lever bed-side; see
  #9/#10. M40's plan drew that boundary before the measurement existed.
- **Decision Transformer:** the sweep's one live contradiction — DT-wins
  claims (arXiv 2305.14550) are directly rebutted at sparse rewards (arXiv
  2507.10174: filtered-BC MLPs match or beat DT at lower cost; "no regime
  where DT is clearly preferable"), and DT's data appetite (5× data for
  2.5× score) plus compute cost disqualify it at our scale regardless of
  who is right.
- **LLM-agent / in-context self-improvement mechanisms** (Reflexion,
  ExpeL, skill libraries…): all five failed to self-improve in Pokemon TCG
  itself (PTCG-Bench, arXiv 2605.29653). Side finding worth keeping: legal-
  action masking was their single most valuable component (118 rating
  points) — our rules-layer action masking is load-bearing; never strip it
  as part of a rule-stack strip.
- **Auxiliary opponent-prediction heads without a consumer:** measurably
  harmful (BRExIt, arXiv 2206.00113). Any opponent-modeling work must ship
  its consumer (rules trigger or search) in the same milestone.
