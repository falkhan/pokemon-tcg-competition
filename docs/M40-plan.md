# M40 plan — STUB. Leave the band, or stop pretending we are trying to

**Status: STUB, drafted 2026-08-02, rewritten the same day after Piotr's
steer: "there's no point building for a modest increment — we've been stuck
in the same ELO slot for a long time."** That is correct, and the first
version of this plan was wrong to recommend banking +24 ELO. Nothing here is
scheduled; decisions at the end are unanswered.

---

## 1. Why we are stuck — and it is not policy quality

The campaign's scores: M30/M35 **817**, M37 **736**, M38 **665**, M39 Ship A
**780**. Five milestones oscillating inside one ~150-point band. The usual
story is "the policy needs to be better". The M39 data says something more
specific and more actionable.

**Two facts that cannot both be about our policy:**

| | |
|---|---|
| our WR vs `m39_bc_top` — a BC clone of **900–1058-band** pilots of our own list | **0.532** (panel of 3 draws, n=3600) — *we are above parity* |
| our live record vs **800+** opponents, M37+M38 | **0-6** |

We beat a "900-band" bed and have never once beaten a real 800+ player.
Both cannot be true of the same opponents, so the bed is not what its label
says.

**M39 accidentally measured the compression factor.** D1's head-to-head put
the 1000-band grim clone over the 750-band grim clone at **0.623 (z=+12.04)**
— clones *do* transmit relative skill, so cloning is not saturated. But 0.623
is ≈86 ELO of separation from demonstrators ~250 ELO apart. **A BC clone
retains roughly a third to a half of its demonstrator's edge.**

Apply that to every bed we own and the campaign's history stops being a
mystery:

> **Our entire offline apparatus is a ~700-band opponent set wearing
> 700–1000-band labels. We have spent five milestones optimizing win rate
> against opposition at the band we are already in. You cannot exit a band by
> optimizing against it.**

It also explains why the gates keep half-transferring: Ship A's +115 came
from *removing* rules (undoing damage is measurable against any opponent),
while M37's and M38's predicted wall/grim gains — which required actually
outplaying a stronger opponent — did not transfer at all.

**Stated as a hypothesis, with its own weakness up front (G-9):** the 0-6 is
n=6, and no live cell may gate a decision at any n. The compression estimate
rests on one head-to-head. **M40's first job is to test this properly, not to
assume it** — see S3. But it is the only story that fits both numbers, and if
it is right then "a modest increment" was never the choice; the choice was
between a real attempt and a rounding error.

## 2. What a step change has to come from

If the beds are the ceiling, then anything that *optimizes against the beds*
inherits the ceiling — which is every lane M37–M39 ran. The mechanisms that
do not fall into two groups:

- **missing or broken faculties** (opponent-independent, so no clone
  ceiling): **S1** the agent cannot assemble a multi-step turn; **S5** it
  cannot see the opponent's bench threat or play history; **S6** it is
  served a stale plan vector it was never trained on. All three are
  structural, all three have tested code already in the tree, and **S5 and
  S6 are the same regression** — the M24 switch to replay-BC corpora,
  which nothing was checking.
- **better opposition / better instruments**: **S2** self-play, **S3**
  opponents above the clone ceiling, **S4** the live ladder.

The first group is where I would look first: cheaper, lower novelty risk,
and neither depends on the bed question being resolved.

### S1 — give the neural line the within-turn combo solver it has never had

**The single largest unexploited structural gap in the codebase, and it is
not new research — it is shipping infrastructure we already own to a lineage
that never received it.**

From [matchrunner.py:390](../rl/matchrunner.py:390) and
[turn_solver.py:531](../rl/turn_solver.py:531): *"the shipped neural bundle
is a greedy one-action argmax, which is the exact failure this module was
written to fix — a multi-prize lethal that needs item → attach → attack is
never assembled. The solver has only ever shipped in the RULES bundle, so the
neural line never got it."*

**We have been shipping a policy that cannot assemble a two-step lethal since
M11.**

Screen run 2026-08-02 (`solved:` vs `model-conserve:` on `m38_bc_wall`,
n=30/arm):

| arm | W-L vs the wall bed | mean move | p99 | max |
|---|---|---|---|---|
| `model-conserve` (live) | 9-21 (**0.300**) | 0.62 ms | 1.3 ms | 8.0 ms |
| **`solved` (net + combo solver)** | **16-14 (0.533)** | **5.92 ms** | 28.9 ms | 117.9 ms |

The 0.300 reproduces the panel gate's 0.296 exactly, so the control is sound.
**+23pp on our worst matchup.**

**This is a SCREEN, not a measurement.** n=30 resolves ~35pp (G-12), and this
campaign has corrected four small-n over-reads in the last week. It does not
enter a ship decision until a panel battery says so. But it is the largest
screen delta the campaign has produced, on the matchup we are worst at, from
code that already exists.

Why it is plausible rather than lucky: this is **structural action-space
competence, not policy quality** — the same class as PTCG-Bench's finding
that legal-action masking was worth **118 rating points**, their single
largest component. And it is opponent-independent, so unlike every M37–M39
lane it does not inherit the clone ceiling.

**It reopens a documented stop-invest line and therefore needs Piotr's
explicit sign-off.** The BACKLOG says *"search-free policy nets beat search
at CPU budgets; the solver stays a label/analysis instrument, not a pilot
component."* Those citations are about **replacing** a policy with tree
search (GO-MCTS at 25–42 s/turn; ReBeL on 90 DGX-1 machines). Completing a
turn the policy has already chosen, at 6 ms inside a 50 ms budget, is a
different intervention. Our own M22c evidence against it predates the entire
replay-BC lineage and was measured on a net that no longer exists.

Open risks, to be closed before any ship: the **117.9 ms max move** against
the real per-move limit; whether the gain survives on beds other than wall;
and whether the solver's own action model is faithful on the current net.

### S5 — adopt the encoder we already built and then abandoned

**We shipped opponent memory in M21, then silently regressed off it in M24
and have never shipped it since.** Found 2026-08-02 while answering "does the
pilot see the opponent's card ids / hp / max damage?".

What the **v3** encoder our live net uses gives us about each opposing
Pokémon (active *and* bench): card id at a learnable embedding site, 36 card
features, **current hp and maxHp**, energy count and per-type energy counts,
attached tools. Plus, for the **active only**, `_combat_features`' best
affordable damage against us after weakness/resistance, KO flags and
one-attach-from-KO.

What the **v4** encoder adds, and our lineage does not have
([encoders.py:657](../rl/encoders.py:657), `V4_EXTRA_DIM = 341`):

| block | what it is | why it matters here |
|---|---|---|
| `_slot_extras` — 12 slots × 5 | appearThisTurn, evo-stack depth, special-energy count, attack-ready-NOW, **best affordable damage per slot** | **the opponent's BENCH gets a damage projection.** In v3 a benched threat is an id and an energy count; the net has to learn the threat model itself. M39's loss anatomy is 43% sweeps where we set up too slowly — mispricing the incoming attacker is exactly that shape |
| `OppMemory` — `N_MEM = 51` + 5 id slots | last-4 opponent played card ids, last opponent attacker id, their last energy-attach target (7-way), charging signals | **their charging target is their announced next attacker.** This is the learned, end-to-end version of what E3's classifier would approximate |

**Why the lineage lost it.** M21's B2/B3 ships (54846434 / 54849475) were v4.
From M24 the campaign switched to replay-BC clones, and
[replay_bc.py:114](../rl/replay_bc.py:114) encodes with `encode_state_v3` —
it **never reads `obs.logs` at all** (zero references in the file). So every
harvest corpus since M24 is v4-blind, and every net trained on one is a v3
net. The shipped `m38_w9294_cont3` confirms it: no `enc_ver` buffer, input
1708 = 1361 numeric + 27 plan + 20 ids × 16. This was never a decision — it
is a side effect of which encoder the corpus builder happened to call.

**And the data is already sitting in the cache.** Probed the cached replays:
**828 of 844 observations (98.1%) carry a non-empty `obs.logs`**, with real
entries (`{'cardId': 1225, 'playerIndex': 0, 'serial': 41, 'type': 4}`). So
this is not a collection campaign — it is a re-encode of a corpus we already
own.

**Why this is a step-change candidate and not a tidy-up.** It is the same
class as S1: *structural observability*, not policy quality, and therefore
**opponent-independent — it does not inherit the clone ceiling** that §1 says
capped the last five milestones. Together S1 and S5 are the two places where
the agent is missing a faculty rather than missing training.

**Cost, honestly.** The vehicle is the one that works, not the one that has
always collapsed: `migrate_v3_to_v4` already exists and is exercised
(`m38_ft_init.pt`), and the M21 warm-start invariant zero-inits the new
columns so the migrated net is *bit-identically the old net at init*. So this
is warm-start + low-dose fine-tune with retention — M39's proven recipe — not
a from-scratch train (every ≤340k-row fresh train has collapsed).

**The one real technical risk, and it must be probed before this is
scoped.** `OppMemory`'s contract is locked to the LIVE prompt window:
*"observe() must be called EXACTLY ONCE per own prompt, before encoding"*,
with prefix-dedupe because sub-prompt chains re-deliver the previous window
verbatim. Replaying that offline over `iter_replay_decisions` is only valid
if the replay log stream is windowed the same way it is live. **If it is not,
the memory features would be silently wrong — an encoder that lies is worse
than one that is blind**, and this campaign has lost milestones to instruments
that lied. First deliverable of S5 is therefore a *fidelity probe*, not a
corpus: reconstruct memory features offline for a game we also have live and
assert they match.

**This re-prices E3.** The M40 draft proposed building an n-gram/naive-Bayes
archetype classifier to feed the racemode trigger. S5 gets the same
information into the net end-to-end, using tested code, with no new runtime
component and no BRExIt consumer problem. **E3 should not be scoped until S5
is priced** — and if S5 lands, E3 may be redundant.

### S6 — the plan head is a TRAIN/SERVE MISMATCH, and it has been one since M24

Found 2026-08-02 by asking "what else did we ship and then lose?". This is the
same regression date as S5 and the same root cause — the switch to replay-BC
corpora — but it is worse, because S5 is a *missing input* while this is an
*actively wrong* one.

**The evidence, in three facts:**

1. **The plan head has received no gradient in five milestones.** `plan_head`
   and `plan_enc` are **byte-identical** across `m28_winners` →
   `m38_w9294_cont3` → `m39_retain_b` → `m39_vsloss`, while `state_enc` and
   `option_enc` change with every fine-tune.
2. **Every corpus in the lineage trains at plan = 0.** Replay shards carry no
   `plans` column at all (`bc_m38_w9294`, `bc_m39_vsloss`, `bc_m39_br_*` all
   confirmed), and `BCDatasetV3` fills the default —
   [plan_iter.py:574](../rl/plan_iter.py:574): *"Old plan-less shards load
   with shaped defaults: plans=zeros (== 'no plan')"*. So 100% of training
   rows since M24 have a zero plan vector.
3. **The shipped pilot serves a NON-zero plan every turn.**
   [submission/main.py:218](../submission/main.py:218) `score_plans` →
   argmax → the chosen plan vector conditions the policy trunk at every own
   MAIN prompt.

**So the net is trained exclusively at plan=0 and served a non-zero plan
chosen by a head whose last gradient came from a different lineage, a
different deck, and the solver-teacher corpus M38 proved regresses the
champion at any dose.** The conditioned branch is out-of-distribution by
construction.

**Screen (n=200/cell, single-process so it is reproducible, seed 31):**

| bed | plan AS SHIPPED | plan ZEROED | delta |
|---|---|---|---|
| **wall** | 0.270 | **0.330** | **+6.0pp** |
| m28 (mirror) | 0.565 | **0.615** | **+5.0pp** |
| **top (900+)** | 0.505 | **0.550** | **+4.5pp** |
| grim | 0.675 | 0.655 | −2.0pp |

Three of four positive, including our worst matchup and the ceiling bed.
**SCREEN, not a measurement** — n=200 resolves ~14pp per cell (G-12) and no
single cell clears it. But the direction is consistent, and unlike a bare
win-rate wobble it has a mechanism established *before* the screen was run.

**Why this may be the best item in the plan.** It is a **removal**: one line
in the pilot, no retraining, no corpus, no new runtime code, and it applies to
every net in the lineage including `retain_b`. This campaign's largest live
gain to date — Ship A, **+115** — was also a removal, and removals transfer
because they do not depend on out-playing anyone.

**What the fix is not.** The plan machinery was not a mistake; the docstring
is explicit that plan-less data is meant to regularize *the plan=0 fallback*
while `plan_iter collect` data trains *the conditioned branch*. The
regression is that we stopped producing plan-carrying data at M24 and never
turned the conditioned branch off. Three options, in ascending cost:
(a) **serve plan=0** — one line, testable today; (b) retrain the head on
plan-carrying corpora — expensive, and the plan lane's value on this lineage
is unproven; (c) strip the plan machinery from the pilot entirely — bigger
diff, only after (a) settles the question.

**Standing check this earns (proposed G-14).** `ship_verify` checks weights,
deck, twin parity, fix-name resolution and now rule firing — **nothing checks
that the inputs the net is SERVED match the inputs it was TRAINED on.** A
one-time assertion comparing serve-time input statistics against the training
corpus would have caught this at M24 and would have caught S5 too. **Both of
this milestone's silent regressions are the same missing check.**

### S2 — self-play iteration (the only mechanism that exceeds demonstration)

M39 proved imitation is exhausted on this corpus (5 arms, 2 corpora, every
one negative on the 900+ panel). Self-play is the textbook answer, and it is
the one lane the campaign has never run properly. It needs, in order:

1. **E0 — the value net on the harvest corpus** (BACKLOG sweep #6, the
   enabler three parked items share). Pre-registered exit criterion: it must
   out-rank the outcome proxy **on the loss families** or the whole branch
   dies on day two rather than on a slot.
2. **A real exploration design.** M39's `bestresp` failed for a diagnosable
   reason: 8% uniform-random exploration produced a corpus the net already
   agreed with **93.5%** of the time, and it then lost to the very beds it
   was collected against. Policy-temperature or ε-over-top-k sampling, not
   uniform noise.
3. **Opponent diversity beyond three clones per family** — the Showdown paper
   hit opponent-distribution overfitting even with realistic partners.

Honest cost: this is the expensive lane and it may return nothing inside one
milestone. Online PPO stays parked (327M env steps ≈ 2 weeks of engine).

### S3 — build opponents ABOVE the clone ceiling, and re-measure the gap

**Prerequisite for believing anything S1 or S2 reports**, and the direct test
of §1's hypothesis. If every bed is 700-band, then S1's +23pp and any S2 gain
are measured against the wrong opposition and may not transfer either.

Cheapest construction, using what S1 builds anyway: **clone + solver
composites** (`solved:<bed_ckpt>:<bed_deck>`). If the composite beats its own
plain clone the way our composite beats ours, we have opponents meaningfully
above the clone ceiling for the first time, at zero new research.

Then re-run the P0.8 gap read against a solver-augmented top bed. Two
readings, both decision-relevant:

- the gap re-opens (we fall well below parity) → §1 confirmed, the campaign's
  offline numbers have been systematically optimistic, and **every historical
  bed absolute needs an asterisk**;
- the gap holds → §1 is wrong, our beds are honest, and the 0-6 live record
  needs a different explanation (matchup structure, or six unlucky games).

Either way M40 learns something the last three milestones could not.

## 3. CORRECTION: submission slots are not scarce, and never were

**Piotr, 2026-08-02: slots reset DAILY to 5.** Four remain today.

**This invalidates the premise of the last three planning documents.** The
M39 plan opens its budget section with *"Slots are the scarce resource;
calendar is not"* and derives the two-ship cadence, the reserve slot, and
part of G-7's justification from it. Drafts v1 and v2 of this stub inherited
it — and v1's recommendation to **bank +24 ELO rather than swing** was
driven *entirely* by a scarcity that does not exist. That recommendation was
wrong on a false premise, not on a judgement call, and Piotr's steer landed
on the right answer before the premise was corrected.

### What the budget actually is

| resource | quantity | binding? |
|---|---|---|
| submissions | 5/day × 42 days ≈ **210** | **no** |
| calendar | **42 days** to 2026-09-13 | yes |
| **decision-grade live reads** | **~3 / ~8 / ~15** at 10 / 30 / 55 games per day (n=150 per G-10) | **YES — this is the real constraint** |

**Slots were never the scarce resource. READ BANDWIDTH is.** A ship is free;
*knowing whether it worked* costs 3–15 days depending on accrual, and the
accrual rate is the one number that decides how many real experiments M40
gets. M38 accrued 54 games in a day, Ship A ~10 in its first hours — that
factor-of-five spread is the difference between 3 and 15 experiments and
should be measured, not assumed (see decision 3).

### What genuinely changes

1. **The "reserve slot" concept is deleted.** There is nothing to protect.
   The `retain_b × package` floor (+3.47pp, z=+7.44) is a config that can be
   shipped any day; it is a fallback in *time*, not in *budget*.
2. **A failed swing costs days of ladder time, not a slot.** The entire
   downside argument against aggressive attempts evaporates. Ship the swing;
   if the read is bad, ship the floor the next day.
3. **Exploratory ships become affordable for the first time in the
   campaign.** Configs that were never worth "a slot" — `raceash`,
   single-rule ablations, an S1 variant at a tighter solver budget — are now
   worth a day of ladder each.
4. **Live results can be REPLICATED.** G-12's core lesson is "replicate a
   surprising significant result before it enters a plan", and the campaign
   has never once been able to afford that live. A config can now be
   re-shipped for an independent second sample. Given that this campaign has
   twice misread the same matchup in opposite directions at n≤8, this may be
   the single most valuable thing the corrected budget buys.
5. **G-7 (one lane per slot) survives — its justification changes.** It was
   argued from slot scarcity; it should now be argued from *attribution*,
   which was always the real reason. With abundant slots it is close to free.
6. **G-10 (declared dwell) becomes MORE important, not less.** Cheap slots
   plus continuous reading is a multiple-comparisons machine. Declaring the
   read-n before the sample is the only thing standing between us and
   shipping-until-a-number-looks-good.

### And the capability this unlocks — the ladder as the high-band instrument

§1 argues our beds cap at ~700-band and that is why we cannot leave the band.
**The live ladder does not cap.** It is populated with real 800+ pilots — the
exact opposition no BC clone can represent. We have never been able to use it
as an instrument because a read cost a scarce slot; at 5/day it becomes a
usable, if slow and coarse, measuring device.

It cannot give per-decision gradients, and G-9 still forbids gating on
per-matchup live cells at any n. What it *can* now do:

- **settle S1 and S2 against real high-band opposition** rather than against
  clones we suspect are 200 ELO light;
- **replicate**, which turns a single live result from an anecdote into
  evidence;
- **test §1's own thesis directly** — ship a config whose offline gate is
  strongly positive and see whether the live delta matches the offline delta.
  Systematic offline-over-live inflation *is* §1, measured.

This deserves its own phase in the real plan (call it **S4 — ladder-as-
instrument**), and it may be worth more than S2.

## 4. Pre-registered kills (draft)

- **S1 does not replicate on a panel battery** (n≥1200/draw across ≥3
  families) → the screen was noise; drop it and S3 still stands on its own.
- **S1's max move exceeds the live per-move limit and cannot be tuned under
  it without losing the gain** → kill; a timeout is a lost game.
- **S5's offline memory features do not match the live ones on a shared
  game** → the replay log window differs from the live one; kill the
  re-encode rather than train on features that lie.
- **S6 does not replicate on a panel battery** → the plan head is inert
  rather than harmful; leave it and drop the lane. (Note the asymmetry:
  a null result here still argues for serving plan=0, because an
  out-of-distribution input with no measured benefit is pure risk.)
- **E0's value net cannot out-rank the outcome proxy on the loss families** →
  S2 dies with it.
- **S2 collection produces a corpus the net agrees with >90% of the time** →
  the exploration design failed again; kill at collection time, not after
  gating. M39's corpus B announced this as `init val_acc 0.935` and nobody
  read it as a kill signal until the gate agreed.
- **Nothing beats the `retain_b × package` floor (+3.47pp, z=+7.44) by
  decision day** → ship the floor. With daily slots that costs a day of
  ladder time, not an opportunity — which is the whole reason the swing is
  the cheap option and not the reckless one.

## 5. Standing guardrails

G-1…G-13 carry over unchanged. Three that M39 earned and M40 will lean on:
**G-11** (write the mechanism probe *before* the deciding battery — it
rescued `conserve`, now +115 live, and predicted the racemode2+4 synergy);
**G-12** (n=800 resolves ~10pp; S1's screen is n=30); **G-13** (no
single-clone bed — and the same scepticism now extends to E0's value net and
to any solver-composite bed).

## 6. Decisions for Piotr (unanswered)

1. **Sign off on reopening the search lane (S1)?** It contradicts a
   documented stop-invest line, on evidence I have argued is about a
   different intervention. It is also the cheapest large lever on the table
   and needs no new research. **Recommendation: yes — a panel battery costs a
   day and settles it.**
1a. **S6 — serve plan=0?** The cheapest item in the plan by a wide margin:
   one line, no retraining, and it applies to every net in the lineage.
   **Recommendation: run the panel battery immediately** — if it holds it
   is shippable the same day as a single-lane change, and daily slots
   mean that costs nothing.
1b. **S5 — re-encode the harvest corpus as v4 and warm-start onto it?**
   No stop-invest line to reopen and no new runtime component; the encoder,
   the memory module, the migration and the tests all exist and shipped
   once. **Recommendation: yes, gated on the fidelity probe passing** — and
   run the probe before anything else in M40, because it is a day of work
   that decides whether a whole lane exists.
2. **Does S2 run in M40, or does M40 spend its weeks on S1+S3 and hand S2 to
   M41?** Running both risks the M38 failure mode (a milestone losing its
   cycle to the attractive lane). **Recommendation: E0 yes regardless (cheap,
   and it is the enabler for everything downstream); full S2 only if E0's
   exit criterion passes AND S1 has already been settled.**
3. ~~Confirm the slot budget.~~ **ANSWERED — slots reset daily to 5.** See §3.
   The replacement question is the one that now sets M40's experiment count:
   **what is the actual game-accrual rate per day?** At 10/day M40 gets ~3
   decision-grade reads; at 55/day it gets ~15. That is the difference
   between a one-shot milestone and an iterative one, and it is measurable
   from the existing listings cache in an hour. **Do this first.**
4. **The target.** With this framing I am no longer recommending re-pointing
   to 820–850. If §1 is right, the ceiling was never a property of our
   policy, and 1000 stays the honest target. If S3 refutes §1, the target
   question returns with better information than it has now.
6. **Does the campaign switch to an iterative live cadence?** If accrual
   supports ~8–15 reads, M40 stops being "one big attempt" and becomes a
   *sequence* of single-variable ships read at declared dwell — which is what
   the guardrails were designed for and what the campaign has never been able
   to afford. **Recommendation: yes, if decision 3 supports it** — with the
   hard condition that every ship still declares its read-n in advance (G-10),
   because cheap slots plus continuous reading is exactly the
   multiple-comparisons trap that discipline exists to prevent.
5. **Nothing finalises before Ship B's n=150 read** — its wall cell carries a
   falsifiable mechanism prediction (opponent deck-outs ~30% live). It is
   also the cleanest available test of whether *any* bed-derived number
   transfers, which is §1's question in live form.

---

## BACKLOG disposition (draft)

| item | proposed M40 disposition |
|---|---|
| **Within-turn solver on the neural line** (not previously a BACKLOG item — it was a code comment) | **S1, headline candidate** |
| Value-net retrain / gen-2 (sweep #6) | **E0 — enabler, runs regardless** |
| Offline best-response (sweep #2) | **S2**, gated on E0 + a real exploration design |
| Advantage-filtered BC (sweep #1) | blocked on E0; **retention × α=0.25 is the free untested cross** M39 left behind |
| Retention (sweep #3) | **no longer an item — it is the default** |
| **Plan-head train/serve mismatch (S6)** — not a BACKLOG item; nothing was checking for it | **cheapest candidate in the plan.** Same M24 root cause as S5; fix is a one-line removal |
| **Encoder v4 adoption (S5)** — not previously a BACKLOG item; the regression was invisible | **headline candidate alongside S1.** M21 shipped it, M24 silently dropped it via `replay_bc`'s encoder choice, and the logs to rebuild the corpus are already cached (98.1% non-empty) |
| Opponent-deck inference (sweep #4) | **DO NOT SCOPE until S5 is priced** — S5 delivers the same information end-to-end with tested code and no new runtime component |
| ROIDA loser replays (sweep #5) | blocked on E0 |
| Online PPO (sweep #7) | parked — 327M env steps |
| Architecture inductive bias (sweep #8) | parked — needs a self-play-scale corpus (S2 could make one) |
| Heavy test-time search / DT / LLM self-improvement / deep equilibrium | **stop-invest, unchanged** — S1 is *not* this, and the plan says why |
| `rl/`↔`tcg/` duplication, W_COUNTER, `setup_plans_late` | housekeeping; calendar, not slots, is what it competes against now |
| **S4 — ladder-as-instrument** (new, §3) | the corrected slot budget makes the live ladder usable as the high-band opponent set no clone can be. May be worth more than S2. |
