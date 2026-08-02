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
inherits the ceiling — which is every lane M37–M39 ran. Three mechanisms do
not:

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

## 3. Slot structure — what makes the aggressive bet rational

The first draft framed this as "bank +24 or gamble". That was a false choice,
because **the measured gain does not expire**:

| slot | use |
|---|---|
| **3** | **the step change** — the best of S1 / S2 by the gate, on a panel roster that S3 has validated |
| **4 (reserve)** | **the already-gated floor: `m39_retain_b` + `conserve,racemode2,racemode4` = +3.47pp, z=+7.44, G-4 already paid** |

The +3.47pp config is measured, exported-ready, and will still be measured in
six weeks. Holding it in slot 4 converts the reserve from dead weight into a
real safety net, and that is precisely what makes spending slot 3 on a swing
the *conservative* choice rather than a reckless one. **Verify the slot count
first (decision 3) — the whole structure depends on it.**

## 4. Pre-registered kills (draft)

- **S1 does not replicate on a panel battery** (n≥1200/draw across ≥3
  families) → the screen was noise; drop it and S3 still stands on its own.
- **S1's max move exceeds the live per-move limit and cannot be tuned under
  it without losing the gain** → kill; a timeout is a lost game.
- **E0's value net cannot out-rank the outcome proxy on the loss families** →
  S2 dies with it.
- **S2 collection produces a corpus the net agrees with >90% of the time** →
  the exploration design failed again; kill at collection time, not after
  gating. M39's corpus B announced this as `init val_acc 0.935` and nobody
  read it as a kill signal until the gate agreed.
- **Nothing beats the slot-4 floor by decision day** → ship the floor from
  slot 3 and keep 4 in reserve. Not a failure; a correctly-priced bet losing.

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
2. **Does S2 run in M40, or does M40 spend its weeks on S1+S3 and hand S2 to
   M41?** Running both risks the M38 failure mode (a milestone losing its
   cycle to the attractive lane). **Recommendation: E0 yes regardless (cheap,
   and it is the enabler for everything downstream); full S2 only if E0's
   exit criterion passes AND S1 has already been settled.**
3. **Confirm the slot budget.** "4 remaining" entered at M39 drafting and has
   never been re-verified. The whole slot-3/slot-4 structure depends on it.
4. **The target.** With this framing I am no longer recommending re-pointing
   to 820–850. If §1 is right, the ceiling was never a property of our
   policy, and 1000 stays the honest target. If S3 refutes §1, the target
   question returns with better information than it has now.
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
| Opponent-deck inference (sweep #4) | strong slot-4 candidate if Ship B's wall claim transfers |
| ROIDA loser replays (sweep #5) | blocked on E0 |
| Online PPO (sweep #7) | parked — 327M env steps |
| Architecture inductive bias (sweep #8) | parked — needs a self-play-scale corpus (S2 could make one) |
| Heavy test-time search / DT / LLM self-improvement / deep equilibrium | **stop-invest, unchanged** — S1 is *not* this, and the plan says why |
| `rl/`↔`tcg/` duplication, W_COUNTER, `setup_plans_late` | housekeeping, only if slot 3 resolves early |
