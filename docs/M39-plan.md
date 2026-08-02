# M39 plan — build the instrument for 1000, recover the regression, seed the lane that can exceed demonstration

Drafted 2026-08-01 from the [M38 post-mortem](m38-post-mortem.md), a critical
review of the [M38 diary](M38.md) and [BACKLOG](BACKLOG.md).
**Status: P0 + P1 + G-13 + P2 COMPLETE. SHIP A SHIPPED (sub 55172160,
2026-08-01). SHIP B EXPORTED 2026-08-02 and awaiting Piotr's replay review —
NOT submitted.** Ship B is `m38_w9294_cont3` + fix string
`conserve,racemode2,racemode4` — a **one-lane rules change** (G-7), +2.37pp
weighted (z=+5.03) on the G-13 panel roster. P3 (policy lane) is gating; its
result feeds slot 3, not this ship. Live progress in [M39.md](M39.md).

**Revision 1, 2026-08-01 (BACKLOG pass).** Re-reviewed against the full
[BACKLOG](BACKLOG.md) including its 2026-08-01 research sweep, which the
first draft did not cover. Changes: finding #9 (the drafted AWR arm was a
verified no-op) and #10; a new BACKLOG-integration section dispositioning
every item; a hard action-masking constraint on the P1 strip; P2c
(opponent-deck inference, stretch); and a rewritten P3 running a harvested
and a manufactured corpus in parallel. New decision points 6–8.

**Revision 2, 2026-08-01 (ceiling re-point; Piotr's call).** Campaign target
raised from 800 to **implied ELO 1000**, with the attempt aimed at
**submission slot 3, not the next ship**. M39's job becomes the two things
that make that attempt real — a measuring instrument above 850, and a lane
that can exceed the demonstrator band — plus the cheap recovery of the M38
regression. P2c cut to M40. New slot-budget section and guardrail G-7.

**Revision 3, 2026-08-01 (decision scopes).** Corrected a material error in
P1: earlier drafts said deckguard/conserve was not part of the gacfr3 stack
and could be retained — but **gacf = deckguard/ash/conserve/benchfloor**,
the exact rules G5 measured at .278 vs plain's .323, so "keep the economy
rules" is an *unmeasured* config, not a conservative one. Added P0.7 (6-cell
per-rule ablation) to fund the strip decision with attribution instead of an
aggregate. Decision points rewritten with scope / options / recommendation /
rationale, grounded in the campaign's own measurements.

**Revision 4, 2026-08-01 (validation protocol).** Computed confidence
intervals on the live ship records for the first time: **M37 [646, 819] and
M38 [566, 753] overlap across nearly their full width**, and the wall cell
the M38 post-mortem headlined (3-6 → 1-5) is Fisher p=0.60. At n=50 a live
read detects only a ~200 ELO swing. New standing doc
[VALIDATION.md](VALIDATION.md) (four-tier pre-ship protocol, pre-registered
post-ship read schedule); new guardrails G-9/G-10/G-11; "converged"
redefined from n≥50 to **n≥150**. This does not change the plan's
direction — that rests on offline gates at n=800 and mechanism forensics —
but it changes what we may conclude from every live read.

**Revision 5, 2026-08-01 (Ship A shipped).** P0 closed; Ship A submitted as
**conserve-only, not the full strip decision 1 had settled** — the powered
3-arm gate inverted it (+1.74pp vs live, z=+3.42) after a mechanism probe
rescued a rule a significance threshold had discarded. Two new standing
guardrails from P0's own failures: **G-12** (offline power / matchrunner is
not run-reproducible) and **G-13** (bed strength is a training lottery;
beds must be panels). P0.9 instrument extension delivered a negative result.

## Status at a glance (2026-08-01)

| phase | state | outcome |
|---|---|---|
| **P0** measurement integrity | ✅ **COMPLETE** | roster was invalid at 77.5% coverage → archaludon bed built; stall bed killed as unbuildable; instrument built and read (**gap ≤10pp = best pre-registered branch**) |
| **P0.7** strip ablation | ✅ COMPLETE | full-strip verdict — later **overturned** by P1-inv |
| **P0.8** the instrument | ✅ COMPLETE | `m39_bc_top`, 329 seats, val .682 |
| **P0.9** instrument extension | ✅ COMPLETE, **negative** | `m39_bc_topgrim` built but no harder than the 750 bed; wall high-band unbuildable |
| **P1 / Ship A** | 🚀 **SHIPPED — sub 55172160** | cont3 + `conserve`; +1.74pp weighted vs live (z=+3.42); dwell declared n=150. **Live read 2026-08-02: n=10, 5W-5L, implied ELO 741, CI [557, 926] — no verdict permitted (G-9)** |
| **D1** matchup-filtered bed | ✅ COMPLETE | `--opp-deck-hash` landed (+2 tests); found **G-13: bed strength is a training lottery** |
| **G-13 bed panels** | ✅ **IMPLEMENTED 2026-08-02** | 7 new draws (2.5 min total); `m39_panel_gate.sh` + panel support in `m39_decide.py`. First run: the wall panel spreads **10.8pp** and `m38_bc_wall` is the SOFTEST draw |
| **P2** anti-deck-out | ✅ **COMPLETE — SHIPS** | `racemode2` alone KILLED (−1.3pp on wall); `racemode4` +1.48pp; **the pair +2.37pp weighted, z=+5.03, wall panel +11.2pp**. Synergistic — neither ships alone |
| **P3** targeted fine-tune | ✅ **COMPLETE** | Both corpora LOSE alone (A −3.10pp, B −0.85pp); **both are rescued by champion-shard mixing (P3-C): retain_a +0.80pp, `retain_b` +1.85pp z=+4.01**. The binding constraint was **catastrophic forgetting**, not volume or demonstrator band — a third answer the A/B diagnostic did not enumerate. **α-ladder NOT null**: α=0.25 beats α=0 by +2.28pp, qualifying M28's winners-only law. **Ceiling trigger FIRES — no arm moved the 900+ panel** (all five negative), so per the pre-registered branch **slot 3 is a self-play / enabler-debt milestone, not another corpus iteration** |
| **P4 / Ship B** | 🚀 **SHIPPED — sub 55182097** | **rules-only, one lane** (G-7): cont3 + `conserve,racemode2,racemode4`; +2.37pp weighted (z=+5.03); dwell declared n=150. **Ship A came back at leaderboard 780.4 vs M38's 664.8 (+115.6)** — M39 meets its own pre-registered success bar ("780 with a clear slot-3 thesis") before Ship B reports |

**Four of my own conclusions were corrected mid-milestone**, all the same
shape — a small-n measurement read as if it were settled. They produced
G-12 and G-13 and are recorded in [M39.md](M39.md) rather than quietly
fixed: the "−5pp rule stack at the top band" (didn't replicate at n=2400),
"+11.2pp cont3 over champion" (+6.0pp powered, and ≈0 on the weighted mix),
"BC cloning saturates" (refuted at z=+12.04), and "conserve is inert vs
grim" (it fires there most of all).

## Objective

**Campaign target: implied ELO ≥ 1000, converged (n ≥ 150, per G-10).** The top pilot
of our exact deck list (hash 9294d9d8) sits at **1058**, so the target is
"match the best known pilot of this deck" — attainable in this vehicle, and
not attainable by the next ship.

**The arithmetic of 1000.** Implied ELO ≈ `avg_opp + 700 × (WR − 0.5)` —
the formula reproduces M38 exactly (672.2 + 700 × −0.019 = 659):

| your avg_opp | WR needed for 1000 |
|---|---|
| 800 | 0.79 |
| 850 | 0.71 |
| 900 | 0.64 |

Against that: **0-6 in the 800+ band** across M37+M38, 0.43 in the
700-band, currently 659. M35 peaked at 817 on a hot pool and the campaign
has never held 800.

**Why 1000 is not a next-ship target, and what that implies.** Two
structural blockers, both of which M39 exists to attack:

1. **We cannot see the band we are targeting.** Every bed we own is a
   700–850 clone. Our entire knowledge of 800+ is a 6-game, zero-win
   sample — equally consistent with "5pp short" and "hopeless," which imply
   completely different milestones. Fixing this costs **zero submission
   slots** and is the highest-information action available (P0.8).
2. **Imitation cannot reach 1000.** BC's ceiling is its demonstrator band
   and it lands below it. Our corpus tops at 1058 with 231 seats @800+, so
   even excellent BC on it projects to ~800–850 — and M38 already spent a
   cycle proving the in-family version of that. Breaching 1000 needs a
   mechanism that *exceeds* demonstration: offline best-response (P3-B) or
   self-play iteration (M40).

**Odds, stated for the record so the milestone is judged honestly:**
breaching 1000 on the next ship, **<5%**. Within the remaining 4 slots,
**~20%** with strong execution — and that estimate is dominated by what the
P0.8 instrument reports, which we can know in ~2 days.

**M39's own ship objective** (distinct from the campaign target): recover
the M38 regression, close the measurable loss mass, and hand slot 3 an
instrument plus an attribution model. The M38 loss anatomy prices the
near-term levers:

| lever | live evidence (M38, 54 games) | plausible gain |
|---|---|---|
| wall+grim+stall block | 2-13 (~13% WR, ~28% of games) | → 50% ≈ +11pp WR ≈ +75 ELO |
| deck-races from winning positions | 5 losses (18% of all losses) | subset of above + mirror/garchomp |
| 700-band sweeps (43% of losses) | policy-quality ceiling | the remaining ~+40–60 ELO |

**Conclusion the plan is built on: the rules/econ lane alone caps out around
~740–760, and imitation on the current corpus caps around ~850. Neither
reaches 1000; both are prerequisites for the slot-3 attempt that might.**
Lanes ship separately so each is measured live (M38's single-shot ship left
them confounded until the post-mortem untangled them).

## Submission-slot budget (4 remaining)

Slots are the scarce resource; **calendar is not** — M38 shipped 07-31 and
had 54 ladder games by 08-01, so a converged read costs ~1 day. That makes
sequential single-variable ships nearly free in time, and it is why the plan
spends a slot on attribution rather than bundling.

| slot | ship | milestone | rationale |
|---|---|---|---|
| ~~1~~ | ✅ **SPENT — Ship A, sub 55172160** (cont3 + `conserve`) | M39 | shipped 08-01; decision read declared at n=150 |
| 2 | **Ship B — best P3 net × surviving P2 rules** | M39 | closes the measurable loss mass |
| 3 | **the 1000 attempt** | M40 | funded by the P0.8 instrument + slots 1–2 attribution |
| 4 | **untouched reserve** | — | insurance; never spent on a hypothesis |

Everything else in M39 — the high-band beds, both P3 corpora, every training
arm — is **offline compute, not slots.** It costs machine time and cannot
harm a live agent. That asymmetry is the reason the arm matrix is large
while the ship count is small.

## Critical-review findings that shape this plan

From auditing the M38 diary and BACKLOG against the post-mortem — these are
the *process* defects; the post-mortem already ranks the *product* defects.

1. **A pre-registered decision rule fired and was overridden.** G5 measured
   the rule stack at −7pp on the wall bed; decision-3's trigger ("strip only
   if actively harmful") was explicitly MET in the diary — and gacfr3
   shipped anyway on minimal-diff grounds. Live wall went 1-5. Guardrail G-1
   below makes triggers binding.
2. **The pooled gate was honest but mis-weighted, and the data to weight it
   existed at gate time.** The +6.1pp was carried by iono/dragapult
   (+14/+17pp, ~7% of live games); re-weighted by the M37 live mix (already
   in the M37 post-mortem) the delta was ≈0. This was knowable before
   submit. Guardrail G-3: the weighted pool IS the gate number.
3. **Watch items were picked by offline z-score, not live share.** Rocket
   (−6pp offline, 0 live games) was THE watch item; wall/grim/stall (28% of
   live games) had none. Guardrail G-2.
4. **QC predicted nothing.** 11W-1L against a roster covering zero of the
   top-3 live loss families. The multi-deck QC rule was followed to the
   letter and still measured the wrong thing.
5. **Bed *construction* age was exempt from the drift law.** The grim clone
   was re-pinned (numbers refreshed) but the clone itself is an M26-era
   600-band artifact; live grim pilots at 738–838 flipped 6-2 → 1-6. Third
   bed-fidelity transfer failure (garchomp M35, wall-solver M37, grim M38).
   Guardrail G-6: beds used in ship gates must be ≤1 meta old.
6. **The M37 ranked recommendations lost to the attractive lane.** M38 spent
   its cycle on the teacher-fix (H1) — which delivered instruments and two
   clean kills but zero live ELO — while rec #2 (racemode4) was never built
   and rec #4 (AWR) never launched. M39 schedules directly from the
   loss-mass ranking.
7. **Post-mortem rec #4 (cross-family fine-tune) contains a vehicle error
   this plan corrects.** BC imitates the demonstrator's *seat*: fine-tuning
   our alakazam policy on grim/wall winner seats teaches piloting *their*
   decks — action distributions we never face — the same
   demonstration-doesn't-transfer failure as the teacher distill, dressed in
   live data. The transferable corpus is **same-deck alakazam winner seats
   filtered to the losing matchups** (how strong alakazam pilots beat grim/
   wall/stall). Cross-family seats are harvested for **beds only** (P0),
   never for labels (P3).
8. **BACKLOG re-scopes** (strike/move on milestone start): "Rule-stack
   strip" → P1. "Kanga-only wall blindspot" → P2a, generalized. "racemode4"
   → P2b. "Universal margin raceconserve" → subsumed by P2a (post-mortem
   concurs). "Band gate refresh" → P0 ride-along. **Grim entry on the
   stop-investing list: REVOKED** (1-6 live is new evidence; the stop-list
   logic itself worked as designed). **"Gen-2 collect + value-net retrain"
   needs re-scoping, not just deferral**: M38's central negative result —
   solver-teacher labels cannot beat the winners lineage at any dose — cuts
   the value of any further solver-teacher corpus. If a value-net retrain
   returns, it trains on the harvest corpus. W_COUNTER stays parked (E0b).
9. **The drafted AWR arm was a no-op — verified in code, not inferred.**
   `--outcome-weight ALPHA` ([plan_iter.py:1243](../rl/plan_iter.py:1243),
   impl [plan_iter.py:907](../rl/plan_iter.py:907)) scales rows where
   `results < 1` — i.e. seats that did NOT win. Its own docstring states
   "alpha=0 reproduces winners-only exactly". The drafted P3 corpus was
   **winner seats only** (as `bc_m38_w9294` was —
   [post-mortem:103](m38-post-mortem.md)), so every row has `results == 1`,
   the mask is empty, and `m39_vsloss_awr` would have trained a
   bit-identical net to `m39_vsloss` at full cost. Corrected in P3: the
   vs-loss corpus is built **without** `--winners-only` and alpha becomes a
   real ladder (α=0 ≡ the winners-only baseline). This also delivers most
   of BACKLOG #5 (use the discarded loser seats) at zero harvest cost.
   *(Note the separate v2-arch hook `--weighting none|score|winner`,
   [bc.py:544](../rl/bc.py:544) — same degeneracy on a winners-only corpus.
   Do not confuse the two.)*
10. **The plan was drafted against the BACKLOG's *deferred* section and
    skipped its *research sweep*** (BACKLOG lines 53–196, added the same
    day). Four of its high-priority items bear directly on M39's two
    weakest points — the thin-corpus risk in P3 and the ~1-epoch dose
    ceiling — and one of its adopted negative results is a safety
    constraint on P1. Integrated below.

## BACKLOG integration (revision 2026-08-01)

Every BACKLOG item, dispositioned. The selection rule: pull in what plausibly
moves **the 29% deck-race block or the 43% sweep mass** this milestone at
costs the schedule can absorb; park research-grade items behind their
enablers. Strike these lines from BACKLOG.md at P0 launch.

| BACKLOG item | disposition in M39 | why |
|---|---|---|
| Rule-stack strip | **P1** (Ship A) | already scheduled |
| Kanga-only wall blindspot | **P2a**, generalized | already scheduled |
| Silent fix-name ignore | **P1 ride-along** | we edit fix strings this cycle |
| Band gate refresh | **P0 ride-along** | already scheduled |
| Sweep #1 advantage-filtered BC | **P3, corrected** (finding #9) | the drafted arm was null; α-ladder is the fix |
| Sweep #2 offline best-response vs loss beds | **P3-B, promoted to a parallel pre-registered arm** | see below |
| Sweep #3 retention-regularized fine-tune | **P3-C, cheap arm** (data mixing only) | attacks the dose law that caps every policy gain |
| Sweep #4 opponent-deck inference | ~~P2c~~ → **CUT to M40** (revision 2) | only item adding new runtime code to a ship; too expensive at 4 slots |
| Sweep negative: action masking is load-bearing | **P1 hard constraint** | prevents a catastrophic strip |
| Sweep #5 ROIDA loser replays | partial, free (P3's α-ladder keeps loser rows); PU-discriminator + TD version stays parked | needs the value net |
| Sweep #6 value-net retrain (+ gen-2 collect) | **deferred to M40**, now named as the blocking enabler | see "enabler debt" below |
| Sweep #7 online PPO best-response | parked; **P3-B is its offline approximation** | 327M env steps ≈ 2 weeks of engine |
| Sweep #8 architecture inductive bias | parked; P3-B is a candidate enabler (manufacturable corpus) | every ≤340k-row fresh-train arm has collapsed |
| Deep equilibrium search / test-time search / DT / LLM-agent self-improvement | **stop-invest, cited** | do not spend a milestone here |
| `rl/` ↔ `tcg/` duplication | parked (housekeeping milestone) | not a ride-along |
| W_COUNTER retune, `setup_plans_late` re-eval | parked (E0b / gen-2) | no gen-2 collect this milestone |

### The three substantive additions

**P3-B — offline best-response (BACKLOG sweep #2), promoted from "fallback"
to a pre-registered parallel arm.** The drafted P3 has a single point of
failure the plan itself admits: the harvested vs-loss corpus may be thin,
and we do not learn *how* thin until the P0 census returns. If it comes back
at 20 seats, the entire policy lane stalls with no plan B in flight and M39
degenerates into a rules-only milestone capped at the ~740–760 the objective
section already rejects. P3-B removes that failure mode: `plan_iter collect`
games against `m38_bc_wall` / `m39_bc_grim` / `m39_bc_archaludon`, keep the
winning seats, low-dose fine-tune. **No new infrastructure** — collect,
filter, and fine-tune all exist and were exercised in M38. Corpus size
becomes manufacturable on demand against precisely the 2-13 loss block.
Evidence: BC-init + best-response vs a FIXED opponent took an exploiter
42%→90% (arXiv 2404.16689); synthetic self-play data carried the Showdown
agent 58%→64–80%.

Two cautions, both pre-registered as gate conditions: (i) **exploiter
overfit is measured and real** — 0.90 at 32 decks collapsed to 0.54 at 1024
in the same paper — so P3-B gates on the **full weighted roster (G-3),
never the target bed**, and a target-bed-only win is a KILL, not a ship;
(ii) we would be imitating **our own net's** winning seats, so in-family
label quality caps the ceiling (M38's demonstrator-strength lesson). That
second caution is exactly why P3-B is a *parallel* arm and not a
replacement: the harvested corpus has 800–1058-band demonstrators, P3-B has
on-demand volume at champion-band quality. Which axis matters more is the
open question, and running both answers it.

**P3-C — retention by data mixing (BACKLOG sweep #3), one cheap arm.** The
~1-epoch dose law is the binding constraint on every policy gain in this
campaign, and M38's w9294_cont 10-epoch collapse is a textbook
state-coverage gap (arXiv 2402.02868, ICML 2024 Spotlight: retention lets
the full transfer happen). The cheapest form needs **no new loss code** —
blend champion-corpus shards into the fine-tune via the existing multi-dir
`--data` path ([plan_iter.py:1202](../rl/plan_iter.py:1202), used as
`m38_gen1_{a,b}`). One arm, one hyperparameter (mix ratio), and it is the
only lane that would let P3-B's manufacturable volume actually be absorbed
— today a bigger corpus is worthless to us because we can only spend one
epoch on it. Per the source: **EWC underperformed BC-retention in both
testbeds — do not try it**; and a KL anchor can destroy the gains the
fine-tune was for (StratFormer, single source), so v1 is pure data mixing,
no anchor.

**~~P2c~~ — opponent-deck inference (BACKLOG sweep #4). CUT to M40 in
revision 2; rationale retained here because the case for it is still good
and M40 should pick it up unchanged.** The racemode/conserve triggers key on **visible board
ids**, which is why Fan Rotom went 0-2 as an unrecognized variant. P2a fixes
that by enumerating more ids — a patch that is one new stall variant away
from failing again. A classifier over the opponent's *played cards* fires
conserve **before** the wall/stall board shows. In a deck race decided by
cards burned across the whole game, firing 3–4 turns earlier plausibly
captures a large share of P2b's ~5–6 card ceiling, and it generalizes to
variants we have never seen. Mechanism is cheap and CPU-trivial (n-gram /
naive Bayes over played card ids; >95% top-1 by turns 3–5 on 50k Hearthstone
replays), trains on the existing replay cache, and the meta is
archetype-few. BRExIt's warning is satisfied by construction: **the consumer
ships in the same milestone** (it is P2a's trigger, not an auxiliary head).

Scope discipline, per finding #6 — the attractive lane is how M38 lost its
cycle. With 4 slots left it was cut outright rather than left to a date
check, because it is the only addition that would put new runtime code into
a shipped agent. Its v1 remains deliberately conservative: it may only fire a
trigger the id-set would eventually fire anyway (strictly *earlier* firing
of an already-validated rule, never a new behavior), confidence-gated to
suppress early-game misprediction. That makes it a timing change with a
one-line off switch, not a new policy surface.

### Enabler debt (explicit, for M40)

Three parked sweep items share one blocker: **a value net trained on the
harvest corpus** (#6). Without it we have no per-decision advantage estimate,
so #1 degrades to outcome-as-advantage-proxy (what P3's α-ladder does), #5's
TD component cannot be built, and #7 has no critic. If M39's policy lane
underperforms, the M40 fork is not "more corpora" — it is paying this debt.

## Deck decision (2026-08-01, discussed with Piotr): STAY on alakazam — with a falsification test

Question raised: should M39 swap decks, and would the planned pilot work
transfer if we ever do?

**Call: stay.** Grounds:

1. **The deck is not the ceiling.** The harvest shows other pilots flying
   the exact same list (hash 9294d9d8) to 947 / 977 / **1058**, and
   alakazam+kadabra is the top-scoring family in the meta recon (903+). We
   are targeting 800 in a vehicle that demonstrably cruises past 1000 in
   other hands — the gap is the pilot.
2. **Our best asset is deck-bound.** The 800–1058-band winners corpus (231
   seats) is the richest demonstration pool we've found for ANY deck (the
   wall corpus tops out ~830). Imitation's ceiling is the demonstrator
   band; alakazam has the highest one available.
3. **Swap cost vs stay cost is asymmetric and shrinking.** A swap resets
   the corpus, the m28→cont3 lineage, the mirror 7-2 (~17% of live games),
   and the rules stack. Meanwhile M38's harvest+clone pipeline cut the
   cost of a future swap from weeks to days — staying now is low-regret.

**The honest counterargument, kept live:** the ~2.5 cards/turn engine burn
is deck-intrinsic; the wall/stall deck-race weakness comes bundled with the
deck. The 900+ pilots manage it — but "manage or dodge" is an open question
the P3 corpus build answers for free:

- **Falsification test (rides on P3, no extra cost):** the vs-loss corpus
  census — how many 800+ alakazam winner seats exist AGAINST
  grim/wall/stall? Plenty → the matchups are winnable with better lines,
  stay-decision confirmed. Nearly none → top pilots lose or dodge them too;
  the econ package (P2) is the only lever, and a deck hedge becomes an M40
  agenda item with data behind it.
- **Optional hedge (P5, default OUT of scope — Piotr's call):**
  clone-scouting. The `m38_bc_wall` recipe (138 seats → a pilot that beats
  our champion 2:1 in its matchup) can BC-clone the best wall or dragapult
  seats and run them on the weighted roster — a day of compute for an
  actual number on "what a swapped-deck pilot would be worth." Parked in
  BACKLOG unless Piotr pulls it in.

### Transferability of the M39 work (if a swap ever comes)

- **Fully deck-agnostic** — all of P0 (harvest snowball, clone-bed
  construction, weighted gate, QC roster, bed-era guardrail), the campaign
  laws (dose ~1 epoch, winners-lineage-not-teacher-distill, re-measure
  rules per net), and the P3 *vehicle* (low-dose fine-tune on same-deck
  winner seats works for any deck with harvestable winners).
- **Transferable mechanism, deck-specific parameters** — the P2 rules
  lane. Trigger sets are keyed to OPPONENT ids (meta-side: Crustle, Tusk,
  Trevenant, Fan Rotom) and transfer to any deck we pilot; the demote
  lists are OUR card ids (Enriching Energy, Poké Pad, Dawn, Sacred Ash…)
  and would be re-instantiated per deck. Roughly 70% pattern / 30% this
  engine. A lean-engine deck might not need racemode4 at all.
- **Not transferable** — checkpoints and corpora (the learned policy is
  alakazam lines; a new deck means fresh BC from its winner seats, not a
  fine-tune of cont3). And one thing that follows us deck-independently in
  the wrong way: the 43% sweep mass (supporter under-play, slow setup) is
  a property of BC imitation quality, not of this deck — swapping does not
  escape the P4 design question.

## Process guardrails (standing, binding from M39 on)

- **G-1 Binding triggers.** A pre-registered decision trigger that fires
  flips the default. Overriding requires a diary entry titled `OVERRIDE`
  with rationale, written *before* export.
- **G-2 Watch items by live share.** Ranked by (live opponent share ×
  offline effect), never offline z alone. Every family ≥10% of the last
  live sample gets a watch item or an explicit waiver.
- **G-3 Weighted gate.** Ship gates pool per-bed WR weighted by the live
  opponent mix (config committed as `data/m39_live_mix.json`, from the
  pooled M37+M38 samples: lucario ~.19, mirror ~.17, grim ~.13, wall ~.11,
  archaludon ~.09, dragapult ~.07, stall ~.04, iono ~.02, rest ~.18). The
  weighted number is THE gate; unweighted reported for continuity. A gate
  whose bed roster covers <80% of live-mix mass is invalid.
- **G-4 Re-measure rules against every new net.** Any net change re-runs the
  G5-style on/off matrix before ship — the M38 law: rules tuned on an old
  net's behavior can actively harm a new one.
- **G-5 Pool-drift control.** The previous sub stays in the monitor as the
  in-window comparator; a new ship is judged against the parent's
  same-window ELO, not its frozen number.
- **G-6 Bed era check.** Every bed in a ship gate must be built from the
  current or previous meta's live seats, or be re-cloned first.
- **G-7 One lane per slot.** With 4 submissions left, no ship changes more
  than one lane (net / rules / config) unless the previous ship's read is
  converged at its declared dwell (G-10). At ~3 days per read this costs no slots and is
  exactly the discipline M37/M38 lacked — M38 shipped a new net *and* a
  rule stack its own gate said to strip, and the slot returned no
  attribution until the post-mortem untangled it.
- **G-8 Band coverage.** No gate may claim evidence about a band it has no
  bed for. Claims about 800+ performance require the P0.8 high-band beds;
  before those exist, the honest statement is "unmeasured," not "expected
  to hold."
- **G-9 Statistical honesty.** No live claim without a 95% CI. No gating
  decision on a live cell with n < 30, and **no matchup-level gating from
  live data at any n** — offline beds are the instrument for matchup
  claims. See [VALIDATION.md](VALIDATION.md) §0: M37 and M38's implied-ELO
  intervals overlap across nearly their full width, and the wall cell the
  M38 post-mortem headlined (3-6 → 1-5) has Fisher p=0.60.
- **G-10 Dwell time — declared per ship, before submit** (Piotr's call
  2026-08-01: per-ship, not a fixed rule). Each pilot **declares its
  decision-read n in the ship commit, before it goes live**, chosen from
  the CI table in [VALIDATION.md](VALIDATION.md) §0 against the effect that
  ship is trying to detect. The declaration is what makes this safe: the
  hazard in a per-ship call is reading continuously and stopping when the
  number looks good, which is a multiple-comparisons machine. Declaring
  the target in advance keeps the flexibility and removes the hazard.
  Reading *early* is always allowed for catastrophe detection and mechanism
  probes; what is fixed in advance is the n at which a **verdict** may be
  taken. Reference points: n≈54 → ±93 ELO, n≈150 → ±56, n≈250 → ±43.
  **The campaign-target claim (implied ELO ≥ 1000) requires n ≥ 150
  regardless** — that claim is too large to rest on a ±93 interval.
- **G-12 Offline power** (added 2026-08-01 from P0.7). `matchrunner
  --workers 8` is NOT run-reproducible — the seed is a label, not a pin, so
  every cell is an independent sample. The campaign's habitual **n=800
  2-seed cell resolves only ~10pp**, while most rules effects we chase are
  ~5pp (needs n≥2400). A delta inside a cell's MDE is "no signal", never
  "no effect" and never a confirmation. Replicate a surprising significant
  result before it enters a plan or a ship decision, and state how many
  contrasts were tested. See [VALIDATION.md](VALIDATION.md) §2b.
- **G-13 Bed panels** (added 2026-08-01 from D1). **Bed strength is a
  training lottery, not a property of a corpus.** Three clones of the same
  grim list from near-identical data scored .671 / .623 / .546 against the
  same net at n=2400 each — a **12.5pp spread against a 2.0pp binomial CI**;
  one fewer training epoch on identical data moved it 4.8pp. Therefore: no
  gate bed is a single clone. Rosters use **≥3 draws per family**, the panel
  mean is the cell value, and the spread across draws is reported. The fix
  costs nothing — it is a *redistribution* of match budget (n=800 against
  each of three draws rather than n=2400 against one). Gate **deltas** are
  largely insulated (the same fixed bed serves both arms, so the draw
  cancels); **absolute** bed numbers are not, and carry ~±6pp until this
  lands. **NOT YET IMPLEMENTED — see the P2/P3 prerequisite note.**
- **G-11 Mechanism proof per ship.** Every ship carries a fire /
  behavioral-diff probe for its changed behavior, run pre-ship offline and
  re-run post-ship on live replays. A ship whose mechanism cannot be
  probed is a ship whose live result cannot be attributed.

## Phases

### P0 — Measurement integrity ✅ **COMPLETE 2026-08-01** (was BLOCKING)

**Status: all 8 items done. Results in [M39.md](M39.md); four findings
changed the plan and one changed Ship A's premise.**

| # | item | outcome |
|---|---|---|
| P0.1 | harvest round 3 | ✅ +600 episodes; seats 2594 → 3908. Corpus A now ≈127 loss-family winner seats, **over the ≥100 target** |
| P0.2 | new beds | ✅ `m39_bc_grim` (195, val .662) + **`m39_bc_archaludon` (137, val .744, unplanned but forced)**; ~~`m39_bc_stall`~~ **killed — unbuildable** |
| P0.3 | live mix | ✅ `data/m39_live_mix.json`; **planned roster measured at 77.5% coverage — below the G-3 floor**, which forced the archaludon bed |
| P0.3b | gate harness | ✅ `m39_gate.sh` + `m39_decide.py`; coverage 82.3%; decoder fixed to carry a CI on the weighted delta |
| P0.4 | fresh pins | ✅ champion / cont3 / shipcfg across 10 beds |
| P0.5/6 | QC legs + band refreeze | ✅ wall/grim/archaludon/top900 QC bundles; `band_decode` refrozen + marked superseded |
| P0.7 | strip ablation | ✅ **FULL STRIP** per the pre-registered rule; no rule cleared the keep-bar |
| P0.8 | THE INSTRUMENT | ✅ `m39_bc_top` (329 seats, val .682); **gap ≤10pp = the best pre-registered reading** |

**P0.9 (added 2026-08-01 on Piotr's call to extend the instrument):**
`m39_bc_topgrim` built from grim hash 3121746f at score >=1000 (513 seats
cached, 200 extracted, val 0.674). It is the SAME list as `m39_bc_grim`
(700-850), so the pair forms a controlled pilot-skill ladder -- identical
deck, two altitudes -- which no previous instrument could separate. A wall
high-band bed is **not buildable**: 57 and 35 seats on the two candidate
lists, under the same 100-seat floor that killed stall.

**P1-inv (Piotr's call: investigate Ship A further, 2026-08-01):** three
candidate ship configs -- `plain`, `conserve-only`, `shipcfg` -- across the
full 9-bed weighted roster at 6 seeds (n=2400/cell), taking the weighted-pool
MDE from 2.5pp to ~1.4pp. `conserve` is settled by gate cell rather than
judgement. **Mechanism probe already in:** `scripts/conserve_probe.py` shows
conserve fires 13x/12 games on the 900+ mirror bed and 8x on mirror, but
**0 times on wall and 0 on grim** -- and the WR cost of removing it tracks
that exactly (-2.75pp on top, -0.29pp on wall). An inert rule cannot cost
2.75pp, so the P0.7 ledger's "strip conserve" line is likely to be
overturned on evidence.

**The finding that changes the milestone:** the Ship A strip shows
**−1.10pp ±1.73 (z=−1.24) on the weighted pool — no detectable
difference.** It helps on wall (+2.5pp pooled, z≈+2.2) and leans mildly
negative on lucario and mirror. Ship A's premise ("the cheapest real gain
available") is **not supported**; the decision on whether to spend slot 1
on it goes to Piotr.

**Standing methodological result → new guardrail G-12:** `matchrunner
--workers 8` is not run-reproducible (five identical runs at seed 1:
0.471/0.494/0.526/0.506/0.537), so the campaign's habitual n=800 2-seed
cell resolves only ~10pp while most rules effects are ~5pp. Two of my own
P0 conclusions were corrected as a direct result.

---

#### Original P0 specification (retained for the record)


The post-mortem's P0, plus the guardrail infrastructure. Everything M39
measures depends on this; it goes first and alone.

1. **Harvest round 3** (snowball from the M38 sample's sub ids): grim/marnie
   winner seats at 738–838 (CoCoSh, yujinki, Hamachi…), fan-rotom and
   hop-stall seats, 800+ mirror seats, wall top-ups. Also chase the
   903–1058 mirror band (P3 feedstock).
2. **New beds** — **REVISED BY MEASUREMENT 2026-08-01, see [M39.md](M39.md)**:
   `m39_bc_grim` ✅ built (195 seats @≥700, val 0.662 vs the M26 relic's
   0.565; relic retired per G-6 with a MILESTONES epoch marker).
   ~~`m39_bc_stall`~~ **KILLED — unbuildable**: 9 seats at band, 22 across
   all bands; the "combined stall bed" fallback does not save it, since
   pooling bands is the exact infidelity that made the M26 grim clone lie.
   **`m39_bc_archaludon` ADDED** ✅ (137 seats, val 0.744) — archaludon
   measured at **12.2% of live games with no bed**, and without it the
   roster covers 77.5% of live-mix mass, which G-3 makes INVALID. Keep
   `m38_bc_wall` (validated live-faithful).
3. **Weighted gate harness**: `data/m39_live_mix.json` + weighted pooling in
   the gate script (extend `scripts/m38_gate.sh` → `scripts/m39_gate.sh`).
4. **Fresh pins** on the full roster: M38 ship config (cont3+gacfr3),
   cont3 plain, plain champion. These are the baselines every M39 arm is
   judged against.
5. **QC battery fix**: add wall, grim, and stall legs (3 games each) to
   `scripts/qc_battery.py` alongside tuned/iono/dragapult + previous-ship
   mirror — QC must cover the top-3 live loss families forever after.
6. Ride-along: re-freeze `band_decode.py` on the current 700-band sample.
7. **Per-rule strip ablation (new in revision 3, funds decision 1).** G5
   measured the stack in aggregate — `plain` / `gacf` / `gacfr3` — and its
   own entry says per-rule attribution beyond racemode3 is **incomplete**.
   Six leave-one-out cells on `m38_bc_wall` at n=400 (drop each of
   telepath / deckguard / ash / conserve / benchfloor / racemode3 from the
   full stack), plus the plain and full pins we already have. Cheap offline
   compute, no slots, and it converts "strip the stack" from a judgement
   call into a measurement. Expected outcome given G5 (no rule showed a
   significant gain anywhere): strip all. The ablation exists to catch the
   case where that expectation is wrong.
8. **THE INSTRUMENT (new in revision 2; highest-information item in M39, zero
   slots).** Clone the top-band mirror seats into **`m39_bc_top`** — the
   903–1058 alakazam winner seats, with `m39_bc_top1058` as a separate
   single-pilot bed if that seat count supports it. Recipe is proven:
   `m38_bc_wall` turned 138 seats into a pilot that beats our champion 2:1
   in its matchup, so the machinery needs no new code.

   **What it buys.** Today "0-6 above 800" is our entire knowledge of the
   target band, and n=6 with zero wins cannot distinguish a 5pp gap from a
   40pp gap. Those imply different milestones — the first says close the
   loss mass and climb; the second says imitation is exhausted and only
   self-play iteration reaches 1000. **Run the champion, the cont3 pin, and
   the M38 ship config against `m39_bc_top` at n=800 before any M39 arm
   trains**, and diary the number the day it lands. It is the single
   measurement that most changes what M40 does with slot 3.

   Pre-registered readings:
   - **gap ≤ 10pp** → the 800+ band is reachable by closing known loss mass;
     M39 proceeds as planned and slot 3 is a genuine 1000 attempt.
   - **gap 10–25pp** → imitation gets us to ~850; the 1000 attempt needs
     P3-B's best-response lane to carry it, and M40 is scoped around that.
   - **gap > 25pp** → the corpus cannot teach the target band. Slot 3 is not
     a 1000 attempt on this lineage; the M40 agenda becomes self-play
     iteration + the enabler debt, and we say so early rather than
     discovering it on a burnt slot.

Exit criteria: new beds pinned with 2-seed n=800 champion baselines;
**`m39_bc_top` gap measured and diaried**; **strip ledger written from the
P0.7 ablation, not from the G5 aggregate**; QC battery runs green
mechanically; weighted-gate harness smoke-tested.

### P1 — Ship A ✅ **SHIPPED 2026-08-01 — sub 55172160**

**Shipped config: `m38_w9294_cont3` + fix string `conserve`** (not PLAIN —
see the overturn below). Deck `alakazam_v2_h4` (md5 `ad014c58`). Single
variable vs 55146658: the fix string. Tarball
`dist/submission_neural_20260801_234656.tar.gz`, ship commit `456f640`.
Diary: [M39.md](M39.md).

**The plan said PLAIN; the measurement said otherwise.** Decision 1 was
settled as a full strip on the P0.7 ledger (no rule cleared the keep-bar;
`conserve` missed at z=−1.91, one of 12 contrasts). Piotr called
investigate-further on the weak Ship A premise, and the powered 3-arm gate
inverted it:

| comparison (9-bed weighted roster, n=2400/cell) | delta | z |
|---|---|---|
| **conserve-only vs gacfr3 (live)** | **+1.74pp ±1.00** | **+3.42** |
| **conserve-only vs plain (full strip)** | **+1.03pp ±1.00** | **+2.03** |
| plain vs gacfr3 | +0.71pp ±1.00 | +1.39 (n.s.) |

Positive on 6 of 8 beds at ~+2–3pp, including every big-share one. **A
mechanism probe, not a significance threshold, is what got this right**:
`scripts/conserve_probe.py` showed conserve fires in 11–13% of trigger-true
states on the 900+ and grim beds and 0.2% on wall — active exactly where its
removal cost win rate. G-11 earned its place as a mandatory gate here.

Expected live effect ≈ **+12 implied ELO**. **G-10 dwell declared before the
sample: decision read at n=150** (~3 days, ±56 ELO); a 1-day read (±93)
could not resolve a 12-ELO effect in either direction.

Original P1 specification retained below for the record.

- **CORRECTION (revision 3).** Earlier drafts of this plan said
  "deckguard/conserve is NOT part of the gacfr3 economy stack; it stays."
  **That is wrong.** The shipped string is
  `telepath,deckguard,ash,conserve,benchfloor,racemode3`
  ([submission/main.py:113](../submission/main.py:113)) and **gacf =
  deck*g*uard, *a*sh, *c*onserve, bench*f*loor** — those four rules ARE the
  `gacf` cell G5 measured at .278 against plain's .323
  ([M38.md:430](M38.md:430)). "Keep deckguard/conserve" is therefore not a
  conservative reading of G5; it is an **unmeasured config**, and shipping
  an unmeasured config on gate evidence is the exact M38 error.
- The **per-rule strip ledger** defines "plain" precisely against the G5
  arm, byte-for-byte. Two facts it must pin, both currently ambiguous:
  (i) G5's "cont3 plain" was the *no-rules* baseline — confirm whether
  `telepath` (an ATTACH fix, M26, applied by a different function) was off
  in that cell too; (ii) per-rule attribution beyond racemode3 is
  explicitly **incomplete** in the diary, which is why P0 runs the
  6-cell ablation below rather than reasoning about which rule to keep.
- **HARD CONSTRAINT on the strip (BACKLOG sweep, adopted negative result):
  legal-action masking is load-bearing and is NEVER in scope for a strip.**
  It was the single most valuable component of the PTCG-Bench agents (118
  rating points) — more than any of the mechanisms that failed there. The
  strip ledger must list the masking rules explicitly in a KEEP column, so
  "plain" can never be read as "unmasked". This is a one-line safeguard
  against the milestone's largest own-goal.
- Gate on the P0 roster, weighted (G-3). Requirement: no weighted-pool
  regression vs the M38 ship config pin; wall/stall cells ≥ the plain pins.
- Export → full four-tier pre-ship protocol ([VALIDATION.md](VALIDATION.md)
  §1) → **STOP for Piotr's replay review and explicit go** → submit,
  monitor row in the ship commit.
- **Mechanism proof (G-11), and Ship A is the ideal case for it.** A
  config-only change makes the Tier-2 behavioral diff nearly a proof:
  pre-ship, the diff against the M38 pilot must show divergences **only**
  in states where the stripped rules fired; post-ship, the inverted fire
  probe (`scripts/racemode_fire_probe.py` generalized) must show 0 fires
  for the stripped rules across the live sample. That verification is
  n-independent — it holds regardless of how wide the ELO interval is,
  which matters because the ELO interval will be wide.
- Watch items (per G-2): wall+stall WR (expect improvement — this is the
  live test of G5's −7pp), mirror (does losing the stack cost the 7-2?),
  grim (no rules-lane change expected yet).

This ship doubles as the live falsification test of the G5 measurement and
as a fresh pool-drift control for Ship B.

### P2 — Anti-deck-out package (rules lane, on real beds) — ⏳ NOT STARTED

**PREREQUISITE ADDED 2026-08-01: implement G-13 bed panels first.** P2 and
P3 are both gated on the M39 beds, and D1 showed a single clone's strength
is a lottery draw with a ~12.5pp spread. Gate *deltas* are largely
insulated, but P2's kill gate is phrased as an absolute ("wall+grim+
archaludon improves ≥5pp"), which is exactly the kind of number the lottery
corrupts. Panels are free — redistribute the same match budget across ≥3
draws per family — and they must land before either lane's gate is read.

Targets the 29% deck-race loss mass, 5 of them thrown from winning
positions. Two independent, separately-measured changes:

- **P2a — stall/pressure coverage.** Add Fan Rotom and Hop's
  Phantump/Trevenant (+ Kangaskhan variants, the BACKLOG's original
  blindspot) to `_RACEMODE_PRESSURE_IDS`; ship the margin-gated pressure
  branch — `racemode2`'s split logic already exists at
  [plan.py:559](../rl/plan.py:559), never shipped live. Trigger sets get a
  regression test enumerating covered opponent ids so the next new stall
  variant is a one-line diff.
- **P2b — racemode4 burn-source demote.** Demote the *real* burn (M37
  audit, twice confirmed): Enriching Energy (−4/attach), Poké Pad, surplus
  Dawn/Hilda once set up, Sacred Ash before deck ≤ 2. Margin/state-gated,
  NOT blanket — the engine is needed when not in a deck race. Ceiling ~5–6
  cards/game; 5 of 8 M38 deck-race losses were within a ≤9-card margin.

- **~~P2c — opponent-deck inference~~ → CUT to M40** (revision 2). It was
  the only addition that puts *new runtime code* into a shipped agent — a
  classifier plus a `plan.py` branch — for what is ultimately a timing
  improvement on a rule we already have. New live code is cheap when slots
  are plentiful and expensive when four remain. The design as scoped
  (strictly-earlier firing of an already-validated trigger, default OFF)
  stands as written; it moves to BACKLOG for M40 with no rework.

Measurement: single-variable cells on `m38_bc_wall` + **`m39_bc_grim` +
`m39_bc_archaludon`** (the stall bed is unbuildable — [M39.md](M39.md)) +
mirror (mirror deck-outs exist too), n=800 2-seed. Kill gate **reworded to
match where the deck-race mass actually is**: a variant ships only if
**wall+grim+archaludon** improves ≥5pp with no weighted-pool harm (G-3) —
and per G-4, the surviving package re-runs the on/off matrix on whatever
net P3 produces.

### P3 — Targeted fine-tune (policy lane, corrected corpus recipe) — ⏳ NOT STARTED

Status note: **corpus A is no longer the thin-corpus risk the plan hedged
against.** The P0.1 snowball took the loss-family winner-seat census to
grim 76 / wall 27 / archaludon 24 (≈127 before the loser rows decision 8
keeps), over the ≥100 target. The `--opp-deck-hash` filter this lane needs
is **already implemented and tested** (it landed early for D1). Corpus B
(best-response) remains scheduled as the volume lane and the A/B
diagnostic. Same G-13 panel prerequisite as P2.

The vehicle that worked (low-dose fine-tune on live winner seats, dose law
~1 epoch) aimed at the corpus flaw that kept M38's gains in-family — with
finding #7's correction applied.

**Two corpora, run in parallel, because they fail independently.** The
harvested corpus (A) has the best demonstrators available but unknown
volume; the manufactured corpus (B) has unlimited volume at champion-band
quality. The P0 census decides how much weight each carries — but both are
scheduled from the start, so a thin census does not stall the lane.

**Corpus A — `bc_m39_w9294_vsloss` (harvested).** Same-deck (9294d9d8 +
6934f4) seats **filtered to games against grim/wall/stall/dragapult
opponents** — how strong alakazam pilots beat what beats us, in our own
action space. **Built WITHOUT `--winners-only`** (finding #9): keeping the
loser seats is what makes the α-ladder a real experiment instead of a
bit-identical rerun, and it recovers the ~50% of decisions the harvest
currently discards (BACKLOG sweep #5, the free half). Snowball until ≥100
winner seats; if the census says thin, corpus B carries the lane and A
blends in as the high-band minority.

**Corpus B — `bc_m39_bestresp` (manufactured; BACKLOG sweep #2).**
`plan_iter collect` vs `m38_bc_wall` / `m39_bc_grim` / `m39_bc_archaludon`
(≤8 workers per the parallelism cap; fresh `--out` dir per the no-resume
rule), keep winning seats, low-dose fine-tune. Existing tooling end to end.

Arms — all on the cont3 lineage, champion recipe, best-val≈epoch-1:

| arm | corpus | knob | question it answers |
|---|---|---|---|
| `m39_vsloss` | A | α=0 (≡ winners-only) | does the vs-loss filter beat the general 800+ corpus? |
| `m39_vsloss_a25` | A | `--outcome-weight 0.25` | do the kept loser seats carry signal? (sweep #1/#5) |
| `m39_bestresp` | B | α=0 | can we manufacture what we cannot harvest? (sweep #2) |
| `m39_retain` | winner of the above **+ champion shards** | multi-dir mix | does retention break the 1-epoch dose law? (sweep #3) |
| `m39_mirror903` *(optional)* | 903–1058 mirror | — | demonstrator-band depth |

α-ladder note: only α ∈ {0, 0.25} in v1. AFBC (arXiv 2110.04698) prefers
the **binary** filter over exp-weighting precisely because temperature
sensitivity is a trap; α=0 *is* the binary filter here, and 0.25 is the one
probe of whether loser rows carry signal. A true per-decision advantage
filter needs the value net — enabler debt, not attempted here.

`m39_retain` runs **after** the first three gate, since it mixes into
whichever corpus wins; it is a second-round arm, not a fourth parallel
train. Mix ratio is its single hyperparameter; **no KL anchor, no EWC** in
v1 (both cautioned in the source evidence).

- Gate: weighted pool on the P0 roster vs the cont3-plain pin. **The
  grim/wall/stall/dragapult cells are the point — a pooled win carried by
  in-family cells does not ship** (the M38 lesson, now mechanically
  enforced by G-3).
- **Ceiling cell (revision 2): every arm additionally reports its
  `m39_bc_top` number against the P0.8 baseline.** This does not gate
  Ship B — closing the 700-band loss mass is Ship B's job and the top bed
  is not the live mix. It is the **slot-3 selection signal**: the arm that
  moves the high-band cell is the M40 candidate even if another arm wins
  the weighted pool, and if *no* arm moves it, imitation is confirmed
  exhausted for the 1000 target and M40 is a self-play milestone. Report
  both numbers side by side in the diary; never blend them.
- **P3-B exploiter-overfit control: `m39_bestresp` must win on the full
  weighted roster. A win on the target beds with weighted-pool harm is a
  KILL, not a ship** — 0.90 at 32 decks → 0.54 at 1024 (arXiv 2404.16689)
  is exactly the trap, and our beds are three clones.
- Kill condition: no arm moves the loss-matchup cells ≥3pp pooled → the
  lane is dead for imitation **at this corpus quality**, and the P4 design
  question fires early. Because A and B fail independently the kill is
  diagnostic: A-fails/B-wins ⇒ volume was the constraint; A-wins/B-fails ⇒
  demonstrator strength was; **both fail ⇒ the ceiling is the objective,
  not the data**, and M40 pays the enabler debt (value net → real
  advantages) instead of harvesting more.

### P4 — Ship B + the sweep-mass design decision — ⏳ NOT STARTED (slot 2 of 4)

- Ship B candidate: best P3 net × surviving P2 rules, G5-style on/off
  matrix (G-4), weighted gate, expanded QC, Piotr review, submit.
- **Success check against the objective**: converged implied ELO vs the
  P1 ship (G-5 in-window comparison). 800+ band results are now visible in
  QC-adjacent terms via the weighted gate's high-band beds.
- **If the 43% sweep mass hasn't moved** (P3 kill or live flatness), the
  scoping question goes to Piotr with the fresh evidence, candidates in
  post-mortem order: opponent-conditional fine-tune arms (per-family heads
  or opponent-deck features), vs accepting the BC ceiling and pushing the
  econ/rules lane. This is a design fork, not a default — standing rule:
  prompt Piotr.

## Planned code changes (file-by-file)

Grounded against the current tree 2026-08-01; verified line refs may drift
once P0 lands. Twin rule applies throughout: any `rl/*` change regenerates
`submission/rl/*` via the export and rides with the twin-parity tests.

### P0 — measurement integrity

- **`data/m39_live_mix.json`** ✅ GENERATED: committed live-mix weights
  (pooled M37+M38, n=147) + a `coverage` field the gate checks against the
  80% rule (G-3). Generator `scripts/m39_live_mix.py` lands with the P0
  commit. **Measured coverage of the PLANNED roster: 77.5% — below the
  floor**, which is what forced the archaludon bed.
- **`scripts/m39_gate.sh`** (NEW, from `m38_gate.sh`): bed roster swaps in
  `m39_bc_grim` / `m39_bc_archaludon`, drops the M26 grim relic. ✅ WRITTEN.
- **`scripts/m39_decide.py`** (NEW, follows the `m3X_decide.py`
  convention): weighted pooling over per-bed cells using the live-mix
  json; prints weighted AND unweighted pools + per-cell z vs control;
  refuses to emit a verdict if roster coverage < 80% of mix mass.
- **Bed construction** ✅ DONE (no `rl/` change needed — existing seat-side
  filters sufficed, as predicted). Recipe used, matching the `m38_bc_wall`
  precedent exactly (hand-aware, ALL seats not winners-only — a bed clones
  how an opponent plays, not how they win):
  `replay_bc build --deck-hash <h> --min-score <s> --hand-aware` then
  `plan_iter train --init checkpoints/m28_winners.pt --epochs 10 --lr 3e-4`.
  | bed | hash(es) | min-score | seats | val |
  |---|---|---|---|---|
  | `m39_bc_top` (instrument) | 9294d9d8 | 900 | 329 | 0.682 |
  | `m39_bc_grim` | 3121746f | 700 | 195 | 0.662 |
  | `m39_bc_archaludon` | a905b537 + b65e607c | 700 | 137 | 0.744 |
  Deck CSVs already existed: `clone54618168` (9294d9d8), `grim_live`
  (3121746f), `archaludon` (b65e607c).
- **`scripts/m39_census.py`** (NEW, unplanned but load-bearing): seats per
  family per band + the vs-loss census. It is what proved the stall bed
  unbuildable and the archaludon bed necessary, and it imports
  `m39_live_mix.classify` so census and gate weights cannot drift.
- **`scripts/m39_topgap.sh`**, **`scripts/m39_strip_ablation.sh`**,
  **`scripts/m39_abl_repair.sh`**, **`scripts/m39_build_qc_beds.py`**,
  **`scripts/live_ci.py`** (all NEW) — the P0 batteries and instruments.
- **`rl/matchrunner.py`**: +5 leave-one-out spec kinds (`abl-no-*`) for the
  P0.7 ablation. Additive registry entries only; matchrunner is an
  instrument and is NOT in the shipped twin set, so no re-export.
  **Plus the instrument (P0.8):** the same recipe with
  `--only-deck-hash 9294d9d8 --winners-only --min-score 903` →
  `checkpoints/m39_bc_top.pt`, and `--min-score 1000` →
  `checkpoints/m39_bc_top1058.pt` if the seat count supports a
  single-pilot bed. Build these **first** — the gap measurement gates what
  the rest of the milestone is for.
- **`scripts/m39_strip_ablation.sh`** (NEW, P0.7): six leave-one-out cells
  on `m38_bc_wall` at n=400 over the shipped string
  `telepath,deckguard,ash,conserve,benchfloor,racemode3`, plus the plain
  and full pins. Pure config sweep via `PKM_ATTACH_FIXES` — **no code
  change**, the predicates in `rl/plan.py` are untouched. Output is the
  strip ledger's evidence table.
- **`scripts/m39_topgap.py`** (NEW, small): runs champion / cont3-plain /
  M38-ship-config against `m39_bc_top` at n=800 2-seed and prints the gap
  against the pre-registered ≤10 / 10–25 / >25 pp readings. Its output is
  a diary entry the day it lands (incremental-diary rule), not a
  milestone-end summary.
- **`scripts/ship_verify.py`** (NEW, standing — [VALIDATION.md](VALIDATION.md)
  §1 Tier 1): runs the artifact-correctness battery **against the exported
  tarball** — twin parity, deck md5, weights bit-identical to the gated
  checkpoint, fix-string match, `prize_semantics_probe`, fix-name
  resolution, and the latency/budget measurement. Today these are done
  ad hoc per milestone and one of them (from-tarball verify) only happened
  in M37; this makes the whole tier one command.
- **`scripts/behavior_diff.py`** (NEW, standing — §1 Tier 2, G-11): runs
  parent and candidate on identical seeds, counts decision divergences and
  classifies them by which fix fired. The pre-ship half of every ship's
  mechanism proof; `racemode_fire_probe.py` generalizes into the live half.
- **`scripts/live_ci.py`** (NEW, small — §2, G-9): implied ELO with 95% CI
  and Fisher exact on per-matchup cells, from the cached replays. Makes the
  honest number the easy number to quote.
- **`scripts/qc_battery.py`**: add a `MODEL_AGENTS` dict
  (`{name: (checkpoint, deck_csv)}`) with wall/grim/stall legs beside the
  existing `SAMPLE_AGENTS` + prev-ship mirror; spawn model opponents via
  the `model:<ckpt>:<deck>` spec already used by the gate batteries. QC
  covers the top-3 live loss families permanently.
- **`scripts/band_decode.py`**: re-freeze weights on the current 700-band
  sample (data refresh; constants updated in place).
- **`MILESTONES.md`**: epoch marker retiring `m25_bc_grim_54861775.pt` as
  a gate bed (G-6).

### P1 — strip ship

- **No `rl/plan.py` logic change.** The ship config drops the gacfr3 fix
  string to match the G5 "plain" arm byte-for-byte; deckguard/M30 conserve
  is not part of that stack and stays. The per-rule strip ledger lands as
  a table in this doc before export.
- **Ride-along hardening** (BACKLOG, M37 audit): unknown names in
  `_ATTACH_FIXES` / play-fix strings currently silently ignored — fail
  loudly (`ValueError`) on unrecognized names, + test. We are editing fix
  configs this milestone; this is the moment the typo-guard pays.
- Export via `tcg.shipping` as usual; `prize_semantics_probe` standing
  regression; monitor row in the ship commit.

### P2a — stall/pressure coverage

- **`rl/plan.py`**: add Fan Rotom + Hop's Phantump/Trevenant + Kangaskhan
  variant ids (from the harvested decklists) to `_RACEMODE_STALL_IDS` /
  `_RACEMODE_PRESSURE_IDS` ([plan.py:382](../rl/plan.py:382),
  [plan.py:401](../rl/plan.py:401)). Ship config moves `racemode3` →
  `racemode2` ([plan.py:559](../rl/plan.py:559), exists, never shipped
  live): blanket conserve vs `_RACEMODE_WALL_IDS`, margin-gated vs the
  pressure set. Verify the wall-side behavior of r2 ≡ r3 on the wall bed
  before relying on it (single cell, cheap).
- **`tests/`**: NEW coverage regression test enumerating every opponent id
  the racemode triggers cover, so the next stall variant is a one-line
  diff caught by review, not a live 0-2.

### P2b — racemode4 burn demote

- **`rl/plan.py`**: NEW `PLAY_FIX_RACEMODE4 = "racemode4"` +
  `_RACEMODE4_BURN_IDS` — the measured burn sources (Enriching Energy,
  Poké Pad, surplus Dawn/Hilda once setup-complete, Sacred Ash) demoted
  ONLY under the racemode2 trigger conditions (opponent wall/pressure set
  + deck-margin gates) — state-gated, not blanket: the engine is needed
  when not in a deck race. Sacred Ash additionally gated to deck ≤ 2 (it
  was played at deck 0/2 in M37 losses). Target ~5–6 cards/game saved.
- **`tests/`**: fixture tests per demote condition (in-race vs
  out-of-race, setup-complete vs not).

### ~~P2c — opponent-deck inference~~ (CUT to M40, revision 2)

Design retained in BACKLOG so M40 picks it up without rework: a
count/naive-Bayes archetype posterior over the opponent's played card ids
(`rl/opp_id.py`, committed as a data table, not trained at runtime),
consulted by the racemode trigger only when the id-set has not already
fired, `posterior ≥ threshold`, behind a play-fix flag defaulting OFF, with
a contract test asserting behavior is *identical* on every fixture where
the id-set trigger never fires.

### P3 — targeted fine-tune

- **`rl/replay_bc.py`**: NEW `--opp-deck-hash` filter on `build()` —
  restrict to seats whose OPPONENT's decklist hash matches (multi-valued:
  grim/wall/stall/dragapult hashes). `build()` currently filters only the
  imitated seat ([replay_bc.py:329](../rl/replay_bc.py:329)); this is the
  one real `rl/` addition of the lane. Pairs with the existing
  `--only-deck-hash 9294d9d8 --min-score 800 --hand-aware` — **and NOT
  with `--winners-only`** (finding #9): corpus A keeps loser seats so
  `--outcome-weight` has rows to act on.
- **`scripts/m39_vsloss_census.py`** (NEW, small): counts 800+ winner
  seats by opponent family — doubles as the deck-decision falsification
  data. Runs before corpus build; **its output selects the A/B corpus
  weighting in `m39_train.sh`**.
- **`scripts/m39_collect_bestresp.sh`** (NEW; BACKLOG sweep #2): wraps
  `plan_iter collect` against the P0 clone beds (`--opponents` accepts the
  bed checkpoints, [plan_iter.py:1187](../rl/plan_iter.py:1187)),
  `--workers 8` MAX, **fresh `--out` dir** (collect has no resume — a
  relaunch into an existing dir clobbers shards), then a winners-only
  `replay_bc build` over the result. No `rl/` change.
- **`scripts/m39_train.sh`** (NEW, from `m38_train.sh`): arms `m39_vsloss`
  (α=0), `m39_vsloss_a25` (`--outcome-weight 0.25` —
  [plan_iter.py:1243](../rl/plan_iter.py:1243), no code change),
  `m39_bestresp`, optional `m39_mirror903`; then the second-round
  `m39_retain` (champion shards mixed via multi-dir `--data`, as m38 gen-1
  a+b — no new loss code). Thin-corpus fallback is now corpus B, not a
  blend-and-hope.
- **`scripts/m39_gate.sh`**: arm cells vs the cont3-plain pin, weighted
  verdict via `m39_decide.py`. `m39_bestresp` additionally reports its
  target-bed cells **separately** from the weighted pool so the
  exploiter-overfit kill is legible at a glance.

### P4 — ship B

- No new code: G5-style on/off matrix as extra `m39_gate.sh` cells,
  export via `tcg.shipping`, expanded QC, monitor row
  (`notebooks/model_monitor.ipynb` MODELS dict) in the ship commit.

### P5 (optional, default OUT) — clone-scout

- No new code if pulled in: `scripts/m39_build_beds.sh` recipe pointed at
  the best wall/dragapult seats + a `m39_gate.sh` roster run.

## Ship cadence

Two submits, **slots 1 and 2 of 4**. **Slot 1 is SPENT** — Ship A shipped
2026-08-01 as sub **55172160** (cont3 + `conserve`), and its declared
decision read is **n=150**. **Ship B (P2×P3)** is the milestone headline and
takes slot 2. Slot 3 is M40's 1000 attempt; slot 4 is untouched reserve.

Per G-7 + G-10, **Ship B does not go out until Ship A's n=150 read lands**
(~3 days from 08-01). Earlier reads are permitted for catastrophe detection
and the G-11 live mechanism probe — does `conserve` fire in live replays at
the ~11–13% rate the offline probe measured? — but no strength verdict
before n=150.

Both M39 submits run the **full four-tier pre-ship protocol**
([VALIDATION.md](VALIDATION.md) §1) — artifact correctness from the
tarball, mechanism proof (G-11), the weighted strength gate, then the QC
smoke test and Piotr's manual replay review and explicit go (no
exceptions) — and the **post-ship read schedule** (§2), with the monitor
row landing in the ship commit.

## What would kill this milestone (pre-registered)

**Resolved so far (2026-08-01):**

- ~~P0 harvest can't produce faithful grim/stall beds~~ — **PARTIALLY FIRED
  and handled.** The grim bed built fine (195 seats at band); the **stall bed
  was unbuildable** (9 seats) and was killed rather than built thin. Coverage
  was restored above the G-3 floor by the unplanned archaludon bed, not by
  the fallback this line anticipated.
- ~~P0.8 can't produce a 903+ bed~~ — **did not fire.** 329 seats.
- ~~P0.8 reports a >25pp gap~~ — **did not fire.** ≤10pp, the best branch;
  slot 3 remains a genuine 1000 attempt. Held loosely pending G-13 panels,
  since absolute bed numbers carry the lottery variance.
- ~~P1 strip regresses the weighted pool~~ — **did not fire**, but nor did
  the strip help: plain vs gacfr3 was +0.71pp (n.s.). The lane was rescued
  by a third config (conserve-only) this line never contemplated.

**Still live:**

- **P0.8 can't produce a 903+ bed** (too few top-band seats even after the
  round-3 snowball) → we cannot instrument the target band by cloning, and
  the 1000 question becomes un-measurable offline. Fall back to the widest
  available high-band bed, label the gap number provisional per G-8, and
  raise the sequencing question with Piotr before slot 3 is committed.
- **P0.8 reports a >25pp gap** → not a milestone kill, but a *target* kill:
  slot 3 stops being a 1000 attempt on this lineage and M40 is re-scoped to
  self-play iteration + the enabler debt. Better to learn this in P0 than
  on a burnt slot — this is the whole reason the instrument is built first.
- P1 strip regresses the weighted pool → gacfr3 interactions are
  net-positive after all; keep the stack, P2 proceeds anyway (its rules are
  new, not re-tunes).
- P3 all-arms kill (**both corpus A and corpus B**) → imitation exhausted at
  this corpus quality; P4 fork fires early, and M40's agenda is the enabler
  debt (value net → real advantages), not more harvesting.
- `m39_bestresp` wins its target beds but harms the weighted pool → textbook
  exploiter overfit; killed, and the result is diaried as a campaign law
  (offline best-response needs roster-wide gating, never bed-local).
- Both ships flat live at converged n → the ceiling is the training
  objective, not the data; slot 3 becomes a self-play/objective milestone
  rather than another corpus iteration.

## Decision points for Piotr (before P0 launch)

Each decision states its **scope** (what is actually being chosen, and what
is not), the **options**, a **recommendation**, and the **rationale** —
grounded in RL/IL practice and in what this campaign has already measured.
Summary first:

**ALL DECISIONS SETTLED 2026-08-01.** Piotr resolved 1, 6+8, 10 and 13 by
direct answer; 2, 3, 4, 5 and 11 were taken as confirmed on the stated
recommendations. The rationale for each is retained below because the
*reasoning* is what future milestones reuse, not the verdict.

| # | decision | resolution |
|---|---|---|
| 1 | Strip scope | ✅ **Full strip to G5-plain, ablation-verified** (P0.7 runs first) |
| 2 | Two-submit cadence | ✅ Confirmed |
| 3 | Deck variant | ✅ `alakazam_v2_h4`, unchanged both ships — standing per-ship confirmation still applies at export |
| 4 | P3 corpus recipe | ✅ Same-deck only; **no cross-family arm** |
| 5 | P5 clone-scout | ✅ OUT of M39 |
| 6 | P3-B as parallel arm | ✅ **Confirmed — the headline lane** |
| 8 | α-ladder | ✅ Runs last, lowest priority; null expected and pre-interpreted |
| 10 | Instrument sequencing | ✅ **Parallel with Ship A prep; blocks P3 arm training only** |
| 11 | G-7 / G-8 binding | ✅ Confirmed |
| 13 | Validation protocol | ✅ Adopted, **with G-10 modified: dwell declared per ship** (Piotr's call), not fixed at n≥150 |

---

### 1. Strip scope (P1 / Ship A)

**Scope.** Which entries of `telepath,deckguard,ash,conserve,benchfloor,racemode3`
survive into Ship A. This is a **config-string change, not a code change** —
`_ATTACH_FIXES` is read from an env-var default
([submission/main.py:113](../submission/main.py:113)), the predicates stay
in `rl/plan.py` untouched. Not in scope: the engine's legal-action layer,
which is not part of this string at all (see the masking constraint in P1 —
it exists to stop "plain" being misread, not because the fix stack touches
legality).

**Options.** (a) full strip to G5-plain; (b) partial strip, keeping some
economy rules; (c) keep the stack.

**✅ RESOLVED, THEN OVERTURNED BY MEASUREMENT.** Settled as (a) full strip;
the powered 3-arm gate then showed **conserve-only beats the full strip by
+1.03pp (z=+2.03)** and the live config by +1.74pp (z=+3.42), so **Ship A
shipped as `conserve`, not plain**. The reasoning below was sound on the
evidence available and is retained — what changed it was a mechanism probe,
not a re-argument. Original recommendation: (a) full strip, byte-for-byte
with G5's `cont3 plain`
cell, after the P0.7 ablation confirms no individual rule carries a real
gain.**

**Rationale.**

*What our own data says.* G5's matrix is `plain .323 > gacf .278 > gacfr3
.254` on the wall bed, and **no rule showed a significant gain anywhere
measured** (mirror +2.7 ns, rocket +3.5 ns, tuned −1.8 ns). We are removing
things with measured harm and no measured benefit. Option (b) is the
dangerous one: because `gacf` *is* deckguard/ash/conserve/benchfloor, any
"keep the economy rules" variant is a config **G5 never measured**, and
shipping an unmeasured config on gate evidence is precisely the M38 error
this plan's finding #1 is about.

*Why RL practice says the same.* A hand-coded fix stack layered over a
learned policy is a **behavior prior / policy constraint**, and the standing
result in offline RL is that constraints calibrated against one policy's
state-visitation distribution are mis-specified when the policy changes —
the same distribution-shift argument Kumar et al. (arXiv 2204.05618) use to
locate offline RL's advantage over BC. The gacf rules were tuned on
`m28_winners`' behavior; `cont3` is a different net that visits different
states, so the rules now fire in states they were never calibrated for.
That is not a hypothesis — G5 measured it, and it is already written into
this plan as guardrail G-4.

*Why the downside is bounded.* Reversibility is near-total: the fix string
is a config default, so if Ship A regresses, slot 2 restores any subset with
zero code risk and a one-line diff. Compare that to the cost of *not*
stripping — a fourth consecutive ship where the rules lane is confounded
with everything else.

**Caveat that makes this recommendation conditional, not automatic.** The
diary states per-rule attribution beyond racemode3 is **incomplete**, so
"strip everything" is currently an inference from an aggregate. P0.7's
6-cell leave-one-out ablation costs no slots and settles it. If a rule turns
up with a genuine gain, keep that one and say so in the ledger. Expected
outcome is strip-all; the ablation is insurance against acting on an
aggregate we have not decomposed.

---

### 2. Two-submit cadence

**Scope.** Whether Ship A goes out before P2/P3 complete, spending slot 1 on
a single-variable config change.

**✅ RESOLVED — confirmed.**

**Rationale.** This is standard **single-factor experimental design under a
hard budget**, and the budget is what makes it correct rather than
fastidious. With 4 slots the marginal information per slot dominates the
marginal ELO per slot: a bundled ship that lands at 750 tells us nothing
about *which* half worked, and we would have 2 slots left to find out. M38
is the worked example — new net plus a rule stack its own gate said to
strip, and the slot returned no attribution until the post-mortem untangled
it. The usual objection to sequential ships is calendar cost, and here that
objection is empirically dead: M38 shipped 07-31 and had 54 ladder games by
08-01, so a converged read costs ~1 day. Ship A also serves as G-5's
pool-drift control for Ship B, which we need regardless.

---

### 3. Deck variant

**Scope.** The *family* decision (stay on alakazam) is made and documented.
This is only the standing per-ship variant confirmation: `alakazam_v2_h4`
(deck md5 `ad014c58`, hash 9294d9d8).

**✅ RESOLVED — alakazam_v2_h4 unchanged for both ships.** (The standing per-ship deck confirmation still applies at export time.)

**Rationale.** Two reasons, one experimental and one about the corpus.
Experimentally, Ship A's entire purpose is to isolate the fix string; a deck
change makes it a two-variable ship and forfeits the attribution we are
spending the slot to buy. On the corpus side, the harvest and every P3 arm
are keyed to hash 9294d9d8 — changing the piloted list while fine-tuning on
seats from a different one introduces exactly the seat/action-space mismatch
that finding #7 corrects for cross-family data. If a variant change is ever
wanted, it belongs on its own slot with nothing else moving.

---

### 4. P3 corpus recipe — same-deck vs cross-family

**Scope.** Whether to also run a small cross-family arm (fine-tuning our
alakazam policy on grim/wall/stall *winner* seats) as a falsification cell
alongside the same-deck vs-loss filter.

**✅ RESOLVED — same-deck only. No arm spent on the
cross-family cell.**

**Rationale.** BC minimizes divergence from the demonstrator's
state-conditional action distribution — it imitates a *seat*, not a
strategy. Cross-family seats are drawn from a different action space (their
cards, their lines) and a different state distribution than we will ever
occupy, so the arm optimizes a loss whose optimum is not the behavior we
want. That is the textbook covariate-shift/coverage-gap failure the ICML
2024 retention work (arXiv 2402.02868) formalizes, and it is the mechanism
finding #7 identifies.

The decisive point is that **we already ran the falsification.** M38's
teacher-distill was exactly this shape — imitate a demonstrator whose
decisions we do not face — and it failed at every dose tested
(.123–.264 against control .517–.631). Spending compute to re-derive a known
negative is the "attractive lane" failure of finding #6. Cross-family seats
remain valuable for **beds** (P0), where imitating their seat is the entire
point.

---

### 5. P5 clone-scout

**Scope.** Whether to BC-clone the best wall/dragapult seats to price what a
deck swap would be worth, this milestone.

**✅ RESOLVED — OUT of M39.**

**Rationale.** Largely subsumed since it was written. P0.8 now builds
top-band alakazam beds, so we get high-band opposition without it, and the
P3 census answers the underlying question — whether 800+ alakazam pilots
beat grim/wall/stall or dodge them — for free. Running it now would also
answer a question we have pre-committed not to act on this milestone
(the family decision is made). Revisit at M40 only if the census says top
pilots dodge our loss matchups, which is the one reading that would make a
swap live again.

---

### 6. P3-B (offline best-response) as a parallel arm

**Scope.** Whether best-response collection against our own clone beds is a
scheduled arm or a contingency if the harvest runs thin. **Revision 2
raises the stakes on this one:** with the target at 1000, P3-B stops being
insurance and becomes the headline lane.

**✅ RESOLVED — confirmed as a parallel arm, and it is the most
important thing M39 builds.**

**Rationale.** The ceiling argument decides it. **BC cannot exceed its
demonstrator band**, our corpus tops at 1058 with 231 seats @800+, and
imitation lands below its demonstrators — so the entire harvested-corpus
lane projects to ~800–850 no matter how well we execute it. Every mechanism
that has ever exceeded its demonstrations does so by *improving against an
opponent* rather than *matching a demonstrator*: BC-init plus best-response
against a fixed opponent took an exploiter 42%→90% (arXiv 2404.16689), and
synthetic self-play data carried the Showdown agent from ~58% to 64–80% —
the same staged recipe we are already running (BC → offline improvement).
P3-B is that step, built entirely from tooling we exercised in M38.

It also removes a real single-point failure: the harvested vs-loss corpus
has unknown volume until the census returns, and without P3-B a thin census
strands the policy lane and reduces M39 to a rules-only milestone capped
around 740–760.

**The two cautions stay pre-registered as gate conditions**, because they
are the known failure modes rather than hypotheticals: exploiter overfit is
measured (0.90 at 32 decks → 0.54 at 1024, same paper), so P3-B gates on
the full weighted roster and a target-bed-only win is a **kill**; and we
imitate our own net's wins, so in-family label quality caps it — which is
exactly why it runs *parallel* to the harvested corpus rather than replacing
it. A and B trade off volume against demonstrator strength, and running both
is what tells us which one was the binding constraint.

---

### 8. The α-ladder (dropping `--winners-only` from corpus A)

**Scope.** Whether corpus A keeps loser seats so `--outcome-weight` has rows
to act on. α=0 reproduces winners-only exactly, so this adds an arm rather
than replacing the baseline.

**✅ RESOLVED — runs as the lowest-priority arm; expect a
null result. Cut it first if compute is contended with P3-B or the
instrument.**

**Rationale, including why I am lukewarm on my own addition.** The case for
it: it is the only probe we have of whether loser rows carry signal, AFBC
(arXiv 2110.04698) finds the binary filter competitive at ~100k expert
samples which brackets our corpus, and the harvest currently discards ~50%
of its decisions (BACKLOG #5).

The case against, which is why it is last: without a value net our
"advantage" is **game-level outcome**, not per-decision advantage. A losing
seat still played many good decisions and a winning seat played bad ones, so
outcome-as-proxy is a very noisy estimator — and the Kumar et al. result
that motivates offline-RL-over-BC assumes you actually have advantages. This
is the enabler debt: the real version of this arm is unavailable until the
value net is retrained on the harvest corpus. It also sits against M28's
winners-only filter, the campaign's one resolved-positive weights change.

**Pre-register the interpretation now:** a null α result means
*outcome-as-advantage-proxy carries no signal*, **not** *offline-RL-style
weighting doesn't work here*. Those are different claims and only the first
is testable this milestone. Recording it here so a null does not get filed
as a kill for the whole lane.

---

### 10. Instrument sequencing — ✅ RESOLVED: parallel, blocks P3 only

**Resolution.** The instrument runs **concurrently with Ship A prep** and
gates **P3 arm training only**. Ship A trains no arms — it is a config
change — so sequencing the instrument first costs zero delay to slot 1.
This is strictly better than the "everything waits" framing in revision 2,
which would have delayed Ship A ~2 days for no measurement benefit, since
Ship A's gate does not use the top bed.

```
Ship A path:  P0.7 ablation -> strip ledger -> export -> QC -> submit
Instrument:   harvest 903+ seats -> m39_bc_top -> gap read   (concurrent)
P3 arms:      BLOCKED until the gap number lands
```

**Rationale (retained — this is the reasoning future milestones reuse).**

**Rationale.** Define the evaluation protocol before optimizing against it.
The cost of inverting that order is the best-evidenced failure in our own
campaign history: **three bed-fidelity transfer failures** — garchomp (M35),
wall-solver (M37), grim (M38) — where an offline bed lied and cost a
milestone. G-6 and G-8 exist because of them, and we are about to spend two
slots and a full arm matrix against beds that **cannot see the band we have
committed to reaching**. Right now "0-6 above 800" is our entire knowledge
of the target, and n=6 with zero wins cannot separate a 5pp gap from a 40pp
one — readings that imply different milestones and different uses of slot 3.

Cost asymmetry settles it: ~2 days and zero slots to run it first, versus
discovering on slot 3 that the corpus could never teach the target band.

---

### 11. G-7 (one lane per slot) and G-8 (band coverage)

**✅ RESOLVED — both bind as standing guardrails from M39 on.**

**Rationale.** G-7 is decision 2 generalized into a rule so it does not have
to be re-argued per ship, and its cost is ~1 day of read time. G-8 is the
generalization of the bed-fidelity failures: it forbids claiming evidence
about a band we have no bed for, which is the error mode behind all three.
Both are cheap, and both bind the *default* rather than forbidding a call —
G-1 already provides the documented-override path.

---

### 13. Validation protocol (NEW, revision 4)

**Scope.** Adopting [VALIDATION.md](VALIDATION.md) as the standing pre- and
post-ship protocol for every pilot, plus guardrails G-9 (statistical
honesty), G-10 (dwell time), G-11 (mechanism proof) — and the redefinition
of "converged" from n≥50 to **n≥150**.

**✅ RESOLVED: adopted, with G-10 modified.** Piotr's call 2026-08-01 —
**dwell is decided per ship, not fixed at n≥150.** Implemented with one
safeguard that preserves the flexibility while removing the hazard: the
decision-read n is **declared in the ship commit before the pilot goes
live**. The risk in a per-ship call was never the flexibility, it was
reading continuously and stopping when the number looked good — which
inflates the false-positive rate and leaves no trace in the record.
Declaring in advance costs nothing and eliminates that. Early reads stay
permitted for catastrophe detection and mechanism probes; only the
*verdict* n is fixed ahead of time. One floor retained: the
implied-ELO-≥-1000 claim needs n ≥ 150 regardless, because a claim that
size cannot rest on a ±93 interval.

**Rationale.** The numbers are computed from our own ship records, not
assumed. M37's implied-ELO interval is [646, 819] and M38's is [566, 753] —
they overlap across nearly their full width, so the "regression to 659" the
last two milestones were re-planned around is a working hypothesis, not an
established fact. The wall cell that the M38 post-mortem headlined (3-6 →
1-5) has Fisher p=0.60. At n=50 a live read detects only a ~200 ELO swing;
every effect this campaign targets is smaller.

Three consequences follow, and they are what the protocol encodes:

1. **The offline weighted gate is the strength instrument, permanently.**
   At n=800/bed it can resolve what live reads at n=50–150 cannot. Live
   validation confirms transfer *direction*, proves *mechanism*, and
   catches *catastrophe* — the campaign's repeated error has been asking it
   for magnitude.
2. **Mechanism probes are our best live instrument because they are
   n-independent.** M37's `racemode_fire_probe` (9/9 trigger-true, 0 fires
   / 723 prompts) is a proof of behavior, not a sample estimate. G-11 makes
   one mandatory per ship, written once pre-ship and re-run post-ship.
3. **Dwell time is free power.** Slots are consumed by submitting, not by
   waiting; 1 day gives ±93 ELO, 3 days ±56, 5 days ±43. Shipping the next
   pilot at n=50 discards a halving of the error bar for nothing.

**What this does not change:** the plan's direction. P1/P2/P3 rest on
offline gates at n=800 and on mechanism forensics (deck-out counts, burn
audits), neither of which depends on the live cells. What changes is what
we are allowed to *conclude* from each live read — and, per G-9, that
per-matchup live cells become forensic hypothesis-generators rather than
gates. Note the same cell has now been misread twice in opposite
directions: grim was "self-resolved" at 6-2 (M37) and "the bed lied" at 1-6
(M38), both at n ≤ 8.

---

### RESOLVED (recorded, no action needed)

7. ~~P2c stretch or drop~~ — **cut to M40** (revision 2).
9. **Campaign target is 1000, aimed at slot 3** — Piotr's call 2026-08-01:
   keep 1000 committed, relax "next milestone" rather than relax
   confidence. Recorded so no future reader mistakes M39's ~750–800 ship
   objective for a lowered campaign target.
12. **What "success" means for M39**, pre-registered so it is not
    re-litigated at milestone end: Ship A recovers toward ~719+, Ship B
    closes measurable loss mass toward ~750–800, and the P0.8 gap number
    plus the ceiling cells tell M40 whether slot 3 is an imitation attempt
    or a self-play one. **Landing at 780 with a clear slot-3 thesis is a
    successful M39. Landing at 820 with no attribution is not.**
