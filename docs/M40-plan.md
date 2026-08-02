# M40 plan — STUB. The last real attempt, and what M39 says it should be

**Status: STUB, drafted 2026-08-02 from [M39.md](M39.md), the
[M39 plan](M39-plan.md) and [BACKLOG](BACKLOG.md). Nothing is scheduled.**
Every phase below is a candidate; the decision points at the end are
unanswered and this plan does not proceed until Piotr resolves at least
decisions 1 and 2. Written in the M39 convention (scope / options /
recommendation / rationale) so the *reasoning* survives even where the
verdict changes.

**Do not start P0 until decisions 1–2 are answered.** M39's own finding #6
is that the attractive lane is how a milestone gets lost, and M40 has less
room to lose one than any milestone so far.

---

## The situation, stated plainly

| fact | value | source |
|---|---|---|
| live now | **Ship A 55172160 = 780.4** | leaderboard, 08-02 |
| in flight | Ship B 55182097 (dwell n=150 declared) | expected ≈ +17 ELO |
| campaign target | **implied ELO 1000** | M39 plan revision 2, Piotr's call |
| **gap after Ship B lands as expected** | **≈ 200 ELO** | 1000 − ~795 |
| **submission slots left** | **2** — slot 3 (the attempt) + slot 4 (reserve) | M39 slot budget; **CONFIRM this is still right** |
| calendar | ~6 weeks (`…-challenge-strategy` closes 2026-09-13) | competition page |

**Slots, not calendar, are the binding constraint — and slot 4 is
insurance, so M40 realistically gets ONE shot.** That single fact should
drive every choice below.

## What M39 settled, and what each finding obliges M40 to do

| M39 finding | evidence | what it obliges |
|---|---|---|
| **The ceiling trigger FIRED — imitation is exhausted at 900+** | 5 arms, 2 independently-failing corpora, **every one negative** on the `m39_bc_top` panel | M40 is **not** another harvest/fine-tune round. That branch was pre-registered and it fired. |
| **Retention is the lever, not dose rationing** | mixing champion shards swung corpus A **+3.90pp** and corpus B **+2.70pp**, turning both losing corpora into wins | retention is a **default** on every future fine-tune, never an arm |
| **Loser rows carry signal** | α=0.25 beats winners-only by **+2.28pp** | `--outcome-weight 0.25` becomes the default too; **both M39 retention arms used the worse α=0**, so the retention×α cross is unexplored and free |
| **A measured, unshipped +3.47pp already exists** | `m39_retain_b` + `conserve,racemode2,racemode4` = **+3.47pp weighted, z=+7.44**, G-4 already paid | slot 3 has a **safe floor option** that needs no new research — see decision 1 |
| **Bed strength is a training lottery (G-13)** | wall panel spread 10.8pp; `m38_bc_wall` is the softest of three draws | panels are standing; every new bed ships as ≥3 draws |
| **Mechanism probes beat significance thresholds — twice** | `conserve` (rescued at z=−1.91, now +115 live); racemode2+4 synergy predicted by fire counts | G-11 stays mandatory, and a probe is written **before** the deciding battery |
| **`plan_iter collect` cannot label a best-response corpus** | it uses `_teacher_step` (solver teacher) in BOTH modes | any self-play lane uses `scripts/m39_collect_bestresp.py`, not `collect` |
| **Self-imitation did not beat its own fixed opponent** | `m39_bestresp` −2.7pp on the wall beds it was collected against | 8% uniform exploration is too weak. If the self-play lane runs, its exploration and its label source are the design, not a detail. |

## Candidate phases (none scheduled)

### E0 — pay the enabler debt: value net on the harvest corpus

**BACKLOG sweep #6, promoted to headline by the ceiling result.** Three
parked items share one blocker and this is it: without per-decision
advantages, #1 degrades to outcome-as-proxy (what M39's α-ladder did), #5's
TD half cannot be built, and #7 has no critic.

- Train a value head on the harvest corpus (the 3,908-seat cache), **not**
  on solver-teacher data — M38's central negative result.
- Validate it the way M39 validated beds: **panels** (≥3 draws) and a
  held-out-band check. A value net is an instrument, and M39's whole lesson
  is that un-panelled instruments lie by ~6–12pp.
- **Exit criterion, pre-registered:** the value net must rank held-out
  outcomes better than the outcome-proxy baseline on the *loss families*
  specifically. If it cannot, everything downstream of it is dead and M40
  says so on day two rather than on a burnt slot.

Cost: days, zero slots. Risk: the old value lineage is prize-phobic and a
retrain may inherit it.

### E1 — advantage-filtered BC, done properly (BACKLOG sweep #1)

Only reachable after E0. M39 ran the outcome-proxy version and got a real
but sub-control result (α=0.25 at −0.82pp). With true advantages this becomes
the binary filter `1{A>0}` that AFBC prefers, on a corpus that retention has
already shown how to absorb.

**Composition that M39 never tested and should run first because it is
nearly free: retention × α=0.25.** Both retention arms used α=0; the α ladder
was run without retention. The cross is one training run.

### E2 — self-play / best-response, if the lane is entered at all

M39's `bestresp` failed, and the failure is diagnostic rather than fatal:
8% uniform-random exploration produced a corpus the net already agreed with
93.5% of the time. A serious version needs at least one of —

- **exploration that is not uniform** (sample from the policy at temperature,
  or ε-greedy over the top-k, so explored actions are plausible);
- **a critic** (E0) so the filter is per-decision rather than per-game;
- **opponent diversity** — the Showdown paper hit opponent-distribution
  overfitting with realistic partners, and our beds are three clones per
  family. G-13 panels give us 3× diversity already; that is probably still
  too few.

Online PPO (sweep #7) stays parked: 327M env steps ≈ 2 weeks of engine
flat-out, which does not fit inside one slot's preparation.

### E3 — opponent-deck inference (BACKLOG sweep #4, cut from M39 with its design intact)

The one item that arrives pre-designed: an n-gram / naive-Bayes archetype
posterior over the opponent's *played* card ids, consulted by the racemode
trigger only when the id-set has not already fired, confidence-gated, behind
a flag defaulting OFF, with a contract test asserting identical behaviour on
every fixture where the id-set trigger never fires.

**M39 raised its value.** The rules lane is now the campaign's best-measured
lever (racemode2+4 = +2.37pp weighted, +11.2pp on the wall panel), and the
package's trigger is exactly what E3 would fire earlier. It also directly
addresses M39's two admitted blind spots — `kanga` and `stall`, 4.1% of live
games, **unmeasurable offline because no bed is buildable** (19 and 34
seats). A classifier generalises where an id list cannot.

BRExIt's warning is satisfied by construction: the consumer ships in the same
milestone.

### E4 — housekeeping, only if a slot-3 decision leaves room

`rl/` ↔ `tcg/` duplication (~200 KB, seven pairs, manual parity tests); the
three pre-existing `test_meta_eval` failures; W_COUNTER retune and
`setup_plans_late` re-evaluation (both gated on a gen-2 collect that only
happens if E2 runs).

## Pre-registered kill conditions (draft)

- **E0 value net cannot out-rank the outcome proxy on the loss families** →
  the enabler debt is unpayable at this corpus quality; E1 and E5 die with
  it, and slot 3 falls back to decision 1's floor option.
- **E2 self-play produces a corpus the net agrees with >90% of the time** →
  the exploration design failed again; kill before training, not after
  gating. (This is checkable at collection time — M39's corpus B announced
  its own weakness as `init val_acc 0.935` and nobody read it as a kill
  signal until the gate agreed.)
- **Any arm wins the weighted pool but not the loss families** → the M38
  failure mode; not a ship.
- **Nothing beats +3.47pp on the weighted gate by slot-3 decision day** →
  ship the already-measured `retain_b × package` and spend the milestone's
  remaining time on E3/E4. This is a *good* outcome, not a failure.

## Standing guardrails (inherited, binding)

G-1…G-13 from [M39-plan.md](M39-plan.md#process-guardrails-standing-binding-from-m39-on)
carry over unchanged. Three that M39 proved the hard way and M40 should
expect to lean on:

- **G-11 (mechanism proof per ship)** — it rescued `conserve` (+115 live) and
  predicted the racemode2+4 synergy. Write the probe *before* the battery.
- **G-12 (offline power)** — n=800 resolves ~10pp; most effects are ~2–5pp.
- **G-13 (panels)** — no single-clone bed, and the same scepticism now
  extends to **any** trained instrument, including E0's value net.

## Decision points for Piotr (unanswered — this plan does not start without 1 and 2)

### 1. What is slot 3? — the milestone's only real question

**Scope.** Slot 3 is the last real submission (slot 4 is reserve). Two
candidates, and they are not close in risk:

| option | expected | evidence | risk |
|---|---|---|---|
| **(a) bank the measured gain** — `m39_retain_b` + `conserve,racemode2,racemode4` | **+3.47pp, z=+7.44** ≈ +24 ELO | already gated on the panel roster; G-4 already paid | two-lane, so it needs Ship B's n=150 read first (G-7) |
| **(b) spend it on the E0→E1/E2 lane** | unknown; nothing in the BACKLOG has 200-ELO evidence at our scale | literature at 100–1000× our data scale | could return nothing and burn the last real slot |

**Recommendation: (a), with (b) run offline as the M41/stretch lane.**
The arithmetic is uncomfortable and should be stated rather than dressed up:
**+24 ELO against a ~200 ELO gap.** Option (b) is the only one that could in
principle close it, but M39 just spent a full milestone establishing that
imitation on this corpus cannot, and (b)'s remaining mechanisms are exactly
the ones the sweep rated as needing an enabler we do not yet have. Banking
(a) is not defeatism — it is refusing to trade a measured +24 for a lottery
ticket when the reserve slot is the only thing left behind it.

**This is Piotr's call and I have deliberately not pre-committed the plan
to it.**

### 2. Does the 1000 target still stand?

**Scope.** The target was raised 800 → 1000 in M39 revision 2, aimed at
slot 3. Slot 3 is now the last real slot and the gap is ~200 ELO.

**Options.** (i) hold 1000 and accept that M40 probably misses it;
(ii) re-point to a defensible landing (~820–850) and judge M40 against that;
(iii) hold 1000 but move the attempt to a hypothetical M41 with more slots —
only meaningful if the slot budget is not what we think it is.

**Recommendation: (ii), stated in advance.** M39 was judged a success by a
pre-registered bar it actually met (780 with a slot-3 thesis). M40 deserves
the same treatment, and a target that arithmetic says is unreachable in the
remaining budget makes the milestone unjudgeable rather than ambitious.
**Counter-argument kept live:** Piotr raised the target on purpose in M39 to
avoid relaxing confidence, and it produced the campaign's best result — so
the case for holding 1000 is not sentimental.

### 3. Confirm the slot budget

The "4 remaining" figure entered the plan at M39 drafting and has never been
re-verified against the competition's actual submission allowance. Decisions
1 and 2 both hinge on whether 2 is really the number. **Verify before
answering 1.**

### 4. Does the self-play lane get built at all?

Even under decision 1(a), E0+E2 can run offline at zero slot cost. The
question is whether it is worth the machine time and the attention given the
M39 result, or whether E3 (opponent-deck inference, pre-designed, feeding
the campaign's best-measured lever) is the better use of the same weeks.

**Recommendation: E0 yes (it is the enabler for everything and it is
cheap), E3 yes, E2 only if E0's exit criterion passes.**

### 5. Ship B's read, and what it changes

Ship B's declared dwell is n=150. Its wall cell carries a specific
falsifiable prediction — **opponent deck-outs at ~30% in live wall replays**.
If that transfers, the rules lane is the campaign's proven vehicle and E3
becomes the obvious slot-4 candidate. If it does not, the fourth bed-fidelity
failure in a row means offline beds cannot be trusted for the rules lane
either, and that is a bigger finding than anything else in this plan.

**Nothing in M40 should be finalised before that read.**

---

## BACKLOG disposition (draft, for the real plan to confirm)

| item | proposed M40 disposition |
|---|---|
| Value-net retrain / gen-2 (sweep #6) | **E0 — headline enabler** |
| Advantage-filtered BC (sweep #1) | **E1**, blocked on E0; run retention×α=0.25 first (free) |
| Offline best-response (sweep #2) | **E2**, only if E0 passes; needs a real exploration design |
| Retention (sweep #3) | **no longer an item — it is the default** |
| Opponent-deck inference (sweep #4) | **E3**, design intact from M39 |
| ROIDA loser replays (sweep #5) | blocked on E0; the α-ladder already took the free half |
| Online PPO (sweep #7) | stays parked — 327M env steps |
| Architecture inductive bias (sweep #8) | stays parked — needs a self-play-scale corpus (E2 could make one) |
| `rl/`↔`tcg/` duplication, W_COUNTER, `setup_plans_late` | **E4**, only if slot 3 resolves early |
| Deep equilibrium search / test-time search / DT / LLM self-improvement | **stop-invest, unchanged** |
| Stop-investing list | unchanged, except **grim stays revoked** |
