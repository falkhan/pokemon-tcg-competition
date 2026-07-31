# M37 code audit — post-mortem claims vs. the actual codebase

Forensic code audit, 2026-07-31, against the M37 ship (sub 55065484, gacfr3 +
alakazam_v2_h4). Scope: verify that the code does what
[m37-post-mortem.md](m37-post-mortem.md) and the milestone plans say it does,
and find omissions / wrong data / dead code that could be depressing the
pilot's score.

Method: everything below was measured against the **live `cg` engine**, not the
synthetic `tests/builders.py` fakes. That distinction is the whole story of the
headline finding.

---

## Verdict

The prize-array semantics inversion that M36 found in `rl/plan.py` and deferred
is **systemic — it sits in four places, and only one of them was known.** The
unknown one, in `rl/turn_solver.py`, made the turn solver score every knockout
it could take at **−150,000**, which poisoned the BC label corpus the shipped
policies learn from. The known one, in `rl/plan.py`, turns out to be nearly
harmless in practice. The priority ordering in the M38 recommendations was
therefore inverted.

---

## P0 — `rl/turn_solver.py::score_leaf`: the solver scored its own knockouts at −150,000

### The defect

```python
# before
score  = W_PRIZE    * max(0, snap.op_prizes - len(op_p.prize))   # "prizes I took"    +100_000
score += W_MY_PRIZE * max(0, snap.my_prizes - len(me_p.prize))   # "prizes I conceded" -150_000
```

The two terms were swapped. A player's `.prize` is the prizes **that player
still has to take**, and it drains for whoever scores the KO — so during my own
turn only *my* array can move.

### Semantics re-verified live (two independent probes)

| probe | result |
|---|---|
| end-state of pilot-vs-pilot games | the **winner's own** array reaches 0 |
| every prize change vs. the acting seat | **58 / 58** drains belonged to the seat that had just acted |

M36's finding is correct. Note that `sample-agent/main.py:283` — the
competition's own reference agent — assumes the opposite and is **wrong**;
`_root_snapshot`'s docstring cited it as the authority, which is how the
inversion entered.

### Measured impact, over 41,570 real search leaves (8 games, solver pilot)

| observation | before | after |
|---|---|---|
| leaves where the **opponent's** prizes dropped (the `+100_000` term) | **0** | 0 |
| leaves where **my** prizes dropped (a KO I scored) | 1,291 | 832 |
| … of those, scored **negative** | **1,291 / 1,291** | **0 / 832** |
| `solve_turn` calls clearing `MIN_OVERRIDE_SCORE` (4 games) | **0 / 18** | **7 / 18** |
| best line score, max | 220 | 1e9 |

The `W_PRIZE` reward was unreachable dead code. A 1-prize KO scored −150,000; a
3-prize KO scored −450,000.

### Why it mattered so much

1. **`_dfs` could never choose a prize line.** It seeds `best_score` with the
   stand-pat leaf (≈0) and only replaces on `score > best_score`. Any
   prize-taking line scored −150k, i.e. strictly worse than doing nothing. Only
   the `W_WIN = 1e9` terminal short-circuit got through — the solver took a KO
   **only when that KO won the game outright**.
2. **`MIN_OVERRIDE_SCORE` was unreachable**, so the M7.4a headline capability
   ("finds the multi-prize lethal greedy misses") was inert in production.
3. **Blast radius is the training corpus, not live inference.** `turn_solver`
   is not in the shipped neural bundle, but `rl/plan_iter.py:109` uses
   `solve_turn_line` to generate the expert-iteration labels, and matchrunner's
   `solver`/`solver-dev` specs back the offline beds. The teacher has been
   systematically labelling *"don't take prizes unless it ends the game."*

That is a precise description of the loss modes the M37 post-mortem calls
chronic and could not explain: **9 sweeps where we took ≤1–2 prizes (30%), 9
contested races lost (30%), 60% of loss mass attributed to "policy quality, not
rules."** It also explains the campaign's characteristic shape — an agent that
out-digs, sets up, and wins by deck-out rather than by prizes.

### Why 30 milestones of green tests missed it

`tests/test_turn_solver.py` encoded the same inversion
(`two_prizes = leaf(op_prizes=2, …)`, `leaf(my_prizes=3) < one_prize  #
conceding a prize costs`). The tests build states with `tests/builders.py`
fakes, so they can drop `op_prizes` during our own turn — **a transition the
real engine never produces.** A green suite was not evidence.

**Fixed**, plus `_root_snapshot`'s docstring (which actively pointed the next
reader back at the sample agent) and the T4 trigger at `turn_solver.py:153`,
which tested `len(op.prize) <= 2` — the *opponent* closing, not us.

---

## P0b — NEW, surfaced by the fix: `MIN_OVERRIDE_SCORE` is mis-calibrated

`MIN_OVERRIDE_SCORE = W_PRIZE - 1 = 99_999` assumes the prize term is the only
contributor to the total. It is not: `W_COUNTER` (−1,000 × prize value, per
exposed Pokémon), `W_RACE` (−10 × turns-to-first-KO) and `W_DECK_LOW`
(−5,000 per card drawn at deck ≤ 6) ride on the same score. **Any negative
tiebreak mass above 1 point drops a genuine 1-prize line under the bar and
`solve_turn` declines it.** A 1-prize KO where the opponent can return-KO
scores ~97,000 and is rejected.

This shows in the post-fix numbers above: 6 of the 7 lines that now clear the
bar are outright game wins; only 1 is a prize line.

**Deliberately NOT retuned in this change.** It is a tuned constant and moving
it needs its own pre-registered A/B (the `DEV_OVERRIDE_MARGIN` precedent,
docs/M8.md). It is pinned by
`test_score_leaf_prize_direction_is_not_inverted` so that retuning fails the
test and forces a deliberate update. **This is the top M38 follow-up** — the P0
fix is only half-effective until the bar is recalibrated.

---

## P1 — `rl/collector.py:456` + `tcg/selfplay.py:229`: PPO prize shaping had the wrong sign

```python
delta = (6 - len(op.prize)) - (6 - len(me.prize))   # "prizes I took - they took"
```

This evaluates to `len(me.prize) - len(op.prize)` — the exact expression
`rl/encoders.py:101` correctly labels *"prize race (negative = I'm ahead)"*. It
is the negation of the intent, so `PRIZE_SHAPING * (delta - prev_delta)`
rewarded the **opponent** taking prizes. Affects the PPO arms (M22c, M28 C1),
not the current BC-based ship. **Fixed in both twins.**

---

## P2 — `rl/plan.py:109,116`: real, known, and smaller than the docs imply

M36's original finding, still unfixed and still shipping (it reaches Kaggle via
`submission/rl/plan.py` → `encode_plan` v[22]/v[25]).

**Measured over 1,288 real plan candidates**: `wins` disagreed with the
corrected value **0 times (0.0%)**, `concedes` **2 times (0.2%)**. The
`lethal` / `return_ko` gates and near-equal prize counts absorb almost all of
it.

**Deliberately left unfixed** — M36's reasoning still holds (feature-
distribution shift on a frozen net) and the measurement now backs it. Keep it
as the training-arm ride-along the M38 plan already schedules, but it should
**not** be prioritised above P0/P0b, which is the opposite of the current
ordering.

---

## Structural / documentation drift

| item | status |
|---|---|
| `tcg/__init__.py` claimed `rl.export`/`rl.gate` were "the live path for the submission pipeline" | **Wrong and inverted** — `build_submission.{sh,ps1}` both drive `tcg.shipping`. Fixed. |
| `rl/export.py` CLI is a live footgun | Its `export()` copies weights, features, deck and `cg/` but **not** the `rl/` package into `submission/` (unlike `tcg/shipping.py:142-152`). Following its own docstring ships stale `plan.py`/`encoders.py` **with no error**. Given the M18.1/M22c fossil-deck history: CLI now fails loudly; importable API kept for the parity tests. |
| `rl/gate.py` CLI | Same treatment — it also lacks the v4 memory-drift and plan-head parity gates. |
| `CLAUDE.md` project shape | `rl/network.py` does not exist; `submission/combat.py` and `submission/plan.py` are `submission/rl/*`. Already logged in ARCHITECTURE.md's drift list (item 6). Fixed. |
| `submission_rules/main.py` build instruction | Pointed at `rl.export`. Fixed. |
| `rl/` ↔ `tcg/` duplication | ~200 KB across seven pairs, each held together by a manual old-vs-new parity test. Unfinished M6 migration. Documented, not touched. |

### Not dead code (reference count is misleading)

`rl/behavior.py` (24 KB) and `rl/gate.py` have zero inbound imports outside
their own tests, but both are **CLI-invoked instruments** with `__main__` +
argparse. `rl/behavior.py` in particular is the M22b behavioural-counter
instrument whose module docstring explains why it must *not* reuse
`rl.plan.enumerate_plans`. Deleting either on a grep count would have been a
mistake.

---

## What checks out clean

- **Ship state matches the record exactly.** `submission/rl/{__init__,combat,
  encoders,memory,plan}.py` are byte-identical to `rl/*`; `submission/deck.csv`
  matches `decks/alakazam_v2_h4.csv`; `_ATTACH_FIXES` default is
  `telepath,deckguard,ash,conserve,benchfloor,racemode3` = gacfr3.
- **All six shipped fix names resolve** to real branches — no silent typo.
  (Nothing would *catch* one, though: an unrecognised name is silently ignored.
  Minor hardening candidate.)
- **No silently-inert card-fact sets**: `_IS_BASIC_POKEMON` = 595 ids (so
  benchfloor really can fire), `_IS_ENERGY` = 20, and `State.energyAttached`
  genuinely exists, so the telepath/backstop guards bind.
- **racemode3 matches its documented behaviour** — blanket demote gated only on
  `_RACEMODE_WALL_IDS` on the opponent's board, and the post-mortem's own family
  classifier (`scripts/pm_extra.py:20`) keys the wall family on Crustle/Dwebble,
  the same cards. The "mechanically perfect, strategically insufficient" verdict
  is accurate. Latent gap: the trigger needs a wall Pokémon *visible on board*,
  so a Kangaskhan-only wall variant is invisible to it.
- **Test suite**: 662 passed, 1 failed — `test_fetch_offline_error_mentions_runbook`,
  purely missing Kaggle credentials in the audit container, not a code defect.
  Same failure before and after this change.

---

## Consequences for M38 sequencing

The post-mortem ranks the AWR training arm #4, reasoning that 60% of loss mass
is policy quality and "the O-rule seam is thinning". That diagnosis is right,
but the cause is now identifiable: the policy is bad at prize races because
**its teacher was scoring prize-taking at −150,000 per prize**.

1. **Recalibrate `MIN_OVERRIDE_SCORE` (P0b)** — without it the P0 fix only
   half-lands. Needs a pre-registered A/B.
2. **Re-generate the solver corpora.** This change invalidates every
   solver-derived bed and BC corpus. Fresh controls everywhere (drift law).
3. **Then** the AWR arm — running it first would fit AWR to the same poisoned
   labels.
4. The wall BC clone bed (post-mortem P0) is unaffected by this and can proceed
   in parallel; note that `solver:wall`'s strawman problem is now *two*
   independent defects deep.

**No re-ship on this branch.** Nothing here changes the shipped neural agent's
inference behaviour (`turn_solver` is not in the bundle, `rl/plan.py` was left
alone deliberately), so no QC battery is triggered by this change alone.
