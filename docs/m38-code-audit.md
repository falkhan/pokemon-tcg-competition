# M38 pre-work — code audit off the M37 post-mortem

Forensic read of the shipped pipeline against `docs/m37-post-mortem.md`,
`docs/m36-post-mortem.md` and the diary, looking for dead code, inconsistencies
and bugs — specifically anything that could explain why the pilot underperforms
its rule set. Branch: `claude/pipeline-postmortem-review-66anve`.

Two engine-verified probes were promoted into `scripts/` so every claim here is
re-runnable:

```
uv run python scripts/prize_semantics_probe.py -n 6        # the engine fact
PKM_PRIZE_FIX=0 uv run python scripts/solver_prize_probe.py -n 10
PKM_PRIZE_FIX=1 uv run python scripts/solver_prize_probe.py -n 10
```

---

## The engine fact everything below turns on

`player.prize` is **that player's OWN remaining prizes**. It drains as *they*
take prizes; the winner's own array ends at 0. Probe output (6 games,
generic:lucario vs generic:kyogre) — every prize-out game votes A:

```
winner=1 prize_lens=(6, 0)  -> A (own-needs)
winner=1 prize_lens=(3, 0)  -> A (own-needs)
winner=1 prize_lens=(2, 0)  -> A (own-needs)
winner=0 prize_lens=(5, 6)  -> deck-out (uninformative)
```

This is what M36 pinned from diag end-states and the kyogre probe
(`docs/M36-plan.md`, "prize-array semantics pinned"). It is **not** what
`sample-agent/main.py` assumes (`len(op_state.prize) <= prize` as the
game-winning KO test) — and the sample agent is where our convention was
inherited from. Three of our modules inherited the error with it; three others
(`rl/encoders.py`, `rl/postmortem.py`, O11 `gustveto`) are correct. That split
is why it survived 37 milestones: the surfaces Piotr reads (post-mortems, the
prize-race feature, the gust veto) all agree with reality.

---

## Finding 1 (headline) — the within-turn solver is paid to NOT take prizes

`rl/turn_solver.score_leaf` credited its two prize terms to the wrong arrays:

```python
score  = W_PRIZE    * max(0, snap.op_prizes - len(op_p.prize))   # "prizes I took"
score += W_MY_PRIZE * max(0, snap.my_prizes - len(me_p.prize))   # "prizes I conceded"
```

With own-needs arrays, `my_prizes` drains when **I** take prizes — so
`W_MY_PRIZE = -150_000` is charged for every prize the solver takes, and
`W_PRIZE = +100_000` pays out only when the opponent takes prizes (inside my own
turn: self-KO effects only, i.e. almost never). Every prize-taking line scores
about **-150k against a stand-pat floor of 0**, and `solve_turn` only overrides
greedy at `MIN_OVERRIDE_SCORE = W_PRIZE - 1`.

Measured with `scripts/solver_prize_probe.py` (solver:lucario vs
generic:lucario, seed 7), two independent series:

| series | | trigger fires | overrides | of which prize-taking lines |
|---|---|---|---|---|
| n=10 | shipped (`PKM_PRIZE_FIX=0`) | 229 | **7 (3.1%)** | **0** |
| n=10 | corrected (`PKM_PRIZE_FIX=1`) | 416 | **117 (28.1%)** | **104** |
| n=6 | shipped | 131 | **7 (5.3%)** | **0** |
| n=6 | corrected | 170 | **78 (45.9%)** | **68** |

(Fire counts differ between arms because overriding changes the trajectory.
The W/L these series report is a two-digit sample and is NOT an A/B result —
strength has to come from a proper battery.)

The module exists to assemble the `item -> attach -> attack` multi-prize turn
greedy misses (its own docstring). Shipped, it has never once overridden greedy
for a prize haul — only for lines that win the game outright, which score
`W_WIN` from `cur.result` and bypass the prize terms entirely.

**Blast radius** — the solver is *not* in the submission bundle, so no live
agent decision is affected. What is affected is everything measured or taught
with it:

- **every `solver:` bed**, including the campaign bar (`>= 0.55 vs
  solver:lucario`) and `solver:wall`. M37 called `solver:wall`
  "strawman-invalid, the solver pilot over-digs its own deck" — a solver that
  cannot cash a prize line and keeps scoring development instead is exactly
  what over-digging looks like. The bed did not lie at random; it lied in a
  direction this bug predicts.
- **`plan_iter collect --mode expert` labels** — `solve_turn_line` is the
  teacher for the plan corpora the shipped net was trained on.
- **`score_siblings`** — the M12 ranker labels. M12's own verdict was
  "score_leaf is the bottleneck" and M13's was "score_leaf must be replaced,
  not distilled". They were right about the symptom and never found this cause.

Also fixed under the same switch: the **T4 trigger**, `len(op.prize) <= 2`
labelled "game-closing range", which fires when the *opponent* is two prizes
from winning rather than when we are.

## Finding 2 — the plan features `wins` / `concedes` read the wrong player

`rl/plan._make_plan` compares `target_prize >= len(op.prize)` for `wins` and
`attacker_prize >= len(me.prize)` for `concedes` — both arrays swapped. These
are plan-vector slots 22 and 25, fed to the policy at every MAIN prompt.

Known since M36, which correctly ruled "DO NOT fix in a rules milestone —
feature-distribution shift on a frozen net; queue for the next training arm".
That ruling stands; this branch only makes the fix available and tested, and
corrects the comment that asserted the wrong semantics.

## Finding 3 — PPO prize shaping has the sign flipped

`rl/collector.py` and its `tcg/selfplay.py` twin:

```python
delta = (6 - len(op.prize)) - (6 - len(me.prize))   # "prizes I took - they took"
```

is `they took - I took`. The learner was paid `+PRIZE_SHAPING` per prize the
**opponent** took and docked for its own. Bounded impact — the terminal ±1
dominates a shaping term whose whole-game range is ±0.6, which is why M20/M21
trained at all — but the dense signal pointed backwards for every PPO arm in
the campaign.

## Finding 4 — `MIN_OVERRIDE_SCORE` is stricter than its docstring (follow-on F1)

`MIN_OVERRIDE_SCORE = W_PRIZE - 1` is documented as "override greedy only for
>= 1 prize or a win". The leaf's negative tail (`W_RACE` -10/turn, `W_COUNTER`
-1000 x prize value, `W_DECK_LOW` -5000/card) routinely drags a genuine
one-prize line below it: a clean 1-prize leaf with race 2 scores 99,980 vs a
bar of 99,999. Dormant while finding 1 kept everything far from the bar; live
the moment it is fixed. Pinned in `tests/test_prize_semantics.py`; the bar
wants re-tuning as part of the finding-1 A/B, not before it.

## Finding 5 — `kaggle_ingest` dies instead of reporting (fixed outright)

`import kaggle` authenticates at import time and calls `exit(1)` when
`~/.kaggle/kaggle.json` is absent. `SystemExit` is not an `Exception`, so it
escaped the `except Exception` that produces the actionable `[NET] ... docs/M7.md`
runbook error: `kaggle_ingest refresh` inside a pipeline killed the process with
no message and no Hermes failure ping. `tests/test_kaggle_ingest.py::
test_fetch_offline_error_mentions_runbook` was red on any box without
credentials. Both fetchers now catch `SystemExit` too.

## Finding 6 — the CLAUDE.md "collect clobbers shards" rule is not what the code does

CLAUDE.md's standing rule says `plan_iter collect` "has NO resume: relaunching
into the same `--out` dir clobbers existing shards". The code has offset shard
indices past existing files since M11 (`rl/plan_iter.py`, and the same in
`rl/bc.py` / `rl/setup_value.py`):

```python
shard_idx = sum(1 for _ in out.glob(f"shard_w{worker:02d}_*.npz"))
```

A re-launch **appends**; nothing is overwritten, and per-worker globbing means
no race between workers. The engine takes no seed (`cg.game.battle_start`), so
the appended games are fresh samples, not repeats. The genuine caveat is
different and smaller: `game_ids` restart at 0 each run, so the by-game
train/val split groups run A's game N with run B's game N.

`collect` now prints a `[COLLECT] ... APPENDS` notice with that caveat when the
target dir is populated. **The CLAUDE.md rule wants correcting — left for
Piotr, since it is his instruction file.**

## Finding 7 — the worker cap was a convention, not a guard (fixed outright)

CLAUDE.md caps pools at 8 workers because 12 deadlocked repeatedly (M17, m19b)
via a native `libcg.so` `free(): invalid pointer` in `mp.Pool` — a hang with no
error, the failure mode the heartbeat cannot distinguish from slow progress.
Nothing enforced it. `rl/matchrunner.check_workers` now refuses `workers > 8`
in both `matchrunner.run_pairs` and `plan_iter.collect`, with
`PKM_ALLOW_UNSAFE_WORKERS=1` as the deliberate override.

## Finding 8 — the deck-out classifier undercounts (fixed outright)

`rl/postmortem.classify_end` only calls DECK-OUT at `deck == 0`. M37 filed two
losses as "unclassified" because we ended at `deck == 1`, where the next
mandatory draw empties the deck and the one after kills us — the post-mortem
had to correct the count by hand. `deck <= 1` now classifies, with its own
message so the two cases stay distinguishable.

## Dead code / inconsistency sweep — clean

- No unreferenced top-level defs beyond `RecordingAgent`, `_spec_pilot` and
  `__post_init__` (test-seam and dataclass hooks, all legitimate).
- `rl/plan.py` still carries the killed `racemode`, `racemoder` and `racemode2`
  arms. They are unreachable unless named by a spec kind and each is the
  measured control for `racemode3`'s battery — kept deliberately.
- `submission/rl/{plan,combat,encoders,memory}.py` are byte-identical to their
  `rl/` originals (`test_bundle_twins`), and `submission/main.py`'s default fix
  string matches `modelt-gacfr3` exactly. No ship drift.
- The `rl/` vs `tcg/` twins remain parity-pinned by `tests/test_parity.py`.

---

## What this branch changes

Nothing that a live game sees. Findings 1-3 are behind a **single opt-in env
var, `PKM_PRIZE_FIX`, defaulting OFF**, on the `WHOLE_BOARD_THREAT` precedent
(an env-driven falsification switch, so an A/B needs no mid-run edit and spawn
workers inherit it):

| module | OFF (default) | ON |
|---|---|---|
| `rl/plan.py` | today's `wins`/`concedes` — the features the frozen net was trained on | own-needs arrays |
| `rl/turn_solver.py` | today's leaf + T4 — every historical bed number reproduces | prizes taken pay, prizes conceded charge |
| `rl/collector.py`, `tcg/selfplay.py` | today's shaping sign — M20/M21/M22c reproduce | +0.1 per prize *we* take |

Fixed outright (no measurement depends on the old behaviour): findings 5, 7, 8,
the `[COLLECT]` notice, and every comment/docstring that asserted the wrong
prize convention.

`tests/test_prize_semantics.py` (12 cases) pins both branches of all three
switches plus the finding-4 bar, and asserts that all four modules read the
same switch — so flipping a default is a one-line change with its expected
behaviour already written down.

Full suite: **678 passed** with the flag off (baseline was 661 passed + 1
failed — finding 5's red is now green).

With `PKM_PRIZE_FIX=1` the suite reports **5 failures, all fixture conventions,
not regressions** — these are the tests whose fake boards were built with the
swapped arrays, and they are the exact edit list for whoever flips the default:

```
tests/test_plan.py::TestEnumeratePlans::test_risk_flags_on_sacrificial_plan
tests/test_turn_solver.py::test_headline_finds_the_multi_prize_lethal_greedy_misses
tests/test_turn_solver.py::test_multi_select_prompt_capped_and_lethal_pair_found
tests/test_turn_solver.py::test_trigger_table[T4-my_p3-op_p3-True]
tests/test_turn_solver.py::test_score_leaf_ordering
```

Worth reading `test_score_leaf_ordering` in particular: it asserts
`leaf(op_prizes=3, my_prizes=3) < one_prize  # conceding a prize costs`, i.e.
the suite itself documented "my array draining" as conceding. The bug had a
matching test, which is how it stayed invisible.

## Recommended M38 sequencing

1. **Re-baseline the beds with `PKM_PRIZE_FIX=1` before anything else.** The
   M37 post-mortem's P0 is "wall BC clone first, `solver:wall` is
   strawman-invalid". Finding 1 gives that verdict a mechanism and a cheaper
   first move: re-run the wall/grim/rocket pins with the fixed solver and see
   whether the strawman is repaired rather than replaced. If it is, the clone
   is a smaller job; if it is not, the clone was needed anyway and now rests on
   a teacher that can cash a prize line.
2. **Then the AWR training arm** (the M37 headline) on corpora collected with
   the fixed teacher and the fixed `.prize` features — one arm, both switches
   on, since neither is separable from a re-collect.
3. **Re-tune `MIN_OVERRIDE_SCORE`** inside that A/B (finding 4).
4. PPO arms inherit the shaping fix from the same flag whenever they next run.

Ordering note: a fixed teacher changes the corpus, so the AWR arm and the
prize-fix arm are one variable, not two — pre-register them together or the
attribution is lost.
