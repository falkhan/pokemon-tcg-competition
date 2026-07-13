# M8 plan — Teach the agent setup play ("Beat the ship agent" campaign)

**Charter:** produce ANY agent (neural, search, hybrid) that beats the shipped
solver pilot — the campaign bar is **≥ 0.55 vs `solver:lucario` at n ≥ 800**
(±3.5pp), zero G1 crashes, latency within G6, then the ship gate suite.
Opened after the M7.4b L4 go/no-go review (see docs/M7.md closing entry).

**Why now.** The agent doesn't build positions: trainer/item/stadium usage,
action sequencing (discard-then-retrieve chains, fetch-basics-before-
evolutions), bench/energy development. Two PPO runs failed with the same
rise-then-decay signature. Two facts reframe the problem:

1. **The warm start cloned a bug.** `checkpoints/osv2_bc.pt` was BC'd
   2026-07-11 from the generic pilot with the play-tier bug still in it (real
   PLAY options carry no `area`; `score_play` returned a flat 2000 in every
   real game since M6 — fixed 2026-07-12). Both PPO attempts started from a
   policy that had LEARNED "can't use trainers"; the decay diagnosis is
   confounded. BC refresh is a prerequisite, not an option.
2. **The missed combos are within-turn.** Discard-then-retrieve, fetch-order,
   bench building are single-turn action sequences — the L2 turn solver's
   exact domain (`rl/turn_solver.py`), which today fires only near lethal
   (`MIN_OVERRIDE_SCORE` = 1 prize, triggers T1–T4).

**Constraints.** Experiment legs ≤ 1–2 h wall-clock (16-core laptop); anything
bigger needs explicit sign-off. Lucario first but nothing deck-hardcoded — new
scoring/shaping terms use `rl/combat.py` / `_race_features`-style deck-blind
math only (the deck-analysis pipeline will swap decks later). The RL core
(`compute_gae`/`ppo_update`/`mcts_search`) is Piotr's code: changes there are
proposed in the diary, never made unilaterally.

**Conventions.** Diary = `docs/M8.md`, one dated entry per leg (including
kills): hypothesis, exact command, n/seed, numbers, verdict, routing. Decide
on n=400 (±5pp) minimum, n=800 for ship calls, never n=100 (the floor-gate
seed-noise incident, M7.md 2026-07-12). Fresh recorded seed per leg. Record
which SOLVER VERSION every number was measured against (M8.1 changes it).

---

## M8.0 — Measurement reset (BLOCKS everything; 2 legs ≤ 1.5 h)

The play-tier fix invalidated all pre-07-12 weakness diagnoses.

**Build (forensics only):** `rl/eval.py::play_games` optional `json_prefix`
(dump `env.toJSON()` per game); `rl/postmortem.py batch` subcommand
(directory → `audit_flags` + `classify_end` aggregate table); four new
setup-taxonomy `audit_flags` (facts-not-judgments style): (a) viable evolution
left in hand at turn end, (b) energy attached off the best
`_turns_to_first_ko` racer while it was unloaded, (c) fetch chose a basis-less
evolution, (d) discard-retrieval item unused with key cards in discard.

**Leg 1 — baselines (~75 min):**
```
uv run python -m rl.matchrunner play --a solver:lucario --b generic:lucario -n 800 --workers 14 --seed 31
uv run python -m rl.matchrunner play --a model:checkpoints/osv2_bc.pt:lucario --b solver:lucario -n 400 --workers 14 --seed 32
uv run python -m rl.matchrunner play --a model:checkpoints/ppo_best.pt:lucario --b solver:lucario -n 400 --workers 14 --seed 33
uv run python -m rl.matchrunner play --a solver:lucario --b rule:lucario -n 400 --workers 14 --seed 34
```
**Leg 2 — taxonomy (~60 min):** ~100 kaggle-env games with `json_prefix`,
batch-postmortem the losses → flag-frequency table; hand-check 2–3 episodes
with `--decisions`.

**Routing:** setup-dominated taxonomy → weights for M8.1 leaves and M8.3 φ.
Race/prize-dominated → the "can't set up" framing was mostly the dead bug;
shrink M8.1, prioritize M8.2/M8.3.

## M8.1 — Setup-mode turn solver (engineering; ~1 build day + 2 legs)

All in `rl/turn_solver.py` (pilot twins untouched, bundle-pure):
- `should_solve` **T5**: underdeveloped board (best `_turns_to_first_ko` ≥
  threshold or no ready attacker) AND hand holds ≥ 2 trainers or ≥ 1 Pokémon;
  optional turn gate; shorter dev-solve deadline (~0.2 s).
- `_Snap` extended with root development facts (energy on race-winner,
  evolutions in play, bench size, hand size) so `score_leaf` scores
  development **deltas** (else "pass" ties "develop").
- `score_leaf` development block strictly below `W_THREAT`:
  Δenergy-on-best-racer, Δevolution progress, Δbench, Δhand — weights seeded
  from the M8.0 taxonomy.
- Tiered override: lethal tier keeps `MIN_OVERRIDE_SCORE`; dev tier needs
  `DEV_OVERRIDE_MARGIN` over the stand-pat leaf.
- New `solver-dev:` matchrunner spec so dev-vs-lethal-only A/B runs.

**Gates:** dev-vs-lethal ≥ 0.53 (n=800, p < 0.05), expert + floor
non-regression, latency G6 (mean < 50 ms). One taxonomy-informed tuning pass;
kill after 2 fails → leaf terms migrate to M8.3's φ. If shipped: re-baseline
everything (pool member, promotion opponent, and campaign bar all changed).

## M8.2 — BC refresh from the FIXED teacher + solver-teacher probe (2 legs)

- `rl/bc.py`: `--teacher solver` routing via
  `rl/matchrunner.py::make_pilot(("solver", deck), instance)` in
  `collect_games_v2`. No encoder/architecture changes.
- **Leg A (~90 min):** M7.3 recipe verbatim from the fixed generic pilot →
  `osv2_bc2.pt`. Gates: fidelity ≥ 0.90 (ref 0.914); G3 vs generic n=400
  (ref 0.44); vs `solver:lucario` n=400 (ref from M8.0). **Must beat osv2_bc
  on both strength reads** — it clones a strictly better teacher; if not,
  debug the pipeline before touching PPO.
- **Leg B (~45 min):** 800-game `--teacher solver` probe → fidelity delta.
  Drop > 3pp → solver picks alias (expected tiny: overrides ≈ 1% of prompts);
  use the generic teacher only.

## M8.3 — PPO decay probe (2 legs × 8 iters; NOT "attempt 3")

Bounded falsification of "decay = bug-cloned warm start + missing setup
credit". Cap 2 legs; extension needs sign-off.

- `rl/collector.py`: `_dev_potential(state)` — φ = w·[race delta
  (`_race_features[7]`), ready-attacker fraction (`[3]`), evolution progress,
  bench fraction]; potential-based at the existing race-shaping site. CLI
  `--shaping {race,dev}` threaded through `rl/ppo.py`. No encoder dim changes.
- `rl/ppo.py`: expose `--eval-every` / `--eval-n` in argparse.
- **Leg A:** attempt-2 config, `--start osv2_bc2.pt`, 8 iters, eval-every 2
  (isolates the warm-start effect). **Leg B:** + `--shaping dev --lr 3e-5`.
- **Continue** (sign-off for 25 iters) iff iter-8 vs_solver ≥ baseline + 3pp
  AND iter-8 ≥ iter-4 AND entropy flat. **Kill** on the M2 signature →
  [Piotr's call] leg C: KL-anchor to the BC prior in `ppo_update`
  (β ≈ 0.01–0.05) — proposed only.

## M8.4 — L4 instruments (parallelizable spikes)

The two documented-but-never-executed fixes for the measured MCTS failure
(value head OOD + Snorlax determinization, docs/M2.md, DECISIONS.md).

- **(a) L3 archetype determinizer (~45 min):** `rl/determinize.py` (or extend
  `rl/mcts.py::determinize`'s `opp_deck` path) — infer archetype from revealed
  cards via pooled-FEAT cosine vs `data/kaggle/meta_v1` (the 98% probe
  representation), sample hidden zones from the matched list minus revealed.
  Ground-truth harness in self-play (we control both decks): top-1 archetype
  accuracy + deck Jaccard by turn. **Gate: top-1 ≥ 0.9 by turn 4. Fail → L4
  no-go.**
- **(b) Value on search-visited states (~90 min):** extend
  `rl/value_train.py::collect` to also record K determinized-rollout states
  per real decision, outcome-labeled; retrain (frozen-body recipe unchanged).
  Then the decisive instrument on the historical v1 stack: `make_mcts_agent`
  sims ladder {16, 32, 64} vs own greedy, n=200/point. **Single question: does
  win rate rise monotonically with sims?** Monotone + ≥ 0.55 @64 → M8.5 go
  case. Flat with both fixes → L4 dead on this budget.

## M8.5 — L4 go/no-go (Piotr's call, by construction)

Inputs: M8.0 taxonomy, M8.1 ship result, M8.3 verdict, M8.4 instruments, and
the reference notebook (`reference/reinforcement-learning-and-mcts-sample-code
.ipynb` — study, don't rewrite). The AZ-lite loop is multi-day and needs
explicit sign-off regardless. No-go fallback: M8.1-style solver engineering +
per-deck refreshed BC is already an "any agent that wins" candidate, and the
team's deck-analysis pipeline becomes the next lever.

## Decision tree

```
M8.0 ──▶ M8.1 build ∥ M8.2 leg A ∥ M8.4(a)   (disjoint files; RUNS serialize on cores)
M8.1 PASS → ship via gates, re-baseline; FAIL×2 → leaves → M8.3 φ
M8.2 → osv2_bc2 = M8.3 start; solver-teacher per fidelity probe
M8.3 holds → sign-off to extend; decays → [Piotr] KL-anchor or PPO falsified
M8.4(a) fail → L4 no-go | (b) monotone ≥0.55 → M8.5 go case | flat → L4 no-go
```

## Piotr's calls, explicitly
1. Any `ppo_update`/`compute_gae` change (the M8.3 KL anchor). 2. Any
`mcts_search` change (none proposed). 3. Any run > 2 h. 4. Every ship flip.
