# M12 plan: value-as-ranker probe (M9-plan Leg 4, adapted to the M11 stack)

Status: **started 2026-07-17, branch `feature/m11`.** Diary: [M12.md](M12.md).

## Context

M11 closed at `osv3_plan0c` 0.415 (pinned n=1200): the bar-honoring teacher's
improvement operator fires on ~7% of turns; on the other 93% the student
ceiling is greedy-quality development play. M8.4/M5 measured that a win/loss
value head cannot RANK sibling actions; M9-plan Leg 4 pre-registered the fix —
a pairwise ranking head trained on the solver's sibling scores, offline-gated.
M12 runs that leg on today's stack: labels from the WIDENED `solve_turn_line`
(2.0s, dev-aware scoring — far deeper than the live 0.4s solver), the ranker on
the OptionScorerV3-shaped option pathway, and a consumer that applies the
widened search's CONFIDENT preferences on every turn.

Known dead ends this walks past deliberately (differences noted):
- M5 1-ply value-hybrid (classifier, no margin) — M12 uses ranking loss +
  margin-filtered pairs + confidence-gated override.
- M8.1 dev-tier override everywhere (0.314) — M12 overrides only above a
  learned-confidence margin, greedy otherwise.
- M8.4 MCTS sims ladder (flat) — no inference MCTS here; 1-ply only.

## Design

1. **`score_siblings(obs, deck, deadline_s, dev=True)`** (`rl/turn_solver.py`):
   at the root prompt, for each `_candidate_actions` child: `search_step` one
   ply, `_dfs` the remainder (shared deadline/budget), return
   `list[(action, score)]` aligned to root options. Pure recombination of
   solve_turn_line machinery, same search_begin/end bracket.
2. **Collection** (`rl/rank.py`): teacher self-play (greedy pilots — the
   states must be ON-distribution, the M11 round-1 lesson); at every own MAIN
   prompt with ≥2 options and search available, record (state_ctx, state_ids,
   option encodings, per-option sibling scores; options not in the beam get
   score = NaN → excluded from pairs). Shards: v2 columns + `sib_scores`.
3. **Training** (`rl/rank.py`): fresh `OptionScorerV3`-shaped net (plan input
   zeros — reuse the class, no new arch), **pairwise logistic loss** over
   same-prompt option pairs whose score gap ≥ `RANK_MARGIN=200` (skip aliased
   ties): `-log sigmoid(logit_i - logit_j)` for score_i > score_j + margin.
   → `osv3_rank1.pt`.
4. **Offline gate (Gate 1, kill-fast, zero games):** held-out pairwise
   accuracy on margin-filtered pairs **≥0.70**; kill below (M9-plan Leg 4
   verbatim).
5. **Consumer (Gate 2)** — `rank:<ckpt>:<deck>` matchrunner spec: the solver
   pilot, plus on own MAIN prompts where the solver tier does NOT override:
   1-ply lookahead over the `_candidate_actions` beam, rank the resulting
   states' leaves with the net, override greedy ONLY when
   `top1_logit - top2_logit ≥ CONF_MARGIN` (default 1.0, one value, no sweep).
   Gate: n=400 seed 1 vs `solver:lucario` **≥0.445** (+3pp over the 0.415
   baseline); kill <0.415. Screen-only session; n=800 confirm only on go.

## Budget

Collection 400 games ≈ 30 min (12 workers, one 2.0s widened solve per MAIN
prompt) · train ≈ 15 min · offline gate = minutes · consumer gate ≈ 20 min.
Total well under 2h. No `rl/ppo.py`; edits batched before runs.

## Files

`rl/turn_solver.py` (+`score_siblings`), new `rl/rank.py` (collect/train/CLI),
`rl/matchrunner.py` (+`rank:` spec), `tests/test_rank.py`, `docs/M12.md`.
