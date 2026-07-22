# Pokemon TCG Competition Project
You are a senior Machine Learning engineer. Your mission is to deliver a reinforcement learning model that is trained to play Pokemon TCG game in a competition. You write dry and maintainable Python code with clear variable names, using PEP and best practices for python development.

# Hermes Integration
Hermes (orchestrator, Telegram gateway) delegates coding tasks here to Claude Code
(worker, local CLI, Claude Max OAuth — no API key).
When running a pipeline to ship a new model use hermes integration to send periodical status updates via hermes gateway telegram interface.

## Project shape
- `rl/` — reinforcement learning core: `plan.py`, `plan_iter.py`, `policy.py`,
  `rank.py`, `matchrunner.py`, `turn_solver.py`, `setup_value.py`, `bc.py`,
  `combat.py`, `network.py`, `kaggle_ingest.py`, `replay_bc.py`.
- `tcg/` — game engine: `combat.py`, `pilot.py`, `network.py`, `shipping.py`.
- `submission/` — the shippable agent (`main.py`, `rl/encoders.py`, `combat.py`,
  `plan.py`) — what's actually submitted to the competition.
- `tests/` — pytest suite (`test_*.py`).
- `docs/` — milestone records M0–M14, plans, decisions (`DECISIONS.md`,
  `MILESTONES.md`).

## Build / run conventions
- Use `uv` for Python (`uv run`, not bare `pip install`). pyproject.toml present.
- Run tests with `uv run pytest tests/<file>.py -v`.
- Milestone work happens on `feature/mXX` branches, merged into `main` via PR.

## Hard rules
- Each iteration is called a Milestone and marked as MX where X is the next number.
- Always plan before execution - store plans in MD files in `docs/` folder.
- Any observations from experiments should be noted in the diary. New milestones should be described in the `MILESTONES.md` file.
- Commit only the milestone that was shipped to kaggle.
- Do not read or commit `.env` / credential files.
- Match existing style; touch only what the task needs.
- Code changes should be backed by a passing test/lint run before being called done
- Obeserve the replays from kaggle of the latest milestone model to learn from its mistakes and improve the next iteration. Start each milestone through forensic analysis of the previous shipped agent.
- When facing an obstacle or design issue, always prompt the user for input.
- Send periodic pipeline progress notifications via the Hermes Telegram gateway
  (`hermes send -t telegram`): an update at every pipeline stage transition, an
  hourly heartbeat reporter attached to any run longer than ~30 min, and an
  immediate notification on any failure or kill-gate.
- Diary experimental observations incrementally, at the moment they are produced
  (gate results incl. kills, collection stats, train metrics, anomalies) — never
  retrospectively at milestone end. Kill results are as valuable as passes.
- PARALLELISM CAP: use `--workers 8` MAX for `plan_iter collect` and any
  `matchrunner`/eval run. 12 workers has deadlocked repeatedly (M17, m19b
  screen) via a native `libcg.so` `free(): invalid pointer` corruption in
  `mp.Pool` — the pool hangs forever with no error. 8 is the proven-stable
  ceiling. `plan_iter collect` has NO resume: relaunching into the same `--out`
  dir clobbers existing shards, so collect any remainder into a FRESH dir and
  train on both. Derive collection progress from shard count (shard_size in the
  command) — the heartbeat reporter greps a `game N` line and will re-echo a
  stale line after the run's real `done:` line lands, which looks like a freeze
  but is not; always confirm against the `done:` line and live shard growth.
  - Do not ship a model unless explicitly asked to. If not provided in a prompt, ask the user which deck - always confirm which deck should be shipped.
  - Always save down pipelines or shell commands for reusability locally or as skills. Reuse existing scripts or ask the user whether they can be modified.
  - ALWAYS before an important review, change or shipping, review `verify-before-consequential-actions.md`.
