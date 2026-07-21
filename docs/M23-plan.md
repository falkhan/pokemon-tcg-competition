# M23 Plan — converge against strong opponents, then decide the architecture

## Context

M22 established: every pilot we build sits at 0.17–0.22 vs `rule:dragapult` while the company
sample agent scores ~0.50 on the same deck; the training loop was anchored to weak endogenous
opponents (`solver:lucario`, 45% of B3's training, beaten 2:1 by the sample agent); the one
positive-direction lever never tested to convergence is the strong teacher (M22c-RL stopped at
10 iters with in-loop WR vs `rule:lucario` still climbing 15%→18%, 0.365 at n=200). Live
forensics (docs/m22-post-mortem.md) added: Gap A produces 0–6 shutouts on-ladder vs kangaskhan;
the setup/draw-engine failure (hand empty from turn 4, Carmine unplayed) plausibly dominates the
~76%-mirror field; every solver-piloted offline cell is endogenous (kangaskhan 0.808 offline vs
0.30 live). Nulled levers not to respend: deeper within-turn search, threat-feature tiebreaks,
loss-only behavioral fixes.

M23 = converge the strong-teacher recipe cheaply (Phase 1); its outcome decides whether the next
milestone-sized spend is training-side or architecture-side (Phase 2, contingent). Piotr's
decisions (2026-07-21): promotion trigger switches to vs_teacher (rl/ppo.py edit approved);
gust-boost dropped, plan-coef kept; BC-clone evaluator is **committed** in scope.

Documentation first: copy this plan to `docs/M23-plan.md` (replacing the stub), diary
observations into `docs/M23.md` incrementally, hermes Telegram updates at every stage
transition + hourly heartbeat on the training run, per CLAUDE.md.

## Phase 0 — instrument fixes (before any training)

1. **Fix the `play_series` seat defect** (`rl/matchrunner.py`): `play_series` is slot-fair
   (side a takes seat `g % 2`) but never tells `game_fn` which seat side a received, so any
   per-seat instrumentation records the opponent half the time (bit us twice in M22).
   Pass the seat through (e.g. extend the `game_fn` callback signature with `a_seat`,
   defaulting compatibly), and fix `rl/behavior.py::from_series` to consume it.
   Regression test in `tests/test_matchrunner.py` next to
   `test_play_series_is_slot_fair_and_maps_draws` (:65-80).
2. **Hard-guard `rule:dragapult` out of training pools**: there is NO code guard today —
   only convention protects the campaign's single clean out-of-loop instrument. Add a check in
   `parse_pool` (`rl/collector.py:119-149`) that raises if any spec resolves to the dragapult
   sample agent (pilot `rule`/teacher `dragapult`, or a deck resolving to `decks/dragapult.csv`),
   with an explicit `ALLOW_DRAGAPULT_TRAINING=1` env override. Test in `tests/` beside the
   existing parse_pool dedupe tests. Mirrors the ship-deck lesson: conventions that cost
   milestones become hard errors.
3. **Supporter-play-rate counter** in `rl/behavior.py`: rate of playing a draw supporter on
   turns 1–6 when one is in hand and the supporter-for-turn is unspent, reported by turn bucket.
   Designed as a **descriptive availability rate for cross-pilot contrast** (us vs the sample
   agent, same deck/opponent), NOT a strict-defect flag — the sample agent is the pricing
   reference, and the report must carry that caveat unconditionally (module discipline,
   `rl/behavior.py:1-30`). Tests in `tests/test_behavior.py` using the existing `_obs(...)`
   builder (already threads `supporter_played=`).
4. **Run the diagnostic** (after 1+3): `rl.behavior series` for `model:ppo_current_m22cRL.pt:lucario`
   vs `rule:lucario`, and `rule:lucario` vs itself (n=200 each, workers 8). Diary the contrast —
   this is the before-picture for Phase 1's convergence claim about setup play.

## Phase 1 — converge the strong-teacher leg (PRIMARY)

**One rl/ppo.py edit (Piotr-approved):** promotion gate switches from `vs_solver` to
`vs_teacher` (`rl/ppo.py:508-521`) — the eval line keeps printing both. Rationale: the solver
is out of the pool and vs_solver is the metric M22 demoted; per the M20 law promotions stay
triggers-only, never evidence. Keep the change minimal; suite must stay green.

**Launch** (recover exact M22c flags from `runs/m22cRL_pipeline.log` before launching; deltas
only, everything else identical to the M22c leg):

- `--start ppo_current_m22cRL.pt` (weights-only warm start; fresh AdamW is expected behavior)
- `--tag m23p1` — **fresh tag mandatory**: reusing `m22cRL` clobbers `ppo_current_m22cRL.pt`
  at iter 0 (`rl/ppo.py:469-470`)
- `--iterations 20` (≈30 total on this recipe), `--games-per-iter 400`, `--workers 8` (hard cap)
- KL **omitted** entirely (`--kl-coef` default 0 → no anchor built; drop `--kl-anneal-to`)
- `--gust-boost 0` (dropped); `--plan-coef`/`--plan-tau` kept at M22c values
- `--eval-every 1` with `--eval-games 200` — per-iter convergence curve is the point of the
  leg; vs_teacher eval is hardcoded n=100 (`rl/ppo.py:512`), fine for a trend line
- `--opponents` (weights normalized by `parse_pool`; dedupe warning expected to stay silent):
  `rule:lucario=0.45  rule:iono=0.15  rule:tuned=0.05  mirror=0.20  past=0.10  random:kyogre=0.05`
  — solver slices fully removed; `rule:iono` adds a cross-deck sample-agent opponent;
  `mirror=` self-strengthens as a free curriculum; NO dragapult (now enforced by Phase 0.2).

**Smoke first**: `--iterations 1 --games-per-iter 40 --eval-games 40` under a throwaway tag to
verify the mixture parses, the promotion edit fires on vs_teacher, and per-iter logging looks
right. Then the real run (~1.5–2.5h judging by M22c's 42 min for 10 iters; heartbeat reporter
attached, hermes updates at start/promotions/finish, diary incremental).

**Pre-registered readout** (decided BEFORE the gates run):

| observation | verdict |
|---|---|
| vs_teacher climbs toward ~0.45+ AND `rule:dragapult` moves >5.5pp vs 0.1888 | training lever VALIDATED — iterate the recipe; architecture change deferred |
| vs_teacher plateaus well below teacher AND dragapult flat | bottleneck is ARCHITECTURE → scope Phase 2 (plan-head redesign) as M24 |
| mixed (one moves, not the other) | diary it, no ship, decide with Piotr |

**Gate battery** (via measure-agent skill; jsonl decode 0=WIN; all 2-seed):

- PRIMARY: `rule:dragapult` n=400 × seeds 1,2 (baselines: M22c 0.1888, B3 0.172; MDE 5.5pp)
- `rule:lucario` n=200 × 2 (convergence corroboration; M22c baseline 0.365)
- `solver:lucario` mirror n=400 × 2 (informational only — contaminated metric)
- `random:kyogre` n=200 × 2 (collapse floor, comparative ≥0.90-ish)
- latency check before any ship talk

**Ship policy**: only on Piotr's explicit call; `build_submission.sh --deck lucario`, verify the
"paired with deck" line AND md5 of bundled deck.csv == `aef8da62…`, review
`verify-before-consequential-actions.md` first.

## Phase 2 — plan-head redesign (CONTINGENT, scope only if Phase 1 forks to architecture)

Not planned in detail here by design. If triggered: make the plan head drive coordinated
multi-step lines with gradient from an opponent that punishes incoherence; encoder-v5-scale
cost (migrate warm start → PPO leg → full re-pin). Write `docs/M24-plan.md` when earned.

**UPDATE 2026-07-21 (Piotr):** before any Phase 2 scoping, run the signal audit in
`docs/m23-signal-audit-plan.md` — the clone result (an OptionScorerV2 with no plan head
playing 0.59 vs the sample agent) shows capacity is not the bottleneck, so "which part needs
rewriting" must be answered by measurement (shadow-eval, credit-assignment probes, and the
V3-distillation discriminator), not by defaulting to the plan head.

## Phase 3 — BC-clone a leaderboard opponent as evaluator (COMMITTED)

Goal: the first evaluator that is neither our solver nor a company rule agent — aimed at the
pilot-quality blindness proven by the 0.808-offline/0.30-live kangaskhan gap.

1. **Audit data volume**: `uv run python -m rl.kaggle_ingest bc-shards --min-score 600` —
   reports opponent seats with (observation, action) streams. Pick the clone target from
   `targets`/leaderboard output: highest-n strong opponent (prefer a strong solrock pilot —
   most games — or the kangaskhan pilot if volume allows).
2. **Add a per-opponent selection filter to `rl/replay_bc.py::build`** (~:381-400): filter
   episodes/seats by `submission_id_{seat}` (and/or deck hash). Metadata already in
   `episodes.parquet`; only the predicate is missing. Tests beside `tests/test_meta_eval.py` /
   existing replay_bc tests.
3. **Train the clone** with the existing BC path (`rl/bc.py` infra), tag `bc_clone_<subid>`.
4. **Validate fidelity BEFORE any gate status**: held-out action accuracy on its own replays,
   plus behavioral comparison (behavior counters on clone-piloted series vs the live replays).
   If fidelity is poor, record it and stop — a bad clone as a gate would be a new endogeneity
   trap wearing an exogenous costume. Its measurement role, if validated: an additional
   informational floor next to dragapult, never a replacement.

## Live monitoring (parallel, no compute)

B2/B3/M22c-RL keep accruing; re-run `notebooks/m22_ab_monitor.ipynb` (cell 2 then 3) daily-ish;
the refusal gate holds any verdict below MDE. If a Phase 1 candidate ships, add its sub id to
the notebook `ARMS` and the model monitor.

## Verification

- Every code change lands with tests: `uv run pytest tests/test_matchrunner.py
  tests/test_behavior.py tests/test_collector*.py -v` for Phase 0; full suite
  (`uv run pytest tests/ -q`, last count 574 green) before the training launch and before any
  ship. rl/ppo.py promotion edit: extend the existing ppo test coverage minimally or add a
  targeted unit test if none covers promotion.
- Smoke leg (1 iter × 40 games) validates the launch config end-to-end before the real run.
- Gate results diaried in `docs/M23.md` at the moment produced, incl. kills; hermes updates at
  every stage transition and on any failure.

## Explicitly out of scope / do-not-do

- No `rule:dragapult` in any training pool (now enforced in code).
- No deeper within-turn search, no threat-feature consumers, no loss-only behavioral fixes.
- No ship without explicit instruction + deck verification ritual.
- No `kaggle_ingest meta` runs (silently repoints every meta-eval).
- No conclusions from sub-MDE deltas anywhere, including the per-iter n=100 teacher curve.
