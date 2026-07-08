# M1 Plan — Behavior Cloning: "Beat Random"

## Context

M0 proved the full pipeline with a random-weights agent (see `docs/M0.md`): numpy-only
`submission/`, parity-gated build (`build_submission.ps1` → `rl/export.py` + `rl/gate.py`),
replay browser, and a first Kaggle submission. The agent plays legally but loses ~100% to
`random` because its arbitrary initialization systematically avoids attaching Energy
(1,439 ATTACH options offered → 2 taken).

**M1 replaces the random weights with a policy that imitates the rule-based Mega Lucario
sample agent** (`sample-agent/main.py`) via supervised learning (behavior cloning).
Success gate: **≥95% win rate vs the `random` baseline over 100 games.**

Decisions made with Piotr:
- **Two-stage**: Stage A trains on *current* encoders to prove the collect→train→export→gate
  pipeline and set a baseline; Stage B enriches the encoders (discard pile etc.) and
  retrains, measuring the improvement directly.
- **Division of labor**: Piotr writes the BC training core (his first real torch training
  loop — the learning centerpiece); Claude builds the non-learning infrastructure
  (teacher loader, data collection, gate extensions, export wiring, diagnostics).

---

## Stage A — BC v0 on current encoders

### A1. Teacher loader — `rl/teacher.py` (new, Claude)

The teacher is `sample-agent/main.py` (rule-based Mega Lucario). Two loading gotchas
discovered in M0 exploration:

- It reads its deck as `"../deck.csv"` **relative to CWD** (falling back to the Kaggle
  path). Loading must run with `cwd=sample-agent/` (or the loader temporarily chdirs)
  so `../deck.csv` resolves to the project-root `deck.csv`.
- It keeps **module-level mutable state** (`plan`, `pre_turn`, `ability_used`). For
  teacher-vs-teacher self-play, the two players MUST be two *separate module instances*
  (`importlib.util.spec_from_file_location` twice, distinct module names) or their
  globals interleave and corrupt decisions.

Deliverable: `load_teacher(instance_name) -> callable` returning an isolated agent.

### A2. Data collection — `rl/bc.py` collection half (Claude)

Use the **direct engine loop** (`cg.game.battle_start` / `battle_select` /
`battle_finish`) rather than kaggle_environments — the official RL sample
(`reference/reinforcement-learning-and-mcts-sample-code.ipynb`, dissected in
ARCHITECTURE.md §5.4) does exactly this and it's much faster per game.
Engine constraint: one battle per process (`cg/sim.py` global) — single process is fine
for Stage A volumes; multiprocessing is a stretch goal.

Per decision, **encode immediately** (don't store raw obs dicts — they're huge):
- `state_ctx` = `encode_state(obs.current)` + `encode_context(...)` (575 floats today)
- `options` = stacked `encode_option(...)` (N×53)
- `label` = index of the teacher's **first pick** (top-1 imitation; the play-time
  argsort ranking follows from scores)
- `game_id` (for the train/val split), `player_result` (±1 final outcome, for the
  optional value-head loss)

Record **both players'** decisions (both are the teacher). Store as sharded `.npz`
files in `data/bc/` (~200 games/shard). Start with **1,000 games** (~150k decisions,
roughly 1 GB, ~20–30 min); scale only if validation accuracy is still climbing.

### A3. Training core — `rl/bc.py` training half (**Piotr**, Claude reviews)

The instructive part. Components to write, in order:

1. **Dataset + collate**: load shards; each item is (state_ctx, options[N,53], label).
   Batch by padding options to the batch max N and building a boolean pad mask.
2. **Forward + masked loss**: `logits, value = model(state_ctx, options)`; set pad
   logits to `-inf` (or `-1e9`) *before* softmax; loss = `F.cross_entropy(logits, label)`.
   This is the variable-action-space subtlety — masking IS the action space.
3. **Optional but recommended — value loss**: `huber(value, player_result)` added with a
   small weight (~0.5). Free critic warm start for M2's PPO.
4. **Split by `game_id`, not by decision** (~90/10). Decisions within a game are
   correlated; splitting by row leaks.
5. **Loop**: AdamW (lr 3e-4), batch 256, a few epochs; log train/val loss and top-1
   accuracy per epoch to TensorBoard (`torch.utils.tensorboard.SummaryWriter`).
6. **Checkpoint**: save `state_dict` to `checkpoints/bc_v0.pt` when val accuracy improves.

Expected outcome: val top-1 accuracy plateauing somewhere in the 70–90% range (the
teacher's discard-count logic is *invisible* to current encoders — that gap is the
Stage B motivation). Reading that pairs with this work: PyTorch "Datasets & DataLoaders"
and "Optimizing Model Parameters" tutorial pages (links in ARCHITECTURE.md §9).

### A4. Export wiring (Claude)

`rl/export.py`: if `checkpoints/` contains a BC checkpoint, load it into `OptionScorer`
before `save_npz`; otherwise keep the seeded random init (M0 behavior). Everything
downstream (gates, packaging, submission) is already built and unchanged.

### A5. Evaluation (Claude builds helpers; run together)

- Add `option_type_report(agent, n_games)` to `rl/eval.py` — productize the M0 ATTACH
  diagnostic (offered vs chosen counts per OptionType in MAIN context).
  **The headline before/after artifact**: BC should snap the chosen-distribution toward
  the teacher's (ATTACH taken constantly, END rare).
- `play_games(policy, "random", 100, replay_prefix="m1_vs_random")` → **gate: ≥95%**.
- `play_games(policy, teacher, 100)` → baseline vs teacher (expect well under 50%;
  this number is M2's yardstick).
- Skim 2–3 replays in `replays/index.html` — watch it attach Energy and attack on curve.

---

## Stage B — encoder enrichment, retrain, compare

### B1. Enrich `rl/encoders.py` (Claude drafts, walk through together)

Add to the state vector (teacher-relevant first):
- **Discard piles, both players**: pooled card-feature sums (like the hand pool) +
  explicit count of Basic Fighting Energy in own discard (Mega Lucario's attack scales
  on it — the single most teacher-relevant feature).
- **Per-Pokémon-slot**: energy-type counts (10 floats) instead of just `len(energies)`;
  attached-tool pooled features; status conditions on actives (5 bools per player).
- **Stadium**: card features of the stadium in play.

STATE_DIM grows from 511 to roughly 1,100–1,300 (exact number computed in code as now,
from constants). OPTION_DIM unchanged.

### B2. Sync + strengthen the parity gate (Claude)

- Copy updated encoder functions into `submission/main.py` (same copy-discipline as M0).
- **New gate in `rl/gate.py`**: today's parity test feeds *random vectors* through both
  forward passes — it catches weight/shape drift but NOT encoder logic drift. Add
  `encoder_parity_check()`: play a few scripted steps via `battle_start`/`battle_select`
  to get a real mid-game observation, encode it with `rl.encoders` AND the submission
  module, assert `np.allclose`. This makes train/serve encoder skew (M0 lesson #5)
  structurally impossible to ship, which matters exactly now that encoders change.

### B3. Recollect, retrain, compare (Piotr runs; both analyze)

- Recollect (encodings changed → old shards are stale) and retrain → `checkpoints/bc_v1.pt`.
- Compare v1 vs v0: val accuracy delta, vs-random win rate, vs-teacher win rate,
  option-type report. Expected: accuracy up (teacher's discard logic now visible),
  vs-teacher win rate up.

---

## Wrap-up

- `.\build_submission.ps1 -Message "M1: behavior-cloned from Mega Lucario teacher"`.
- Write `docs/M1.md` diary entry (same shape as M0.md: what/decisions/lessons/next),
  including the before/after option-type tables and all win rates.
- Copy this plan into `docs/M1-plan.md` for future reference (user request).
- Update memory (`pokemon-tcg-project-state`) with M1 status.

## Files touched

| File | Change | Who |
|---|---|---|
| `rl/teacher.py` | new — isolated teacher module loader | Claude |
| `rl/bc.py` | collection half (Claude) + training core (**Piotr**) | both |
| `rl/eval.py` | add `option_type_report()` | Claude |
| `rl/export.py` | load BC checkpoint when present | Claude |
| `rl/encoders.py` | Stage B feature additions | Claude (reviewed together) |
| `submission/main.py` | Stage B: re-copy enriched encoders | Piotr (copy discipline practice) |
| `rl/gate.py` | add `encoder_parity_check()` | Claude |
| `docs/M1.md`, `docs/M1-plan.md` | diary + saved plan | Claude |

## Verification (end-to-end)

1. **Collection sanity**: one shard collected → decisions/game ≈ 70–100, labels in range,
   no NaNs; teacher self-play win rates ~50/50 across the shard (globals isolation works).
2. **Training sanity**: loss decreases epoch 1; val accuracy ≫ 1/avg-N (random-pick
   baseline); overfit check on a tiny subset (accuracy → ~100% proves the loss/mask
   plumbing before burning time on full runs).
3. **Gates**: `python -m rl.gate` green (incl. new encoder parity in Stage B).
4. **Behavioral**: `option_type_report` before/after; ≥95% vs random over 100 games;
   replay eyeball test.
5. **Bundle isolation test** (from M0): extract fresh tar.gz to temp dir, run a game with
   only the bundle importable.

## Known risks / gotchas

- Teacher module globals → must use two isolated instances (A1).
- Teacher deck path is CWD-relative (A1).
- Split train/val **by game** or validation lies (A3).
- Stage B invalidates Stage A shards — recollect, don't mix.
- `np.argsort` returns `np.int64` — submission already casts to `int`; keep it that way
  in any new agent wrapper.
- Engine allows one battle per process — collection is sequential for now.
