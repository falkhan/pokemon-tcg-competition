# Pokémon TCG AI Battle — RL Agent

An agent for the Kaggle [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
competition (`cabt` simulation environment). It receives game observations and returns legal
action indices — first its 60-card deck, then one decision at a time across a full match,
with a ±1 win/loss reward at the end.

**Status (M20, 2026-07-19):** champion `ppo_best_m20legB.pt` + `decks/lucario.csv`, live as
Kaggle submission 54836093. Mirror strength **0.488** vs `solver:lucario` — the campaign's
best neural result, against a bar of 0.55. The neural track has not yet beaten our own
rules-based bundle on the live leaderboard; that is the central open problem. M21 (encoder
v4, plan-head-in-PPO) is in flight on `feature/m21`.

## How it works

Full design in **[ARCHITECTURE.md](ARCHITECTURE.md)**; the short version:

- **A pointer network, not a fixed policy head.** Each decision presents a variable-length
  list of legal options, so the net scores each presented option against an encoded state.
  Action masking is implicit.
- **The teacher is a search, not an agent.** A within-turn solver runs at 5× the live budget
  and labels the states it visits. Training is mostly supervised; PPO fine-tunes on top.
  Imitating a fixed rule agent was measured out early — fidelity caps around 55% when the
  teacher's reasoning is unobservable.
- **Turn plans are explicit inputs.** The net commits to an `(attacker, target, attack)` plan
  at each turn's first MAIN prompt and holds it, which keeps labels consistent with respect
  to the observation.
- **Factory/product split.** Training lives in `rl/` + `tcg/` (PyTorch allowed); the shipped
  `submission/` bundle is numpy-only. A parity gate proves the numpy forward pass matches the
  torch model before anything is packaged.
- **A policy and its deck are one artifact** — they ship together, always.

Milestone-by-milestone history: **[docs/MILESTONES.md](docs/MILESTONES.md)**, with a diary per
milestone in [`docs/`](docs/) and a fast index of findings in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Setup

### 1. Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) (the package manager this repo uses)
- A [Kaggle](https://www.kaggle.com) account — an API token is only needed to submit or
  harvest replays from the CLI

### 2. Download the competition files (required)

The competition-provided files are **not in this repository** (all gitignored):

- `cg/` — the game engine bindings (`api.py`, `sim.py`, `utils.py`, compiled `libcg.so`).
  Nothing runs without them.
- `data/EN_Card_Data.csv` — the card database, needed to regenerate the feature tables.
- The rules/API PDFs.

```bash
uv run kaggle competitions download -c pokemon-tcg-ai-battle
unzip pokemon-tcg-ai-battle.zip -d pokemon-tcg-ai-battle
```

Then place them where the project expects:

1. Copy the **`cg/` folder** from the sample submission into the repository root (it's
   bundled in the official
   [Mega Lucario ex agent](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck)'s
   `submission.tar.gz`, alongside `main.py` and `deck.csv`).
2. Copy **`EN_Card_Data.csv`** into `data/`.

The build pipeline syncs `cg/` into the bundle automatically — Kaggle's agent runtime does
**not** provide the engine, so it must ship inside the tarball.

### 3. Install

```bash
uv sync
uv run pytest tests/ -q     # 474 tests, all offline against a fake engine
```

## Common commands

### Measure an agent

Everything routes through one battle loop with a shared spec grammar
(`model:<ckpt>:<deck>`, `solver:<deck>`, `generic:<deck>`, `rule:<agent>`, `random:<deck>`,
`ext:<path>:<deck>`). Details and current baselines: `.claude/skills/measure-agent/`.

```bash
# mirror series vs the canonical opponent
uv run python -m rl.matchrunner play \
  --a model:checkpoints/<CKPT>.pt:lucario --b solver:lucario \
  -n 400 --workers 10 --seed 1 --checkpoint runs/<name>_s1.jsonl

# meta co-gate: weighted win rate vs the frozen harvested-meta deck pool
uv run python -m rl.replay_bc meta-eval \
  --a model:checkpoints/<CKPT>.pt:lucario --workers 12 --seed 0
```

> ⚠️ **The jsonl `results` code is `0 = side-a WIN`, 1 = loss, 2 = draw.** Summing the array
> gives the *loss* count — a real result of 0.384 reads as 0.620 if you get this backwards.
> Use the decoder snippet in the `measure-agent` skill.

Screen at n=400 on one seed, confirm on a **fresh** seed, then pool. Single-seed numbers are
not trustworthy: the previous champion's pinned 0.534 was seed-0 inflation and measures ~0.465
across two seeds.

### Train a candidate

```bash
# 1. expert collection — the widened turn solver labels states
uv run python -m rl.plan_iter collect --mode expert \
  --games 800 --out data/plan_<name> --workers 12 --seed 1 \
  --opponents self ext:data/external/buddy:decks/lucario.csv rule:lucario \
  --value-ckpt checkpoints/osv3o_setupval1.pt

# 2. supervised train, warm-started from the previous champion
uv run python -m rl.plan_iter train \
  --data data/plan_<name> data/plan_m16 data/plan_m15 \
  --name osv3o_<name> --init checkpoints/<prev>.pt --epochs 5 --lr 1e-4

# 3. optional PPO fine-tune (M20+)
uv run python -m rl.ppo --start <ckpt>.pt --tag <name> \
  --iterations 8 --games-per-iter 400 --lr 3e-5 --kl-coef 0.1
```

Checkpoints store no metadata — **record `val_acc`/`plan_acc` in the diary immediately**.
Spawned workers re-import from disk, so batch code edits *before* launching a run.
Full recipe: `.claude/skills/train-ship/`.

### Build & ship

```bash
# build only -> dist/submission_<agent>_<stamp>.tar.gz
./build_submission.sh --checkpoint <CKPT>.pt --deck lucario

# build + upload (upload is opt-in; Kaggle limits submissions per day)
./build_submission.sh --checkpoint <CKPT>.pt --deck lucario --message "M20: ..."

# the rules-based bundle instead of the neural one
./build_submission.sh --agent rules --deck lucario --message "..."
```

> ⚠️ **Always pass `--deck` explicitly.** `tcg/shipping.py` still defaults to `kyogre`, an
> M1-era fossil — and `decks/kyogre.csv` is actually a Mega Abomasnow water deck, not Kyogre.
> M16 and M18 both shipped it by accident, permanently confounding M16's live score. Verify
> the deck md5 out of the built tarball before submitting.

A failed gate aborts the build, so a broken or train/serve-skewed package cannot be produced.
`build_submission.ps1` is the PowerShell twin; all logic lives in `tcg/shipping.py`.

### Watch games and diagnose losses

```bash
# per-episode loss anatomy of a shipped agent (start every milestone here)
uv run python -m rl.postmortem <episode-id>

# harvest replays + the net's own per-decision logs from Kaggle
uv run python -m rl.kaggle_ingest refresh
uv run python -m rl.kaggle_ingest agent-logs
```

Replays land under `replays/` with an auto-generated `index.html` browser (win-rate table per
matchup, click-to-watch per game). `visualizer.html` feeds a replay's `visualize` JSON to the
official ptcgvis visualizer. Since M19 the shipped agent emits one `NN|{json}` line per
decision on stderr — logits and plan commits — which `agent-logs` joins back to the replay,
so you can see what the net *thought*, not just what it did.

Notebooks in `notebooks/`: `model_monitor.ipynb` (live scores + replay-audited deck identity),
`card_pool_eda.ipynb`, and `deck_analysis.ipynb` (feature extraction into `data/*.parquet`).

## Repository layout

```
├── ARCHITECTURE.md      <- the full design: network, encoders, training loop, ship path
├── CLAUDE.md            <- working agreement / house rules
├── build_submission.sh  <- one-command export -> gate -> package -> (submit)
├── cg/                  <- engine bindings — YOU provide these (setup step 2)
├── rl/                  <- training core + the four modules that ship in the bundle
├── tcg/                 <- readable twins + the authoritative ship pipeline
├── submission/          <- the built bundle: numpy-only main.py + weights + deck
├── sample-agent{,-iono,-tuned,-dragapult}/   <- rule-based teacher agents
├── decks/               <- lucario · kyogre · iono · dragapult + generated candidates
├── checkpoints/         <- trained nets
├── data/                <- training shards, harvested replays, feature parquets
├── docs/                <- milestone diaries M0–M20, DECISIONS.md, MILESTONES.md
├── tests/               <- 474 offline tests + a fake engine
└── reference/           <- official sample notebooks from the competition
```

`rl/` and `tcg/` hold parity-pinned **twins** of several modules (pilot, encoders, combat,
PPO, network). Changing one means changing both — see
[ARCHITECTURE.md §12](ARCHITECTURE.md#12-rl-vs-tcg--the-twin-law).
