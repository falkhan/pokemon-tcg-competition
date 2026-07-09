# Pokémon TCG AI Battle — RL Agent

A reinforcement-learning agent for the Kaggle
[Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
competition (`cabt` simulation environment). The agent receives game observations and
must return legal action indices — first its 60-card deck, then one decision at a time
across a full Pokémon Trading Card Game match, with a ±1 win/loss reward at the end.

The project is built around three learnable components sharing one data layer
(see [ARCHITECTURE.md](ARCHITECTURE.md) for the full design):

- **Play policy** — an option-scoring ("pointer") network: the game presents a
  variable-length list of legal options each turn, and the network scores each one
  against an encoded game state. Trained with behavior cloning, then self-play PPO.
- **Value function** — a critic head on the same encoder, predicting win probability.
- **Deck policy** — starts as a fixed known-good deck (Mega Lucario ex); later a
  mutation bandit over legal deck lists.

A core design decision is the **factory/product split**: training lives in `rl/`
(PyTorch allowed), while the shipped `submission/` package is a frozen numpy-only
artifact — every improvement ships as new data files (`policy_weights.npz`,
`deck.csv`), and a parity gate guarantees the numpy forward pass matches the torch
model before anything is packaged.

## Milestones

From [ARCHITECTURE.md §11](ARCHITECTURE.md#11-milestones), with a diary per milestone
in [`docs/`](docs/):

| Milestone | Goal | Status |
|---|---|---|
| **M0 — watch it fight itself** | Submit a random-weights skeleton with a fixed deck; full legal games end-to-end, zero illegal actions/timeouts, one-command packaging | ✅ done ([docs/M0.md](docs/M0.md)) |
| **M1 — beat random** | Behavior-clone the rule-based sample agent; >95% vs `random` | ✅ done — BC agent reaches ~90% vs random and ~70% vs its own teacher ([docs/M1.md](docs/M1.md)) |
| **M2 — beat the teacher** | Self-play PPO with prize-based reward shaping; >50% vs the rule-based Mega Lucario agent | ❌ PPO stalled; supervised value head works; pivoted to search ([docs/M2.md](docs/M2.md)) |
| **M3 — combat features** | Combat-lookahead features to rescue Lucario BC | ❌ refuted — fidelity flat, BC ceiling is structural ([docs/M3.md](docs/M3.md)) |
| **M4 — deck search** | Self-play deck evolution (hill-climb, openskill) | ✅ built, honest — but 0/30 improving mutations vs the small offline field ([docs/M4.md](docs/M4.md)) |
| **M5 — tune the rule agent** | Heuristic parameter optimization + value-head hybrid | ❌ 0/20 tunings beat defaults; hybrid loses — strategic wall confirmed ([docs/M5.md](docs/M5.md)) |
| **M6 — deck-agnostic pilot** | Generic rule pilot that plays any deck + rule-based submission | ✅ pilot at expert-parity deck ranking; probe submitted ([docs/M6.md](docs/M6.md)) |
| **M7 — the deck factory loop** | Deck builder from the full pool + pilot training loop + Kaggle replay ingestion (real-meta field, lookahead search) | 📋 planned ([docs/M7-plan.md](docs/M7-plan.md)) |
| **M∞ — winning BIGLY** 🏆 | | |

## How to run

### 1. Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) (the package manager this repo uses)
- A [Kaggle](https://www.kaggle.com) account (to download competition files; an API
  token is only needed if you want to submit from the CLI)

### 2. Download the competition files from Kaggle (required!)

The competition-provided files are **not in this repository** (all gitignored):

- `cg/` — the game engine bindings (`api.py`, `sim.py`, `utils.py`, and the compiled
  `libcg.so`). Nothing in this project runs without them.
- `data/EN_Card_Data.csv` — the original card database (needed by
  `deck_analysis.ipynb` to regenerate the feature tables).
- The rules/API PDFs.

Get them via the Kaggle CLI (installed by `uv sync`; needs a
[Kaggle API token](https://www.kaggle.com/settings)) from the
[competition page](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle):

```bash
# Competition data: sample submission (with the cg/ engine), card database, rules PDFs
uv run kaggle competitions download -c pokemon-tcg-ai-battle
unzip pokemon-tcg-ai-battle.zip -d pokemon-tcg-ai-battle
```

Then place the files where the project expects them:

1. Copy the **`cg/` folder** from the sample submission into the repository root
   (it's bundled in the sample agent's `submission.tar.gz` — the official
   [Mega Lucario ex rule-based agent](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck) —
   alongside `main.py` and `deck.csv`).
2. Copy **`EN_Card_Data.csv`** into `data/`.

```
pokemon-tcg-competition/
├── cg/                      <- from the sample submission (api.py, sim.py, utils.py, libcg.so)
├── data/
│   └── EN_Card_Data.csv     <- from the competition data download
├── rl/
├── submission/
└── ...
```

The build pipeline (`rl/export.py`) later syncs `cg/` into `submission/` automatically —
Kaggle's agent runtime does not provide the engine, so it must ship inside the tarball.

### 3. Install dependencies

```bash
uv sync
```

### 4. Common commands

```bash
# Run the pre-submission gates: torch/numpy parity, deck legality,
# and a full Kaggle-style game loaded exactly the way Kaggle loads it
uv run python -m rl.gate

# Export training artifacts (weights npz, card features, deck, cg/) into submission/
uv run python -m rl.export

# Behavior cloning (M1): collect teacher self-play demonstrations, then train
uv run python -m rl.bc collect --games 1000
uv run python -m rl.bc train --epochs 10 --name bc_v1
```

Notebooks: `deck_analysis.ipynb` (card database exploration and feature extraction into
`data/*.parquet`) and `testing.ipynb` (scratch experiments with the environment).

### 5. Build & submit (Windows / PowerShell)

```powershell
# Build only: export -> gates -> dist/submission_<timestamp>.tar.gz
.\build_submission.ps1

# Build + submit to Kaggle (needs Kaggle API credentials)
.\build_submission.ps1 -Message "M1: behavior-cloned baseline"
```

A failed gate aborts the build, so a broken or train/serve-skewed package cannot be
produced.

### Watching games

Replays are saved under `replays/` with an auto-generated `index.html` browser
(win-rate table per matchup, click-to-watch per game). `visualizer.html` feeds a
replay's `visualize` JSON to the official ptcgvis visualizer — see
[ARCHITECTURE.md §6](ARCHITECTURE.md#6-watching-it-play--visualization-via-the-cabt-kaggle-environment).

## Repository layout

```
├── ARCHITECTURE.md      <- full design: policies, training plan, tooling decisions
├── docs/                <- milestone diary (M0.md, M1.md, ...)
├── data/                <- extracted feature tables (parquet); EN_Card_Data.csv is downloaded, not committed
├── rl/                  <- training side (torch): encoders, policy, BC, PPO, gates
├── submission/          <- the shipped agent: numpy-only main.py + weights + deck
├── reference/           <- official sample notebooks from the competition
├── sample-agent/        <- rule-based Mega Lucario agent (the M1 BC teacher)
├── cg/                  <- engine bindings — YOU provide this (see setup step 2)
└── build_submission.ps1 <- one-command export -> gate -> package -> (submit)
```
