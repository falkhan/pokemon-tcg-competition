# Pokémon TCG AI Battle — RL Agent

An agent for the Kaggle [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
competition (`cabt` simulation environment). It receives game observations and returns legal
action indices — first its 60-card deck, then one decision at a time across a full match,
with a ±1 win/loss reward at the end.

**Status (M43, 2026-08-12):** two live submissions — **55265099**
(`m41b_wide_prod` width-143 net + `decks/alakazam_v2_h4.csv`) and **55265105**
(`m41_ogerpon` + `decks/ogerpon.csv`). Best settled implied ELO to date: **817**
(M30/M35); the campaign bar is **1000+**. M43 (honest PPO + ogerpon self-play,
`docs/M43-plan.md`) is in flight on `feature/m43`, running on a fresh box —
the full artifact chain is rebuilt locally from the recipes in this repo.

## How it works

Full design in **[ARCHITECTURE.md](ARCHITECTURE.md)**; the short version:

- **A pointer network, not a fixed policy head.** Each decision presents a variable-length
  list of legal options, so the net scores each presented option against an encoded state.
  Action masking is implicit.
- **Imitation first, then honest RL.** The net clones harvested leaderboard replays
  (`rl/replay_bc.py`) and our own solver-labeled games (`rl/plan_iter.py`); `rl/ppo.py`
  fine-tunes on top with correct losses, a KL anchor to the frozen start, and optional
  invariant potential shaping (`F = γΦ(s′) − Φ(s)`, Φ(terminal)=0).
- **Turn plans are explicit inputs.** The net commits to an `(attacker, target, attack)` plan
  at each turn's first MAIN prompt and holds it, which keeps labels consistent with respect
  to the observation.
- **Factory/product split.** Training lives in `rl/` + `tcg/` (PyTorch allowed); the shipped
  `submission/` bundle is numpy-only. A parity gate proves the numpy forward pass matches the
  torch model before anything is packaged.
- **A policy and its deck are one artifact** — they ship together, always.
- **Measurements are pre-registered.** Gates run from hashed spec JSONs
  (`scripts/gate_spec.py`); editing a bar after seeing the numbers produces a refusal, not
  a kinder verdict.

Milestone-by-milestone history: **[docs/MILESTONES.md](docs/MILESTONES.md)**, with a diary per
milestone in [`docs/`](docs/) and a fast index of findings in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Setup

### 1. Prerequisites

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) (the package manager this repo uses)
- A [Kaggle](https://www.kaggle.com) account with API credentials — needed to harvest
  replays and submit. Either `~/.kaggle/kaggle.json` or the
  `KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables work (the CLI and
  `rl/kaggle_ingest.py` resolve both).
- Optional: an NVIDIA GPU. Training (`plan_iter train`, `rl.ppo`) takes `--device auto`
  and uses CUDA when available; game collection, evaluation and the shipped bundle are
  CPU-only by design.

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
uv run pytest tests/ -q            # ~1300 tests, all offline against a fake engine
uv run python scripts/ci_gate.py   # suite + bundle invariants + instrument coverage
```

## Hard laws (read before running anything)

- **`--workers 8` is the ceiling** for every collector/matchrunner/eval run. 12 workers
  deadlocks a native `libcg.so` corruption inside `mp.Pool` — the pool hangs forever with
  no error (M17, m19b).
- **`plan_iter collect` has NO resume.** Relaunching into the same `--out` clobbers the
  shards already there. Collect any remainder into a FRESH dir and train on both.
  (`scripts/m40_s2_collect.py` IS resumable — interrupt it freely.)
- **The jsonl `results` code is `0 = side-a WIN`, 1 = loss, 2 = draw.** Summing the array
  gives the *loss* count — a real 0.384 reads as 0.620 backwards.
- **Always pass `--deck` explicitly** to the build. The export defaults to `kyogre`, an
  M1-era fossil (actually a Mega Abomasnow list). It shipped by accident twice; the guard
  now hard-errors, but verify the deck md5 out of the built tarball anyway.
- **No submission without the human QC stop** — offline gates are screening, live decides,
  and three consecutive ships have inverted offline→live. See "Ship" below.
- **Diary as you go** (`docs/M<N>.md`): gate results (kills included), collection stats,
  train metrics, anomalies — at the moment they land, never retrospectively.
- Checkpoints store no metadata — **record `val_acc`/`plan_acc` in the diary immediately**.
  Spawned workers re-import from disk, so batch code edits *before* launching a run.

## The pipeline, end to end

Every stage below is idempotent or resumable except where marked. Long runs get an hourly
Hermes heartbeat (`scripts/hermes_heartbeat.sh <tag> <log> <pattern> &`) and a stage-
transition ping. `notebooks/pipeline.ipynb` automates stages 1–2 plus an OpenSkill arena
with per-stage gates and a `runs/pipeline_state.json` handoff; it stops hard at the ship
path, which is human-only.

### Stage 1 — harvest the live meta (network)

```bash
uv run python -m rl.kaggle_ingest leaderboard --top 5          # auth check + targets
uv run python -m rl.kaggle_ingest refresh --subs <our-sub-ids> --max-new 500
uv run python -m rl.kaggle_ingest targets --top-k 30 --min-score 600
uv run python -m rl.kaggle_ingest refresh --subs <ours> --opp-subs <targets> --max-new 400
uv run python -m rl.kaggle_ingest harvest --min-games 3        # -> opp_decks.parquet
# export an opponent deck for beds/games (reproducible, legality-checked):
uv run python scripts/export_opp_deck.py --hash 3631d393       # or --family dragapult
```

Raw episodes cache immutably under `data/kaggle/raw/`; tables land in
`data/kaggle/episodes.parquet` + `opp_decks.parquet`.

### Stage 2 — build BC corpora from replays

```bash
# clone one archetype's seats at a score band (native width, currently 143):
uv run python -m rl.replay_bc build --deck-hash <hash8> --min-score 900 \
  --hand-aware --out data/bc_<name>_w143
# slice to an older net's width when warm-starting narrow nets (byte-exact):
uv run python scripts/m41b_slice_corpus.py data/bc_<name>_w143 data/bc_<name> --width 100
# integrity gates:
uv run python -m rl.replay_bc audit          # G0
uv run python -m rl.replay_bc roundtrip -n 6 # G1 (>=98% alignment)
```

**Width law:** the encoder's option vector grows append-only. Corpora encode at the
current width; nets trained at an older width read their prefix (the forward slices), and
`--init-wide` on `plan_iter train` widens a narrow checkpoint to the data's width with
zero-init columns (day one reproduces the narrow net exactly).

### Stage 3 — train (BC / fine-tune)

```bash
# gate-bed clone (the G-13 panel law: >=3 draws per family, epochs 10/9/8):
uv run python -m rl.plan_iter train --data data/bc_<name> --name <bed_name> \
  --init checkpoints/m28_winners.pt --epochs 10 --lr 3e-4 --device auto

# lane-start fine-tune (wide retrain, the M41b recipe):
uv run python -m rl.plan_iter train --data data/bc_<name>_w143 --name <candidate> \
  --init-wide checkpoints/<narrow>.pt --epochs 14 --seed 3 --lr 3e-4 \
  --outcome-weight 0.25 --device auto
```

`scripts/m43_rebuild.sh` runs the full artifact chain (all corpora → 21 gate beds →
champion cont3 → `m41b_wide_prod`) on a fresh box, idempotently.

### Stage 4 — manufacture self-play rows (Lane B style)

```bash
# resumable; kill-at-collection guard baked in (--kill-above 0.90):
uv run python scripts/m40_s2_collect.py --out data/<name> --games 250 \
  --beds grim_d1 grim_d2 grim_d3 topgrim wall_d1 wall_d2 wall_d3 oger_mirror \
  --arm "model-pz:checkpoints/<net>.pt:decks/<deck>.csv" --deck decks/<deck>.csv --tau 0.6
uv run python scripts/m40_s2_collect.py --out data/<name> --status
```

### Stage 5 — PPO (honest RL on top of the clone)

```bash
uv run python -m rl.ppo --start <ckpt>.pt --tag <name> \
  --learn-deck <deck> --eval-deck <deck> \
  --iterations 50 --games-per-iter 400 --workers 8 --eval-every 5 \
  --kl-coef 0.1 --device auto \
  --opponents "model:checkpoints/m39_bc_grim.pt:grim_live=0.35" ... "mirror=0.25" "past=0.15"
# value-potential shaping arm (invariant form, M43):
#   + --shaping value --shaping-coef 0.05 --phi-ckpt <frozen-start>.pt
```

Inner kill bars, per leg, before any gate is spent: entropy must not RISE across the leg
(TensorBoard `train/entropy`), KL to the frozen start bounded, and the `vs_teacher`
promotion curve must beat the warm-start baseline within 15 iterations.

### Stage 6 — gate (pre-registered, hashed)

```bash
uv run python scripts/gate_spec.py hash docs/specs/<spec>.json   # register BEFORE running
uv run python scripts/gate_spec.py run  docs/specs/<spec>.json --out runs/<name> --workers 8
uv run python scripts/gate_spec.py decode docs/specs/<spec>.json --out runs/<name>
```

The spec names arm, control, beds, n, seed and bars; `run` stamps the hash into every
cell; `decode` refuses a verdict if anything disagrees. Offline deltas are **screening** —
live decides.

### Stage 7 — QC + ship (human stops are mandatory)

```bash
# 1. export with the GATED fix package + verify the artifact:
uv run python -m tcg.shipping export --checkpoint <ckpt>.pt --deck <deck> \
  --fixes "conserve,racemode2,racemode4,planzero"       # = the gated arm's package
uv run python scripts/ship_verify.py --checkpoint <ckpt>.pt --deck <deck> \
  --corpus data/<train-dirs...> --gate-arm "model-c-pkgz:checkpoints/<ckpt>.pt:<deck>"

# 2. multi-deck QC battery (3 games vs every working sample agent + the
#    previous ship tarball as the mirror leg); replays land in replays/:
uv run python scripts/qc_battery.py --prefix m<NN>_qc_<arm> --prev dist/<prev-ship>.tar.gz

# 3. STOP. Manual replay review + explicit go from the project owner. Then:
./build_submission.sh --checkpoint <ckpt>.pt --deck <deck> \
  --fixes "<gated package>" --message "M<NN>: ..."

# 4. In the SAME ship commit: add the new submission id to the MODELS dict in
#    notebooks/model_monitor.ipynb.
```

### Stage 8 — watch the live reads

```bash
uv run python scripts/live_ci.py --sub <id>       # win rate + CI + implied ELO
uv run python -m rl.kaggle_ingest forensics --sub <id>   # W/L by opponent archetype
uv run python -m rl.postmortem <episode-id>       # per-episode loss anatomy
```

Start every milestone with forensics on the previous ship's losses.

## Instruments

| Command | What it does |
|---|---|
| `scripts/ci_gate.py` | The one pre-anything gate: full suite + bundle invariants + instrument-coverage registry |
| `scripts/gate_spec.py` | Hashed pre-registered gate runner (`hash` / `run` / `decode`) |
| `scripts/ship_verify.py` | Tier-1 artifact correctness on the exported bundle, incl. gate/ship fix-set equality (`--gate-arm`) |
| `scripts/qc_battery.py` | Pre-ship multi-deck QC battery incl. previous-ship mirror leg |
| `scripts/live_ci.py` | Live leaderboard read with CIs and implied ELO |
| `scripts/watch_games.py` | Play two specs, emit watchable replays + postmortem JSONs |
| `scripts/export_opp_deck.py` | Reproducible opponent-deck CSV export from the harvest |
| `scripts/m43_rebuild.sh` | Rebuild the full bed/lane-start artifact chain from recipes |
| `scripts/m40_s2_collect.py` | Resumable self-play BC collection with exploration + kill guard |
| `scripts/m40_e0_value.py` | E0: is a value head good enough to serve as critic/Φ? |
| `scripts/hermes_heartbeat.sh` | Hourly Telegram heartbeat for any run >30 min |
| `rl.matchrunner play` | The one battle loop; spec grammar `model[-fixes]:<ckpt>:<deck>`, `rule:<agent>`, `solver:<deck>`, ... |
| `rl.replay_bc meta-eval` | Weighted win rate vs the frozen harvested-meta deck pool |

Measurement discipline (G-9…G-14) lives in `docs/VALIDATION.md`: n≥2400 for ~5pp claims,
panels over single beds, no gating on live cells under n=30, fresh seeds before pooling.

## Watch games and diagnose losses

Replays land under `replays/` with an auto-generated `index.html` browser (win-rate table per
matchup, click-to-watch per game). `visualizer.html` feeds a replay's `visualize` JSON to the
official ptcgvis visualizer. The shipped agent emits one `NN|{json}` line per decision on
stderr — logits and plan commits — which `kaggle_ingest agent-logs` joins back to the replay,
so you can see what the net *thought*, not just what it did.

Notebooks in `notebooks/`: `pipeline.ipynb` (ingest → BC shards → arena, gated),
`model_monitor.ipynb` (live scores + replay-audited deck identity; **updated on every
submit**), `card_pool_eda.ipynb`, and `deck_analysis.ipynb`.

## Repository layout

```
├── ARCHITECTURE.md      <- the full design: network, encoders, training loop, ship path
├── CLAUDE.md            <- working agreement / house rules
├── build_submission.sh  <- one-command export -> gate -> package -> (submit)
├── cg/                  <- engine bindings — YOU provide these (setup step 2)
├── rl/                  <- training core + the five modules that ship in the bundle
├── tcg/                 <- readable twins + the authoritative ship pipeline
├── submission/          <- the built bundle: numpy-only main.py + weights + deck
├── sample-agent{,-iono,-tuned,-dragapult}/   <- rule-based teacher agents
├── decks/               <- 20+ deck lists (alakazam_v2_h4 · ogerpon · beds · candidates)
├── checkpoints/         <- trained nets
├── data/                <- training shards, harvested replays, feature parquets
├── docs/                <- milestone diaries M0–M43, specs/, DECISIONS, MILESTONES, VALIDATION
├── scripts/             <- the instrument layer (see table above) + milestone one-shots
├── notebooks/           <- pipeline.ipynb, model_monitor.ipynb, EDA
├── runs/                <- batteries, logs, TensorBoard, pipeline state
├── tests/               <- ~1300 offline tests + a fake engine
└── reference/           <- official sample notebooks from the competition
```

`rl/` and `tcg/` hold parity-pinned **twins** of several modules (pilot, encoders, combat,
PPO, network). Changing one means changing both — see
[ARCHITECTURE.md §12](ARCHITECTURE.md#12-rl-vs-tcg--the-twin-law).
