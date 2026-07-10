# M7 manual tests — the single reference

Every [ENGINE] (needs `cg/libcg.so`) and [NET] (needs `~/.kaggle/kaggle.json`) run,
in execution order. All commands are PowerShell-safe. Paste results into the
[M7.md](M7.md) results log — the *Gates* row of each section says which number
unlocks what. Run `git pull` + `uv sync` first.

Status legend from past runs: ✅ passed · ❌ failing · ⏳ not yet run.

---

## 0. Offline suite (run after every pull)

```
uv run pytest -q
```

| Gate | Threshold |
|---|---|
| all tests | 320 passed |

**Purpose.** Pins everything that doesn't need the engine: rl↔tcg twin parity
(byte-identical scorers/encoders/nets), the score-ladder constants, deck legality,
shard schemas, and the corrected PPO loss. It proves the *code* is consistent — it
proves nothing about win rates; those are what the sections below measure.

---

## 1. M7.0 — Kaggle ingestion [NET] ⏳ (you're fixing the endpoint URL)

```
uv run python -m rl.kaggle_ingest list --sub 54474043
uv run python -m rl.kaggle_ingest fetch --episode <id-from-list>
#   if the endpoint 403/404s:  kaggle competitions replay <id>
#   uv run python -m rl.kaggle_ingest import-file episode-<id>-replay.json --episode <id>
uv run python -m rl.kaggle_ingest verify --episode <id>
uv run python -m rl.kaggle_ingest refresh --subs 54474043 --max-new 100
uv run python -m rl.kaggle_ingest harvest --min-games 1
uv run python -m rl.kaggle_ingest meta --top-k 8
uv run python -m rl.kaggle_ingest forensics --sub 54474043
```

| Gate | Threshold |
|---|---|
| `verify` | both seats' first actions are 60-int lists passing `validate_deck` (hard assert) |
| `harvest` | ≥ 20 distinct opponent decks |
| `meta` | archetype clusters look sane on eyeballing |

**Hyperparams.** `--max-new` (episodes per refresh; politeness-throttled at ~1 req/s);
`--min-games` (sightings before a deck counts — filters parse flukes); `--top-k`
(meta anchors frozen per snapshot); `ARCHETYPE_COS = 0.95` in `rl/kaggle_ingest.py`
(lower → coarser clusters).

**Purpose & logic.** The env contract makes each agent's *first action* its decklist,
so our own submissions' replays carry real opponent decks — the fix for wall #1
(field diversity). Clustering: exact deck-hash first, then greedy agglomerative
cosine over the L2-normalized pooled FEAT vector of each deck's Pokémon core (the
representation a linear probe identified archetypes with at 98%); two decks merge
when `cos(a,b) = a·b ≥ 0.95`. Go/no-go: decklists extractable → the full meta
pipeline; outcomes only → keep forensics, drop the meta/BC edges. `verify` also
prints `has_opponent_obs`, the BC-on-winners go/no-go.

**Feeds:** `generate --meta`, `league init --meta`, harvested shells, L3 determinization.

---

## 2. M7.1 — deck factory + refinement [ENGINE] ❌ (best template ≈ bc_v1 level)

```
uv run python -m rl.deck_build generate --n 40 --top-k 10        # add --meta data/kaggle/opp_decks.parquet post-M7.0
uv run python -m rl.league add-batch --dir decks/gen
uv run python -m rl.league run --games-per-anchor 40 --workers 4
uv run python -m rl.league standings
uv run python -m rl.deck_search climb --deck decks/gen/deck_<top>.csv   # ~13 min each
uv run python -m rl.league promote --deck decks/gen/refined_<hash>.csv
```

| Gate | Threshold |
|---|---|
| promote check 1 | ≥ 1 deck ≥ 0.50 vs the Lucario deck, same generic pilot, n=400 |
| promote check 2 | field fitness ≥ 0.55 vs the full anchor field |
| batch shape | ≥ 3 distinct archetypes above the anchor-field median ordinal |

**Hyperparams.** Generation: `--n`, `--top-k` (lines kept), `--meta` (weakness
histogram source). Line scoring in `rl/deck_build.py`: `score = dpe·(1 +
COVERAGE_WEIGHT·coverage) − PRIZE_PENALTY·(prizes−1) − SETUP_PENALTY·setup` with
`dpe = best-attack damage / its energy cost`; knobs `SETUP_PENALTY=8` (drop if the
pilot's closing improves), `PRIZE_PENALTY=10`, `COVERAGE_WEIGHT=1.0`,
`VARIABLE_DISCOUNT=0.5` (printed damage on conditional attacks is best-case).
Climb: `--proposals 50`, `--games-per-opp 30`, `--min-gain 0.02`.

**Purpose & math.** Templates give global coverage (every candidate legal by
construction); the climb refines trainer/energy slots on FIELD fitness — never
mirror-only, the documented overfit trap. `min_gain=0.02` vs noise: each proposal
plays 7 anchors × 30 = 210 games, so the standard error of a win rate near 0.5 is
√(0.25/210) ≈ 3.4pp — requiring +2pp *plus* beating the champion's own measured
number rejects most 1σ noise while still accepting real gains within ~50 proposals.

---

## 3. M7.2 — league + gates [ENGINE] ✅ (self-test passed 2026-07-10)

```
uv run python -m rl.league selftest --games 40 --workers 4
uv run python -m rl.league run --games-per-anchor 40 --workers 4
uv run python -m rl.league standings
uv run python -m rl.league gate --entry <entry_id> [--n-scale 0.25]
```

| Gate | Threshold | n |
|---|---|---|
| selftest | anchors reproduce the M6 ordering: {lucario_expert ≈ tuned} > generic+lucario > bc_v1 > random | 40/pair |
| G1 legality | 0 crashes / illegal picks | 200 |
| G2 vs random | ≥ 0.90 | 200 |
| G3 vs generic, same deck | ≥ 0.55 (M7.3 entry: ≥ 0.50) | 400 |
| G4 vs Lucario expert | ≥ 0.35 | 400 |
| G5 held-out decks | ≥ generic − 3pp on 5 unseen decks | 200/deck |
| G6 time budget | mean move < 50 ms | over G1–G4 |

**Hyperparams.** `--games-per-anchor` (rating precision: openskill σ shrinks roughly
as 1/√games — 40/anchor took candidate σ to ~1.9 in your run); `--n-scale` shrinks
every gate n for smoke runs and is *recorded in the JSON* so scaled results are
honestly labeled; `TOP_PEERS=3` (post-anchor peer schedule).

**Purpose & math.** The league is a persistent PlackettLuce table whose frozen anchor
population pins the rating scale week over week (`ordinal = μ − 3σ` is a conservative
lower bound). The gate n's come from the binomial CI: at n=400 the 95% interval on a
measured 0.55 is ±1.96·√(0.55·0.45/400) ≈ ±4.9pp — enough to resolve the G3/G4
thresholds from noise. Gates write `data/league/gates/<entry>.json` (gates-as-data,
citable in the diary).

---

## 4. M7.2b — pilot race-math gates [ENGINE] ❌ floor (0.695), ✅ expert (0.305)

```
uv run python -m rl.matchrunner play --a generic:lucario --b generic:floor_zero_damage -n 200 --workers 4
uv run python -m rl.matchrunner play --a generic:lucario --b rule:lucario -n 400 --workers 4
uv run python -m rl.matchrunner play --a generic:lucario --b random:kyogre -n 200 --workers 4
# loss forensics when the floor misses (paste the loss lines):
uv run python -m rl.matchrunner play --a generic:lucario --b generic:floor_zero_damage -n 50 --diag
```

| Gate | Threshold | Last measured |
|---|---|---|
| floor (vs zero-damage deck) | ≥ 0.90 | 0.695 ❌ |
| vs Lucario expert | > 0.30 | 0.305 ✅ |
| vs random | ~0.90 sanity | 0.855 |

**Hyperparams** (all in `tcg/constants.py`, mirrored inline in `rl/generic_pilot.py`
— change BOTH): `SCORE_CHIP_CLOSE_BASE=2300` (close-mode chip tier; must stay above
trainers ≤2200 and below play-Pokémon 2400 — usable range 2250–2390);
`SCORE_ATTACH_RACE_CLOSER_ACTIVE/BENCH = 2750/2680` (the bench tier requires an
attack-ready active — the starvation guard); `PROMOTE_TURN_PENALTY=50` ×
`PROMOTE_TURNS_CAP=4`; `_op_board_harmless` strictness (e.g. add `turn > 2`);
`RACE_TURN_CAP=10` (race-feature normalization).

**Purpose & math.** The pilot scores every option on one priority ladder; these
constants ARE the policy. Race math: `turns_to_first_ko = max(energy_gap, 1) +
hits_to_ko − 1` (an attacker one energy short still fires this turn — attach precedes
attack), with `hits_to_ko = ⌈hp / charged_best_damage⌉`. The floor deck cannot KO, so
every floor loss is a self-inflicted deck-out — the `--diag` end-state lines
(deck/prizes remaining) identify the mechanism to fix next.

---

## 5. M7.3 — deck-conditioned BC loop [ENGINE] ⏳

```
uv run python -m rl.league population --top 10
uv run python -m rl.bc collect --teacher generic --decks data/league/population.json --games 3000
uv run python -m rl.bc train --arch v2 --epochs 10
uv run python -m rl.league add --deck decks/gen/deck_<best>.csv --pilot model:checkpoints/osv2_bc.pt --id osv2_bc+<name>
uv run python -m rl.league run --games-per-anchor 40 --workers 4
uv run python -m rl.league gate --entry osv2_bc+<name> --n-scale 0.5
```

| Gate | Threshold |
|---|---|
| fidelity | val top-1 ≥ 0.80 (BC should *match* the teacher, not beat it) |
| G3 | ≥ 0.50 (parity with the generic teacher) |
| G5 | ≥ generic − 3pp on held-out decks (the deck-conditioning proof) |

**Hyperparams.** `--games` (each game yields ~35 decisions × 2 recorded seats; 3000
games ≈ 200k decisions across the population — raise it if a deck's per-deck val acc
lags); `--epochs 10`, `lr=3e-4`, `batch_size=256`; `EMBED_DIM=16` (id-embedding
width); `VALUE_LOSS_WEIGHT=0.5`; population `--top` (deck diversity vs data-per-deck
trade).

**Purpose & math.** Loss = masked cross-entropy over each decision's ragged option
menu (padded logits set to −10⁹ so softmax ignores them) + 0.5·Huber(value, ±1
outcome). The split is BY GAME, not by row — rows within a game are correlated, and a
row-split leaks. Watch the per-deck val accuracies (`d0:0.83 d1:0.79 ...`): uniform
high fidelity = the embeddings + deck pools condition on the deck; one lagging deck =
underrepresented data, not architecture. Why this can work where Lucario-expert BC
capped at 55%: the generic teacher's scoring is a function of *observable* state —
no hidden AttackPlan to alias.

---

## 6. M7.4b — PPO retry (fixed loss, diverse field) [ENGINE] ⏳

```
# 5-iteration smoke first — the §9 check is "value_loss moving":
uv run python -m rl.ppo --start osv2_bc.pt --decks data/league/population.json --value-ckpt checkpoints/bc_v1_value.pt --iterations 5 --games-per-iter 400
# then the real run (hours; promotion gate at 0.55 vs the current best):
uv run python -m rl.ppo --start osv2_bc.pt --decks data/league/population.json --value-ckpt checkpoints/bc_v1_value.pt --iterations 50
# gate the result like any pilot:
uv run python -m rl.league add --deck decks/gen/deck_<best>.csv --pilot model:checkpoints/ppo_best.pt --id ppo_v2+<name>
uv run python -m rl.league gate --entry ppo_v2+<name>
```

| Gate | Threshold |
|---|---|
| smoke | `value_loss` decreasing over 5 iters; entropy NOT rising |
| G3 | ≥ 0.55 (now beats the teacher) |
| G4 | ≥ 0.35 vs the Lucario expert |
| escape hatch | two failed honest attempts → L4 AlphaZero-lite go/no-go review |

**Hyperparams.** `--lr 1e-4` (first knob if ratios explode/stall); `ENTROPY_COEF=
0.001` (sparse ±1 rewards give a weak advantage signal — a large entropy bonus
overpowers it and diffuses the policy toward random, the measured M2 failure);
`VALUE_COEF=0.5`; `CLIP_EPS=0.2`; GAE `GAMMA=0.99`/`LAM=0.95` (bias↔variance dial);
`PRIZE_SHAPING=0.1`; `--race-shaping <coef>` on the collector (start 0.05 if setup
actions still get no credit); `--games-per-iter 400`; pool weights in
`default_pool` (mirror 0.4 / experts 0.15+0.15 / generic 0.1+0.1 / random 0.1).

**Purpose & math.** The corrected objective is `L = L_clip + 0.5·L_value −
0.001·H`, with `L_clip = −E[min(r·Â, clip(r, 1±0.2)·Â)]`, `r = π_new/π_old`. The old
code computed `L_policy · 0.5 · L_value − ...` — a *multiplicative* loss whose policy
gradient is scaled by the value error and flips sign with it; every stalled PPO run
(M2–M4) trained on that, which is why "PPO failed" was never an honest measurement.
The critic warm-start loads the supervised value head (the one proven training
success: MSE 1.0→0.35, sign-acc 0.87) so GAE gets sane baselines from iteration 0.
Race shaping is potential-based, `F = c·(φ(s') − φ(s))` with φ = the race delta —
it telescopes out of episode returns, so it credits setup without changing the
optimal policy (Ng et al. 1999). Multi-deck self-play + the generic pool members
attack the old failure's other half: overfitting to 3 opponents.

---

## 7. Results discipline

After each session, paste the numbers into the [M7.md](M7.md) results log. What
unlocks what: M7.0's harvest → regenerate decks (`--meta`) and league anchors
(`init --meta`), then RE-RUN sections 2–3 (promotions must hold on the newest meta).
Section 4's floor fix gates nothing downstream but raises every pilot number.
Section 5's fidelity gates section 6's start checkpoint. Two honest failures at
section 6 = the L4 review, per plan.
