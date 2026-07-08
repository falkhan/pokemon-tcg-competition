# RL Agent Architecture — Pokémon TCG AI Battle (Kaggle `cabt`)

Plan for building a reinforcement-learning agent we can submit to the Kaggle simulation,
initially just watching it play against itself, then learning until it starts winning
BIGLY.

> **Living docs:** this file is the intended design. What we actually tried, measured, and
> changed our minds about lives in the milestone diaries (`docs/M0.md` … `docs/M2.md`) and,
> in fast-index form, `docs/DECISIONS.md`. When a measurement contradicts the plan below,
> add a DECISIONS entry and update the relevant section here — keep them in sync.

---

## 1. The big picture

The agent has to close this loop:

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          │
  ┌───────────┐     ┌────────────┐     ┌─────────────┐     ┌───────┴──────┐
  │  DECK     │────▶│  SELF-PLAY │────▶│  LEARNING   │────▶│  DECK        │
  │  POLICY   │     │  (cabt env)│     │  (play pol.,│     │  REEVALUATION│
  │  build 60 │     │  many games│     │   value fn) │     │  win-rate per│
  │  cards    │     │            │     │             │     │  card/archet.│
  └───────────┘     └────────────┘     └─────────────┘     └──────────────┘
```

Three learnable components ("policies"), one shared data layer:

| Component | Job | Learns from |
|---|---|---|
| **Deck policy** `π_deck` | Emit a legal 60-card list | Deck-level win rates across games |
| **Play policy** `π_play(a\|s)` | Pick option indices at every decision point | Game outcomes (self-play RL) |
| **Value function** `V(s)` | Estimate win probability of a state | Same trajectories (critic / bootstrapping) |
| **Data layer** (done ✅) | id → numeric features | `deck_analysis.ipynb` → `data/cards_features.parquet`, `data/attacks_features.parquet`, `validate_deck()` |

---

## 2. The environment contract (what we must satisfy)

From `cg/api.py` and the `cabt` kaggle environment
(`kaggle_environments/envs/cabt/cabt.py`):

- **Agent signature**: `def agent(obs_dict: dict) -> list[int]`
- **First call**: `obs["select"] is None` → return the **deck** (60 card IDs).
- **Every other call**: return a list of **option indices** with
  `minCount <= len(ret) <= maxCount`, each `0 <= i < len(select.option)`, no duplicates.
- **Reward** (from `cabt.json`): `+1` win, `-1` loss, `0` draw — a single terminal reward.
- **Budget**: `remainingOverageTime: 600` seconds for the whole episode beyond the
  per-step allowance — the policy must be *fast* (no giant model, no unbounded search).
- The engine also exposes a **forward model** (`search_begin` / `search_step` in
  `cg.api`) — we can simulate ahead from any observation if we predict the hidden zones
  (opponent hand/deck/prizes). This unlocks search-based improvements later (Phase 3).

**Key structural fact**: the action space is *not* fixed. Each step presents a variable-
length list of `Option` objects (PLAY / ATTACK / EVOLVE / CARD / YES / …) under a
`SelectContext` (MAIN / TO_HAND / SWITCH / … — 49 contexts and growing). So the play
policy is a **pointer network / option-scoring** problem: score each presented option,
not a fixed softmax head.

---

## 3. Policies in detail

### 3.1 Play policy `π_play` — option scorer

For each decision, encode *(game state, context, option)* → score. Choose top-k
(`maxCount` of them, or fewer when allowed).

```
score(option_i) = MLP( [ state_vec ; context_onehot ; option_vec_i ] )
π(a=i | s)      = softmax over the presented options only   (built-in action masking)
```

**State encoding** (`state_vec`, fixed width, all from `Observation.current` +
`cards_features.parquet` lookups):

| Block | Contents |
|---|---|
| Global | turn, my/opp prize counts, deck counts, hand counts, energyAttached, supporterPlayed, stadium id-features, whose turn |
| My active (1 slot) | card features (38-dim) + hp/maxHp + n energies + energy-type counts + status conditions + tools |
| My bench (5 slots, zero-padded) | same per-slot encoding |
| Opp active + bench | same, minus hidden info |
| My hand (multiset) | sum/mean-pool of card feature vectors + count |
| My discard / opp discard | pooled card features (energy count in discard matters a lot) |

**Option encoding** (`option_vec_i`): OptionType one-hot + features of the referenced
card (via `get_card`-style resolution) + attack features from `attacks_features.parquet`
when `type == ATTACK` + area/index scalars.

### 3.2 Value function `V(s)` — the critic

Same state encoder, scalar head, trained to predict the terminal ±1. Used for:
- **Advantage estimation** in PPO (GAE), and
- **Progress metric**: watching `V(s)` trend over a game tells us if the network
  understands who's ahead (sanity check even before win rates move).

### 3.3 Deck policy `π_deck` — start dumb, stay legal

Deck building is a separate, *slower* loop. Curriculum:

1. **Phase 0 (now)**: fixed known-good deck (e.g. the Mega Lucario ex list in
   `main.py` / `deck.csv`). Isolate play learning from deck learning.
2. **Phase 1**: deck **mutation bandit** — keep a population of legal decks; after each
   evaluation batch, mutate the worst (swap 1–4 cards, always re-check with
   `validate_deck()`), keep the best. Elo/TrueSkill per deck.
3. **Phase 2 (optional)**: learned deck generator (autoregressive: pick 60 cards, mask
   illegal picks using the same `validate_deck` logic incrementally).

The **deck reevaluation** signal: per-deck win rate, plus per-card credit — e.g. mean
outcome of games whose deck contained card *c* vs games without it (a crude but
effective "card impact" statistic across a diverse population).

---

## 4. Training plan — how we make sure it actually learns

### 4.1 Algorithm: self-play PPO with a shaped curriculum

Sparse ±1 terminal reward on 100+ decision games is learnable but slow. We de-risk it:

1. **Warm start via behavior cloning (BC)**: the repo already contains a decent
   rule-based agent (`sample-agent/main.py`). Run it vs itself, log
   `(obs, chosen option)` pairs, train `π_play` to imitate. This alone should beat
   `random_agent` massively and gives PPO a sane starting point.
2. **Reward shaping (small, potential-based)**: `Δ(opp prizes remaining − my prizes
   remaining)` each step, weight ~0.1, plus the terminal ±1. Prize deltas are the
   game's natural progress bar.
3. **Self-play with opponent pool**: play the current policy against a pool
   {current, past checkpoints, rule-based agent, random}. Prevents strategy collapse
   and gives stable eval baselines.
4. **PPO** (clipped) on trajectories; league-style checkpointing every N games.

### 4.2 The "is it learning?" dashboard

Track after every training batch:

| Metric | Expectation |
|---|---|
| Win rate vs `random_agent` | → ~100% quickly (else bug) |
| Win rate vs rule-based sample agent | the real milestone; > 50% = we're cooking |
| Win rate vs frozen past self (100 games) | > 55% steadily = still improving |
| Mean game length | should *drop* as play gets more lethal |
| Avg prizes taken per game | should rise toward 6 |
| `V(s₀)` calibration | ~0 at start of mirror games |
| Illegal/timeout actions | must stay 0 |

### 4.3 Ensuring the loop stays honest

- **Determinism harness**: fixed-seed eval suites, separate from training games.
- **Checkpoint + evaluate before promote**: a new policy only replaces "best" if it
  beats it ≥ 55% over ≥ 200 games (SPRT-style early stop optional).
- **Watch actual games** (Section 6) — numbers lie, replays don't.

---

## 5. Pseudo-code

### 5.1 Play-policy agent (the thing we submit)

```
function agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK                      # 60 card ids, pre-validated

    s   = encode_state(obs.current)                       # fixed-width float vec
    ctx = onehot(obs.select.context)
    scores = []
    for opt in obs.select.option:
        o = encode_option(opt, obs)                       # card/attack features
        scores.append(policy_net(s, ctx, o))

    k = choose_count(obs.select)         # usually maxCount; sometimes fewer is legal
    if TRAINING: pick = sample_topk_without_replacement(softmax(scores), k)
    else:        pick = argsort(scores, desc)[:k]
    return pick
```

### 5.2 Self-play training loop

```
initialize policy_net (from BC checkpoint), value_net, opponent_pool = {rule_based, random}

for iteration in 1..N:
    trajectories = []
    for game in 1..GAMES_PER_ITER:
        opponent = sample(opponent_pool ∪ {policy_net})
        env = make("cabt")
        env.run([policy_agent, opponent])                 # or symmetric self-play
        R = terminal_reward(env)                          # ±1 / 0
        trajectories += [(s_t, ctx_t, options_t, a_t, r_t, R)]   # r_t = prize shaping

    advantages = GAE(trajectories, value_net)
    ppo_update(policy_net, value_net, trajectories, advantages)

    if iteration % EVAL_EVERY == 0:
        wr = evaluate(policy_net vs best_so_far, n=200)
        if wr >= 0.55: best_so_far = checkpoint(policy_net); opponent_pool.add(best_so_far)
        log_dashboard_metrics(); save_replay_html(sample_games)

    if iteration % DECK_EVERY == 0:                       # Phase 1+
        deck_population = mutate_and_select(deck_population, elo_scores, validate_deck)
```

### 5.3 Deck mutation (Phase 1)

```
function mutate(deck):
    repeat:
        d = copy(deck)
        remove random 1..4 cards
        add cards sampled ∝ card_impact_score (from cards_features + past win stats)
        if validate_deck(d): return d
```

### 5.4 Reference: the official RL + MCTS sample (`reference/reinforcement-learning-and-mcts-sample-code.ipynb`)

The competition hosts published a complete, working RL agent — worth studying closely
because it independently validates several of our design choices and hands us a ready
implementation for the parts we've deferred:

**What it is.** A small transformer (d_model=128, 1 encoder + 1 decoder layer) trained
AlphaZero-style: self-play games where a **determinized MCTS** (10 simulations/move via
the engine's `search_begin`/`search_step` forward model) produces the training targets —
the value head learns a TD(λ)-blend of the final ±1 result, the policy head learns
"child value minus root value" from the search tree. No PPO; search *is* the policy
improvement operator.

**What it confirms about our plan:**
- **Option scoring is the right shape.** Its decoder scores each candidate action against
  the encoded state via cross-attention — the same pointer-style idea as our
  `OptionScorer`, just with attention instead of concatenation.
- **Card-ID embeddings work.** It feeds sparse card-ID features through `EmbeddingBag`,
  i.e. learned per-card vectors — exactly the "first capacity upgrade" we planned when
  our hand-built 36 features plateau.
- **Hidden information is handled by guessing, not ignoring.** For MCTS it fills unknown
  zones with placeholders (own deck: random order of the known 60; opponent's hand/deck:
  literally Snorlax and Basic Energy) — determinization doesn't need to be clever to help.

**What we can lift directly:**
- The **MCTS node/backprop/PUCT loop** (~100 lines) for milestone M4 — it plugs into our
  policy/value heads as priors with minimal changes.
- The **direct engine loop** via `cg.game.battle_start`/`battle_select` instead of
  kaggle_environments — much less overhead per game; the M2 trajectory collector should
  use this.
- Its handling of **multi-select decisions**: when `maxCount > 1` it enumerates up to 64
  *combinations* of options as candidate actions and scores each combination — more
  principled than our current independent top-k picks.

**Where we deliberately differ (for now):** we start with BC + PPO on engineered features
because it's simpler to debug and teaches the fundamentals; the sample jumps straight to
search-guided learning with embeddings. If PPO stalls at M2/M3, converging toward the
sample's recipe is the escape hatch — and the two approaches share our data layer,
encoders-discipline, and submission pipeline unchanged.

---

## 6. Watching it play — visualization via the `cabt` kaggle environment

The env already ships everything we need to *watch* games:

```python
from kaggle_environments import make

env = make("cabt")
env.run([my_agent, my_agent])          # self-play, watch it fight itself

# 1) Standalone interactive replay (already proven in testing.ipynb -> resulte.html):
with open("replays/iter_0042_game_007.html", "w") as f:
    f.write(env.render(mode="html"))

# 2) Inline in a notebook:
env.render(mode="ipython", height=700)

# 3) The official visualizer: env stores the engine's visualize data on step 0 —
#    that's the JSON `visualizer.html` POSTs to https://ptcgvis.heroz.jp
import json
vis = env.steps[0][0]["visualize"]
json.dump(vis, open("replays/iter_0042_game_007.json", "w"))
# then open visualizer.html in a browser and pick that JSON (it also accepts the
# full env JSON since it reads steps[0][0]["visualize"] itself).
```

**Replay policy**: every eval batch, dump 3 HTML replays — (best win, worst loss,
median game) — named `iter_{i}_...html`. Reviewing losses is where the insight is:
misplayed Energy attachments, wasted Supporters, bad retreat decisions show up
instantly on the board even when metrics look fine.

Learning curves (win rates, game length, V-calibration) live in the notebook /
TensorBoard; replays explain *why* the curves move.

---

## 7. Sample code

### 7.1 Minimal submittable policy agent (`main.py` shape)

```python
import os
import numpy as np
from cg.api import (Observation, OptionType, SelectContext,
                    to_observation_class, all_card_data)

# --- load artifacts (also present under /kaggle_simulations/agent/ on Kaggle) ---
BASE = os.path.dirname(os.path.abspath(__file__))
W = np.load(os.path.join(BASE, "policy_weights.npz"))     # tiny MLP: fits limits easily
DECK = [int(x) for x in open(os.path.join(BASE, "deck.csv")) if x.strip()]

CARDS = {c.cardId: c for c in all_card_data()}
FEAT = np.load(os.path.join(BASE, "card_features.npy"))   # row = card_id, from notebook

def mlp(x):
    h = np.maximum(0, x @ W["w1"] + W["b1"])
    return float(h @ W["w2"] + W["b2"])

def encode_state(state) -> np.ndarray:
    me = state.players[state.yourIndex]; op = state.players[1 - state.yourIndex]
    def poke_vec(p):
        if p is None: return np.zeros(FEAT.shape[1] + 3)
        return np.concatenate([FEAT[p.id], [p.hp / 340, p.maxHp / 340, len(p.energies) / 5]])
    slots = [me.active[0] if me.active else None] + list(me.bench) + [None] * (5 - len(me.bench))
    opsl  = [op.active[0] if op.active else None] + list(op.bench) + [None] * (5 - len(op.bench))
    hand  = np.sum([FEAT[c.id] for c in me.hand], axis=0) if me.hand else np.zeros(FEAT.shape[1])
    glob  = np.array([state.turn / 30, len(me.prize) / 6, len(op.prize) / 6,
                      me.deckCount / 60, op.handCount / 15,
                      float(state.energyAttached), float(state.supporterPlayed)])
    return np.concatenate([glob, hand] + [poke_vec(p) for p in slots + opsl])

def encode_option(o, obs) -> np.ndarray:
    v = np.zeros(17 + FEAT.shape[1])            # OptionType one-hot + card features
    v[int(o.type)] = 1.0
    cid = o.cardId
    if cid is None and o.index is not None:     # resolve card behind the option
        try:
            ps = obs.current.players[o.playerIndex or obs.current.yourIndex]
            card = {2: ps.hand, 3: ps.discard, 4: ps.active, 5: ps.bench}[int(o.area)][o.index]
            cid = card.id if card else None
        except Exception:
            cid = None
    if cid: v[17:] = FEAT[cid]
    return v

def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK
    s = encode_state(obs.current)
    ctx = np.zeros(64); ctx[int(obs.select.context)] = 1.0
    scores = [mlp(np.concatenate([s, ctx, encode_option(o, obs)]))
              for o in obs.select.option]
    order = np.argsort(scores)[::-1]
    return [int(i) for i in order[:obs.select.maxCount]]
```

### 7.2 Self-play data collection

```python
import json
from kaggle_environments import make

def play_games(agent_a, agent_b, n_games: int, replay_prefix=None):
    results = []
    for g in range(n_games):
        env = make("cabt")
        env.run([agent_a, agent_b])
        rewards = [env.state[0].reward, env.state[1].reward]  # +1 / -1 / 0
        results.append(rewards)
        if replay_prefix and g < 3:                            # keep a few replays
            with open(f"{replay_prefix}_{g}.html", "w") as f:
                f.write(env.render(mode="html"))
            json.dump(env.steps[0][0]["visualize"],
                      open(f"{replay_prefix}_{g}.json", "w"))
    wins = sum(r[0] > r[1] for r in results)
    return wins / n_games, results
```

### 7.3 Trajectory logging for PPO / BC

Wrap the agent so every decision is recorded; the env replays give the terminal reward:

```python
class RecordingAgent:
    def __init__(self, inner): self.inner, self.traj = inner, []
    def __call__(self, obs_dict):
        action = self.inner(obs_dict)
        if obs_dict.get("select") is not None:
            self.traj.append({"obs": obs_dict, "action": action})
        return action
# after env.run: zip(rec.traj, terminal_reward) -> training tuples
```

---

## 8. Tooling decisions

**No RL framework — hand-rolled PPO on top of PyTorch.** Decided 2026-07 (see also
§2/§3 for the constraints that drive this):

- Our action space is a **variable-length option list** scored by a pointer-style
  network. Stable-Baselines3, Tianshou, TorchRL and RLlib all assume fixed
  `Discrete`/`Box` action spaces and a Gym-style `env.step()` API; the cabt env is a
  callback model (`agent(obs_dict) -> list[int]`) with nested-dict observations. We'd
  spend more code wrapping than the ~300 lines a PPO loop costs.
- Self-play with opponent pools isn't native to SB3; RLlib supports it but is a heavy
  Ray dependency and historically painful on Windows.
- **CleanRL is our reference, not a dependency** — copy the single-file PPO and adapt
  the action head.

**Stack** (all in `pyproject.toml`, CPU-only is fine — the model is a small MLP and the
bottleneck is the C game engine):

| Package | Role |
|---|---|
| `torch` (2.12, CPU) | training-side network + optimizer |
| `tensorboard` | the §4.2 "is it learning?" dashboard |
| `openskill` | ratings for opponent-pool checkpoints and deck population |
| `numpy` | **the only inference dependency** — submission loads `.npz` weights |
| `polars` | data layer (already used by `deck_analysis.ipynb`) |

Notes:
- JAX + flax + optax are already in the venv (transitive deps) and would also work,
  but torch was chosen for debuggability and the volume of reference PPO code. There's
  no JAX `vmap`-the-env win available anyway: the engine is a ctypes DLL.
- The engine keeps **one global battle per process** (`cg/sim.py` → `Battle.battle_ptr`),
  so parallel self-play = `multiprocessing`, one env per worker. Design the collector
  around this from day one.
- Keep torch out of `submission/`: export with `rl.policy.save_npz`, run inference with
  the hand-rolled numpy MLP (§7.1).

---

## 9. Learning path & documentation (team is new to RL and PyTorch)

Curated, in the order worth reading. The starred items are the 20% that gives 80%.

### PyTorch (write your first network)
- ★ **Learn the Basics** — official 8-part intro (tensors → datasets → autograd →
  training loop): https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- **60-Minute Blitz** — the classic quickstart:
  https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- **TensorBoard with PyTorch** (how we'll plot the dashboard):
  https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
- Torch's own RL tutorial (DQN on CartPole — good for seeing an env loop, we use PPO
  instead): https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

### RL fundamentals (understand what PPO is doing)
- ★ **OpenAI Spinning Up** — the best practical intro; read "Intro to RL" parts 1–3,
  then the PPO page: https://spinningup.openai.com/
  - PPO specifically: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- ★ **Karpathy, "Deep RL: Pong from Pixels"** — policy gradients built from scratch in
  one readable post: https://karpathy.github.io/2016/05/31/rl/
- **Hugging Face Deep RL Course** — free, hands-on, unit 8 is PPO from scratch:
  https://huggingface.co/learn/deep-rl-course
- Sutton & Barto, *Reinforcement Learning: An Introduction* (free PDF) — the reference
  book; skim ch. 13 for policy gradients: http://incompleteideas.net/book/the-book-2nd.html
- Papers (short, readable): PPO https://arxiv.org/abs/1707.06347 · GAE (the advantage
  estimator we use) https://arxiv.org/abs/1506.02438 · AlphaZero (for Phase-3 search)
  https://arxiv.org/abs/1712.01815

### Implementation guides (the difference between "compiles" and "learns")
- ★ **CleanRL PPO docs + annotated single-file implementation** — our template:
  https://docs.cleanrl.dev/rl-algorithms/ppo/
- ★ **"The 37 Implementation Details of PPO"** — legendary checklist of the tricks that
  make PPO actually work; read before writing `rl/ppo.py`:
  https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- **Andy Jones, "Debugging RL Without the Agonizing Pain"** — probe tests for when
  win rates won't move: https://andyljones.com/posts/rl-debugging.html

### Our environment & tooling
- **Official sample agents** — the best documentation of the engine API in existence:
  - `reference/reinforcement-learning-and-mcts-sample-code.ipynb` — full RL agent
    (transformer + MCTS + self-play training); dissected in §5.4.
  - `sample-agent/` (Mega Lucario) and
    [A Sample Rule-Based Agent Iono's Deck](https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck)
    — rule-based agents; our M1 behavior-cloning teacher and the reference for
    submission packaging (`main.py` + `cg/` + `deck.csv` in the tarball).
- **kaggle-environments** (env API, `make/run/render`):
  https://github.com/Kaggle/kaggle-environments
- **Competition data & rules** (agent contract, submission format): the competition's
  Kaggle page + `pokemon-tcg-ai-battle/` PDFs in this repo; engine API is documented
  inline in `cg/api.py`.
- **openskill.py** (rating the opponent pool): https://openskill.me/
- **uv** (package manager this repo uses): https://docs.astral.sh/uv/

Suggested 2-week on-ramp: PyTorch Basics (day 1–2) → Spinning Up parts 1–3 + Karpathy
(day 3–5) → reproduce CleanRL PPO on CartPole (day 6–8) → read the 37-details post,
then start on `rl/bc.py` with the team (week 2) — BC is supervised learning, so it's
the gentlest first contact with the codebase.

---

## 10. Repo layout (target)

```
Pokemon/
├── ARCHITECTURE.md            <- this file
├── deck_analysis.ipynb        <- data exploration + feature prep (done)
├── data/
│   ├── EN_Card_Data.csv
│   ├── cards_features.parquet     <- id -> 38 features (done)
│   └── attacks_features.parquet   <- attackId -> cost/damage vecs (done)
├── cg/                        <- game engine bindings (given)
├── rl/
│   ├── encoders.py            <- encode_state / encode_option (done, numpy-only)
│   ├── policy.py              <- option-scoring net + value head (done, torch)
│   ├── bc.py                  <- behavior cloning from sample agent (stub)
│   ├── ppo.py                 <- self-play training loop (stub)
│   ├── deck_search.py         <- validate_deck + legal mutations (done)
│   └── eval.py                <- play_games, RecordingAgent, replay dumps (done)
├── replays/                   <- self-generated replay pages + index.html browser
├── reference/                 <- official sample notebooks (see §5.4)
├── docs/                      <- milestone diary (M0.md, ...)
└── submission/
    ├── main.py                <- 7.1-style agent (keep `agent` the LAST callable!)
    ├── cg/                    <- engine bindings, bundled — Kaggle does NOT provide them
    ├── deck.csv
    ├── policy_weights.npz
    └── card_features.npy
```

## 11. Milestones

1. **M0 — watch it fight itself (this week)**: submit the 7.1 skeleton with *random*
   weights + fixed deck; run `env.run([agent, agent])` locally, dump replays, confirm
   zero illegal actions / timeouts end-to-end.
2. **M1 — beat random**: BC on rule-based agent logs → >95% vs `random_agent`.
3. **M2 — beat the teacher**: PPO self-play with prize shaping → >50% vs the
   rule-based Mega Lucario agent.
4. **M3 — deck loop**: mutation bandit over decks, per-card impact stats feeding back.
5. **M4 — search (optional)**: use `search_begin`/`search_step` with sampled hidden
   states (determinized MCTS) at low visit counts, policy/value as priors — budget
   permitting under the 600 s overage cap. The official sample (§5.4) implements
   exactly this loop; adapt it rather than writing from scratch.
6. **M∞ — winning BIGLY** 🏆
