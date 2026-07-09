# M7 Plan — The Deck Factory Loop: deck building + pilot training + real-meta ingestion

**Status:** 🚧 Active (2026-07-09) — M7.0 infrastructure landed. Implementation diary: [M7.md](M7.md).

## Context

M0–M6 (diaries in this folder, fast index in [DECISIONS.md](DECISIONS.md)) established,
with measurements:

| Approach | Module | Result | Why it stopped |
|---|---|---|---|
| Behavior cloning | `rl/bc.py` | `bc_v1` (~90% vs random, LB 640) | fidelity collapses on the experts' unobservable lookahead (~55% ceiling on Lucario) |
| Combat-lookahead features | `rl/encoders.py` (M3) | fidelity 55%→56.4%, *worse* win rate | state-level aggregates don't drive the expert's fine decisions |
| Self-play PPO | `rl/ppo.py` | stalled after 1 modest gain | noisy advantages on sparse ±1; critic never learned; **suspected loss bug** (`loss = policy_loss * VALUE_COEF * value_loss`, docstring says `+`) |
| Supervised value head | `rl/value_train.py` | **works** — MSE 1.0→0.35, sign-acc 0.87 | the one clear success; critic was trainable all along |
| Determinized MCTS | `rl/mcts.py` | 47–53% vs its own greedy policy; worse with more sims | value head OOD on search states; Snorlax-placeholder determinization |
| Rule+value hybrid | `rl/hybrid.py` | loses to stock rule agent at every margin | 0.87-sign-acc classifier can't *rank* sibling actions |
| Deck mutation search | `rl/deck_search.py` (M4) | 0/30 improving mutations on the tuned seed | bottlenecked by opponent diversity, not method |
| Heuristic param tuning | `sample-agent-tuned/` (M5) | 0/20 proposals beat defaults | the sample agent is already tuned for the simulable field |
| Deck-agnostic generic pilot | `rl/generic_pilot.py` (M6) | 95% vs random; **ranks decks at expert parity** (Lucario-vs-Iono 77% ≈ expert 76%); ~25–30% vs the Lucario expert | the unlock M7 builds on |

Two walls, named:

1. **Opponent-field diversity** — the offline field (2 rule pilots + bc_v1) can't represent
   the real ~4,500-team meta; every search overfits to it. *The #1 measured bottleneck.*
2. **The generic-vs-expert pilot gap** (~25–30%) — driven by the experts' multi-turn
   planning (`AttackPlan`), which no method so far has reproduced.

M7 attacks both, plus the lever we've never automated: **learning from the real
leaderboard**. A rule-agent probe (submission id 54474043) is already pending; its
episodes are downloadable.

---

## 1. The loop

```
                    ┌────────────────────────────────────────────┐
                    │   KAGGLE  (leaderboard = honest fitness)   │
                    └───────▲───────────────────────────┬────────┘
                    submit  │                           │ episodes (JSON)
                            │                           ▼
 ┌──────────────┐   ┌───────┴───────┐          ┌────────────────────┐
 │ deck factory │   │ ship pipeline │          │  rl/kaggle_ingest  │  M7.0
 │ rl/deck_build│   │ export + gate │          │  decks / steps /   │
 │    M7.1      │   │  (existing)   │          │  outcomes tables   │
 └──────┬───────┘   └───────▲───────┘          └─────────┬──────────┘
        │ candidate decks   │ champion (deck,pilot)      │ harvested meta
        ▼                   │                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                    LEAGUE  rl/league.py   M7.2                  │
 │  persistent openskill table over (deck, pilot) entries          │
 │  fixed ANCHOR field: rule experts + generic pilot + bc_v1 +     │
 │  random + harvested-meta bots          gates as data (G1–G6)    │
 └───────────────┬────────────────────────────────▲────────────────┘
                 │ promising decks                │ new pilots
                 ▼                                │
 ┌─────────────────────────────────────────────────────────────────┐
 │            PILOT TRAINING + LOOKAHEAD  M7.2b/M7.3/M7.4          │
 │  race-math lookahead → deck-conditioned OptionScorer v2 (BC     │
 │  from generic pilot across MANY decks) → within-turn combo      │
 │  solver → fixed-PPO / AlphaZero-lite                            │
 └─────────────────────────────────────────────────────────────────┘
```

One sentence: *generate decks that make sense → pilot them → league-rate against a
diverse fixed field → train pilots on the winners → harvest the real meta from Kaggle
episodes → counter-build → repeat*, with the leaderboard as the only fitness function we
trust and every offline proxy explicitly designed not to be gamed (frozen anchors,
slot-fair, large n).

Two ordering principles, both derived from DECISIONS.md:

- **Ingestion first (M7.0).** Cheap, independent, and it de-risks everything else: if we
  can read opponent decks from episodes, the field-diversity problem changes category.
- **Deck work only pays if the pilot doesn't wash it out** — M6 made the generic pilot
  deck-legible, so deck generation (M7.1) can run in parallel with pilot work using the
  generic pilot as the evaluator.

Everything runs on the direct engine loop (~50k games/hr, multiprocessing, one battle
per process — the `rl/collector.py` pattern), so the plan optimizes for signal quality
(large n, slot-fair) over cleverness.

---

## 2. Component A — deck factory (`rl/deck_build.py`, new)

### 2.1 What "a deck that makes sense" means, operationally

A deck the pilot can actually execute needs:

1. **A complete evolution line as the damage core** — walk `evolves_from_id` chains in
   `data/cards_features.parquet` (`is_basic`/`is_stage1`/`is_stage2`), with counts shaped
   like real decks: 4-3 for stage-1 lines, 4-3-2 (or 3-2-2 + candy-equivalent) for
   stage-2. (We cannot identify Rare Candy from features — see 2.3.)
2. **Matched energy** — the attackers' `energy_type_id` and attack costs
   (`attacks_features.parquet`) determine basic-energy type and count (10–14; more for
   expensive attackers, since the pilot attaches ~1/turn). Mono- or dual-type only.
3. **A consistency shell** — draw/search trainers. The hard part (2.3).
4. **Tech / secondary attacker** — a second line of a different type chosen for
   **weakness coverage** of the meta (M6 measured the pool: Fire/Fighting/Lightning hit
   220/188/155 cards; replace the pool histogram with the *harvested-meta* histogram
   once M7.0 lands).

### 2.2 Attacker-line scoring

```python
# rl/deck_build.py
@dataclass
class Line:
    stages: list[int]          # card_ids basic -> final
    final: int
    energy_type: int
    peak_damage: int           # max_damage of final
    setup_cost: int            # min energy to best attack + n_stages
    hp: int
    prize_liability: int       # prizes_on_ko of final (ex/mega = 2-3)

def enumerate_lines() -> list[Line]                 # walk evolves_from_id chains
def score_line(line, meta_types: dict[int, float]) -> float
    # damage-per-energy * weakness-coverage(meta_types)
    #   - prize_liability_penalty - setup_cost_penalty
```

`meta_types` starts as the pool-wide type histogram and is replaced by the harvested-meta
histogram after M7.0 — the first concrete feedback edge from ingestion into deck
building. `setup_cost_penalty` starts strong (M6's residual weakness is slow setups —
the floor-test self-deck losses) and is revisited as the pilot's closing improves.

### 2.3 The trainer-shell problem — three sources

`cards_features.parquet` cannot tell Ultra Ball from a blank item (`is_item/is_supporter/
is_stadium/is_tool` is all we have). Three sources, in deployment order:

| Source | Cost | Quality |
|---|---|---|
| (1) Shell library mined from the known decks (`decks/*.csv`) | ~0 | proven-good, but only 3–4 samples |
| (2) Per-archetype `hill_climb`+`mutate_flex` refinement (exists) | ~9 min / 50 proposals | learned, but only as good as the eval field |
| (3) **Harvested shells**: trainer-frequency across winning harvested decks | free after M7.0 | the real answer — 4,500 teams already did this search |

Ship (1) as the template default now, run (2) as the per-candidate refinement, swap in
(3) the moment ingestion lands. Persist as `data/shells.json`
(`{"lucario_engine": [ids...], "meta_staples_v1": [ids...]}`).

### 2.4 Search strategy over decks

- **Template enumeration** (new, global coverage): top-K lines × shells ×
  {no-tech, best-coverage-tech} → dedupe → 20–60 candidates per iteration, every one
  `validate_deck`-legal by construction.
- **`hill_climb` with `mutate_flex`** (exists, local refinement) on the top 3–5
  templates; keep the ≥57%-over-150-games accept rule. Population `evolve()` stays
  available but non-default (DECISIONS: openskill at small samples promoted a worse deck).
- **Fitness is always vs the diverse fixed anchor field, never mirror-only** (the
  documented mirror-overfit failure). New in `rl/deck_search.py`:

```python
def field_fitness(deck, field: list[OpponentSpec], games_per_opp=60,
                  pilot="generic") -> tuple[float, dict]:
    # slot-fair vs each field member; (weighted mean wr, per-opponent breakdown)
```

`OpponentSpec` generalizes `rl/collector.py`'s picklable tuples — `("rule", agent, deck)`,
`("model", ckpt, deck)`, `("generic", deck)`, `("random", deck)` — defined once, imported
by collector, league, deck search, and gates (today the collector has its own copy).

---

## 3. Component B — pilot training ("learn to play a new deck")

### 3.1 The options, honestly compared

**(a) Generic rule pilot as-is** — the baseline evaluator. 95% vs random, expert-parity
deck ranking, ~25–30% vs experts. Zero cost, already shippable. The floor every other
option must beat *on the candidate decks* to justify itself.

**(b) Deck-conditioned neural pilot — the core bet (M7.3).** Key fact from the code:
`rl/encoders.py` already represents cards by feature rows (`FEAT[card_id]`), so the
representation is nominally deck-agnostic — bc_v1's problem was that it only ever *saw*
Kyogre states. Two upgrades:

1. **Card-ID embeddings**: `nn.Embedding(1268, 16)` concatenated with the feature row
   wherever a card enters state/option encoding. The 38 features are nearly blind for
   trainers; embeddings let a multi-deck pilot learn per-card behavior from data. Export
   stays numpy-trivial (embedding = matrix row lookup, ~20k floats in the npz). This
   changes `STATE_DIM`/`OPTION_DIM` → the whole `encoder_parity_check` discipline applies.
2. **Deck-context features**: pooled FEAT-sums of my full 60-card list and of my
   *remaining* deck (list minus hand/board/discard — all observable). Cheap, helps
   mulligan/fetch decisions. NOT a 1,268-wide per-id count vector — the reverted-feature
   lesson (pooled features already carry archetype at 98% probe accuracy; wide sparse
   ids overfit).

Training: **BC warm-start from the generic pilot as teacher** on self-play across the
whole deck population (both seats are the teacher → both recordable). Fidelity should be
high — unlike the Lucario expert's hidden `AttackPlan`, the generic pilot's scoring is a
function of observable features plus the same `rl/combat.py` lookahead we already encode,
so there is no invisible reasoning to alias. Result: a pilot ≈ teacher strength but
*differentiable* — improvable by (d)/(e) where the frozen rule pilot is not. "Learning a
new deck" = evaluate directly (gate G5) or briefly fine-tune (c).

**(c) Per-deck fine-tuning** — take (b)'s net, fine-tune on the champion deck's
self-play for a few epochs at ship time; keep an unseen-deck regression eval against
catastrophic forgetting. Not a loop-inner step.

**(d) AlphaZero-lite** — the big lookahead bet; see §3b (L4).

**(e) PPO retry, bug fixed, diverse field (M7.4b).** Two things changed since PPO
stalled: (i) the suspected loss bug — fix `loss = policy_loss * VALUE_COEF * value_loss`
to `+` in **both** `rl/ppo.py` and `tcg/ppo.py` and flip
`tests/test_ppo.py::test_ppo_update_parity_pins_the_loss_quirk` to pin the corrected
form; (ii) the field — PPO previously overfit to 3 opponents; with harvested-meta
opponents and multi-deck self-play the on-policy signal is qualitatively different. Apply
the known fixes: entropy 0.001, critic warm-started via `rl/value_train.py` (proven)
instead of trusting PPO to learn it.

**Recommendation:** (a) as evaluator now → L1 lookahead into the teacher (§3b) → (b) as
the M7.3 deliverable → (e) as the first cheap improvement pass → (d) only if (e) fails
twice and the leaderboard says the ceiling matters. (c) at ship time.

### 3.2 Concrete changes

- Encoders v2 (`encode_state_v2`/`encode_option_v2`, `EMBED_DIM=16` id channels +
  deck-context pools) — land side-by-side, keep v1 intact (bc_v1/value artifacts depend
  on it). Prefer landing in `tcg/encoders.py` first to inherit the parity-test discipline.
- `rl/policy.py`: `OptionScorerV2(hidden=256, embed=16)` — same pointer architecture,
  wider inputs.
- `rl/bc.py`: `--teacher generic` mode (teacher = generic pilot on a per-game deck
  sampled from the candidate population; record BOTH seats) and `--decks
  data/league/population.json`.
- `rl/collector.py`: per-game deck sampling instead of the fixed `LEARN_DECK`; a
  `("generic", deck)` opponent spec; write `deck_ids` into each shard so training can
  condition/split by deck.

### 3b Strategic lookahead — setup turns and combo turns

The experts' documented edge is multi-turn planning (`AttackPlan` in
`sample-agent/main.py`), and the requirement is explicit: the agent must value actions
with **no immediate reward** — benching and powering a future attacker, and chaining
trainer + items into a single lethal multi-prize turn (e.g. damage-boost items into a KO
on a 3-prize mega).

*Why every current method misses this:* BC can't imitate unobservable lookahead
(fidelity collapses — DECISIONS 2026-07-08); full-game MCTS failed for identified
reasons (value head OOD on search states; Snorlax-placeholder determinization; more sims
made it worse); 1-ply value lookahead lost to greedy; PPO's sparse ±1 gives setup
actions almost no credit.

Four options, in escalation order — L1/L2 are engineering with proven mechanisms, L3/L4
are the research bets that need dedicated time:

**(L1) Generalized rule lookahead — "race math" in the generic pilot (cheap, first).**
Port the expert's `AttackPlan` idea deck-agnostically on the existing `rl/combat.py`
core: for each of my Pokémon, turns-to-attack-readiness (energy gap vs attach-1/turn)
and turns-to-KO vs the opponent's active/bench; the same in reverse for the opponent →
a **prize-race table**. Feed it into `score_attach` / `score_card` / bench-promotion
(charge the attacker that wins the race; bench the line that matters) and into
`encode_state` as features — the M3 combat features were 1-turn only; these are k-turn.
Proven (it IS the experts' edge), and it upgrades the BC teacher and every evaluator
downstream. Directly targets the "set up a powerful attacker on the bench" behavior.

**(L2) Within-turn combo search — deterministic, solves the lethal-combo-turn case.**
Structural fact: *within my own turn*, my action sequences (items/supporter/abilities/
attach/evolve/retreat → attack) are fully observable and near-deterministic — only my
own deck's draw order is hidden, and the engine re-prompts after every action, so the
plan can be recomputed after each reveal. Use the engine forward model
(`search_begin`/`search_step`) as a **depth-limited turn solver**: beam/DFS over my own
turn's sequences, score end-of-turn states (prizes taken this turn ≫ lethal-next-turn
setup ≫ board development), and always answer "is there a sequence that KOs for 2–3
prizes / wins right now?" before settling for setup. Cost is bounded (fire the full
enumeration only when a damage-boost card or near-lethal is detected) and fits the 600s
overage budget. Highest lookahead leverage per CPU cycle: **no determinization of
opponent hidden zones needed at all**. This is exactly the "trainer + multiple items to
boost damage → KO → 3 prizes in one turn" scenario.

**(L3) Archetype-inferred determinization (prerequisite for L4; synergy with §5).**
MCTS failed partly on placeholder determinization. With harvested meta decklists, infer
the opponent's archetype from revealed cards (pooled features already identify the
opponent at 98% in the linear probe) → sample their hidden hand/deck/prizes from the
matched harvested list minus revealed cards, several samples per decision. Ingestion
becomes a lookahead enabler, not just an eval-field fix.

**(L4) AlphaZero-lite — search as the *training-time* policy-improvement operator.**
Self-play where L2+L3-powered MCTS produces the targets: the value head trained
supervised on **search-visited states** (fixes the OOD failure; supervised value training
is the one proven method), the policy trained on visit counts; distill back into the
greedy pointer network so the *shipped* agent has lookahead internalized at zero
inference cost (MCTS-at-inference stays optional within budget). The official reference
notebook (`reference/reinforcement-learning-and-mcts-sample-code.ipynb`) implements
exactly this loop — ~100-line MCTS core, TD(λ) value targets, 64-combination
multi-select handling; **study/adapt it rather than rewrite**. Dedicated research time
is budgeted in M7.4b before committing.

*(Supporting, any RL pass)* **Potential-based reward shaping on board development** —
energy attached to viable attackers, evolution-line progress, prize-race delta — so
setup actions get credit without corrupting the optimal policy.

### 3.3 Pilot gates (per pilot P on deck D; all slot-fair, direct engine)

| Gate | Threshold | n | Why |
|---|---|---|---|
| G1 legality/safety | 0 illegal picks, 0 crashes | 200 | ship safety |
| G2 vs random | ≥90% | 200 | absolute floor (experts cap ~88–95%) |
| G3 vs generic pilot, same deck | ≥55% (M7.3: ≥50% parity) | 400 | did learning beat the teacher |
| G4 vs Lucario expert (its deck) | ≥35% (M7.3) → ≥45% (M7.4) | 400 | the expert gap, staged honestly (now 25–30%) |
| G5 unseen-deck generalization | ≥ generic −3pp on 5 held-out decks | 200/deck | proves deck-conditioning; guards fine-tune forgetting |
| G6 time budget | mean move <50ms; p99 game overage <60% of 600s | measured over G2–G4 | Kaggle timeout safety |

n=400 resolves a 55% threshold at ±5pp (95% CI).

---

## 4. Component C — the league loop (`rl/league.py` + `rl/matchrunner.py`, new)

A persistent openskill (PlackettLuce, as in `rate_population`) table whose **entries are
(deck, pilot) pairs**, two classes:

- **Anchors (frozen, never removed):** Lucario expert, Iono expert, tuned-Lucario,
  generic+Lucario, generic+Iono, bc_v1+Kyogre, random+Kyogre — and, after M7.0,
  `generic+harvested_deck_k` for the top-K meta decks, versioned (`meta_v1`, `meta_v2`, …).
  Anchors pin the rating scale across weeks and are the anti-mirror-overfit field.
- **Candidates:** new decks (piloted by generic) and new pilots (on their decks), rated
  vs anchors first (fixed schedule, e.g. 40 games each), then top peers.

```python
# rl/league.py — persisted in data/league/league.json
@dataclass
class Entry:
    entry_id: str        # "generic+lucario", "osv2_it3+aggro_fire_a"
    pilot: OpponentSpec
    deck_hash: str       # sha1 of sorted ids; deck at data/league/decks/<hash>.csv
    mu: float; sigma: float; games: int
    frozen: bool         # anchors
    origin: str          # "template" | "hillclimb" | "harvested" | "checkpoint"
```

All games run through one shared multiprocessing **match runner** extracted from the
`rl/collector.py` worker pattern (`rl/matchrunner.py`), replacing the three separate
battle-loop implementations (collector, `deck_search.matchup` with its hardcoded
Lucario pilots, eval).

**Promotion rules**

- A **deck** is promoted when `field_fitness ≥ 0.55` vs the full anchor field under the
  generic pilot AND ≥50% vs the current champion deck (same pilot, n=400).
- A **pilot** is promoted when it passes G1–G6.
- A **(deck, pilot) pair** is ship-eligible when both hold together, the existing
  `rl.gate` suite (parity, deck legality, bundle isolation, gate game) is green, and it
  beats the currently-shipped pair ≥55% offline.
- **Gates are data**, not prints: `data/league/gates/<entry>.json` records thresholds,
  n, date, and field version, so the diary can cite them.
- Promotions must hold on the *newest* meta version after each ingestion refresh.
- Every submission id is logged in `data/league/submissions.json` — it feeds ingestion.

---

## 5. Component D — Kaggle replay ingestion (`rl/kaggle_ingest.py`, new) — build FIRST

### 5.1 Mechanics ([NET] — validate before building the parser)

The official Kaggle CLI (already a dependency) supports simulation replays directly:

```bash
kaggle competitions submissions -c pokemon-tcg-ai-battle --csv   # our submission ids
kaggle competitions episodes <submission_id>                     # episode list
kaggle competitions replay <episode_id>       # -> episode-<id>-replay.json
kaggle competitions logs <episode_id> <agent_index>              # per-agent logs
```

Fallbacks for bulk/other-team metadata: the raw endpoints
(`competitions.EpisodeService/ListEpisodes|GetEpisodeReplay`) and the public
**Meta Kaggle** dataset (`Episodes.csv`, `EpisodeAgents.csv`, refreshed daily) to find
top-team episode ids without hammering the API. Politeness: cache every response to
disk keyed by episode id, throttle ~1 req/s, resumable; plan for hundreds of episodes,
not tens of thousands.

### 5.2 What's extractable (confidence-ranked; schema verified in M7.0)

1. **Opponent 60-card decklists — HIGH.** The first action of each agent IS its deck
   (the env contract). `steps[0][i]["action"]` (off-by-one to verify). This alone fixes
   wall #1.
2. **Outcomes / per-episode scores / matchup graph — HIGH.** Per-opponent-archetype win
   rates for our submissions → counter-tuning and loss forensics.
3. **Both agents' actions per step — HIGH** (always serialized).
4. **Both agents' observations — MEDIUM, verify.** If the opponent seat's observations
   are populated, we can BC directly on *winning opponents' decisions* (subject to the
   known BC-fidelity ceiling — their hidden reasoning aliases like the Lucario expert's
   did). If blanked, meta decks piloted by the generic pilot still deliver the bigger win.
5. **The `visualize` JSON** — if embedded, the existing `rl/replay.py` browser gives
   loss forensics for free.

### 5.3 Module and schema

```python
# rl/kaggle_ingest.py   (requests/CLI + polars; CLI: python -m rl.kaggle_ingest <cmd>)
def list_episodes(submission_id=None, team_id=None) -> pl.DataFrame
def fetch_episode(episode_id, cache=DATA/"kaggle/raw") -> dict     # cached, throttled
def parse_episode(raw, our_submission_ids) -> ParsedEpisode
def refresh(our_subs, top_team_ids=(), max_new=500) -> None
def harvest_decks(min_games=3) -> pl.DataFrame     # -> data/kaggle/opp_decks.parquet
def build_meta_field(top_k=8, dedupe_by="archetype") -> list[OpponentSpec]
def bc_shards_from_opponents(min_score, out=DATA/"bc_meta") -> list[str]   # if obs present
def forensics(submission_id) -> report   # losses by archetype, game length, first-KO
```

```
data/kaggle/raw/episode_<id>.json.gz     immutable cache
data/kaggle/episodes.parquet             episode_id, sub/team ids, scores, rewards, n_steps
data/kaggle/opp_decks.parquet            episode_id, seat, team, deck ids, deck_hash, archetype
data/kaggle/meta_v<N>/                   frozen snapshot: deck csvs + manifest.json
```

Archetype clustering: exact-list hash first, then cosine over the pooled FEAT vector of
the Pokémon core (the representation the probe validated). Weight the meta field by
observed frequency × opponent score.

### 5.4 Feedback edges (the point of all this)

- `build_meta_field` → league anchors + collector opponent pool + `field_fitness` field.
- `opp_decks` type histogram → `deck_build.score_line` weakness targeting.
- Harvested trainer staples → `data/shells.json` (§2.3 source 3).
- Harvested decklists → **L3 archetype determinization** (§3b) for search.
- `bc_shards_from_opponents` → optional fine-tuning on high-scoring opponents.
- `forensics` → the M7 diary; a curated **combo-position suite** (positions where a
  lethal existed and was missed) becomes the L2 turn-solver's test set.

---

## 6. Orchestration + budget

CLI entry points (existing `python -m rl.<mod>` convention):

```
python -m rl.kaggle_ingest refresh --subs 54474043 ...
python -m rl.deck_build generate --n 40 --meta data/kaggle/opp_decks.parquet
python -m rl.league add --deck decks/gen/<hash>.csv --pilot generic
python -m rl.league run --games-per-anchor 40
python -m rl.league standings
python -m rl.deck_search climb --seed-deck <hash> --field meta_v1
python -m rl.bc --teacher generic --decks data/league/population.json --arch v2
python -m rl.ppo --checkpoint checkpoints/osv2_bc.pt --field meta_v1
python -m rl.league gate --entry <id>
python -m rl.export --agent <...> && python -m rl.gate --agent <...>    # existing ship path
```

One loop iteration (~1 day wall-clock): ingest refresh → generate + league-rate decks
(~30 min engine) → hill-climb top decks (~30 min) → retrain/fine-tune pilot (hours CPU)
→ gate → ship if it beats the champion → read board + forensics → diary.

Budget reality at ~50k games/hr: rating 40 template decks vs 7 anchors × 40 games ≈
11,200 games ≈ 13 min; hill-climb (50×150) ≈ 9 min; BC collection (30 decks × 200) ≈
7 min; full gate suite ≈ 2 min per candidate. **The engine is not the bottleneck —
torch-on-CPU training and Kaggle's episode drip-rate are.** Design accordingly: big
honest n everywhere, few precious training runs.

---

## 7. Milestones

- **M7.0 — Ingestion spike (2d).** *Status 2026-07-09: infrastructure + 21 offline
  tests landed (`rl/kaggle_ingest.py`); the [NET] verification runs via the runbook in
  [M7.md](M7.md) — this sandbox has no Kaggle access (risk 8).* Fetch episodes for
  54474043 (+ bc_v1's old sub); verify the JSON schema; extract ≥20 opponent decks +
  outcomes. **Go/no-go:** decklists extractable → full plan; only outcomes → keep
  forensics, drop meta-field/BC edges, recalibrate gates to the old anchor field. Also
  settles: are opponent observations present (unlocks BC-on-winners)?
- **M7.1 — Deck factory (2–3d).** *Status 2026-07-09: `deck_build.py` + shells v1 +
  `field_fitness` landed with 15 tests; 40 candidates generated to `decks/gen/`; the
  [ENGINE] gate runs via the runbook in [M7.md](M7.md).* `deck_build.py` + shells v1;
  40 candidates league-rated under the generic pilot. **Gate:** ≥1 generated deck ≥50%
  vs the Lucario deck same-pilot (n=400) — factory reaches parity with the human-tuned
  seed; ≥3 distinct archetypes above the anchor-field median.
- **M7.2 — League + gates (1–2d, overlaps).** *Status 2026-07-09: `league.py` +
  `matchrunner.py` landed with 35 tests (suite 277); anchors bootstrapped into
  `data/league/league.json`; deck_search loops delegate to the matchrunner; the
  [ENGINE] self-test runs via the runbook in [M7.md](M7.md).* `league.py`,
  `matchrunner.py`, persistent standings, gates-as-data. **Gate (self-test):** anchors
  reproduce the known ordering (Lucario expert > tuned > generic+Lucario > bc_v1 >
  random).
- **M7.2b — Race-math lookahead L1 (1–2d, parallel).** *Status 2026-07-09: race
  primitives + scorer rewiring landed in both rl/ and tcg/ (parity-pinned, 294 tests);
  `decks/floor_zero_damage.csv` committed; the [ENGINE] gates run via the runbook in
  [M7.md](M7.md); the k-turn `encode_state` features are deferred to M7.3 encoders-v2
  per risk 4 (v1 must stay byte-identical).* Prize-race table on `rl/combat.py` →
  generic-pilot scorers + k-turn state features. **Gate:** floor test (vs zero-damage
  deck) ≥90% (currently 75% — the self-deck/setup weakness is a lookahead failure) and
  vs-Lucario-expert above 30%.
- **M7.3 — Deck-conditioned pilot v1 (3–5d).** Encoders v2 (+embeddings, +deck pools),
  `OptionScorerV2`, BC from the L1-improved generic pilot across the M7.1 population.
  **Gate:** G5 pass and G3 ≥50% (parity with teacher — BC alone should *match*, not beat).
- **M7.4a — Within-turn combo solver L2 (3–4d incl. research).** Forward-model turn
  search, lethal-first. **Gate:** on the curated combo-position suite (from forensics),
  finds the multi-prize lethal the greedy pilot misses; end-to-end ≥55% vs the same
  pilot without L2; p99 time within budget.
- **M7.4b — Improvement pass (1wk, decision-gated).** Fix the PPO loss in `rl/ppo.py` +
  `tcg/ppo.py` (+ flip the pin test); critic warm-start via `rl/value_train.py`; PPO vs
  the meta_v1 field with multi-deck self-play + setup shaping. In parallel: L3
  archetype determinization. **Gate:** G3 ≥55%, G4 ≥35%. Two failed honest PPO attempts
  → **L4 AlphaZero-lite go/no-go review** (budgeted research: dissect the reference
  notebook's MCTS/value-target loop before committing — don't drift into it).
- **M7.5 — Counter-meta ship loop (ongoing).** Weakness-target the harvested meta, gate,
  ship, read the board. **M7 success criterion:** a submission scoring above the
  rule-agent probe's final score (and above bc_v1's 640), with forensics explaining *why*.

## 8. Risks

1. **Episode schema differs / endpoint throttled for cabt** — M7.0 exists to find out
   first; fallback path defined.
2. **Opponent observations absent** → the BC-on-winners edge dies; harvested decks
   piloted by the generic pilot still fix field diversity (the bigger win).
3. **Generic pilot as BC teacher caps the neural pilot** — by design; escape hatches are
   (e)/(d). If G4 stays ~30% after M7.4, the honest conclusion may be that M6-style
   heuristic pilot engineering is the better marginal spend.
4. **Encoder-v2 churn** invalidates bc_v1/value artifacts and numpy parity — keep v1
   intact, land v2 side-by-side (the `tcg/` discipline), extend `encoder_parity_check`
   before any v2 ship.
5. **Meta snapshot staleness** — version `meta_vN`, re-validate promotions on refresh.
6. **`rl/` vs `tcg/` drift** — every touched module has a parity-pinned twin; changes
   land in both + tests. New modules (`kaggle_ingest`, `league`, `deck_build`,
   `matchrunner`) get ONE home (decide at implementation with the tcg/ graduation plan).
7. **Turn-solver blowup** — multi-select contexts explode the within-turn branching;
   bound with beams + the lethal-trigger heuristic, and measure G6 before shipping L2.
8. **This sandbox has no engine** — every win-rate number above is [ENGINE]-only;
   network steps are [NET]. Nothing here can be validated end-to-end without them.

## 9. Verification checklist

[ENGINE] = needs `cg/libcg.so`; [NET] = needs Kaggle access; unflagged = runs under the
fake-`cg` test stubs.

1. `validate_deck` on every generated candidate; template counts sum to 60 (unit test).
2. Line enumeration spot-checked against the known Lucario line (`decks/lucario.csv`).
3. [NET] One episode of 54474043 fetched; assert first-step actions are 60-int lists
   passing `validate_deck`; document the actual schema in the module docstring.
4. [NET] ≥20 decks harvested; archetype clusters eyeballed.
5. [ENGINE] `matchrunner` reproduces `rl/eval.play_games` numbers (generic vs random
   ~95%; unswapped player-0 bias ~61%).
6. [ENGINE] League self-test: anchor ordering matches the M6 measurements.
7. Encoders v2 dims computed from constants; v1 path byte-identical (existing parity
   tests); overfit-a-tiny-batch sanity for `OptionScorerV2`.
8. [ENGINE] BC-from-generic fidelity ≥80% val top-1; gates G1–G6 as specified.
9. PPO fix: pin test updated in BOTH copies; [ENGINE] 5-iter smoke run shows value_loss
   moving.
10. [ENGINE] L2 turn solver beats no-L2 on the combo suite; G6 timing holds.
11. [ENGINE] Full ship path for the winning pair (`rl.export`, `rl.gate`), then [NET]
    submit.
