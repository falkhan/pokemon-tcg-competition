"""Parallel self-play collection for PPO with an OPPONENT POOL (M2 Path A).

Same collector as the old ``rl/collector.py``, plus ``write_shard`` — the
.npz decision-shard writer that used to be transcribed inline in three
modules (rl/bc.py, rl/collector.py, rl/value_train.py).

Why a pool: pure mirror self-play only teaches the agent to beat copies of itself, but
the leaderboard is full of other decks/strategies. So the LEARNING policy plays one seat
(sampled actions, recorded), and the OTHER seat is sampled per game from a pool of
opponents: the current policy (mirror), past promoted selves, the rule agents on their
own decks, and random. Only the learning seat's decisions are recorded — PPO is on-policy,
so we can only update on actions taken by the current policy with known logprobs.

The cg engine holds ONE battle per process (cg/sim.py global) -> multiprocessing, one
engine per worker. Everything heavy is imported/loaded inside the worker; only picklable
config (paths, deck names, spec tuples) crosses the process boundary.

Opponent spec tuples (picklable):
  ("model", checkpoint_path, deck_name)   a neural opponent (current or past self)
  ("rule",  agent_name, deck_name)        a rule agent (lucario/iono) on some deck
  ("random", deck_name)                   uniform-random legal moves

Shard layout = BC fields + PPO fields (actions/logprobs/values/rewards/players); see
tcg/ppo.py load_shards. `players` here is always the learning seat for that game.

Usage:  python -m tcg.selfplay --games 400 --workers 4 --checkpoint checkpoints/bc_v1.pt
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

from tcg.decks import ROOT, load_deck

OUT_DIR = ROOT / "data" / "ppo"
PRIZE_SHAPING = 0.1             # per-prize reward shaping (terminal ±1 dominates)
LEARN_DECK = "kyogre"           # the deck the learning policy pilots (our champion base)
WORKER_SEED_BASE = 1000         # worker i seeds python/torch RNGs with BASE + i
LOGPROB_EPS = 1e-12             # keeps log(prob) finite for near-zero probabilities
RECENT_SELVES = 3               # how many past promoted checkpoints join the pool


def write_shard(path: Path, columns: dict[str, list],
                int32_columns: frozenset[str], float32_columns: frozenset[str]) -> None:
    """np.savez_compressed with the repo's decision-shard conventions.

    Per-decision fixed-width rows (``states``, and encoders-v2's ``state_ids``)
    are stacked; per-option ragged menus (``options``, v2's ``option_ids``) are
    concatenated flat with per-decision lengths riding along in ``n_options``;
    every other column becomes a typed 1-D array.
    """
    arrays = {}
    for name, values in columns.items():
        if name in ("states", "state_ids"):
            arrays[name] = np.stack(values)
        elif name in ("options", "option_ids"):
            arrays[name] = np.concatenate(values)
        elif name in int32_columns:
            arrays[name] = np.array(values, dtype=np.int32)
        elif name in float32_columns:
            arrays[name] = np.array(values, dtype=np.float32)
        else:
            raise ValueError(f"column {name!r} has no declared dtype")
    np.savez_compressed(path, **arrays)


PPO_INT32_COLUMNS = frozenset({"n_options", "actions", "game_ids", "players"})
PPO_FLOAT32_COLUMNS = frozenset({"logprobs", "values", "rewards"})


def default_pool(checkpoint: str, learn_deck: str = LEARN_DECK):
    """Build (specs, weights): current self (mirror) + rule experts + random + recent selves."""
    specs = [("model", str(checkpoint), learn_deck),   # mirror vs current policy
             ("rule", "lucario", "lucario"),           # strong fighting deck
             ("rule", "iono", "iono"),                 # lightning deck
             ("random", "kyogre")]                     # weak/varied baseline
    weights = [0.4, 0.2, 0.2, 0.1]
    past = sorted((ROOT / "checkpoints").glob("ppo_it*.pt"))[-RECENT_SELVES:]
    for path in past:
        specs.append(("model", str(path), learn_deck))
        weights.append(0.1 / len(past))
    weight_array = np.array(weights, dtype=np.float64)
    return specs, (weight_array / weight_array.sum()).tolist()


def play_worker(args: tuple) -> str:
    """One worker process: play its share of games, write one shard, return its path.

    Module-level (not nested) so the "spawn" multiprocessing context can pickle it.
    """
    worker_id, n_games, checkpoint, learn_deck_name, specs, weights, out_dir, seed = args

    # Everything heavy is imported here, inside the worker process.
    import random
    import torch
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    from tcg.encoders import encode_context, encode_option, encode_state
    from tcg.network import OptionScorer
    from tcg.teachers import load_teacher

    rng = random.Random(seed)
    torch.manual_seed(seed)

    def load_model(path):
        model = OptionScorer()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    learner = load_model(checkpoint)
    learn_deck = load_deck(learn_deck_name)

    def run_model(model, observation, sample: bool):
        """Return (picks, action, logprob, value, state_ctx, option_vectors) for a model's turn."""
        state_ctx = np.concatenate([
            encode_state(observation.current),
            encode_context(observation.select.context)]).astype(np.float32)
        option_vectors = np.stack([encode_option(option, observation)
                                   for option in observation.select.option]).astype(np.float32)
        with torch.no_grad():
            logits, value = model(torch.from_numpy(state_ctx).unsqueeze(0),
                                  torch.from_numpy(option_vectors).unsqueeze(0))
            logits = logits.squeeze(0)
            probs = torch.softmax(logits, dim=0)
            action = int(torch.multinomial(probs, 1)) if sample else int(torch.argmax(logits))
            logprob = float(torch.log(probs[action] + LOGPROB_EPS))
        k = observation.select.maxCount
        order = torch.argsort(logits, descending=True).tolist()
        picks = [action] + [i for i in order if i != action]
        return picks[:k], action, logprob, float(value), state_ctx, option_vectors

    # Build each distinct opponent once (cache models / rule instances by spec).
    opponent_cache: dict = {}

    def get_opponent(spec):
        if spec not in opponent_cache:
            kind = spec[0]
            if kind == "model":
                opponent_model = load_model(spec[1])
                act = lambda obs_dict, model=opponent_model: run_model(
                    model, to_observation_class(obs_dict), sample=False)[0]
                opponent_cache[spec] = (act, load_deck(spec[2]))
            elif kind == "rule":
                agent = load_teacher(f"opp_{worker_id}_{len(opponent_cache)}",
                                     agent=spec[1], deck=spec[2])
                opponent_cache[spec] = (agent, load_deck(spec[2]))
            else:  # random
                act = lambda obs_dict: rng.sample(
                    range(len(obs_dict["select"]["option"])),
                    obs_dict["select"]["maxCount"])
                opponent_cache[spec] = (act, load_deck(spec[1]))
        return opponent_cache[spec]

    columns: dict[str, list] = {name: [] for name in (
        "states", "options", "n_options", "actions",
        "logprobs", "values", "rewards", "game_ids", "players")}

    for game in range(n_games):
        spec = specs[rng.choices(range(len(specs)), weights=weights)[0]]
        opponent_act, opponent_deck = get_opponent(spec)
        learn_seat = game % 2                          # alternate seats -> slot-fair data
        deck_p0, deck_p1 = ((learn_deck, opponent_deck) if learn_seat == 0
                            else (opponent_deck, learn_deck))

        obs_dict, start = battle_start(deck_p0, deck_p1)
        if start.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected deck (errorType={start.errorType})")

        rows: list[dict] = []
        prev_delta = 0.0
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            if seat == learn_seat:
                observation = to_observation_class(obs_dict)
                picks, action, logprob, value, state_ctx, option_vectors = \
                    run_model(learner, observation, sample=True)
                me = observation.current.players[seat]
                opponent = observation.current.players[1 - seat]
                # prizes I took minus prizes they took; reward the per-step change
                delta = (6 - len(opponent.prize)) - (6 - len(me.prize))
                rows.append(dict(state=state_ctx, options=option_vectors,
                                 action=action, logprob=logprob, value=value,
                                 reward=PRIZE_SHAPING * (delta - prev_delta)))
                prev_delta = delta
                obs_dict = battle_select([int(i) for i in picks])
            else:
                obs_dict = battle_select([int(i) for i in opponent_act(obs_dict)])

        result = obs_dict["current"]["result"]         # 0/1 winner, 2 draw
        battle_finish()
        if rows:                                        # terminal reward on learner's last row
            rows[-1]["reward"] += 0.0 if result == 2 else (1.0 if result == learn_seat else -1.0)

        for row in rows:
            columns["states"].append(row["state"])
            columns["options"].append(row["options"])
            columns["n_options"].append(len(row["options"]))
            columns["actions"].append(row["action"])
            columns["logprobs"].append(row["logprob"])
            columns["values"].append(row["value"])
            columns["rewards"].append(row["reward"])
            columns["game_ids"].append(game)
            columns["players"].append(learn_seat)

    out = Path(out_dir) / f"ppo_shard_w{worker_id:02d}.npz"
    write_shard(out, columns, PPO_INT32_COLUMNS, PPO_FLOAT32_COLUMNS)
    return str(out)


def collect(n_games: int, checkpoint: str, n_workers: int = 4,
            learn_deck: str = LEARN_DECK, pool=None, out_dir: Path = OUT_DIR) -> list[str]:
    """Collect n_games across n_workers, learning policy vs an opponent pool."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs, weights = pool if pool is not None else default_pool(checkpoint, learn_deck)
    per_worker = [n_games // n_workers + (1 if i < n_games % n_workers else 0)
                  for i in range(n_workers)]
    jobs = [(i, per_worker[i], checkpoint, learn_deck, specs, weights, str(out_dir),
             WORKER_SEED_BASE + i)
            for i in range(n_workers) if per_worker[i] > 0]

    context = mp.get_context("spawn")
    with context.Pool(len(jobs)) as worker_pool:
        return worker_pool.map(play_worker, jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str,
                        default=str(ROOT / "checkpoints" / "bc_v1.pt"))
    args = parser.parse_args()

    import time
    start_time = time.time()
    shards = collect(args.games, args.checkpoint, args.workers)
    elapsed = time.time() - start_time
    print(f"{args.games} games in {elapsed:.0f}s "
          f"({3600 * args.games / elapsed:.0f} games/hr) -> {shards}")
