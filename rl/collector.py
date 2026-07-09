"""Parallel self-play collection for PPO with an OPPONENT POOL (M2 Path A).

Why a pool: pure mirror self-play only teaches the agent to beat copies of itself, but
the leaderboard is full of other decks/strategies. So the LEARNING policy plays one seat
(sampled actions, recorded), and the OTHER seat is sampled per game from a pool of
opponents: the current policy (mirror), past promoted selves, the rule agents on their
own decks, and random. Only the learning seat's decisions are recorded — PPO is on-policy,
so we can only update on actions taken by the current policy with known logprobs.

The cg engine holds ONE battle per process (cg/sim.py global) -> multiprocessing, one
engine per worker. Everything heavy is imported/loaded inside the worker; only picklable
config (paths, deck names, spec tuples) crosses the process boundary.

Opponent spec tuples (picklable; canonical vocabulary now in rl/matchrunner.py —
this worker's battle loop migrates onto it in M7.3 together with per-game deck
sampling, since it records training tensors mid-game on top of match running):
  ("model", checkpoint_path, deck_name)   a neural opponent (current or past self)
  ("rule",  agent_name, deck_name)        a rule agent (lucario/iono) on some deck
  ("random", deck_name)                   uniform-random legal moves

Shard layout = BC fields + PPO fields (actions/logprobs/values/rewards/players); see
rl/ppo.py load_shards. `players` here is always the learning seat for that game.

Usage:  python -m rl.collector --games 400 --workers 4 --checkpoint checkpoints/bc_v1.pt
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "ppo"
DECK_DIR = ROOT / "decks"
PRIZE_SHAPING = 0.1
LEARN_DECK = "kyogre"           # the deck the learning policy pilots (our champion base)


def _deck(name: str) -> list[int]:
    return [int(x) for x in (DECK_DIR / f"{name}.csv").read_text().split() if x.strip()]


def default_pool(checkpoint: str, learn_deck: str = LEARN_DECK):
    """Build (specs, weights): current self (mirror) + rule experts + random + recent selves."""
    specs = [("model", str(checkpoint), learn_deck),   # mirror vs current policy
             ("rule", "lucario", "lucario"),           # strong fighting deck
             ("rule", "iono", "iono"),                 # lightning deck
             ("random", "kyogre")]                     # weak/varied baseline
    weights = [0.4, 0.2, 0.2, 0.1]
    past = sorted((ROOT / "checkpoints").glob("ppo_it*.pt"))[-3:]   # recent promoted selves
    for p in past:
        specs.append(("model", str(p), learn_deck))
        weights.append(0.1 / len(past))
    w = np.array(weights, dtype=np.float64)
    return specs, (w / w.sum()).tolist()


def _play_worker(args: tuple) -> str:
    worker_id, n_games, checkpoint, learn_deck_name, specs, weights, out_dir, seed = args

    import random
    import torch
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    from rl.encoders import encode_context, encode_option, encode_state
    from rl.policy import OptionScorer
    from rl.teacher import load_teacher

    rng = random.Random(seed)
    torch.manual_seed(seed)

    def load_model(path):
        m = OptionScorer(); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
        return m

    learner = load_model(checkpoint)
    learn_deck = _deck(learn_deck_name)

    def run_model(model, obs, sample: bool):
        """Return (picks, action, logprob, value, state_ctx, opts) for a model's turn."""
        sc = np.concatenate([encode_state(obs.current), encode_context(obs.select.context)]).astype(np.float32)
        opts = np.stack([encode_option(o, obs) for o in obs.select.option]).astype(np.float32)
        with torch.no_grad():
            logits, value = model(torch.from_numpy(sc).unsqueeze(0), torch.from_numpy(opts).unsqueeze(0))
            logits = logits.squeeze(0)
            probs = torch.softmax(logits, dim=0)
            action = int(torch.multinomial(probs, 1)) if sample else int(torch.argmax(logits))
            logprob = float(torch.log(probs[action] + 1e-12))
        k = obs.select.maxCount
        order = torch.argsort(logits, descending=True).tolist()
        picks = [action] + [i for i in order if i != action]
        return picks[:k], action, logprob, float(value), sc, opts

    # Build each distinct opponent once (cache models / rule instances by spec).
    opp_cache: dict = {}
    def get_opponent(spec):
        if spec not in opp_cache:
            kind = spec[0]
            if kind == "model":
                m = load_model(spec[1])
                fn = lambda od, _m=m: run_model(_m, to_observation_class(od), sample=False)[0]
                opp_cache[spec] = (fn, _deck(spec[2]))
            elif kind == "rule":
                agent = load_teacher(f"opp_{worker_id}_{len(opp_cache)}", agent=spec[1], deck=spec[2])
                opp_cache[spec] = (agent, _deck(spec[2]))
            else:  # random
                fn = lambda od: rng.sample(range(len(od["select"]["option"])), od["select"]["maxCount"])
                opp_cache[spec] = (fn, _deck(spec[1]))
        return opp_cache[spec]

    cols = {k: [] for k in ("states", "options", "n_options", "actions",
                            "logprobs", "values", "rewards", "game_ids", "players")}

    for game in range(n_games):
        spec = specs[rng.choices(range(len(specs)), weights=weights)[0]]
        opp_fn, opp_deck = get_opponent(spec)
        learn_seat = game % 2                          # alternate seats -> slot-fair data
        d0, d1 = (learn_deck, opp_deck) if learn_seat == 0 else (opp_deck, learn_deck)

        obs_dict, start = battle_start(d0, d1)
        if start.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected deck (errorType={start.errorType})")

        rows, prev_delta = [], 0.0
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            if seat == learn_seat:
                obs = to_observation_class(obs_dict)
                picks, action, logprob, value, sc, opts = run_model(learner, obs, sample=True)
                me = obs.current.players[seat]; op = obs.current.players[1 - seat]
                delta = (6 - len(op.prize)) - (6 - len(me.prize))     # prizes I took - they took
                rows.append(dict(state=sc, opts=opts, action=action, logprob=logprob,
                                 value=value, reward=PRIZE_SHAPING * (delta - prev_delta)))
                prev_delta = delta
                obs_dict = battle_select([int(i) for i in picks])
            else:
                obs_dict = battle_select([int(i) for i in opp_fn(obs_dict)])

        result = obs_dict["current"]["result"]         # 0/1 winner, 2 draw
        battle_finish()
        if rows:                                        # terminal reward on learner's last row
            rows[-1]["reward"] += 0.0 if result == 2 else (1.0 if result == learn_seat else -1.0)

        for r in rows:
            cols["states"].append(r["state"]); cols["options"].append(r["opts"])
            cols["n_options"].append(len(r["opts"])); cols["actions"].append(r["action"])
            cols["logprobs"].append(r["logprob"]); cols["values"].append(r["value"])
            cols["rewards"].append(r["reward"]); cols["game_ids"].append(game)
            cols["players"].append(learn_seat)

    out = Path(out_dir) / f"ppo_shard_w{worker_id:02d}.npz"
    np.savez_compressed(
        out,
        states=np.stack(cols["states"]),
        options=np.concatenate(cols["options"]),
        n_options=np.array(cols["n_options"], dtype=np.int32),
        actions=np.array(cols["actions"], dtype=np.int32),
        logprobs=np.array(cols["logprobs"], dtype=np.float32),
        values=np.array(cols["values"], dtype=np.float32),
        rewards=np.array(cols["rewards"], dtype=np.float32),
        game_ids=np.array(cols["game_ids"], dtype=np.int32),
        players=np.array(cols["players"], dtype=np.int32),
    )
    return str(out)


def collect(n_games: int, checkpoint: str, n_workers: int = 4,
            learn_deck: str = LEARN_DECK, pool=None, out_dir: Path = OUT_DIR) -> list[str]:
    """Collect n_games across n_workers, learning policy vs an opponent pool."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs, weights = pool if pool is not None else default_pool(checkpoint, learn_deck)
    per = [n_games // n_workers + (1 if i < n_games % n_workers else 0) for i in range(n_workers)]
    jobs = [(i, per[i], checkpoint, learn_deck, specs, weights, str(out_dir), 1000 + i)
            for i in range(n_workers) if per[i] > 0]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool_:
        return pool_.map(_play_worker, jobs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "bc_v1.pt"))
    args = p.parse_args()

    import time
    t0 = time.time()
    shards = collect(args.games, args.checkpoint, args.workers)
    dt = time.time() - t0
    print(f"{args.games} games in {dt:.0f}s ({3600 * args.games / dt:.0f} games/hr) -> {shards}")
