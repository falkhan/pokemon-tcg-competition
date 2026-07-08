"""Parallel self-play game collection for PPO (M2 Phase 0/1).

The cg engine holds ONE battle per process (cg/sim.py global), so parallelism means
multiprocessing: each worker process loads the policy checkpoint itself, runs games
via the direct engine loop (fast path, same as rl/bc.py collection), and writes its
trajectories to its own shard file. Nothing torch/engine crosses process boundaries —
only file paths and small config tuples, which keeps Windows spawn-pickling happy.

Shard layout extends the BC layout (rl/bc.py) with PPO fields per decision:
  states, options, n_options, game_ids     as in BC shards
  actions   (D,) int32     chosen option index (the FIRST pick, like BC labels)
  logprobs  (D,) float32   log pi(action | state) under the collecting policy
  values    (D,) float32   value-head output at the decision
  rewards   (D,) float32   per-decision shaped reward: 0.1 * prize-delta since the
                           player's previous decision, +/-1 terminal on each game's
                           last decision per player
  players   (D,) int32     which seat made the decision (for GAE per player)

Usage:  python -m rl.collector --games 400 --workers 4 --checkpoint checkpoints/bc_v1.pt
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "ppo"
PRIZE_SHAPING = 0.1


def _play_worker(args: tuple) -> str:
    """Runs in a child process: play n_games of policy self-play, write one shard."""
    worker_id, n_games, checkpoint, deck_path, out_dir, seed = args

    # All heavyweight imports happen inside the worker (fresh engine + torch per process).
    import torch
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    from rl.bc import _load_deck
    from rl.encoders import encode_context, encode_option, encode_state
    from rl.policy import OptionScorer

    torch.manual_seed(seed)
    model = OptionScorer()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    deck = _load_deck() if deck_path is None else \
        [int(x) for x in Path(deck_path).read_text().split() if x.strip()]

    cols = {k: [] for k in ("states", "options", "n_options", "actions",
                            "logprobs", "values", "rewards", "game_ids", "players")}

    for game in range(n_games):
        obs_dict, start = battle_start(deck, deck)
        if start.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected deck (errorType={start.errorType})")

        game_rows = []                    # per-decision dicts, rewards filled as we go
        prev_prizedelta = [0.0, 0.0]      # per seat: (opp prizes taken) - (mine taken) proxy

        while obs_dict["current"]["result"] < 0:
            obs = to_observation_class(obs_dict)
            seat = obs.current.yourIndex
            me = obs.current.players[seat]
            op = obs.current.players[1 - seat]

            state_ctx = np.concatenate([encode_state(obs.current),
                                        encode_context(obs.select.context)])
            opts = np.stack([encode_option(o, obs) for o in obs.select.option])

            with torch.no_grad():
                logits, value = model(torch.from_numpy(state_ctx).unsqueeze(0),
                                      torch.from_numpy(opts).unsqueeze(0))
                logits = logits.squeeze(0)
                probs = torch.softmax(logits, dim=0)
                action = int(torch.multinomial(probs, 1))      # SAMPLE (exploration)
                logprob = float(torch.log(probs[action] + 1e-12))

            # prize-delta shaping: how the prize race moved since this seat's last decision
            prizedelta = (6 - len(op.prize)) - (6 - len(me.prize))  # prizes I took - they took
            shaped = PRIZE_SHAPING * (prizedelta - prev_prizedelta[seat])
            prev_prizedelta[seat] = prizedelta

            game_rows.append(dict(state=state_ctx, opts=opts, action=action,
                                  logprob=logprob, value=float(value), reward=shaped,
                                  seat=seat))

            # engine contract: submit between minCount and maxCount picks; we lead with
            # the sampled action and fill the rest greedily
            k = obs.select.maxCount
            order = torch.argsort(logits, descending=True).tolist()
            picks = [action] + [i for i in order if i != action]
            obs_dict = battle_select([int(i) for i in picks[:k]])

        result = obs_dict["current"]["result"]        # 0/1 winner, 2 draw
        battle_finish()

        # terminal reward lands on each seat's LAST decision
        last_of_seat = {}
        for idx, row in enumerate(game_rows):
            last_of_seat[row["seat"]] = idx
        for seat, idx in last_of_seat.items():
            terminal = 0.0 if result == 2 else (1.0 if result == seat else -1.0)
            game_rows[idx]["reward"] += terminal

        for row in game_rows:
            cols["states"].append(row["state"])
            cols["options"].append(row["opts"])
            cols["n_options"].append(len(row["opts"]))
            cols["actions"].append(row["action"])
            cols["logprobs"].append(row["logprob"])
            cols["values"].append(row["value"])
            cols["rewards"].append(row["reward"])
            cols["game_ids"].append(game)
            cols["players"].append(row["seat"])

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
            deck_path: str | None = None, out_dir: Path = OUT_DIR) -> list[str]:
    """Self-play n_games split across n_workers processes. Returns shard paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per = [n_games // n_workers + (1 if i < n_games % n_workers else 0)
           for i in range(n_workers)]
    jobs = [(i, per[i], checkpoint, deck_path, str(out_dir), 1000 + i)
            for i in range(n_workers) if per[i] > 0]

    ctx = mp.get_context("spawn")                     # Windows default; explicit anyway
    with ctx.Pool(len(jobs)) as pool:
        shards = pool.map(_play_worker, jobs)
    return shards


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
