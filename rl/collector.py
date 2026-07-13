"""Parallel self-play collection for PPO with an OPPONENT POOL (M2 Path A).

Why a pool: pure mirror self-play only teaches the agent to beat copies of itself, but
the leaderboard is full of other decks/strategies. So the LEARNING policy plays one seat
(sampled actions, recorded), and the OTHER seat is sampled per game from a pool of
opponents: the current policy (mirror), the rule experts, the GENERIC pilot on the
known decks (M7.4b), past promoted selves, and random. Only the learning seat's
decisions are recorded — PPO is on-policy, so we can only update on actions taken by
the current policy with known logprobs.

The cg engine holds ONE battle per process (cg/sim.py global) -> multiprocessing, one
engine per worker. Everything heavy is imported/loaded inside the worker; only picklable
config (paths, deck names/id-lists, spec tuples) crosses the process boundary.

Opponents are built through rl/matchrunner.make_pilot (the canonical OpponentSpec
home), so every spec kind it supports works here: ("model", ckpt, deck) incl. v2 and
pre-M3 checkpoints, ("rule", agent, deck), ("generic", deck), ("random", deck).

M7.4b additions:
- per-game LEARNER deck sampling from a population file (multi-deck self-play — the
  field-diversity fix; REQUIRED for encoders-v2 checkpoints, whose deck-context
  features need the deck list);
- encoders-v2 checkpoints (embedding.weight in the state dict) are auto-detected:
  decisions are encoded with encode_state_v2/encode_option_v2 and shards gain
  state_ids / option_ids / deck_idx columns (rl/ppo.py trains OptionScorerV2 on them);
- optional potential-based race shaping (docs/M7-plan.md §3b): reward +=
  coef * (phi_t - phi_{t-1}) with phi = the race-delta feature, so SETUP actions get
  credit without corrupting the optimal policy (the shaping telescopes out of returns).

Shard layout = BC fields + PPO fields (actions/logprobs/values/rewards/players
[+ state_ids/option_ids/deck_idx for v2]); see rl/ppo.py load_shards. `players` is
always the learning seat for that game.

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
    """Build (specs, weights): current self (mirror) + rule experts + the generic
    pilot on the known decks (M7.4b field diversity) + the LIVE solver ship agent
    + random + recent selves.

    M7.5 attempt-2 rebalance: attempt 1 spent 50% of games on mirror+random,
    optimized the mirror, and REGRESSED vs every rules pilot (G3 0.44->0.145 —
    docs/M7.md 2026-07-12). You become what you train against: the pool now
    majority-weights the rule-based opponents, with the ship agent heaviest."""
    specs = [("model", str(checkpoint), learn_deck),   # mirror vs current policy
             ("rule", "lucario", "lucario"),           # strong fighting deck
             ("rule", "iono", "iono"),                 # lightning deck
             ("generic", "lucario"),                   # deck-agnostic pilot, strong deck
             ("generic", "iono"),                      # ... and the lightning deck
             ("solver", "lucario"),                    # the LIVE ship agent — the bar
             ("random", "kyogre")]                     # weak/varied baseline
    weights = [0.2, 0.15, 0.1, 0.15, 0.1, 0.25, 0.05]
    past = sorted((ROOT / "checkpoints").glob("ppo_it*.pt"))[-3:]   # recent promoted selves
    for p in past:
        specs.append(("model", str(p), learn_deck))
        weights.append(0.1 / len(past))
    w = np.array(weights, dtype=np.float64)
    return specs, (w / w.sum()).tolist()


def _dev_potential(obs) -> float:
    """M8.3 leg-B shaping potential: the M8.1 dev-tier terms as a bounded
    state potential (the killed tier's leaf math migrates here per plan —
    docs/M8.md 2026-07-13). phi in ~[0, 1.65]; taxonomy weights: race progress
    dominates, then attack-readiness, evolution, bench insurance."""
    from rl.turn_solver import _dev_facts
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active and op.active[0] is not None else None
    race, ready, bench, evos, _hand = _dev_facts(me, op_active)
    return ((10.0 - race) / 10.0
            + 0.3 * min(ready, 1)
            + 0.2 * min(evos, 2) / 2.0
            + 0.15 * min(bench, 3) / 3.0)


def _play_worker(args: tuple) -> str:
    (worker_id, n_games, checkpoint, learn_deck_name, learn_decks, specs, weights,
     out_dir, seed, race_shaping, shaping) = args

    import random
    import torch
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish
    from rl.encoders import (_race_features, encode_context, encode_option,
                             encode_option_v2, encode_state, encode_state_v2)
    from rl.matchrunner import make_pilot
    from rl.policy import OptionScorer, OptionScorerV2

    rng = random.Random(seed)
    torch.manual_seed(seed)

    sd = torch.load(checkpoint, map_location="cpu")
    is_v2 = "embedding.weight" in sd
    if is_v2 and not learn_decks:
        raise ValueError("encoders-v2 checkpoints need a deck population "
                         "(collect(..., decks_file=...)): the deck-context "
                         "features require the learner's deck list")
    learner = OptionScorerV2() if is_v2 else OptionScorer()
    learner.load_state_dict(sd)
    learner.eval()
    fixed_deck = _deck(learn_deck_name)

    def run_learner(obs, deck):
        """Sampled decision: (picks, action, logprob, value, sc, sids, opts, oids)."""
        if is_v2:
            num, sids = encode_state_v2(obs.current, deck)
            sc = np.concatenate([num, encode_context(obs.select.context)]).astype(np.float32)
            pairs = [encode_option_v2(o, obs) for o in obs.select.option]
            opts = np.stack([p[0] for p in pairs]).astype(np.float32)
            oids = np.stack([p[1] for p in pairs])
            with torch.no_grad():
                logits, value = learner(
                    torch.from_numpy(sc).unsqueeze(0),
                    torch.from_numpy(sids.astype(np.int64)).unsqueeze(0),
                    torch.from_numpy(opts).unsqueeze(0),
                    torch.from_numpy(oids.astype(np.int64)).unsqueeze(0))
        else:
            sids = oids = None
            sc = np.concatenate([encode_state(obs.current),
                                 encode_context(obs.select.context)]).astype(np.float32)
            opts = np.stack([encode_option(o, obs) for o in obs.select.option]).astype(np.float32)
            with torch.no_grad():
                logits, value = learner(torch.from_numpy(sc).unsqueeze(0),
                                        torch.from_numpy(opts).unsqueeze(0))
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=0)
        action = int(torch.multinomial(probs, 1))
        logprob = float(torch.log(probs[action] + 1e-12))
        order = torch.argsort(logits, descending=True).tolist()
        picks = ([action] + [i for i in order if i != action])[:obs.select.maxCount]
        return picks, action, logprob, float(value), sc, sids, opts, oids

    # Build each distinct opponent once (matchrunner handles every spec kind,
    # incl. generic pilots and v2 / pre-M3 model checkpoints).
    opp_cache: dict = {}

    def get_opponent(spec):
        if spec not in opp_cache:
            opp_cache[spec] = make_pilot(spec, instance=f"cw{worker_id}_{len(opp_cache)}")
        return opp_cache[spec]

    col_names = ["states", "options", "n_options", "actions",
                 "logprobs", "values", "rewards", "game_ids", "players"]
    if is_v2:
        col_names += ["state_ids", "option_ids"]
    if learn_decks:
        col_names += ["deck_idx"]
    cols: dict[str, list] = {k: [] for k in col_names}

    for game in range(n_games):
        if learn_decks:
            deck_idx = rng.randrange(len(learn_decks))
            learn_deck = learn_decks[deck_idx]
        else:
            deck_idx, learn_deck = -1, fixed_deck
        spec = specs[rng.choices(range(len(specs)), weights=weights)[0]]
        opp_fn, opp_deck = get_opponent(spec)
        learn_seat = game % 2                          # alternate seats -> slot-fair data
        d0, d1 = (learn_deck, opp_deck) if learn_seat == 0 else (opp_deck, learn_deck)

        obs_dict, start = battle_start(d0, d1)
        if start.errorPlayer >= 0:
            raise ValueError(f"battle_start rejected deck (errorType={start.errorType})")

        rows, prev_delta, prev_phi = [], 0.0, None
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            if seat == learn_seat:
                obs = to_observation_class(obs_dict)
                picks, action, logprob, value, sc, sids, opts, oids = \
                    run_learner(obs, learn_deck)
                me = obs.current.players[seat]; op = obs.current.players[1 - seat]
                delta = (6 - len(op.prize)) - (6 - len(me.prize))     # prizes I took - they took
                reward = PRIZE_SHAPING * (delta - prev_delta)
                prev_delta = delta
                if race_shaping:
                    # Potential-based setup shaping (M7-plan §3b): phi = the race
                    # delta; F = coef*(phi' - phi) telescopes out of the return,
                    # so it credits board development without changing the
                    # optimal policy.
                    phi = (_dev_potential(obs) if shaping == "dev"
                           else float(_race_features(obs.current)[7]))
                    if prev_phi is not None:
                        reward += race_shaping * (phi - prev_phi)
                    prev_phi = phi
                rows.append(dict(state=sc, state_ids=sids, opts=opts, option_ids=oids,
                                 action=action, logprob=logprob, value=value,
                                 reward=reward))
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
            if is_v2:
                cols["state_ids"].append(r["state_ids"])
                cols["option_ids"].append(r["option_ids"])
            if learn_decks:
                cols["deck_idx"].append(deck_idx)

    arrays = dict(
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
    if is_v2:
        arrays["state_ids"] = np.stack(cols["state_ids"])
        arrays["option_ids"] = np.concatenate(cols["option_ids"])
    if learn_decks:
        arrays["deck_idx"] = np.array(cols["deck_idx"], dtype=np.int32)
    out = Path(out_dir) / f"ppo_shard_w{worker_id:02d}.npz"
    np.savez_compressed(out, **arrays)
    return str(out)


def _load_population(decks_file) -> list[list[int]]:
    import json
    from rl.matchrunner import resolve_deck
    return [resolve_deck(d) for d in json.loads(Path(decks_file).read_text())["decks"]]


def collect(n_games: int, checkpoint: str, n_workers: int = 4,
            learn_deck: str = LEARN_DECK, pool=None, out_dir: Path = OUT_DIR,
            decks_file=None, race_shaping: float = 0.0,
            shaping: str = "race") -> list[str]:
    """Collect n_games across n_workers, learning policy vs an opponent pool.

    decks_file: population.json — the learner samples a deck per game from it
    (multi-deck self-play; required for encoders-v2 checkpoints). race_shaping:
    coefficient of the potential-based setup-shaping term (0 = off).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    specs, weights = pool if pool is not None else default_pool(checkpoint, learn_deck)
    learn_decks = _load_population(decks_file) if decks_file else None
    per = [n_games // n_workers + (1 if i < n_games % n_workers else 0) for i in range(n_workers)]
    jobs = [(i, per[i], checkpoint, learn_deck, learn_decks, specs, weights,
             str(out_dir), 1000 + i, race_shaping, shaping)
            for i in range(n_workers) if per[i] > 0]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool_:
        return pool_.map(_play_worker, jobs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "bc_v1.pt"))
    p.add_argument("--decks", type=str, default=None,
                   help="population.json for per-game learner deck sampling")
    p.add_argument("--race-shaping", type=float, default=0.0)
    args = p.parse_args()

    import time
    t0 = time.time()
    shards = collect(args.games, args.checkpoint, args.workers,
                     decks_file=args.decks, race_shaping=args.race_shaping)
    dt = time.time() - t0
    print(f"{args.games} games in {dt:.0f}s ({3600 * args.games / dt:.0f} games/hr) -> {shards}")
