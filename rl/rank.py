"""M12 value-as-ranker (M9-plan Leg 4 on the M11 stack).

Labels: per-sibling deep scores from the WIDENED search
(`turn_solver.score_siblings`, 2.0s, dev-aware) at own MAIN prompts of
solver-pilot self-play — on-distribution states (the M11 round-1 lesson).
Training: fresh OptionScorerV3-shaped net (plan input zeros), PAIRWISE
LOGISTIC loss over same-prompt option pairs with score gap >= RANK_MARGIN
(skip aliased ties) — the direct fix for the M5/M8.4 "win/loss classifier
can't rank siblings" failure.

Gate 1 (offline, zero games): held-out pairwise accuracy >= 0.70, else kill.
Consumer: matchrunner `rank:<ckpt>:<deck>` — solver pilot + confident net
override on MAIN prompts (see matchrunner).

CLI:
  python -m rl.rank collect --games 400 --out data/rank1
  python -m rl.rank train --data data/rank1 --name osv3_rank1
"""
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cg.api import SelectContext, to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from rl.bc import load_population
from rl.encoders import (N_CONTEXTS, STATE_V2_DIM, encode_context,
                         encode_option_v2, encode_state_v2)
from rl.plan import PLAN_DIM
from rl.policy import OptionScorerV3

ROOT = Path(__file__).resolve().parent.parent

RANK_MARGIN = 200.0        # min deep-score gap for a training/eval pair
WIDE_NODES = 4000
WIDE_DEPTH = 10


def _sibling_scores(obs, deck, deadline_s):
    from rl.turn_solver import score_siblings
    if getattr(obs, "search_begin_input", None) is None:
        return None
    try:
        return score_siblings(obs, deck, deadline_s=deadline_s,
                              max_depth=WIDE_DEPTH, max_nodes=WIDE_NODES)
    except Exception:
        return None


def _collect_chunk(args):
    (lo, hi, decks_file, out_dir, deadline, shard_size, seed, worker) = args
    import random

    from rl.turn_solver import make_solver_pilot

    out = Path(out_dir)
    population = load_population(decks_file)
    rng = random.Random(seed * 20011 + worker)

    columns = ("states", "state_ids", "options", "option_ids", "n_options",
               "sib_scores", "game_ids", "results", "deck_idx")
    shard = {k: [] for k in columns}
    shard_idx = sum(1 for _ in out.glob(f"shard_w{worker:02d}_*.npz"))
    stats = {"games": 0, "rows": 0, "pairs": 0}
    t0 = perf_counter()

    def flush():
        nonlocal shard_idx
        if not shard["states"]:
            return
        np.savez_compressed(
            out / f"shard_w{worker:02d}_{shard_idx:04d}.npz",
            states=np.stack(shard["states"]),
            state_ids=np.stack(shard["state_ids"]),
            options=np.concatenate(shard["options"]),
            option_ids=np.concatenate(shard["option_ids"]),
            n_options=np.array(shard["n_options"], dtype=np.int32),
            sib_scores=np.concatenate(shard["sib_scores"]),
            game_ids=np.array(shard["game_ids"], dtype=np.int32),
            results=np.array(shard["results"], dtype=np.float32),
            deck_idx=np.array(shard["deck_idx"], dtype=np.int32),
        )
        shard_idx += 1
        for v in shard.values():
            v.clear()

    for game in range(lo, hi):
        picks_idx = [rng.randrange(len(population)) for _ in range(2)]
        decks = [population[picks_idx[0]], population[picks_idx[1]]]
        pilots = [make_solver_pilot(d, instance=f"rk{game}_{s}")
                  for s, d in enumerate(decks)]

        obs_dict, start_data = battle_start(decks[0], decks[1])
        if start_data.errorPlayer >= 0:
            raise ValueError("battle_start rejected a deck")

        game_rows = []
        while obs_dict["current"]["result"] < 0:
            player = obs_dict["current"]["yourIndex"]
            obs = to_observation_class(obs_dict)
            n = len(obs.select.option)
            if obs.select.context == SelectContext.MAIN \
                    and obs.select.maxCount == 1 and n >= 2:
                sibs = _sibling_scores(obs, decks[player], deadline)
                if sibs:
                    scores = np.full(n, np.nan, dtype=np.float32)
                    for action, s in sibs:
                        if len(action) == 1:
                            scores[action[0]] = s
                    if np.isfinite(scores).sum() >= 2:
                        state_num, sids = encode_state_v2(obs.current,
                                                          decks[player])
                        sc = np.concatenate(
                            [state_num, encode_context(obs.select.context)]
                        ).astype(np.float32)
                        pairs = [encode_option_v2(o, obs)
                                 for o in obs.select.option]
                        game_rows.append(
                            (sc, sids,
                             np.stack([x for x, _ in pairs]).astype(np.float32),
                             np.stack([i for _, i in pairs]),
                             scores, player))
                        stats["rows"] += 1
                        fin = scores[np.isfinite(scores)]
                        stats["pairs"] += int(
                            (np.abs(fin[:, None] - fin[None, :])
                             >= RANK_MARGIN).sum() // 2)
            picks = pilots[player](obs_dict)
            obs_dict = battle_select([int(i) for i in
                                      picks[:obs.select.maxCount]])

        result = obs_dict["current"]["result"]
        battle_finish()
        stats["games"] += 1
        for sc, sids, opts, oids, scores, player in game_rows:
            shard["states"].append(sc)
            shard["state_ids"].append(sids)
            shard["options"].append(opts)
            shard["option_ids"].append(oids)
            shard["n_options"].append(len(opts))
            shard["sib_scores"].append(scores)
            shard["game_ids"].append(game)
            shard["results"].append(
                0.0 if result == 2 else (1.0 if result == player else -1.0))
            shard["deck_idx"].append(picks_idx[player])

        if (game - lo + 1) % shard_size == 0:
            flush()
        if game - lo + 1 == 25:
            per_game = (perf_counter() - t0) / 25
            print(f"[w{worker}] game 25: {per_game:.1f}s/game -> projected "
                  f"{per_game * (hi - lo) / 60:.0f} min", flush=True)

    flush()
    return stats


def collect(n_games: int, decks_file, out_dir: Path, deadline: float = 2.0,
            workers: int = 12, shard_size: int = 200, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = -(-n_games // workers)
    jobs, lo = [], 0
    for w in range(workers):
        hi = min(lo + chunk, n_games)
        if lo >= hi:
            break
        jobs.append((lo, hi, str(decks_file), str(out_dir), deadline,
                     shard_size, seed, w))
        lo = hi
    if len(jobs) == 1:
        results = [_collect_chunk(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(_collect_chunk, jobs)
    agg = {k: sum(r[k] for r in results) for k in ("games", "rows", "pairs")}
    print(f"done: {agg['games']} games, {agg['rows']} ranked prompts, "
          f"~{agg['pairs']} margin pairs -> {out_dir}", flush=True)
    return agg


# ------------------------------------------------------------------ training

class RankDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path | list):
        cols = {k: [] for k in ("states", "state_ids", "options", "option_ids",
                                "n_options", "sib_scores", "game_ids")}
        starts_list = []
        option_base = game_base = 0
        dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
        for d in dirs:
            for path in sorted(Path(d).glob("*.npz")):
                print(f"loading {path.name}", flush=True)
                shard = np.load(path)
                starts = np.cumsum(shard["n_options"]) - shard["n_options"]
                starts_list.append(starts + option_base)
                cols["game_ids"].append(shard["game_ids"] + game_base)
                for k in cols:
                    if k != "game_ids":
                        cols[k].append(shard[k])
                option_base += len(shard["options"])
                game_base += int(shard["game_ids"].max()) + 1
        for k, v in cols.items():
            setattr(self, k, np.concatenate(v))
        self.starts = np.concatenate(starts_list)

    def __len__(self):
        return len(self.n_options)

    def __getitem__(self, i):
        s, n = self.starts[i], self.n_options[i]
        return (self.states[i], self.state_ids[i], self.options[s:s + n],
                self.option_ids[s:s + n], self.sib_scores[s:s + n])


def collate_rank(batch):
    B = len(batch)
    maxN = max(row[2].shape[0] for row in batch)
    states = torch.zeros(B, STATE_V2_DIM + N_CONTEXTS)
    state_ids = torch.zeros(B, batch[0][1].shape[0], dtype=torch.long)
    options = torch.zeros(B, maxN, batch[0][2].shape[1])
    option_ids = torch.zeros(B, maxN, 2, dtype=torch.long)
    scores = torch.full((B, maxN), float("nan"))
    for i, (sc, sids, menu, oids, sib) in enumerate(batch):
        n = menu.shape[0]
        states[i] = torch.from_numpy(sc)
        state_ids[i] = torch.from_numpy(sids.astype(np.int64))
        options[i, :n] = torch.from_numpy(menu)
        option_ids[i, :n] = torch.from_numpy(oids.astype(np.int64))
        scores[i, :n] = torch.from_numpy(sib)
    return states, state_ids, options, option_ids, scores


def _pair_stats(logits, scores, margin):
    """(correct, total) over pairs with |score gap| >= margin, plus loss."""
    d_score = scores.unsqueeze(2) - scores.unsqueeze(1)       # (B, N, N)
    valid = torch.isfinite(d_score) & (d_score >= margin)
    d_logit = logits.unsqueeze(2) - logits.unsqueeze(1)
    loss = F.softplus(-d_logit[valid]).mean() if valid.any() else None
    correct = int((d_logit[valid] > 0).sum())
    return loss, correct, int(valid.sum())


def train(data_dirs: list, name: str, epochs: int = 6, lr: float = 3e-4,
          batch_size: int = 128, margin: float = RANK_MARGIN) -> float:
    ds = RankDataset([Path(d) for d in data_dirs])
    rng = np.random.default_rng(0)
    unique_games = np.unique(ds.game_ids)
    val_games = set(rng.choice(unique_games,
                               max(1, int(0.1 * len(unique_games))),
                               replace=False).tolist())
    is_val = np.isin(ds.game_ids, list(val_games))
    train_ds = torch.utils.data.Subset(ds, np.nonzero(~is_val)[0])
    val_ds = torch.utils.data.Subset(ds, np.nonzero(is_val)[0])
    print(f"train {len(train_ds)} prompts, val {len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_rank)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_rank)

    model = OptionScorerV3(option_dim=ds.options.shape[1])  # M16: shard width
    zeros = None
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = 0.0

    def evaluate():
        model.eval()
        c = t = 0
        with torch.no_grad():
            for states, sids, options, oids, scores in val_dl:
                z = torch.zeros(states.shape[0], PLAN_DIM)
                logits, _ = model(states, z, sids, options, oids)
                _, ci, ti = _pair_stats(logits, scores, margin)
                c, t = c + ci, t + ti
        return c / max(1, t)

    for epoch in range(epochs):
        model.train()
        running = n_batches = 0
        for states, sids, options, oids, scores in train_dl:
            zeros = torch.zeros(states.shape[0], PLAN_DIM)
            opt.zero_grad()
            logits, _ = model(states, zeros, sids, options, oids)
            loss, _, _ = _pair_stats(logits, scores, margin)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        acc = evaluate()
        print(f"epoch {epoch}: loss {running / max(1, n_batches):.3f}  "
              f"pairwise_acc {acc:.3f}", flush=True)
        if acc > best:
            best = acc
            (ROOT / "checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), ROOT / "checkpoints" / f"{name}.pt")
    print(f"best held-out pairwise_acc {best:.3f}  (Gate 1 bar: 0.70)")
    return best


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--games", type=int, default=400)
    c.add_argument("--decks", type=str, default="data/league/population.json")
    c.add_argument("--out", type=str, required=True)
    c.add_argument("--deadline", type=float, default=2.0)
    c.add_argument("--workers", type=int, default=12)
    c.add_argument("--shard-size", type=int, default=200)
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", type=str, nargs="+", required=True)
    t.add_argument("--name", type=str, required=True)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    if args.cmd == "collect":
        collect(args.games, args.decks, Path(args.out),
                deadline=args.deadline, workers=args.workers,
                shard_size=args.shard_size, seed=args.seed)
    else:
        train(args.data, args.name, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size)
