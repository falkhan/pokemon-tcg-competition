"""M13 Rung 0: outcome-grounded SETUP VALUE (spec: docs/M13-plan.md).

Collection: solver-pilot self-play at LIVE budgets (fast), logging ONE state
per own turn (the first own MAIN prompt) — the setup snapshot. Dated spec
note 2026-07-17: turn-level states instead of every decision (less
within-turn correlation, 5-10x smaller shards).

Training: matched-pair ranking — pairs drawn from DIFFERENT games matched on
(turn bucket, my prizes, opp prizes, own deck); one state from an
eventually-WON game vs one from a LOST game; pairwise logistic loss (M12's
proven fix — never a win/loss classifier, never score_leaf-derived labels).

Gate V0 (offline, zero games): held-out matched-pair accuracy >= 0.62
(chance 0.5 by construction); kill < 0.58 after one iteration.

CLI:
  python -m rl.setup_value collect --games 2000 --out data/setupval
  python -m rl.setup_value train --data data/setupval --name osv3_setupval1 \
      --init-v3 checkpoints/osv3_plan0c.pt
"""
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from cg.api import SelectContext, to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from rl.bc import load_population
from rl.encoders import (encode_context, encode_state_v3, OPTION_V2_DIM,
                         STATE_V2_DIM, N_CONTEXTS, DECK_LOW_AT)
from rl.plan import PLAN_DIM
from rl.policy import OptionScorerV3, option_dim_of

ROOT = Path(__file__).resolve().parent.parent

TURN_BUCKET = 4
MIN_TURN = 3               # openers are aliased — excluded from pairs
PAIR_CAP_PER_BUCKET = 400
GATE_V0 = 0.62

# M33 Stage 1: late-game features for the value trunk (appended via the net's
# extra_dim slot, so state_ctx = [base_state | context | LATE]). These attack the
# documented long-game blindness: turn/30 saturates past turn 30, the value net
# never sees opponent deck count (v4-only), and there is no deck-out clock — so
# the value is near-chance past turn ~32 (VS_MAX_TURN). Kept small + interpretable
# and defined here (not encoders.py) so the SHIPPED encoders stay byte-identical.
LATE_DIM = 5
BASE_STATE_W = STATE_V2_DIM + N_CONTEXTS   # width of `states` when extra_dim=0
LATE_BUCKET = 4            # turn//TURN_BUCKET >= this == "late" (turn >= 16)


def late_game_feats(me, op, turn) -> np.ndarray:
    """5 late-game scalars: opponent deck count (absent in the v3 base state),
    both sides' deck-out-proximity ramps (mirror encoders.DECK_LOW_AT), a
    de-saturated turn (base turn/30 pegs at 1.0 past turn 30), and a late-phase
    flag at the turn-16 win-rate cliff (docs/m31-post-mortem.md)."""
    def low_ramp(dc):
        return max(0.0, min(1.0, (DECK_LOW_AT - dc) / DECK_LOW_AT))
    return np.array([
        getattr(op, "deckCount", 60) / 60.0,
        low_ramp(getattr(me, "deckCount", 60)),
        low_ramp(getattr(op, "deckCount", 60)),
        min(int(turn), 60) / 60.0,
        1.0 if int(turn) >= 16 else 0.0,
    ], dtype=np.float32)


# ---------------------------------------------------------------- collection

def _collect_chunk(args):
    (lo, hi, decks_file, out_dir, shard_size, seed, worker) = args
    import random

    from rl.turn_solver import make_solver_pilot

    out = Path(out_dir)
    population = load_population(decks_file)
    rng = random.Random(seed * 31013 + worker)

    columns = ("states", "state_ids", "turn", "my_prizes", "opp_prizes",
               "game_ids", "results", "deck_idx", "my_deck", "opp_deck")
    shard = {k: [] for k in columns}
    shard_idx = sum(1 for _ in out.glob(f"shard_w{worker:02d}_*.npz"))
    stats = {"games": 0, "rows": 0}
    t0 = perf_counter()

    def flush():
        nonlocal shard_idx
        if not shard["states"]:
            return
        np.savez_compressed(
            out / f"shard_w{worker:02d}_{shard_idx:04d}.npz",
            states=np.stack(shard["states"]),
            state_ids=np.stack(shard["state_ids"]),
            turn=np.array(shard["turn"], dtype=np.int32),
            my_prizes=np.array(shard["my_prizes"], dtype=np.int32),
            opp_prizes=np.array(shard["opp_prizes"], dtype=np.int32),
            game_ids=np.array(shard["game_ids"], dtype=np.int32),
            results=np.array(shard["results"], dtype=np.float32),
            deck_idx=np.array(shard["deck_idx"], dtype=np.int32),
            my_deck=np.array(shard["my_deck"], dtype=np.int32),
            opp_deck=np.array(shard["opp_deck"], dtype=np.int32),
        )
        shard_idx += 1
        for v in shard.values():
            v.clear()

    try:
        _run_games(lo, hi, population, rng, shard, shard_size, out, worker,
                   stats, flush, t0)
    except KeyboardInterrupt:
        flush()                      # Ctrl-C: keep every completed game
    flush()
    return stats


def _run_games(lo, hi, population, rng, shard, shard_size, out, worker,
               stats, flush, t0):
    from rl.turn_solver import make_solver_pilot

    stop_file = out / ".stop"
    for game in range(lo, hi):
        if stop_file.exists():       # graceful drain (wrapper-script Ctrl-C)
            break
        picks_idx = [rng.randrange(len(population)) for _ in range(2)]
        decks = [population[picks_idx[0]], population[picks_idx[1]]]
        pilots = [make_solver_pilot(d, instance=f"sv{game}_{s}")
                  for s, d in enumerate(decks)]

        obs_dict, start_data = battle_start(decks[0], decks[1])
        if start_data.errorPlayer >= 0:
            raise ValueError("battle_start rejected a deck")

        seen_keys = set()
        game_rows = []
        while obs_dict["current"]["result"] < 0:
            player = obs_dict["current"]["yourIndex"]
            obs = to_observation_class(obs_dict)
            key = (obs.current.turn, player)
            if obs.select.context == SelectContext.MAIN \
                    and key not in seen_keys:
                seen_keys.add(key)
                me = obs.current.players[player]
                op = obs.current.players[1 - player]
                num, sids = encode_state_v3(obs.current, decks[player])
                late = late_game_feats(me, op, obs.current.turn)
                sc = np.concatenate(
                    [num, encode_context(obs.select.context), late]
                ).astype(np.float32)
                game_rows.append((sc, sids, obs.current.turn, len(me.prize),
                                  len(op.prize), player,
                                  getattr(me, "deckCount", 60),
                                  getattr(op, "deckCount", 60)))
                stats["rows"] += 1
            picks = pilots[player](obs_dict)
            obs_dict = battle_select([int(i) for i in
                                      picks[:obs.select.maxCount]])

        result = obs_dict["current"]["result"]
        battle_finish()
        stats["games"] += 1
        for sc, sids, turn, myp, opp, player, mydeck, opdeck in game_rows:
            shard["states"].append(sc)
            shard["state_ids"].append(sids)
            shard["turn"].append(turn)
            shard["my_prizes"].append(myp)
            shard["opp_prizes"].append(opp)
            shard["game_ids"].append(game)
            shard["results"].append(
                0.0 if result == 2 else (1.0 if result == player else -1.0))
            shard["deck_idx"].append(picks_idx[player])
            shard["my_deck"].append(mydeck)
            shard["opp_deck"].append(opdeck)

        if (game - lo + 1) % shard_size == 0:
            flush()
        # progress beacon for wrapper UIs: one tiny file per
        # worker, overwritten per game — cheap, crash-safe, poll-friendly.
        (out / f".progress_w{worker:02d}").write_text(str(game - lo + 1))
        if game - lo + 1 == 25:
            per_game = (perf_counter() - t0) / 25
            print(f"[w{worker}] game 25: {per_game:.1f}s/game -> projected "
                  f"{per_game * (hi - lo) / 60:.0f} min", flush=True)


def collect(n_games: int, decks_file, out_dir: Path, workers: int = 12,
            shard_size: int = 500, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".stop").unlink(missing_ok=True)   # stale sentinel from a prior run
    chunk = -(-n_games // workers)
    jobs, lo = [], 0
    for w in range(workers):
        hi = min(lo + chunk, n_games)
        if lo >= hi:
            break
        jobs.append((lo, hi, str(decks_file), str(out_dir), shard_size,
                     seed, w))
        lo = hi
    if len(jobs) == 1:
        results = [_collect_chunk(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(_collect_chunk, jobs)
    agg = {k: sum(r[k] for r in results) for k in ("games", "rows")}
    print(f"done: {agg['games']} games, {agg['rows']} setup states -> "
          f"{out_dir}", flush=True)
    return agg


# ------------------------------------------------------------------ training

def _load_rows(data_dirs):
    base_keys = ("states", "state_ids", "turn", "my_prizes",
                 "opp_prizes", "game_ids", "results", "deck_idx")
    opt_keys = ("my_deck", "opp_deck")   # M33: present only in fresh corpora
    cols = {k: [] for k in base_keys}
    game_base = 0
    for d in data_dirs:
        for path in sorted(Path(d).glob("*.npz")):
            shard = np.load(path)
            for k in opt_keys:            # add optional cols on first sight
                if k in shard.files and k not in cols:
                    cols[k] = []
            cols["game_ids"].append(shard["game_ids"] + game_base)
            # (state_ids widths may mix 12/20 across batches — padded below)
            for k in cols:
                if k != "game_ids":
                    cols[k].append(shard[k])
            game_base += int(shard["game_ids"].max()) + 1
    widths = {a.shape[1] for a in cols["state_ids"]}
    if len(widths) > 1:                        # M15 pad shim: 12 -> 20 ids
        wmax = max(widths)
        cols["state_ids"] = [np.pad(a, ((0, 0), (0, wmax - a.shape[1])))
                             for a in cols["state_ids"]]
    return {k: np.concatenate(v) for k, v in cols.items()}


def build_pairs(rows: dict, rng, exclude_games=None, cap=PAIR_CAP_PER_BUCKET,
                coarse_late=False, late_cap=None):
    """Matched pairs (win_row_idx, loss_row_idx) from DIFFERENT games with
    equal (turn bucket, my_prizes, opp_prizes, deck_idx). Returns index pairs
    plus each pair's turn bucket (for per-bucket reporting).

    M33 Stage 1 (coarse_late): in late buckets (tb >= LATE_BUCKET) DROP deck_idx
    from the match key and use `late_cap`. Prizes stay matched (no prize-count
    leak), but merging across decks lifts the ~3.6 pairs/key late-bucket sparsity
    that leaves the late value near-chance (docs/M33-plan.md)."""
    ok = (rows["turn"] >= MIN_TURN) & (rows["results"] != 0.0)
    if exclude_games is not None:
        ok &= ~np.isin(rows["game_ids"], exclude_games)
    idxs = np.nonzero(ok)[0]
    buckets: dict = {}
    for i in idxs:
        tb = int(rows["turn"][i]) // TURN_BUCKET
        if coarse_late and tb >= LATE_BUCKET:
            key = (tb, int(rows["my_prizes"][i]), int(rows["opp_prizes"][i]))
        else:
            key = (tb, int(rows["my_prizes"][i]), int(rows["opp_prizes"][i]),
                   int(rows["deck_idx"][i]))
        buckets.setdefault(key, ([], []))[0 if rows["results"][i] > 0
                                          else 1].append(int(i))
    pairs, pair_bucket = [], []
    for key, (wins, losses) in buckets.items():
        tb = key[0]
        this_cap = (late_cap or cap) if (coarse_late and tb >= LATE_BUCKET) \
            else cap
        rng.shuffle(wins)
        rng.shuffle(losses)
        n = min(len(wins), len(losses), this_cap)
        for k in range(n):
            if rows["game_ids"][wins[k]] == rows["game_ids"][losses[k]]:
                continue                      # different games only
            pairs.append((wins[k], losses[k]))
            pair_bucket.append(tb)
    return pairs, pair_bucket


def _values(model, rows, idx_list, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx_list), batch):
            sel = idx_list[i:i + batch]
            sc = torch.from_numpy(rows["states"][sel])
            sids = torch.from_numpy(
                rows["state_ids"][sel].astype(np.int64))
            zeros = torch.zeros(len(sel), PLAN_DIM)
            se = model.embedding(sids).flatten(-2)
            s = model.state_enc(torch.cat([sc, zeros, se], dim=-1))
            out.append(model.value_head(s).squeeze(-1))
    return torch.cat(out)


def train(data_dirs: list, name: str, init_v3: str | None = None,
          epochs: int = 6, lr: float = 1e-4, batch_size: int = 256,
          seed: int = 0, late: bool = False, coarse_pairs: bool = False) -> float:
    rows = _load_rows([Path(d) for d in data_dirs])
    print(f"{len(rows['states'])} states, "
          f"{int((rows['results'] > 0).sum())} from wins")
    # M33 Stage 1: a fresh corpus stores state_ctx = [base | context | LATE].
    # `late` feeds the LATE block via the net's extra_dim; baseline slices it off
    # so the A/B differs only in the added features (+ pair construction).
    have_w = rows["states"].shape[1]
    if late:
        if have_w < BASE_STATE_W + LATE_DIM:
            raise ValueError(f"--late needs a LATE-feat corpus (state width "
                             f"{have_w} < {BASE_STATE_W + LATE_DIM})")
        rows["states"] = rows["states"][:, :BASE_STATE_W + LATE_DIM]
        extra_dim = LATE_DIM
    else:
        rows["states"] = rows["states"][:, :BASE_STATE_W]
        extra_dim = 0
    if init_v3 is not None and extra_dim:
        raise ValueError("warm-start (--init-v3) with --late is unsupported "
                         "(state_enc width differs); run --late from scratch")
    print(f"recipe: late={late} coarse_pairs={coarse_pairs} "
          f"extra_dim={extra_dim} state_w={rows['states'].shape[1]}")

    rng = np.random.default_rng(seed)
    unique_games = np.unique(rows["game_ids"])
    val_games = rng.choice(unique_games, max(1, int(0.1 * len(unique_games))),
                           replace=False)
    py_rng = __import__("random").Random(seed)
    train_pairs, _ = build_pairs(rows, py_rng, exclude_games=val_games,
                                 coarse_late=coarse_pairs, late_cap=1200)
    # val pairs come ONLY from val games (no leakage through shared games)
    ok_train_games = np.setdiff1d(unique_games, val_games)
    val_pairs, val_buckets = build_pairs(rows, py_rng,
                                         exclude_games=ok_train_games,
                                         coarse_late=coarse_pairs, late_cap=1200)
    print(f"train pairs {len(train_pairs)}, val pairs {len(val_pairs)}")

    # Setup-value shards carry no `options` column, so the option width must
    # come from the init checkpoint (as every other loader does). This retrain
    # was skipped in M15/M16, so the M16 `rows["options"]` line was never run.
    if init_v3 is not None:
        p = Path(init_v3)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        init_sd = torch.load(p, map_location="cpu")
        option_dim = option_dim_of(init_sd)
        print(f"warm-start trunk from {init_v3} (option_dim={option_dim})")
    else:
        init_sd = None
        option_dim = OPTION_V2_DIM
    model = OptionScorerV3(n_state_ids=rows["state_ids"].shape[1],
                           option_dim=option_dim, extra_dim=extra_dim)
    if init_sd is not None:
        model.load_state_dict(init_sd)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = 0.0

    def evaluate():
        model.eval()
        w_idx = [p[0] for p in val_pairs]
        l_idx = [p[1] for p in val_pairs]
        vw = _values(model, rows, w_idx)
        vl = _values(model, rows, l_idx)
        hit = (vw > vl).numpy()
        by_bucket = {}
        for h, bkt in zip(hit, val_buckets):
            by_bucket.setdefault(bkt, []).append(h)
        detail = " ".join(f"t{b * TURN_BUCKET}-{(b + 1) * TURN_BUCKET - 1}:"
                          f"{np.mean(v):.2f}(n={len(v)})"
                          for b, v in sorted(by_bucket.items()))
        return float(hit.mean()), detail

    for epoch in range(epochs):
        model.train()
        py_rng.shuffle(train_pairs)
        running = n_batches = 0
        for i in range(0, len(train_pairs), batch_size):
            chunk = train_pairs[i:i + batch_size]
            w_idx = [p[0] for p in chunk]
            l_idx = [p[1] for p in chunk]
            sc = torch.from_numpy(
                np.concatenate([rows["states"][w_idx],
                                rows["states"][l_idx]]))
            sids = torch.from_numpy(np.concatenate(
                [rows["state_ids"][w_idx],
                 rows["state_ids"][l_idx]]).astype(np.int64))
            zeros = torch.zeros(len(chunk) * 2, PLAN_DIM)
            opt.zero_grad()
            se = model.embedding(sids).flatten(-2)
            s = model.state_enc(torch.cat([sc, zeros, se], dim=-1))
            v = model.value_head(s).squeeze(-1)
            vw, vl = v[:len(chunk)], v[len(chunk):]
            loss = F.softplus(vl - vw).mean()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        acc, detail = evaluate()
        print(f"epoch {epoch}: loss {running / max(1, n_batches):.3f}  "
              f"pair_acc {acc:.3f}  [{detail}]", flush=True)
        if acc > best:
            best = acc
            (ROOT / "checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(),
                       ROOT / "checkpoints" / f"{name}.pt")
    print(f"best held-out matched-pair acc {best:.3f}  "
          f"(Gate V0 bar: {GATE_V0})")
    return best


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--games", type=int, default=2000)
    c.add_argument("--decks", type=str, default="data/league/population.json")
    c.add_argument("--out", type=str, required=True)
    c.add_argument("--workers", type=int, default=8)   # CLAUDE.md cap (12 deadlocks)
    c.add_argument("--shard-size", type=int, default=500)
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", type=str, nargs="+", required=True)
    t.add_argument("--name", type=str, required=True)
    t.add_argument("--init-v3", type=str, default=None)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--late", action="store_true",
                   help="M33: feed the LATE-game feature block (extra_dim)")
    t.add_argument("--coarse-pairs", action="store_true",
                   help="M33: coarsen late-bucket pair matching (drop deck_idx)")
    args = p.parse_args()
    if args.cmd == "collect":
        collect(args.games, args.decks, Path(args.out), workers=args.workers,
                shard_size=args.shard_size, seed=args.seed)
    else:
        train(args.data, args.name, init_v3=args.init_v3, epochs=args.epochs,
              lr=args.lr, batch_size=args.batch_size, seed=args.seed,
              late=args.late, coarse_pairs=args.coarse_pairs)
