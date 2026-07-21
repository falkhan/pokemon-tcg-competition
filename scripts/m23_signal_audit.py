"""M23 signal audit probes (docs/m23-signal-audit-plan.md).

Read-only diagnostics — no training, no shard mutation. Subcommands:

  s1  shadow-eval agreement by option type over the clone corpus
      (data/bc_clone_54618168), one or more checkpoints. v4 checkpoints get
      the v2-encoded states zero-padded (v4 block + missing ids = the "no
      history" defaults); read the WITHIN-checkpoint class contrast, not
      absolute agreement.
  s2  supporter(t2-t4)<->win correlation over self-play shards (data/ppo).
  s3  behavior-policy sampled action-type frequency in candidate states.
  s5  value-head calibration: corr(V(s), outcome) by turn bucket.
  s7  option-head entropy by turn bucket (exact collection-time logits:
      forward with the stored plans column and the collecting checkpoint).

Usage:
  uv run python scripts/m23_signal_audit.py s1 --ckpt checkpoints/ppo_current_m22cRL.pt ...
  uv run python scripts/m23_signal_audit.py s2|s3|s5 [--data data/ppo]
  uv run python scripts/m23_signal_audit.py s7 --ckpt checkpoints/ppo_current_m23p1.pt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.encoders import (N_CONTEXTS, N_OPTION_TYPES, STATE_V2_DIM,  # noqa: E402
                         V4_EXTRA_DIM)
from rl.plan import PLAN_DIM  # noqa: E402
from rl.plan_iter import _n_ids_of  # noqa: E402
from rl.policy import (OptionScorerV2, OptionScorerV3,  # noqa: E402
                       option_dim_of)

OT_NAMES = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD",
            5: "ENERGY_CARD", 6: "ENERGY", 7: "PLAY", 8: "ATTACH",
            9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT",
            13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION"}
OT_PLAY = 7
TURN_BUCKETS = [(1, 2), (3, 4), (5, 6), (7, 10), (11, 99)]


def _supporter_ids() -> set[int]:
    import polars as pl
    df = pl.read_parquet(ROOT / "data/cards_features.parquet")
    return set(df.filter(pl.col("is_supporter") == 1)["card_id"].to_list())


def load_ckpt(path: Path):
    sd = torch.load(path, map_location="cpu")
    if "plan_enc.0.weight" in sd:
        extra = V4_EXTRA_DIM if "enc_ver" in sd else 0
        model = OptionScorerV3(n_state_ids=_n_ids_of(sd),
                               option_dim=option_dim_of(sd), extra_dim=extra)
    elif "embedding.weight" in sd:
        model = OptionScorerV2(option_dim=option_dim_of(sd))
        # V2 stores no dim attributes; mirror V3's so _forward_shard is uniform
        model.n_state_ids, model.option_dim = 12, option_dim_of(sd)
    else:
        raise ValueError(f"{path}: not a v2/v3/v4 checkpoint")
    model.load_state_dict(sd)
    model.eval()
    return model


def _load_shards(data_dir: Path, pattern: str = "*.npz") -> list[dict]:
    shards = [dict(np.load(p)) for p in sorted(data_dir.glob(pattern))]
    if not shards:
        raise SystemExit(f"no shards under {data_dir}")
    return shards


def _option_class(opt_row: np.ndarray, card_id: int,
                  supporters: set[int]) -> str:
    ot = int(opt_row[:N_OPTION_TYPES].argmax())
    if ot == OT_PLAY:
        return "PLAY_supporter" if card_id in supporters else "PLAY_other"
    return OT_NAMES.get(ot, f"OT{ot}")


def _turn_of(states: np.ndarray) -> np.ndarray:
    return np.rint(states[:, 0] * 30).astype(int)


def _bucket(turn: int) -> str:
    for lo, hi in TURN_BUCKETS:
        if lo <= turn <= hi:
            return f"t{lo}-{hi}" if lo != hi else f"t{lo}"
    return "t?"


def _forward_shard(model, shard: dict, batch_size: int = 512):
    """Yield (row_slice, masked logits (B, maxN) tensor) over one shard.

    Pads states / state_ids / options up to the model's widths (zero = the
    "no info" default in every encoder shim); plans come from the shard when
    present, else zeros (the V3(plan=0)==V2 fallback path).
    """
    n = len(shard["labels"] if "labels" in shard else shard["actions"])
    n_opts = shard["n_options"].astype(np.int64)
    starts = np.cumsum(n_opts) - n_opts
    states = shard["states"].astype(np.float32)
    is_v3 = isinstance(model, OptionScorerV3)
    want_state = STATE_V2_DIM + N_CONTEXTS + (model.extra_dim if is_v3 else 0)
    if states.shape[1] < want_state:
        states = np.pad(states, ((0, 0), (0, want_state - states.shape[1])))
    sids = shard["state_ids"].astype(np.int64)
    if sids.shape[1] < model.n_state_ids:
        sids = np.pad(sids, ((0, 0), (0, model.n_state_ids - sids.shape[1])))
    opts = shard["options"].astype(np.float32)
    if opts.shape[1] < model.option_dim:
        opts = np.pad(opts, ((0, 0), (0, model.option_dim - opts.shape[1])))
    oids = shard["option_ids"].astype(np.int64)
    plans = (shard["plans"] if "plans" in shard
             else np.zeros((n, PLAN_DIM))).astype(np.float32)

    with torch.no_grad():
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            maxN = int(n_opts[lo:hi].max())
            menu = torch.zeros(hi - lo, maxN, opts.shape[1])
            menu_ids = torch.zeros(hi - lo, maxN, 2, dtype=torch.long)
            valid = torch.zeros(hi - lo, maxN, dtype=torch.bool)
            for i in range(hi - lo):
                s, m = starts[lo + i], n_opts[lo + i]
                menu[i, :m] = torch.from_numpy(opts[s:s + m])
                menu_ids[i, :m] = torch.from_numpy(oids[s:s + m])
                valid[i, :m] = True
            args = (torch.from_numpy(states[lo:hi]),)
            if is_v3:
                args += (torch.from_numpy(plans[lo:hi]),)
            args += (torch.from_numpy(sids[lo:hi]), menu, menu_ids)
            logits, _ = model(*args)
            yield slice(lo, hi), logits.masked_fill(~valid, -1e9)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r + two-sided normal-approx p (Fisher z)."""
    n = len(x)
    if n < 4 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return r, p


# ------------------------------------------------------------------------ s1

def s1(args):
    supporters = _supporter_ids()
    shards = _load_shards(ROOT / args.corpus)
    for ckpt in args.ckpt:
        model = load_ckpt(ROOT / ckpt)
        agree, probs, classes, turns = [], [], [], []
        for shard in shards:
            n_opts = shard["n_options"].astype(np.int64)
            starts = np.cumsum(n_opts) - n_opts
            labels = shard["labels"]
            turn = _turn_of(shard["states"])
            for sl, logits in _forward_shard(model, shard):
                p = torch.softmax(logits, dim=1)
                for i in range(sl.stop - sl.start):
                    row = sl.start + i
                    lab = int(labels[row])
                    agree.append(int(logits[i].argmax()) == lab)
                    probs.append(float(p[i, lab]))
                    s = starts[row]
                    classes.append(_option_class(
                        shard["options"][s + lab],
                        int(shard["option_ids"][s + lab, 0]), supporters))
                    turns.append(_bucket(int(turn[row])))
        agree, probs = np.array(agree), np.array(probs)
        classes, turns = np.array(classes), np.array(turns)
        print(f"\n=== S1 {ckpt}  n={len(agree)}  "
              f"overall agreement {agree.mean():.3f}  "
              f"mean p(target) {probs.mean():.3f}")
        print(f"{'class':<16} {'n':>6} {'agree':>7} {'p(tgt)':>7}   "
              + "  ".join(f"{f't{lo}-{hi}':>7}" for lo, hi in TURN_BUCKETS))
        for cls in sorted(set(classes.tolist()),
                          key=lambda c: -int((classes == c).sum())):
            m = classes == cls
            per_bucket = []
            for lo, hi in TURN_BUCKETS:
                bm = m & (turns == f"t{lo}-{hi}")
                per_bucket.append(f"{agree[bm].mean():.3f}({bm.sum()})"
                                  if bm.sum() else "      -")
            print(f"{cls:<16} {m.sum():>6} {agree[m].mean():>7.3f} "
                  f"{probs[m].mean():>7.3f}   " + "  ".join(per_bucket))
        atk = np.isin(classes, ["ATTACK", "ATTACH"])
        eco = np.isin(classes, ["PLAY_supporter", "EVOLVE"])
        if atk.any() and eco.any():
            gap = agree[atk].mean() - agree[eco].mean()
            print(f"pre-registered read: attack-class {agree[atk].mean():.3f} "
                  f"vs card-economy {agree[eco].mean():.3f}  gap {gap:+.3f} "
                  f"(threshold: >+0.20 = localized deficit)")


# --------------------------------------------------------------- s2 / s3 / s5

def _game_rows(shards):
    """Yield (rows_index_array, shard) per (shard, game_id) trajectory,
    preserving row order (the collector appends in decision order)."""
    for shard in shards:
        gids = shard["game_ids"]
        for g in np.unique(gids):
            yield np.nonzero(gids == g)[0], shard


def _outcomes(shards) -> list[tuple[np.ndarray, dict, int]]:
    """(rows, shard, outcome) per game; outcome 1 win / 0 loss, draws dropped.
    Terminal reward on the last row dominates shaping (|r| > 0.5 <=> decided)."""
    out = []
    for rows, shard in _game_rows(shards):
        last = float(shard["rewards"][rows[-1]])
        if last > 0.5:
            out.append((rows, shard, 1))
        elif last < -0.5:
            out.append((rows, shard, 0))
    return out


def s2(args):
    supporters = _supporter_ids()
    shards = _load_shards(ROOT / args.data, "ppo_shard_*.npz")
    plays, avail, wins = [], [], []
    for rows, shard, won in _outcomes(shards):
        n_opts = shard["n_options"].astype(np.int64)
        starts = np.cumsum(n_opts) - n_opts
        turn = _turn_of(shard["states"])
        n_play = n_avail = 0
        for r in rows:
            if not (2 <= turn[r] <= 4):
                continue
            s, m = starts[r], n_opts[r]
            kinds = [_option_class(shard["options"][s + j],
                                   int(shard["option_ids"][s + j, 0]),
                                   supporters) for j in range(m)]
            if "PLAY_supporter" in kinds:
                n_avail += 1
                if kinds[int(shard["actions"][r])] == "PLAY_supporter":
                    n_play += 1
        plays.append(n_play)
        avail.append(n_avail)
        wins.append(won)
    plays, avail = np.array(plays, float), np.array(avail, float)
    wins = np.array(wins, float)
    r, p = _pearson(plays, wins)
    print(f"\n=== S2  games={len(wins)} (draws dropped)  "
          f"winrate {wins.mean():.3f}")
    print(f"supporter plays t2-t4 per game: winners "
          f"{plays[wins == 1].mean():.2f}  losers {plays[wins == 0].mean():.2f}")
    print(f"corr(plays, win) r={r:+.3f}  p={p:.4f}")
    has = avail > 0
    rate = np.divide(plays, avail, out=np.zeros_like(plays), where=has)
    r2, p2 = _pearson(rate[has], wins[has])
    print(f"conditioned on availability (n={int(has.sum())} games): "
          f"take-rate winners {rate[has & (wins == 1)].mean():.3f}  "
          f"losers {rate[has & (wins == 0)].mean():.3f}  "
          f"corr r={r2:+.3f} p={p2:.4f}")
    print("pre-registered read: materially positive corr + policy declines "
          "supporters = credit-assignment failure; ~zero corr = signal absent "
          "-> opponents/exploration")


def s3(args):
    supporters = _supporter_ids()
    shards = _load_shards(ROOT / args.data, "ppo_shard_*.npz")
    cand: dict[str, int] = {}
    took: dict[str, int] = {}
    sup_by_bucket: dict[str, list[int]] = {}
    for shard in shards:
        n_opts = shard["n_options"].astype(np.int64)
        starts = np.cumsum(n_opts) - n_opts
        turn = _turn_of(shard["states"])
        for r in range(len(shard["actions"])):
            s, m = starts[r], n_opts[r]
            kinds = [_option_class(shard["options"][s + j],
                                   int(shard["option_ids"][s + j, 0]),
                                   supporters) for j in range(m)]
            chosen = kinds[int(shard["actions"][r])]
            for k in set(kinds):
                cand[k] = cand.get(k, 0) + 1
                took[k] = took.get(k, 0) + (chosen == k)
            if "PLAY_supporter" in kinds:
                b = _bucket(int(turn[r]))
                sup_by_bucket.setdefault(b, [0, 0])
                sup_by_bucket[b][0] += 1
                sup_by_bucket[b][1] += (chosen == "PLAY_supporter")
    print(f"\n=== S3  sampled-action frequency in candidate states "
          f"({sum(len(s['actions']) for s in shards)} decisions)")
    print(f"{'class':<16} {'candidates':>10} {'sampled':>8} {'freq':>7}")
    for k in sorted(cand, key=lambda k: -cand[k]):
        print(f"{k:<16} {cand[k]:>10} {took[k]:>8} {took[k] / cand[k]:>7.3f}")
    print("supporter-play by turn bucket:")
    for b in sorted(sup_by_bucket):
        c, t = sup_by_bucket[b]
        print(f"  {b:<6} {t}/{c} = {t / c:.3f}")
    f = took.get("PLAY_supporter", 0) / max(1, cand.get("PLAY_supporter", 1))
    print(f"pre-registered read: supporter sampled freq {f:.3f} "
          f"(<0.05 = exploration ceiling)")


def s5(args):
    shards = _load_shards(ROOT / args.data, "ppo_shard_*.npz")
    vals, outs, turns = [], [], []
    for rows, shard, won in _outcomes(shards):
        turn = _turn_of(shard["states"])
        for r in rows:
            vals.append(float(shard["values"][r]))
            outs.append(1.0 if won else -1.0)
            turns.append(_bucket(int(turn[r])))
    vals, outs = np.array(vals), np.array(outs)
    turns = np.array(turns)
    print(f"\n=== S5  value-head calibration, n={len(vals)} decisions "
          f"(draws dropped)")
    print(f"{'bucket':<7} {'n':>6} {'corr(V,outcome)':>16} {'mean|V|':>8}")
    for lo, hi in TURN_BUCKETS:
        m = turns == f"t{lo}-{hi}"
        if m.sum() < 10:
            continue
        r, _ = _pearson(vals[m], outs[m])
        print(f"t{lo}-{hi:<4} {m.sum():>6} {r:>16.3f} "
              f"{np.abs(vals[m]).mean():>8.3f}")
    print("pre-registered read: early-turn corr ~0 while late strong = critic "
          "cannot price setups")


def s6(args):
    """Plan-consistency from plan-PPO shards: rows carry the turn's committed
    plan (all-zeros = none; [0]=has-plan, [1:7]=attacker-slot one-hot,
    [17]=needs_attach) and the board ids (state_ids[0:6] = my active+bench).
    Consistency = the chosen ATTACK's acted card (option_ids[:,0]) is the
    plan's named attacker; for ATTACH rows under needs_attach plans, the
    attach target (option_ids[:,1]) is the named attacker. Chance reference =
    the marginal frequency of that slot's card being chosen regardless of
    plan. Approximation: slots can shift between commit and action."""
    shards = _load_shards(ROOT / args.data, "ppo_shard_*.npz")
    atk_match = atk_total = att_match = att_total = 0
    bench_named = 0
    for shard in shards:
        n_opts = shard["n_options"].astype(np.int64)
        starts = np.cumsum(n_opts) - n_opts
        plans = shard["plans"]
        for r in range(len(shard["actions"])):
            if plans[r, 0] < 0.5:
                continue
            slot = int(plans[r, 1:7].argmax())
            named = int(shard["state_ids"][r, slot])
            if named == 0:
                continue
            s = starts[r]
            chosen = s + int(shard["actions"][r])
            ot = int(shard["options"][chosen, :N_OPTION_TYPES].argmax())
            if ot == 13:                                    # ATTACK (always by
                atk_total += 1                              # the active)
                atk_match += int(shard["state_ids"][r, 0]) == named
                bench_named += slot != 0
            elif ot == 8 and plans[r, 17] > 0.5:            # ATTACH, needs it
                att_total += 1
                att_match += int(shard["option_ids"][chosen, 1]) == named
    print(f"\n=== S6  plan-consistency (plan-PPO shards)")
    if atk_total:
        print(f"ATTACK rows under a committed plan: {atk_match}/{atk_total} "
              f"= {atk_match / atk_total:.3f} attacked with the named "
              f"attacker as active ({bench_named}/{atk_total} plans named a "
              f"bench attacker; slot drift between commit and attack is the "
              f"approximation error)")
    if att_total:
        print(f"ATTACH rows under needs_attach plans: {att_match}/{att_total} "
              f"= {att_match / att_total:.3f} attach to the named attacker")
    print("pre-registered read: consistency at chance level = the plan head "
          "is decorative -> Phase 2 conversation includes DELETING it")


def s7(args):
    shards = _load_shards(ROOT / args.data, "ppo_shard_*.npz")
    model = load_ckpt(ROOT / args.ckpt)
    ents, nents, turns = [], [], []
    for shard in shards:
        n_opts = shard["n_options"].astype(np.int64)
        turn = _turn_of(shard["states"])
        for sl, logits in _forward_shard(model, shard):
            p = torch.softmax(logits, dim=1)
            h = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
            for i in range(sl.stop - sl.start):
                row = sl.start + i
                n = int(n_opts[row])
                if n < 2:
                    continue
                ents.append(float(h[i]))
                nents.append(float(h[i]) / math.log(n))
                turns.append(_bucket(int(turn[row])))
    ents, nents = np.array(ents), np.array(nents)
    turns = np.array(turns)
    print(f"\n=== S7 {args.ckpt}  option-head entropy (stored plans), "
          f"n={len(ents)} multi-option decisions")
    print(f"{'bucket':<7} {'n':>6} {'H':>7} {'H/logN':>7}")
    for lo, hi in TURN_BUCKETS:
        m = turns == f"t{lo}-{hi}"
        if m.sum() < 10:
            continue
        print(f"t{lo}-{hi:<4} {m.sum():>6} {ents[m].mean():>7.3f} "
              f"{nents[m].mean():>7.3f}")
    print("pre-registered read: t1-t4 near-determinism explains exploration "
          "never finding setup lines")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("s1")
    p1.add_argument("--ckpt", nargs="+", required=True)
    p1.add_argument("--corpus", default="data/bc_clone_54618168")
    p1.set_defaults(fn=s1)
    for name, fn in (("s2", s2), ("s3", s3), ("s5", s5), ("s6", s6)):
        pp = sub.add_parser(name)
        pp.add_argument("--data", default="data/ppo")
        pp.set_defaults(fn=fn)
    p7 = sub.add_parser("s7")
    p7.add_argument("--ckpt", required=True)
    p7.add_argument("--data", default="data/ppo")
    p7.set_defaults(fn=s7)
    a = p.parse_args()
    a.fn(a)
