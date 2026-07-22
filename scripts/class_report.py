"""Per-class fidelity report for a BC checkpoint over its training corpus.

Breaks the headline val_acc down by teacher OptionType, and the weak classes
(ATTACH, PLAY) down by acted card id, with top confusions per weak cell.

The val split is reconstructed EXACTLY as rl.plan_iter.train does it
(default_rng(0), 10% of unique game_ids) — pass --data dirs in the SAME ORDER
as the training invocation, the split depends on it.

Usage:
    uv run python scripts/class_report.py \
        --ckpt checkpoints/m25_bc_alakazam_v3h.pt --data data/bc_m25_alakazam_v3h

Acceptance pin (m25_bc_alakazam_v3h + data/bc_m25_alakazam_v3h): headline 0.668,
ATTACH 0.408, Telepath-attach 23/67 = 0.343, PLAY 0.486.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import OptionType                          # noqa: E402
from rl.plan_iter import BCDatasetV3, _n_ids_of, collate_v3   # noqa: E402
from rl.policy import OptionScorerV3, option_dim_of    # noqa: E402

N_OPTION_TYPES = 17          # OptionType one-hot block at the head of an option
CARD_DETAIL_TYPES = ("ATTACH", "PLAY")   # weak classes broken down per card


def load_model(ckpt_path: Path) -> OptionScorerV3:
    sdict = torch.load(ckpt_path, map_location="cpu")
    model = OptionScorerV3(n_state_ids=_n_ids_of(sdict),
                           option_dim=option_dim_of(sdict))
    model.load_state_dict(sdict)
    model.eval()
    return model


def val_indices(ds: BCDatasetV3) -> np.ndarray:
    """The exact split from rl.plan_iter.train (:822-829): seeded, per-game."""
    rng = np.random.default_rng(0)
    unique_games = np.unique(ds.game_ids)
    val_games = set(rng.choice(unique_games,
                               max(1, int(0.1 * len(unique_games))),
                               replace=False).tolist())
    return np.nonzero(np.isin(ds.game_ids, list(val_games)))[0]


def option_meta(ds: BCDatasetV3, row: int, opt_idx: int) -> tuple[int, int]:
    """(OptionType int, acted card id) of option `opt_idx` in row `row`."""
    flat = ds.starts[row] + opt_idx
    return int(ds.options[flat, :N_OPTION_TYPES].argmax()), \
        int(ds.option_ids[flat, 0])


def card_names() -> dict[int, str]:
    with open(ROOT / "data/cards_features.csv") as f:
        return {int(r["card_id"]): r["name"] for r in csv.DictReader(f)}


def report(ckpt: Path, data_dirs: list[Path], batch_size: int = 512) -> dict:
    ds = BCDatasetV3(data_dirs)
    model = load_model(ckpt)
    val_idx = val_indices(ds)
    names = card_names()

    by_type = defaultdict(lambda: [0, 0])            # type -> [correct, total]
    by_card = {t: defaultdict(lambda: [0, 0]) for t in CARD_DETAIL_TYPES}
    confusion = {t: Counter() for t in CARD_DETAIL_TYPES}

    corpus_types = Counter(
        option_meta(ds, i, ds.labels[i])[0] for i in range(len(ds)))

    with torch.no_grad():
        for b0 in range(0, len(val_idx), batch_size):
            idxs = val_idx[b0:b0 + batch_size]
            batch = collate_v3([ds[int(i)] for i in idxs])
            (states, plans, sids, options, oids, valid, labels,
             *_rest) = batch
            logits, _ = model(states, plans, sids, options, oids)
            preds = logits.masked_fill(~valid, -1e9).argmax(dim=1)
            for k, i in enumerate(idxs):
                true_j, pred_j = int(labels[k]), int(preds[k])
                t, cid = option_meta(ds, i, true_j)
                tname = OptionType(t).name
                ok = true_j == pred_j
                by_type[t][0] += ok
                by_type[t][1] += 1
                if tname in CARD_DETAIL_TYPES:
                    by_card[tname][cid][0] += ok
                    by_card[tname][cid][1] += 1
                    if not ok:
                        pt, pcid = option_meta(ds, i, pred_j)
                        confusion[tname][(cid, OptionType(pt).name, pcid)] += 1

    correct = sum(v[0] for v in by_type.values())
    total = sum(v[1] for v in by_type.values())
    headline = correct / max(1, total)

    print(f"ckpt {ckpt.name}  data {[d.name for d in data_dirs]}")
    print(f"rows {len(ds)}  val {total}  headline val_acc {headline:.3f}\n")
    print("class      share   val_acc")
    for t, (c, n) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        share = corpus_types[t] / len(ds)
        print(f"  {OptionType(t).name:10s} {share:5.1%}  {c:4d}/{n:4d} = {c/n:.3f}")
    for tname in CARD_DETAIL_TYPES:
        print(f"\n{tname} val_acc by acted card:")
        for cid, (c, n) in sorted(by_card[tname].items(),
                                  key=lambda kv: -kv[1][1])[:10]:
            print(f"  {names.get(cid, str(cid)):30s} {c:3d}/{n:3d} = {c/n:.3f}")
        print(f"{tname} top confusions (true card -> predicted type/card):")
        for (cid, pt, pcid), n in confusion[tname].most_common(8):
            pname = names.get(pcid, "") if pcid else ""
            print(f"  {names.get(cid, str(cid)):26s} -> {pt:8s} {pname:24s} x{n}")

    return {"headline": headline,
            "by_type": {OptionType(t).name: (c, n)
                        for t, (c, n) in by_type.items()},
            "by_card": {t: dict(d) for t, d in by_card.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data", required=True, nargs="+", type=Path,
                    help="corpus dirs, SAME ORDER as the training invocation")
    args = ap.parse_args()
    report(args.ckpt, args.data)


if __name__ == "__main__":
    main()
