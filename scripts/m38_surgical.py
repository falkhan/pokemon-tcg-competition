"""M38 surgical corpus — prize-relevant rows only (Piotr's pivot call,
2026-07-31, after the full-corpus retrain collapse).

The full-corpus fine-tune overwrote the champion's kaggle-winner behavior
wholesale (gate battery: ft .144 vs control .517 pooled). The plan's
original logic said the poisoned prior is LOCALIZED to prize-closing
decisions — so the corrective dose must be localized too. This filter keeps
only the rows where E0c measured the corrected teacher's labels actually
shifting:

  - chosen option is an ATTACK   (which attack / whether to close)
  - chosen option is a supporter PLAY (the chronic under-play line)

Commit provenance is NOT recoverable from the shards (plan-null 0.97 on
this corpus — committed lines mostly derive null plans), so the filter is
by label class, which is unambiguous in the data.

Usage:
  uv run python scripts/m38_surgical.py [--src data/m38_gen1_a data/m38_gen1_b]
      [--out data/m38_surgical]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_OT_PLAY, _OT_ATTACK = 7, 13    # rl/encoders.py option-type one-hot slots

ROW_KEYS = ("states", "plans", "state_ids", "labels", "game_ids", "results",
            "deck_idx", "n_plan_cands", "plan_labels", "weights",
            "n_options")


def filter_shard(path: Path, supporter_ids: frozenset):
    s = dict(np.load(path))
    n_opt = s["n_options"]
    offsets = np.concatenate([[0], np.cumsum(n_opt)])
    cand_off = np.concatenate([[0], np.cumsum(s["n_plan_cands"])])
    keep = []
    for i, (n, label) in enumererate_guard(n_opt, s["labels"]):
        chosen = s["options"][offsets[i] + label]
        if chosen[_OT_ATTACK] == 1:
            keep.append(i)
        elif chosen[_OT_PLAY] == 1 and \
                s["option_ids"][offsets[i] + label, 0] in supporter_ids:
            keep.append(i)
    if not keep:
        return None
    keep = np.array(keep)
    out = {k: s[k][keep] for k in ROW_KEYS if k in s}
    opt_rows = np.concatenate([np.arange(offsets[i], offsets[i] + n_opt[i])
                               for i in keep])
    out["options"] = s["options"][opt_rows]
    out["option_ids"] = s["option_ids"][opt_rows]
    cand_rows = [np.arange(cand_off[i], cand_off[i] + s["n_plan_cands"][i])
                 for i in keep]
    cand_rows = (np.concatenate(cand_rows) if cand_rows
                 else np.zeros(0, dtype=int))
    out["plan_cands"] = s["plan_cands"][cand_rows]
    return out


def enumererate_guard(n_opt, labels):
    """enumerate() with the label-in-range guard (paranoia: a label past the
    menu would silently select the wrong option row after filtering)."""
    for i, (n, label) in enumerate(zip(n_opt, labels)):
        assert 0 <= label < n, f"row {i}: label {label} outside menu {n}"
        yield i, (n, label)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", nargs="+",
                   default=["data/m38_gen1_a", "data/m38_gen1_b"])
    p.add_argument("--out", default="data/m38_surgical")
    a = p.parse_args()

    from cg.api import CardType, all_card_data
    supporter_ids = frozenset(c.cardId for c in all_card_data()
                              if c.cardType == CardType.SUPPORTER)

    out_dir = ROOT / a.out
    if out_dir.exists() and any(out_dir.glob("*.npz")):
        raise SystemExit(f"{out_dir} already holds shards — fresh dirs only")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_in = total_out = attack = supp = 0
    idx = 0
    for src in a.src:
        for path in sorted((ROOT / src).glob("shard_*.npz")):
            s_in = np.load(path)
            total_in += len(s_in["labels"])
            out = filter_shard(path, supporter_ids)
            if out is None:
                continue
            total_out += len(out["labels"])
            for i, (n, label) in enumererate_guard(out["n_options"],
                                                   out["labels"]):
                off = int(out["n_options"][:i].sum())
                attack += out["options"][off + label][_OT_ATTACK] == 1
                supp += out["options"][off + label][_OT_PLAY] == 1
            np.savez_compressed(out_dir / f"shard_w00_{idx:04d}.npz", **out)
            idx += 1
    print(f"kept {total_out}/{total_in} rows "
          f"({total_out / max(1, total_in):.1%}) — attack {attack}, "
          f"supporter-play {supp} -> {out_dir} ({idx} shards)")


if __name__ == "__main__":
    main()
