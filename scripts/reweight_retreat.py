"""M19.x rare-class reweighting: bump the policy-CE `weights` column on rows
where the teacher's chosen label is a RETREAT option. Reconstructed for
M19.2 from the m19b shards (data/plan_m19_rw, weight=8.0) — verified to
reproduce them byte-for-byte before use (602/27185 retreat rows, 0 shard
mismatches at weight=8.0).

Usage: uv run python scripts/reweight_retreat.py <src_dir> <dst_dir> <weight>
"""
import sys
from pathlib import Path

import numpy as np

_OT_RETREAT = 12  # OptionType.RETREAT one-hot index (rl/encoders.py)


def reweight_dir(src: Path, dst: Path, weight: float) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_retreat = 0
    for f in sorted(src.glob("shard_w*.npz")):
        shard = dict(np.load(f, allow_pickle=True))
        n_opts = shard["n_options"]
        offsets = np.cumsum(n_opts) - n_opts
        chosen = shard["options"][offsets + shard["labels"]]
        is_retreat = chosen[:, _OT_RETREAT] == 1.0
        shard["weights"] = np.where(is_retreat, weight, 1.0).astype(np.float32)
        total_rows += len(shard["labels"])
        total_retreat += int(is_retreat.sum())
        np.savez_compressed(dst / f.name, **shard)
    print(f"{src} -> {dst}: {total_retreat}/{total_rows} rows weighted {weight}")


if __name__ == "__main__":
    src_dir, dst_dir, weight_str = sys.argv[1], sys.argv[2], sys.argv[3]
    reweight_dir(Path(src_dir), Path(dst_dir), float(weight_str))
