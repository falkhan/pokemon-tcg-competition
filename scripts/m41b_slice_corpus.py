"""Slice a BC shard dir's option vectors to a narrower width (M41b R2).

The width experiment needs a control that differs from the arm in EXACTLY one
thing: whether the appended columns exist. Re-encoding twice cannot give that
— the episode cache, the shard order and the row set would all have to match
by luck. Slicing one corpus does give it: same rows, same labels, same order,
same shard boundaries, only the columns removed.

The slice is exact rather than approximate because the encoder law is
append-and-slice: `[0, N)` of the wide encoding IS the narrow encoding,
byte for byte, which `scripts/m42_column_safety.py` proves on real options at
every trained width. So a corpus sliced to 100 is what a width-100 encoder
would have written for those same prompts.

    uv run python scripts/m41b_slice_corpus.py data/bc_x_w143 data/bc_x_w100 --width 100
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402


def slice_shard(src: Path, dst: Path, width: int) -> tuple[int, int]:
    """Copy one shard with `options` truncated to `width`. Returns
    (rows, original width)."""
    with np.load(src) as z:
        data = {k: z[k] for k in z.files}
    old = int(data["options"].shape[1])
    if width > old:
        raise SystemExit(f"{src.name}: cannot widen {old} -> {width} by "
                         "slicing; re-encode instead")
    data["options"] = np.ascontiguousarray(data["options"][:, :width])
    np.savez(dst, **data)
    return int(data["options"].shape[0]), old


def slice_dir(src: Path, dst: Path, width: int) -> dict:
    if not src.is_dir():
        raise SystemExit(f"no such corpus: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.npz"))
    if not shards:
        raise SystemExit(f"{src} holds no .npz shards")
    rows, widths = 0, set()
    for shard in shards:
        n, old = slice_shard(shard, dst / shard.name, width)
        rows += n
        widths.add(old)
    # carry the provenance file across — the registry records the build
    # params, and a sliced corpus has the same provenance at a narrower width
    for extra in src.glob("*.json"):
        shutil.copy(str(extra), str(dst / extra.name))
    return {"shards": len(shards), "rows": rows, "from": sorted(widths),
            "to": width}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--width", type=int, required=True)
    a = ap.parse_args()
    out = slice_dir(a.src, a.dst, a.width)
    print(f"{a.src} -> {a.dst}")
    print(f"  {out['shards']} shards, {out['rows']} rows, "
          f"option width {out['from']} -> {out['to']}")
    if len(out["from"]) > 1:
        print("  NOTE: source shards had MIXED widths; the slice has made "
              "them uniform, which is only correct if every source was >= "
              f"{a.width}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
