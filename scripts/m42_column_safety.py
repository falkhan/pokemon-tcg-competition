"""Append-safety proof: re-encode REAL options and hash the trained columns.

The encoder law this milestone works under is append-and-slice: new features go
on the end, `rl/policy.py` truncates every option to the width its net was
trained at, and so a shipped bundle is bit-identical before and after. M41
proved that once by hand against an already-live submission (78,165 options,
max absolute delta 0.0). This makes it a command.

Run it, change the encoder, run it again, diff the digests:

    uv run python scripts/m42_column_safety.py --out before.json
    ... edit rl/encoders.py / rl/scaling.py ...
    uv run python scripts/m42_column_safety.py --out after.json --compare before.json

`--width` is the trained width to protect (default 100 = OPTION_M28_DIM, what
every live bundle slices to). The digest covers columns [0, width) only; the
whole point is that columns at and beyond it are free to move.

Options come from local replay JSON (`rl/eval.py --json_prefix` output, which
is what `scripts/watch_games.py` and the QC battery write), so this needs no
Kaggle cache and no network.
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from cg.api import OptionType, to_observation_class  # noqa: E402

from rl.encoders import encode_option_v2  # noqa: E402


def _digest(rows: list[np.ndarray]) -> str:
    if not rows:
        return "EMPTY"
    return hashlib.sha256(np.stack(rows).tobytes()).hexdigest()[:32]


def collect(patterns: list[str], width: int, limit: int) -> dict:
    rows: list[np.ndarray] = []
    by_type: dict[str, int] = {}
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    for path in files:
        try:
            raw = json.loads(Path(path).read_text())
        except ValueError:
            continue
        if not isinstance(raw, dict) or not raw.get("steps"):
            continue           # manifest.json and friends live here too
        for step in raw["steps"]:
            for seat in (0, 1):
                st = step[seat] if seat < len(step) else None
                obs_dict = (st or {}).get("observation")
                if not obs_dict or not obs_dict.get("select"):
                    continue
                try:
                    obs = to_observation_class(obs_dict)
                except Exception:      # noqa: BLE001 — a malformed step is not
                    continue           # a reason to abandon the proof
                for opt in (obs.select.option or ()):
                    try:
                        num, _ids = encode_option_v2(opt, obs)
                    except Exception:  # noqa: BLE001
                        continue
                    rows.append(np.asarray(num[:width], dtype=np.float32))
                    name = getattr(OptionType(opt.type), "name",
                                   str(opt.type)) if opt.type is not None else "?"
                    by_type[name] = by_type.get(name, 0) + 1
                    if len(rows) >= limit:
                        return {"n": len(rows), "files": len(files),
                                "width": width, "by_type": by_type,
                                "digest": _digest(rows),
                                "checksum": float(np.stack(rows).sum())}
    return {"n": len(rows), "files": len(files), "width": width,
            "by_type": by_type, "digest": _digest(rows),
            "checksum": float(np.stack(rows).sum()) if rows else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replays", nargs="*",
                    default=["replays/**/*.json"],
                    help="glob(s) of env.toJSON() replay files")
    ap.add_argument("--width", type=int, default=100,
                    help="trained option width to protect (100 = OPTION_M28_DIM)")
    ap.add_argument("--limit", type=int, default=200_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None,
                    help="a previous --out file; exits non-zero on any drift")
    a = ap.parse_args()

    got = collect(a.replays, a.width, a.limit)
    print(f"encoded {got['n']} real options from {got['files']} replay files")
    for name, n in sorted(got["by_type"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:7d}  {name}")
    print(f"\ncolumns [0,{a.width}) digest {got['digest']}")

    if got["n"] == 0:
        print("\nFAIL: no options encoded — the proof is vacuous, not passing. "
              "Check --replays; local replay JSON comes from `rl/eval.py "
              "--json_prefix` (scripts/watch_games.py, the QC battery).")
        return 1

    rc = 0
    if a.compare:
        old = json.loads(Path(a.compare).read_text())
        if old["width"] != got["width"] or old["n"] != got["n"]:
            print(f"\nFAIL: not comparable — before was n={old['n']} "
                  f"width={old['width']}, now n={got['n']} width={got['width']}")
            rc = 1
        elif old["digest"] != got["digest"]:
            print(f"\nFAIL: columns [0,{a.width}) MOVED.\n"
                  f"  before {old['digest']} (sum {old['checksum']:.6f})\n"
                  f"  after  {got['digest']} (sum {got['checksum']:.6f})\n"
                  "  A live bundle trained at this width would change "
                  "behaviour. Move the new features to the END of the vector.")
            rc = 1
        else:
            print(f"\nPASS: columns [0,{a.width}) are bit-identical across "
                  f"{got['n']} real options. Anything at or beyond column "
                  f"{a.width} is free to move — every live bundle slices it off.")
    if a.out:
        Path(a.out).write_text(json.dumps(got, indent=2))
        print(f"  -> {a.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
