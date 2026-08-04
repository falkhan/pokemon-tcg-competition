"""One gate runner over a PRE-REGISTERED, HASHED spec (M41b § II.3d).

Every milestone so far hand-rolled a `*_decide.py` — eight of them now, each a
fork of the last. Forking the decoder means the bar and the read live in the
same file as the numbers, and nothing stops either from being adjusted after
the numbers land. The diaries keep that discipline by hand; this is the
machine version.

The contract, in three commands:

    gate_spec.py hash   spec.json                 # what you pre-register
    gate_spec.py run    spec.json --out runs/x    # stamps the hash into every cell
    gate_spec.py decode spec.json --out runs/x    # refuses if they disagree

`run` writes the spec's hash into each battery checkpoint header, where
`rl.matchrunner.run_pairs` ALREADY refuses to resume a file whose header does
not match. `decode` re-hashes the spec on disk and refuses to emit a verdict
unless every cell it reads was produced under that exact hash, at the n the
spec named. So editing a bar after seeing the result does not produce a
kinder verdict — it produces a refusal.

What is deliberately NOT here: the weighted-pool live-mix machinery of
`scripts/m40_decide.py`. That decoder is pre-registered, proven, and frozen as
the record of what shipped; this runner is the general instrument for new
gates and keeps its read simple and stated (per-cell two-proportion z against
the SAME-BATTERY control, 95% CIs, G-9). A spec that needs live-mix weighting
should say so and reuse that decoder.

Spec shape (JSON):

    {
      "name":       "m41b_stage_b",
      "question":   "is search output learnable by our net at our scale?",
      "arm":        "model:checkpoints/arm.pt:deck",
      "control":    "model:checkpoints/control.pt:deck",
      "beds":       [{"name": "grim", "spec": "model:checkpoints/bed.pt:grim_live"}],
      "n_per_cell": 3000,
      "seed":       1,
      "bars":       {"pass": 0.02, "kill": 0.0}
    }

`bars` are on the ARM MINUS CONTROL delta, in win-rate points: `pass` is the
bar the arm must clear to be called a pass, `kill` the value at or below which
the milestone stops. Both are pre-registered; neither is readable from the
numbers.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Z95 = 1.959963985
REQUIRED = ("name", "arm", "control", "beds", "n_per_cell", "seed", "bars")


def spec_hash(spec: dict) -> str:
    """A canonical hash of the DECISION content of a spec.

    Canonical (sorted keys, no whitespace) so reformatting the file is not a
    tamper, and restricted to the fields that decide the verdict so a typo fix
    in `question` does not invalidate a running battery. Everything that could
    change the answer — arms, beds, n, seed, bars — is inside.
    """
    payload = {k: spec[k] for k in REQUIRED if k in spec}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise SystemExit(f"spec is missing required field(s): {missing}")
    if not spec["beds"]:
        raise SystemExit("spec names no beds — there is nothing to measure")
    for bed in spec["beds"]:
        if "name" not in bed or "spec" not in bed:
            raise SystemExit(f"every bed needs a name and a spec: {bed}")
    bars = spec["bars"]
    if "pass" not in bars or "kill" not in bars:
        raise SystemExit("bars must pre-register both 'pass' and 'kill'")
    if bars["kill"] > bars["pass"]:
        raise SystemExit("kill bar sits above the pass bar — unreadable")
    return spec


def cell_path(out: Path, side: str, bed_name: str) -> Path:
    return out / f"{side}__{bed_name}.jsonl"


def wr(results: list[int]) -> float:
    """Win rate for side a; draws count half (the matchrunner decode law)."""
    if not results:
        return 0.0
    return (sum(1 for r in results if r == 0)
            + 0.5 * sum(1 for r in results if r == 2)) / len(results)


def ci95(p: float, n: int) -> float:
    return Z95 * math.sqrt(max(p * (1 - p), 0.0) / n) if n else 0.0


def two_proportion_z(p_a: float, n_a: int, p_b: float, n_b: int) -> float:
    """Pooled-SE z for arm vs control. G-9: always against the SAME battery's
    control, never a frozen historical pin."""
    if not n_a or not n_b:
        return 0.0
    pool = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    se = math.sqrt(max(pool * (1 - pool), 0.0) * (1 / n_a + 1 / n_b))
    return (p_a - p_b) / se if se > 0 else 0.0


def read_cell(path: Path) -> tuple[list[int], dict]:
    """(results, header) for one battery checkpoint."""
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    results: list[int] = []
    for row in rows[1:]:
        results.extend(row["results"])
    return results, rows[0]


def cmd_hash(spec_path: Path) -> int:
    spec = load_spec(spec_path)
    print(spec_hash(spec))
    return 0


def cmd_run(spec_path: Path, out: Path, workers: int) -> int:
    from rl.matchrunner import parse_spec, run_pairs

    spec = load_spec(spec_path)
    digest = spec_hash(spec)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(spec_path.read_text(encoding="utf-8"),
                                   encoding="utf-8")
    print(f"gate {spec['name']}  spec {digest}  n/cell {spec['n_per_cell']}")

    for side in ("arm", "control"):
        for bed in spec["beds"]:
            path = cell_path(out, side, bed["name"])
            print(f"  {side:<8} vs {bed['name']:<12} -> {path.name}", flush=True)
            run_pairs([(parse_spec(spec[side]), parse_spec(bed["spec"]),
                        spec["n_per_cell"])],
                      workers=workers, seed=spec["seed"],
                      checkpoint=str(path),
                      key_extra={"gate": spec["name"], "spec_hash": digest})
    print("done — decode with: gate_spec.py decode "
          f"{spec_path} --out {out}")
    return 0


def cmd_decode(spec_path: Path, out: Path) -> int:
    spec = load_spec(spec_path)
    digest = spec_hash(spec)
    print(f"=== gate {spec['name']} ===")
    print(f"spec hash   {digest}")
    if spec.get("question"):
        print(f"question    {spec['question']}")
    print(f"bars        pass >= {spec['bars']['pass']:+.4f}   "
          f"kill <= {spec['bars']['kill']:+.4f}   (arm minus control)")

    # --- refuse before reading a single number --------------------------
    refusals: list[str] = []
    cells: dict[str, dict[str, list[int]]] = {"arm": {}, "control": {}}
    for side in ("arm", "control"):
        for bed in spec["beds"]:
            path = cell_path(out, side, bed["name"])
            if not path.exists():
                refusals.append(f"missing cell {path.name}")
                continue
            results, header = read_cell(path)
            stamped = (header.get("extra") or {}).get("spec_hash")
            if stamped != digest:
                refusals.append(
                    f"{path.name} ran under spec {stamped!r}, not {digest!r} "
                    "— the spec changed after the run")
            if len(results) != spec["n_per_cell"]:
                refusals.append(
                    f"{path.name} holds {len(results)} games, spec "
                    f"pre-registered {spec['n_per_cell']}")
            cells[side][bed["name"]] = results

    if refusals:
        print("\nREFUSED — this run does not match its pre-registered spec:")
        for r in refusals:
            print(f"  - {r}")
        print("\nNo verdict is emitted. That is the point: a bar that moves "
              "after the numbers land is not a bar.")
        return 2

    # --- the read -------------------------------------------------------
    print(f"\n{'bed':<14}{'arm':>18}{'control':>18}{'delta':>10}{'z':>8}")
    deltas, n_total = [], 0
    for bed in spec["beds"]:
        a, c = cells["arm"][bed["name"]], cells["control"][bed["name"]]
        p_a, p_c = wr(a), wr(c)
        z = two_proportion_z(p_a, len(a), p_c, len(c))
        deltas.append(p_a - p_c)
        n_total += len(a)
        print(f"{bed['name']:<14}{p_a:>10.4f}+-{ci95(p_a, len(a)):<6.4f}"
              f"{p_c:>10.4f}+-{ci95(p_c, len(c)):<6.4f}"
              f"{p_a - p_c:>+10.4f}{z:>8.2f}")

    pooled_a = [r for bed in spec["beds"] for r in cells["arm"][bed["name"]]]
    pooled_c = [r for bed in spec["beds"] for r in cells["control"][bed["name"]]]
    p_a, p_c = wr(pooled_a), wr(pooled_c)
    delta = p_a - p_c
    z = two_proportion_z(p_a, len(pooled_a), p_c, len(pooled_c))
    print(f"\n{'POOLED':<14}{p_a:>10.4f}+-{ci95(p_a, len(pooled_a)):<6.4f}"
          f"{p_c:>10.4f}+-{ci95(p_c, len(pooled_c)):<6.4f}"
          f"{delta:>+10.4f}{z:>8.2f}")

    bars = spec["bars"]
    if delta >= bars["pass"]:
        verdict, rc = "PASS", 0
    elif delta <= bars["kill"]:
        verdict, rc = "KILL", 1
    else:
        verdict, rc = "INCONCLUSIVE", 1
    print(f"\nVERDICT: {verdict}   delta {delta:+.4f} vs pass "
          f"{bars['pass']:+.4f} / kill {bars['kill']:+.4f}")
    if verdict == "INCONCLUSIVE":
        print("  between the bars — the pre-registered outcome is 'we do not "
              "know', not a rounded-up pass.")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash", help="print the spec's canonical hash")
    h.add_argument("spec", type=Path)
    r = sub.add_parser("run", help="run every cell, stamping the spec hash")
    r.add_argument("spec", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=8)
    d = sub.add_parser("decode", help="verify the run, then emit the verdict")
    d.add_argument("spec", type=Path)
    d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.cmd == "hash":
        return cmd_hash(a.spec)
    if a.cmd == "run":
        if a.workers > 8:                      # the standing parallelism cap
            raise SystemExit("--workers 8 is the proven-stable ceiling")
        return cmd_run(a.spec, a.out, a.workers)
    return cmd_decode(a.spec, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
