"""M39 P0.7 — decode the strip ablation into the per-rule strip ledger.

Answers decision 1 with attribution instead of an aggregate. G5 measured
plain/gacf/gacfr3 only as whole configs and its diary says per-rule
attribution beyond racemode3 is INCOMPLETE, so "strip everything" was an
inference. Each `no_<rule>` arm is the shipped stack MINUS that one rule:

    delta = WR(no_rule) - WR(full)
    delta > 0  ->  removing the rule HELPS  ->  strip it
    delta < 0  ->  the rule is EARNING its slot -> keep it

Two beds, because a rule that helps on one and hurts on the other is a real
finding and pooling would hide it: `wall` (where G5 found the harm) and
`top` (P0.8's 900+ instrument, which independently reproduced the stack cost).

Usage: uv run python scripts/m39_abl_decode.py
"""
import glob
import json
import math

ARMS = ("plain", "full", "no_racemode3", "no_telepath", "no_deckguard",
        "no_ash", "no_conserve", "no_benchfloor")
BEDS = ("wall", "top")


def decode(paths):
    res = []
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                if "results" in d:
                    res += d["results"]
    if not res:
        return None
    return (res.count(0) + 0.5 * res.count(2)) / len(res), len(res)


def z2(p1, n1, p2, n2):
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(pool * (1 - pool), 1e-9) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else 0.0


def main() -> int:
    cells = {(a, b): decode(sorted(glob.glob(f"runs/m39_abl_{a}_{b}_s*.jsonl")))
             for a in ARMS for b in BEDS}

    print("=== M39 P0.7 strip ledger: leave-one-out from the shipped stack ===")
    print("shipped = telepath,deckguard,ash,conserve,benchfloor,racemode3\n")
    print(f"{'arm':<16}{'wall':>16}{'top':>16}   note")
    print("-" * 70)
    for a in ARMS:
        row = f"{a:<16}"
        for b in BEDS:
            c = cells[(a, b)]
            row += f"{c[0]:>10.3f}(n{c[1]})" if c else f"{'PENDING':>16}"
        note = {"plain": "no rules at all (the G5 'plain' arm)",
                "full": "the shipped config = baseline for the deltas"}.get(a, "")
        print(row + f"   {note}")

    full = {b: cells[("full", b)] for b in BEDS}
    print("\n--- delta vs full (positive = REMOVING the rule helps) ---")
    print(f"{'rule removed':<16}" + "".join(f"{b + ' delta':>14}{'z':>7}"
                                            for b in BEDS) + "   verdict")
    print("-" * 78)
    for a in ARMS:
        if a == "full":
            continue
        row, helps, hurts = f"{a:<16}", 0, 0
        for b in BEDS:
            c, f = cells[(a, b)], full[b]
            if not c or not f:
                row += f"{'--':>14}{'--':>7}"
                continue
            d = c[0] - f[0]
            z = z2(c[0], c[1], f[0], f[1])
            row += f"{d:>+14.3f}{z:>+7.2f}"
            if z > 1.64:
                helps += 1
            elif z < -1.64:
                hurts += 1
        verdict = ("STRIP (helps significantly)" if helps and not hurts else
                   "KEEP (earns its slot)" if hurts and not helps else
                   "mixed - inspect" if helps and hurts else "no signal")
        print(row + f"   {verdict}")

    print("\nDecision rule (plan decision 1): default is strip-all; a rule is "
          "kept only if removing it significantly HURTS on a bed. 'no signal' "
          "rules fall to the default and are stripped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
