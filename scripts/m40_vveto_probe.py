"""M40b Track C2 — vveto mechanism + latency probe.

Plays n games of the `model-cz-vv` arm vs a bed (single process so the
module-global VVETO_STATS is readable) and reports: veto engagement
(prompts/opened/stepped/vetoes), per-prompt latency (mean/max vs the G6
50 ms bar and the 600 s/game hard budget), and the W-L (small n — count
only, never a strength claim).

Usage: uv run python scripts/m40_vveto_probe.py [-n 16] [--bed <spec>]
       [--delta 0.10] [--topk 3]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rl.matchrunner as mr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=16)
    ap.add_argument("--checkpoint", default="checkpoints/m38_w9294_cont3.pt")
    ap.add_argument("--deck", default="alakazam_v2_h4")
    ap.add_argument("--bed",
                    default="model:checkpoints/m28_winners.pt:alakazam_v2_h4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--topk", type=int, default=None)
    a = ap.parse_args()
    if a.delta is not None:
        mr.VVETO_DELTA = a.delta
    if a.topk is not None:
        mr.VVETO_TOPK = a.topk

    fn_a, deck_a = mr.make_pilot(
        mr.parse_spec(f"model-cz-vv:{a.checkpoint}:{a.deck}"),
        instance=f"vv{a.seed}_a")
    fn_b, deck_b = mr.make_pilot(mr.parse_spec(a.bed),
                                 instance=f"vv{a.seed}_b")
    wins = 0
    for g in range(a.games):
        fns = (fn_a, fn_b) if g % 2 == 0 else (fn_b, fn_a)
        decks = (deck_a, deck_b) if g % 2 == 0 else (deck_b, deck_a)
        res = mr._engine_game(fns[0], fns[1], decks[0], decks[1])
        wins += int(res == g % 2)

    s = mr.VVETO_STATS
    print(f"vveto probe: n={a.games} vs {a.bed}  "
          f"(delta={mr.VVETO_DELTA}, topk={mr.VVETO_TOPK})")
    print(f"  W-L {wins}-{a.games - wins}   (count only at this n)")
    print(f"  prompts {s['prompts']}  opened {s['opened']}  "
          f"stepped {s['stepped']}  VETOES {s['vetoes']} "
          f"({s['vetoes'] / max(s['prompts'], 1):.1%} of eligible prompts)")
    if s["prompts"]:
        print(f"  veto latency: mean {s['time_sum'] / s['prompts'] * 1e3:.1f} ms"
              f"  max {s['time_max'] * 1e3:.1f} ms   "
              f"(G6 bar: 50 ms mean; hard budget 600 s/game)")
    ok = s["vetoes"] > 0 and s["opened"] > 0
    print("\nmechanism " + ("FIRES — C3 battery is worth running."
                            if ok else "DID NOT FIRE — investigate before "
                            "spending battery time (vacuous-pass rule)."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
