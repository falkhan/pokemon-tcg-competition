"""M37: O12 racemode fire-count probe (the diag companion to m37_battery.sh).

Wraps rl.plan.apply_play_overrides with a counter and plays in-process games
of the gacfr/gacfrr arm vs a trigger bed, reporting per-variant: MAIN prompts
seen, trigger-true states (opp board shows a _RACEMODE_OPP_IDS Pokémon),
margin-gate-open states, and actual FIRES (order changed by the O12 demote,
isolated by diffing against the same call WITHOUT the racemode fix).

A battery null result with zero fires = trigger bug, not a dead lever
(docs/M37-plan.md risk note) — this probe is the check that separates them.

Usage: uv run python scripts/racemode_fire_probe.py [--bed hop] [-n 12]
                                                    [--arm gacfr] [--seed 5]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rl.plan as rp
from rl.matchrunner import parse_spec, play_series

BEDS = {
    "hop": "solver:decks/hops_stall.csv",
    "wall": "solver:decks/greattusk_wall.csv",
    "grim": "model:checkpoints/m25_bc_grim_54861775.pt:"
            "data/kaggle/grimmsnarl_3121746f_deck.csv",
    "garchomp": "model:checkpoints/m37_bc_garchomp.pt:"
                "data/kaggle/garchomp_m37_deck.csv",
    "mirror": "model:checkpoints/m28_winners.pt:clone54618168",
}
RACE_FIXES = frozenset({rp.PLAY_FIX_RACEMODE, rp.PLAY_FIX_RACEMODER,
                        rp.PLAY_FIX_RACEMODE2, rp.PLAY_FIX_RACEMODE3})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", default="hop", choices=sorted(BEDS))
    ap.add_argument("--arm", default="gacfr",
                    choices=["gacfr", "gacfrr", "gacfr2", "gacfr3"])
    ap.add_argument("-n", "--games", type=int, default=12)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    stats = Counter()
    orig = rp.apply_play_overrides

    def counting(obs, ranked, fixes):
        out = orig(obs, ranked, fixes)
        if not (fixes & RACE_FIXES) or obs.select is None \
                or obs.current is None \
                or obs.select.context != rp.SelectContext.MAIN:
            return out
        stats["main_prompts"] += 1
        st = obs.current
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        if rp._opp_board_ids(op) & rp._RACEMODE_OPP_IDS:
            stats["trigger_true"] += 1
            if (me.deckCount < op.deckCount - rp._RACEMODE_MARGIN
                    and rp._RACEMODE_DECK_LO < me.deckCount
                    <= rp._RACEMODE_DECK_HI):
                stats["margin_open"] += 1
            if out != orig(obs, ranked, fixes - RACE_FIXES):
                stats["o12_fires"] += 1
        return out

    rp.apply_play_overrides = counting
    try:
        a = parse_spec(f"modelt-{args.arm}:checkpoints/m28_winners.pt:"
                       f"decks/alakazam_v2_h4.csv")
        b = parse_spec(BEDS[args.bed])
        results = play_series(a, b, args.games, seed=args.seed)
    finally:
        rp.apply_play_overrides = orig

    res = results["results"] if isinstance(results, dict) else results
    w = list(res).count(0)
    print(f"\narm={args.arm} bed={args.bed} n={args.games} "
          f"W-L {w}-{len(list(res)) - w}")
    for k in ("main_prompts", "trigger_true", "margin_open", "o12_fires"):
        print(f"  {k:13s} {stats[k]}")
    if stats["trigger_true"] == 0:
        print("  WARNING: trigger never true — id set or board-read bug?")
    elif stats["o12_fires"] == 0:
        print("  NOTE: trigger true but zero fires — the demote only acts "
              "when a targeted draw ability is the model's top pick.")


if __name__ == "__main__":
    main()
