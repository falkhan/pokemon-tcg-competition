"""M39 P1-inv: does `conserve` actually do anything, and does it prevent
deck-outs? The mechanism half of the conserve decision.

Follows scripts/racemode_fire_probe.py: wraps rl.plan.apply_play_overrides
with a counter and plays in-process (single-process => reproducible, unlike
matchrunner --workers 8, see docs/VALIDATION.md G-12), isolating actual
FIRES by diffing the same call with and without the fix.

Why a mechanism probe at all. The win-rate evidence on conserve is a
coin-flip: removing it cost -2.75pp (z=-1.91) on the 900+ bed and -0.29pp
(z=-0.22) on wall, and it was one of 12 contrasts tested, so p~0.056 is
what chance produces. But conserve's CONTRACT is specific and checkable
without statistics — it demotes the Fezandipiti/Dudunsparce draw ability
once our own deck is at or below _CONSERVE_AT, so we do not draw ourselves
out. Three things this can establish that a WR delta cannot:

  * fires == 0        -> the rule is inert; strip it, the WR wobble is noise
  * fires >> 0 and deck-out rate drops -> it is doing its job
  * fires >> 0 and deck-out rate flat  -> it fires but does not help

`self_deckouts` counts games where WE ran out of cards (our loss mode, 29%
of live losses); `opp_deckouts` is the mirror image and is reported so a
"fewer self deck-outs" result cannot be confused with "won the race".

Usage:
    uv run python scripts/conserve_probe.py --bed wall -n 40
    uv run python scripts/conserve_probe.py --bed top -n 40 --arm plain
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rl.plan as rp  # noqa: E402
from rl.matchrunner import parse_spec, play_series  # noqa: E402

NET = "checkpoints/m38_w9294_cont3.pt"
BEDS = {
    "wall": "model:checkpoints/m38_bc_wall.pt:greattusk_wall",
    "grim": "model:checkpoints/m39_bc_grim.pt:grim_live",
    "archaludon": "model:checkpoints/m39_bc_archaludon.pt:archaludon",
    "top": "model:checkpoints/m39_bc_top.pt:clone54618168",
    "mirror": "model:checkpoints/m28_winners.pt:clone54618168",
}
ARMS = {                      # the three Ship A candidates
    "plain": "model",
    "conserve": "model-conserve",
    "shipcfg": "modelt-gacfr3",
}
CONSERVE = frozenset({rp.PLAY_FIX_CONSERVE})


def make_counting(stats: Counter, orig):
    """The probe's measurement core, extracted (unchanged) so the golden
    fixtures in tests/test_probes_mechanism.py can pin it on constructed
    observations (M41b II.3a). Classifies ONE apply_play_overrides call:
    count the MAIN prompt, track the deck floor, and isolate a real FIRE by
    re-running the identical call without the conserve fix — the same diff
    that defines a fire in the module docstring."""
    def counting(obs, ranked, fixes):
        out = orig(obs, ranked, fixes)
        if obs.select is None or obs.current is None \
                or obs.select.context != rp.SelectContext.MAIN:
            return out
        stats["main_prompts"] += 1
        st = obs.current
        me = st.players[st.yourIndex]
        stats["min_deck_seen"] = min(stats.get("min_deck_seen", 99),
                                     me.deckCount)
        if me.deckCount <= rp._CONSERVE_AT:
            stats["trigger_true"] += 1          # own deck in the demote zone
            if fixes & CONSERVE and out != orig(obs, ranked, fixes - CONSERVE):
                stats["conserve_fires"] += 1    # the demote changed the order
        return out
    return counting


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", default="wall", choices=sorted(BEDS))
    ap.add_argument("--arm", default="conserve", choices=sorted(ARMS))
    ap.add_argument("-n", "--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    stats = Counter()
    orig = rp.apply_play_overrides
    rp.apply_play_overrides = make_counting(stats, orig)
    try:
        a = parse_spec(f"{ARMS[args.arm]}:{NET}:decks/alakazam_v2_h4.csv")
        b = parse_spec(BEDS[args.bed])
        results = play_series(a, b, args.games, seed=args.seed)
    finally:
        rp.apply_play_overrides = orig

    res = list(results["results"] if isinstance(results, dict) else results)
    w = res.count(0)
    print(f"\narm={args.arm} bed={args.bed} n={len(res)} "
          f"W-L {w}-{len(res) - w} (wr={w / max(len(res), 1):.3f})")
    for k in ("main_prompts", "trigger_true", "conserve_fires"):
        print(f"  {k:16s} {stats[k]}")
    print(f"  {'min_deck_seen':16s} {stats.get('min_deck_seen', '-')}"
          f"   (_CONSERVE_AT = {rp._CONSERVE_AT})")

    if args.arm != "plain" and stats["trigger_true"] and not stats["conserve_fires"]:
        print("  NOTE: deck reached the demote zone but the rule never fired "
              "— the demote only acts when a targeted draw ability is the "
              "model's top pick. An inert rule cannot be earning its slot.")
    if not stats["trigger_true"]:
        print("  WARNING: own deck never reached the demote zone in these "
              "games — this bed cannot exercise conserve at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
