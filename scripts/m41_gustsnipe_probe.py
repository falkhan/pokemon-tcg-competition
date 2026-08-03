"""G-11 mechanism probe for O18 `gustsnipe`: does it fire, and on what?

docs/VALIDATION.md G-11: every ship needs a fire/behavioural-diff probe, and
"a ship whose mechanism cannot be probed is a ship whose live result cannot be
attributed." This counts, over real games:

  prompts        own MAIN prompts seen
  trigger_true   _gustsnipe_target(state) says a frail multi-prize body is benched
  gust_offered   ...AND a Boss's Orders PLAY was actually on the menu
  fires          ...AND the override changed the top pick

The gap between trigger_true and gust_offered is the honest ceiling: the rule
can only act when we are holding the card. A rule that triggers constantly but
never has the gust in hand is a dead lever, and that distinction is exactly
what `scripts/racemode_fire_probe.py` was built to expose.

Usage:
    uv run python scripts/m41_gustsnipe_probe.py \
        --arm model-pz-snipe:checkpoints/m41_ogerpon.pt:decks/ogerpon.csv \
        --opp model:checkpoints/m39_bc_grim.pt:grim_live -n 20
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cg.api import OptionType, SelectContext, to_observation_class  # noqa: E402

import rl.plan as rp  # noqa: E402
from rl.matchrunner import _engine_game, make_pilot, parse_spec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, help="spec carrying the gustsnipe fix")
    ap.add_argument("--opp", required=True)
    ap.add_argument("-n", "--games", type=int, default=20)
    a = ap.parse_args()

    stats = Counter()
    targets = Counter()
    orig = rp.apply_play_overrides

    def spy(obs, ranked, fixes):
        out = orig(obs, ranked, fixes)
        try:
            st = obs.current
            if st is not None and obs.select is not None \
                    and obs.select.context == SelectContext.MAIN:
                stats["prompts"] += 1
                if rp._gustsnipe_target(st):
                    stats["trigger_true"] += 1
                    me = st.players[st.yourIndex]
                    hand = me.hand or []
                    offered = [i for i in ranked
                               if obs.select.option[i].type == OptionType.PLAY
                               and rp._hand_card_id(obs.select.option[i], hand)
                               in rp.GUST_IDS]
                    if offered:
                        stats["gust_offered"] += 1
                        if out and out[0] != ranked[0]:
                            stats["fires"] += 1
                            op = st.players[1 - st.yourIndex]
                            for p in (op.bench or ()):
                                if p is None or p.id not in rp._CARD:
                                    continue
                                if rp._CARD[p.id][4] >= rp._GUSTSNIPE_MIN_PRIZES:
                                    targets[f"{rp._CARD[p.id][0] or p.id}"] += 1
        except Exception:  # noqa: BLE001 — a probe must never break the game
            pass
        return out

    # Patched around ONE make_pilot call so the counters bind to that pilot
    # alone (the m40_rule_probe pattern) — the opponent's overrides never count.
    rp.apply_play_overrides = spy
    try:
        fn_a, deck_a = make_pilot(parse_spec(a.arm), "probe_a")
    finally:
        rp.apply_play_overrides = orig
    fn_b, deck_b = make_pilot(parse_spec(a.opp), "probe_b")

    rp.apply_play_overrides = spy
    try:
        for g in range(a.games):
            _engine_game(fn_a, fn_b, deck_a, deck_b) if g % 2 == 0 else \
                _engine_game(fn_b, fn_a, deck_b, deck_a)
    finally:
        rp.apply_play_overrides = orig

    p = stats["prompts"] or 1
    print(f"gustsnipe probe: {a.games} games\n  arm {a.arm}\n  opp {a.opp}\n")
    print(f"  own MAIN prompts       {stats['prompts']:6d}")
    print(f"  trigger_true           {stats['trigger_true']:6d}  "
          f"({stats['trigger_true'] / p:5.1%} of prompts)")
    print(f"  ...gust also in hand   {stats['gust_offered']:6d}")
    print(f"  ...override FIRED      {stats['fires']:6d}")
    if stats["trigger_true"] and not stats["gust_offered"]:
        print("\n  DEAD LEVER: the board condition holds but we never hold the "
              "card. Fire rate is bounded by draw, not by the predicate.")
    if targets:
        print("\n  benched multi-prize targets present when it fired:")
        for name, n in targets.most_common(6):
            print(f"    {n:4d}  {name}")


if __name__ == "__main__":
    main()
