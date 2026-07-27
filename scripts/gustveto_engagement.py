"""O11 gustveto engagement probe (M36 P1 — the trigger is too rare offline
for a strength claim; this counts ENGAGEMENT instead, à la ability_probe).

Runs the CONTROL arm (modelt-gacf — veto off) and counts, at every MAIN
prompt: opponent-needs-<=1-prize states where a Boss's Orders PLAY is legal,
and how often the model's TOP pick is that gust — each such prompt is exactly
one decision the gacfv veto would flip. Reported per game and split by the
game's result.

Usage: uv run python scripts/gustveto_engagement.py \
           [--b model:checkpoints/m28_winners.pt:decks/archaludon.csv] \
           [-n 60] [--seed 41]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import OptionType, SelectContext, to_observation_class
from rl.plan import GUST_IDS, _hand_card_id
from rl.matchrunner import make_pilot, parse_spec, play_series
import rl.matchrunner as mr


class Tap:
    def __init__(self, fn):
        self.fn = fn
        self.game_events = 0          # this game's would-veto count
        self.per_game = []            # (events, opp_le1_gust_prompts)
        self.game_prompts = 0

    def reset_game(self):
        self.per_game.append((self.game_events, self.game_prompts))
        self.game_events = self.game_prompts = 0

    def __call__(self, od):
        picks = self.fn(od)
        obs = to_observation_class(od)
        st, sel = obs.current, obs.select
        if (st is None or sel is None
                or sel.context != SelectContext.MAIN or not picks):
            return picks
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        if len(op.prize or ()) > 1:
            return picks
        hand = me.hand or []
        gust_legal = any(
            o.type == OptionType.PLAY and _hand_card_id(o, hand) in GUST_IDS
            for o in sel.option)
        if not gust_legal:
            return picks
        self.game_prompts += 1
        top = sel.option[picks[0]]
        if top.type == OptionType.PLAY and _hand_card_id(top, hand) in GUST_IDS:
            self.game_events += 1
        return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="modelt-gacf:checkpoints/m28_winners.pt:"
                                   "decks/alakazam_v2.csv")
    ap.add_argument("--b", default="model:checkpoints/m28_winners.pt:"
                                   "decks/archaludon.csv")
    ap.add_argument("-n", "--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()

    spec_a, spec_b = parse_spec(args.a), parse_spec(args.b)
    fn_a, deck_a = make_pilot(spec_a, instance="gveng_a")
    tap = Tap(fn_a)

    def game_fn(fn0, fn1, deck0, deck1, stats, a_seat=None):
        from rl.matchrunner import _engine_game
        r = _engine_game(fn0, fn1, deck0, deck1, stats)
        tap.reset_game()
        return r

    orig = mr.make_pilot
    mr.make_pilot = lambda spec, instance: (
        (tap, deck_a) if spec == spec_a else orig(spec, instance))
    try:
        res = play_series(spec_a, spec_b, args.games, seed=args.seed,
                          game_fn=game_fn)
    finally:
        mr.make_pilot = orig

    w, l, d = res.count(0), res.count(1), res.count(2)
    print(f"series {w}W {l}L {d}D vs {args.b}")
    per_game = tap.per_game[:len(res)]
    events = sum(e for e, _ in per_game)
    prompts = sum(p for _, p in per_game)
    print(f"opp<=1 prompts with gust legal: {prompts} | "
          f"top-pick gust (would-veto): {events} "
          f"({events / max(prompts, 1):.1%})")
    ev_w = sum(e for (e, _), r in zip(per_game, res) if r == 0)
    ev_l = sum(e for (e, _), r in zip(per_game, res) if r == 1)
    print(f"would-veto events in WINS {ev_w} | in LOSSES {ev_l} | "
          f"games with >=1 event: "
          f"{sum(1 for e, _ in per_game if e)}/{len(per_game)}")


if __name__ == "__main__":
    main()
