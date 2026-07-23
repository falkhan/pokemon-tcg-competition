"""Deck-drain forensics: attribute side A's deck consumption by source.

M30 P5 follow-up (Piotr's QC review: b2ga still loses by deck-out). Plays
side A vs side B on the direct engine loop and, for every observation of A,
diffs A's deckCount against the previous one, attributing the delta to the
action A chose in between (PLAY <card>, ABILITY <pokemon>, supporter, ...),
to the mandatory turn-start draw, or to the opponent phase (opponent mill /
triggered abilities — drain no override rule can touch). Reports the drain
table split by game outcome, plus prize/hand state at each deck-out loss
(decking while AHEAD on prizes = economy problem; while behind = strength).

Usage:
    uv run python scripts/deck_drain.py \
        --a modelt-guardash:checkpoints/m28_winners.pt:clone54618168 \
        --b model:checkpoints/m30_bc_rocket_54834745.pt:data/kaggle/rocket_3394cd30_deck.csv \
        -n 40 --seed 1
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import OptionType, SelectContext, to_observation_class
from rl.matchrunner import make_pilot, parse_spec
from rl.plan import _board_pokemon_id, _hand_card_id


def _card_names() -> dict[int, str]:
    with open(ROOT / "data/cards_features.csv") as f:
        return {int(r["card_id"]): r["name"] for r in csv.DictReader(f)}


NAMES = _card_names()


def _label(obs, pick: int) -> str:
    """Human-readable label for the chosen option at a MAIN prompt."""
    st, sel = obs.current, obs.select
    me = st.players[st.yourIndex]
    o = sel.option[pick]
    t = OptionType(o.type)
    if t in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
        cid = _hand_card_id(o, me.hand or [])
        return f"{t.name} {NAMES.get(cid, cid)}"
    if t == OptionType.ABILITY:
        cid = _board_pokemon_id(o, me)
        return f"ABILITY {NAMES.get(cid, cid)}"
    return t.name


class DrainTap:
    """Wraps A's pilot fn; on every prompt, attributes A deckCount deltas."""

    def __init__(self, fn):
        self.fn = fn
        self.reset_game()
        self.drain_by_label: dict[bool, Counter] = {True: Counter(),
                                                    False: Counter()}
        self.refill_by_label: dict[bool, Counter] = {True: Counter(),
                                                     False: Counter()}
        self.uses_by_label: dict[bool, Counter] = {True: Counter(),
                                                   False: Counter()}
        self.low_uses = Counter()      # MAIN labels chosen at deckCount <= 8

    def reset_game(self):
        self.last_deck = None
        self.last_turn = None
        self.pending = None            # label of A's last chosen action
        self.events = []               # (turn, delta, label) this game
        self.choices = Counter()       # MAIN labels chosen this game

    def flush_game(self, won: bool):
        for _, delta, label in self.events:
            if delta < 0:
                self.drain_by_label[won][label] += -delta
            else:
                self.refill_by_label[won][label] += delta
        self.uses_by_label[won].update(self.choices)

    def __call__(self, od):
        obs = to_observation_class(od)
        st = obs.current
        if st is not None:
            me = st.players[st.yourIndex]
            deck, turn = me.deckCount, st.turn
            if self.last_deck is not None and deck != self.last_deck:
                delta = deck - self.last_deck
                if turn != self.last_turn:
                    # own turn started since last prompt: mandatory draw first
                    if delta <= -1:
                        self.events.append((turn, -1, "turn draw"))
                        delta += 1
                    if delta:
                        self.events.append((turn, delta, "opponent phase"))
                else:
                    self.events.append((turn, delta,
                                        self.pending or "untracked"))
            self.last_deck, self.last_turn = deck, turn
        picks = self.fn(od)
        if (st is not None and obs.select is not None
                and obs.select.context == SelectContext.MAIN):
            self.pending = _label(obs, picks[0])
            self.choices[self.pending] += 1
            if st.players[st.yourIndex].deckCount <= 8:
                self.low_uses[self.pending] += 1
        return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="instrumented side (spec)")
    ap.add_argument("--b", required=True)
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import random

    from cg.game import battle_finish, battle_select, battle_start
    random.seed(args.seed)

    fn_a, deck_a = make_pilot(parse_spec(args.a), "drain_a")
    fn_b, deck_b = make_pilot(parse_spec(args.b), "drain_b")
    tap = DrainTap(fn_a)

    games = []
    for g in range(args.n):
        a_seat = g % 2
        fns = (tap, fn_b) if a_seat == 0 else (fn_b, tap)
        decks = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        tap.reset_game()
        obs_dict, start = battle_start(*decks)
        if start.errorPlayer >= 0:
            battle_finish()
            raise ValueError(f"battle_start rejected deck ({start.errorType})")
        try:
            while obs_dict["current"]["result"] < 0:
                seat = obs_dict["current"]["yourIndex"]
                picks = fns[seat](obs_dict)
                obs_dict = battle_select([int(i) for i in picks])
            cur = obs_dict["current"]
            players = cur["players"]
            me, opp = players[a_seat], players[1 - a_seat]
            won = cur["result"] == a_seat
            # RESULT log (type 23): reason 1=prizes, 2=deck-out, 3=benched,
            # 4=card effect — deck_end==0 alone overcounts deck-outs.
            reason = next((lg.get("reason") for lg in obs_dict.get("logs") or []
                           if lg.get("type") == 23), None)
            tap.flush_game(won)
            games.append(dict(
                won=won, draw=cur["result"] == 2, turns=cur.get("turn"),
                reason=reason,
                deck_end=me.get("deckCount"), opp_deck_end=opp.get("deckCount"),
                hand_end=len(me.get("hand") or []),
                my_prizes_left=len(me.get("prize") or []),
                opp_prizes_left=len(opp.get("prize") or [])))
        finally:
            battle_finish()

    n = len(games)
    wins = sum(g["won"] for g in games)
    losses = [g for g in games if not g["won"] and not g["draw"]]
    deckout = [g for g in losses if g["reason"] == 2]
    print(f"A={args.a}\nB={args.b}\n{n} games, {wins}W-{len(losses)}L; "
          f"TRUE deck-out losses (reason=2) {len(deckout)}/{len(losses)}; "
          f"deck_end==0 losses {sum(g['deck_end'] == 0 for g in losses)}")
    print("  loss reasons:", dict(Counter(g["reason"] for g in losses)))
    for won, tag in ((False, "LOST games"), (True, "WON games")):
        drain = tap.drain_by_label[won]
        refill = tap.refill_by_label[won]
        uses = tap.uses_by_label[won]
        total = sum(drain.values())
        print(f"\n  A deck drain in {tag} (total {total} cards):")
        for label, cards in drain.most_common(12):
            per = f" ({cards / uses[label]:.1f}/use, {uses[label]} uses)" \
                if uses[label] else ""
            print(f"    {cards:5d} ({cards / max(total, 1):5.1%})  "
                  f"{label}{per}")
        if refill:
            print(f"  refills in {tag}:", dict(refill.most_common(6)))
    print("\n  MAIN choices made at deckCount<=8 (all games):",
          dict(tap.low_uses.most_common(14)))
    if deckout:
        print("\n  TRUE deck-out losses (turns / hand at end / my-vs-opp "
              "prizes left / opp deck):")
        for g in deckout:
            ahead = ("AHEAD" if g["my_prizes_left"] < g["opp_prizes_left"]
                     else "behind/even")
            print(f"    t{g['turns']}: hand {g['hand_end']}, prizes "
                  f"{g['my_prizes_left']}v{g['opp_prizes_left']} ({ahead}), "
                  f"opp deck {g['opp_deck_end']}")


if __name__ == "__main__":
    main()
