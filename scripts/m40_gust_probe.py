"""M40 deep-dive: is Boss's Orders TARGETING prize-aware?

The engineered stack is prize-blind (rl/combat.py never reads prizes_on_ko)
and the net only sees prize liability as one raw feature among many. This
measures the behavior directly: at every gust target select (the CARD/area-5
prompt following a Boss's Orders play), compare the prizes_on_ko of the
CHOSEN opponent bench target against the best available.

Run over any seat set (same filters as m40_behavior_census) so our live subs
can be diffed against 1000+ pilots of our own list.

Usage:
    uv run python scripts/m40_gust_probe.py --subs 55172160 55182097 55185485
    uv run python scripts/m40_gust_probe.py --hash 9294d9d8 --min-score 1000
"""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402
from cg.api import OptionType, to_observation_class  # noqa: E402

from m40_behavior_census import seat_list  # noqa: E402
from rl.plan import GUST_IDS, _hand_card_id  # noqa: E402
from rl.replay_bc import iter_replay_decisions  # noqa: E402

GUST_TARGET_AREA = 5   # opponent bench (verified on live replays, 2026-08-03)


def load_prizes():
    prizes, names = {}, {}
    with open(ROOT / "data/cards_features.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["card_id"])
            prizes[cid] = int(row["prizes_on_ko"] or 1)
            names[cid] = row["name"]
    return prizes, names


class GustTally:
    """The probe's measurement core, extracted (unchanged) from main() for the
    golden fixtures in tests/test_probes_mechanism.py (M41b II.3a).

    A ONE-decision latch: a Boss's Orders PLAY arms `pending`, and the very
    next decision — if it is a pure opponent-bench CARD menu (area 5) — is
    read as the gust target select, where the chosen target's prizes_on_ko is
    classified against the best available. `pending` must be reset to False at
    each episode boundary (the caller's job, as in main())."""

    def __init__(self, prizes, names):
        self.prizes, self.names = prizes, names
        self.pending = False
        self.gusts = 0
        self.chosen_pv, self.best_pv = [], []
        self.missed_big = 0     # a >=2-prize target was available, we took 1-prize
        self.took_big = 0
        self.picked = Counter()

    def feed(self, obs, action) -> None:
        st, sel = obs.current, obs.select
        if st is None or sel is None:
            return
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        hand = me.hand or []
        if self.pending:
            self.pending = False
            opts = sel.option
            if opts and all(OptionType(o.type) == OptionType.CARD
                            and o.area == GUST_TARGET_AREA for o in opts):
                bench = opp.bench or []

                def pv(o):
                    if o.index is None or o.index >= len(bench) \
                            or bench[o.index] is None:
                        return None
                    return self.prizes.get(bench[o.index].id, 1), \
                        bench[o.index].id
                cands = [pv(o) for o in opts]
                cands_ok = [c for c in cands if c is not None]
                ch = cands[action[0]] if action[0] < len(cands) else None
                if not cands_ok or ch is None:
                    return
                self.gusts += 1
                best = max(c[0] for c in cands_ok)
                self.chosen_pv.append(ch[0])
                self.best_pv.append(best)
                self.picked[self.names.get(ch[1], ch[1])] += 1
                if best >= 2:
                    if ch[0] >= 2:
                        self.took_big += 1
                    else:
                        self.missed_big += 1
            return
        ch = sel.option[action[0]]
        if (OptionType(ch.type) == OptionType.PLAY
                and ch.index is not None and ch.index < len(hand)
                and hand[ch.index].id in GUST_IDS):
            self.pending = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", type=int, nargs="+", default=None)
    ap.add_argument("--hash", default="9294d9d8")
    ap.add_argument("--min-score", type=float, default=1000.0)
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--max-seats", type=int, default=400)
    a = ap.parse_args()
    prizes, names = load_prizes()

    tally = GustTally(prizes, names)
    label = (f"subs {a.subs}" if a.subs
             else f"hash {a.hash} score>={a.min_score:.0f}")
    for ep, seat, _, _ in seat_list(a)[: a.max_seats]:
        path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
        if not path.exists():
            continue
        raw = json.load(gzip.open(path))
        tally.pending = False
        for i, od, action in iter_replay_decisions(raw["steps"], seat,
                                                   Counter()):
            tally.feed(to_observation_class(od), action)

    gusts, chosen_pv, best_pv = tally.gusts, tally.chosen_pv, tally.best_pv
    took_big, missed_big, picked = tally.took_big, tally.missed_big, tally.picked
    print(f"=== gust targeting: {label} ===")
    print(f"gust target selects analyzed: {gusts}")
    if gusts:
        print(f"mean prizes_on_ko: chosen {sum(chosen_pv) / gusts:.2f}   "
              f"best available {sum(best_pv) / gusts:.2f}")
        n_big = took_big + missed_big
        print(f"multi-prize target available: {n_big}/{gusts} "
              f"({n_big / gusts:.0%}) — taken {took_big}, "
              f"PASSED OVER {missed_big} "
              f"({took_big / max(n_big, 1):.0%} conversion)")
        print("targets picked:")
        for n, c in picked.most_common(10):
            print(f"  {c:3d}x {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
