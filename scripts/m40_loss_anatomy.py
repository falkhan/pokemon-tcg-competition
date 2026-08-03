"""M40 deep-dive part 3: anatomy of LIVE losses, grouped by opponent family.

For every loss across the given subs: who scored first and when, the prize
race trajectory, deck consumption on both sides, and hand size — enough to
tell a blowout (never in the game) from a race lost late, a disruption
lock (hand starved), or a deck-out. Complements live_family_forensics (who
beats us) with HOW they beat us.

Usage: uv run python scripts/m40_loss_anatomy.py --subs 55172160 55182097 55185485
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402
from cg.api import SelectContext, to_observation_class  # noqa: E402

from m39_live_mix import FAMILY_SIGNATURES, load_pokemon_names, opponent_deck  # noqa: E402
from rl.replay_bc import iter_replay_decisions  # noqa: E402


def family_of(pokemon_names: set) -> str:
    for fam, sigs in FAMILY_SIGNATURES:
        if any(s in pokemon_names for s in sigs):
            return fam
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", type=int, nargs="+", required=True)
    ap.add_argument("--result", type=int, default=-1, choices=(-1, 1),
                    help="-1 anatomize losses (default), 1 wins — the win "
                         "table calibrates whether first-KO initiative "
                         "decides games in both directions")
    a = ap.parse_args()
    poke_names = load_pokemon_names()
    df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet")

    rows = []
    for r in df.iter_rows(named=True):
        for seat in (0, 1):
            if r[f"submission_id_{seat}"] not in a.subs:
                continue
            ep = int(r["episode_id"])
            path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
            if not path.exists():
                continue
            raw = json.load(gzip.open(path))
            if not raw.get("rewards") or raw["rewards"][seat] != a.result:
                continue
            steps = raw["steps"]
            fam = family_of({poke_names[c]
                             for c in (opponent_deck(steps, seat) or [])
                             if c in poke_names})
            drops = Counter()
            first_our_ko = first_their_ko = None
            prizes_armed = False   # setup prompts carry EMPTY prize lists —
            last = None            # only count a shrink after both sat at 6
            hand_by_turn = {}
            for i, od, action in iter_replay_decisions(steps, seat, drops):
                obs = to_observation_class(od)
                st = obs.current
                if st is None:
                    continue
                last = st
                me = st.players[st.yourIndex]
                opp = st.players[1 - st.yourIndex]
                mp, op = len(me.prize or []), len(opp.prize or [])
                if not prizes_armed:
                    prizes_armed = mp == 6 and op == 6
                else:
                    if first_our_ko is None and mp < 6:
                        first_our_ko = st.turn
                    if first_their_ko is None and op < 6:
                        first_their_ko = st.turn
                hand_by_turn[st.turn] = len(me.hand or [])
            if last is None:
                continue
            me = last.players[last.yourIndex]
            opp = last.players[1 - last.yourIndex]
            mid_hand = [h for t, h in hand_by_turn.items()
                        if 4 <= t <= 10]
            rows.append(dict(
                fam=fam, ep=ep, turns=last.turn,
                our_kos=6 - len(me.prize or []),
                their_kos=6 - len(opp.prize or []),
                first_our_ko=first_our_ko, first_their_ko=first_their_ko,
                deck_end=me.deckCount, opp_deck_end=opp.deckCount,
                hand_mid=(sum(mid_hand) / len(mid_hand)) if mid_hand else None,
                hand_end=len(me.hand or []),
            ))

    kind = "losses" if a.result == -1 else "WINS"
    print(f"=== {len(rows)} {kind} across subs {a.subs} ===\n")
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["fam"]].append(r)
    print(f"{'family':12s}{'n':>3s}{'turns':>7s}{'ourKOs':>7s}{'thKOs':>7s}"
          f"{'1stKO us/them':>15s}{'deckEnd':>8s}{'oppDeck':>8s}"
          f"{'midHand':>8s}{'deckout':>8s}{'shutout':>8s}")
    for fam, rs in sorted(by_fam.items(), key=lambda kv: -len(kv[1])):
        n = len(rs)
        def m(k, d=0):
            vals = [r[k] for r in rs if r[k] is not None]
            return sum(vals) / len(vals) if vals else d
        deckouts = sum(1 for r in rs if r["deck_end"] == 0)
        shutouts = sum(1 for r in rs if r["our_kos"] <= 1)
        fko_us = [r["first_our_ko"] for r in rs
                  if r["first_our_ko"] is not None]
        fko_th = [r["first_their_ko"] for r in rs
                  if r["first_their_ko"] is not None]
        fko = (f"{sum(fko_us) / len(fko_us):4.1f}" if fko_us else " never") + \
              "/" + (f"{sum(fko_th) / len(fko_th):4.1f}" if fko_th else " ?")
        print(f"{fam:12s}{n:>3d}{m('turns'):>7.1f}{m('our_kos'):>7.1f}"
              f"{m('their_kos'):>7.1f}{fko:>15s}{m('deck_end'):>8.1f}"
              f"{m('opp_deck_end'):>8.1f}{m('hand_mid'):>8.1f}"
              f"{deckouts:>8d}{shutouts:>8d}")
    print("\nshutout = we took <=1 prize. deckout = our deck at 0.")
    print("first-KO 'never' = we never took a single prize in ANY loss of "
          "that family.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
