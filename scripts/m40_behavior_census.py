"""M40 deep-dive: behavior census over an arbitrary seat set.

live_postmortem.py generalized: instead of "one of OUR submissions", the seat
set is any filter over cached seats — so the same metric block can be computed
for (a) our live pilots, (b) 900+/1000+ pilots of OUR OWN deck (the band the
campaign wants to reach), and (c) the exact winner seats the champion was
fine-tuned on — and diffed. The diff between (a) and (b) is the concrete
"what do 1000-band pilots of this list do that ours does not".

Usage:
    # our live sub
    uv run python scripts/m40_behavior_census.py --subs 55172160 55182097
    # the aspiration band: any pilot of our list at >=1000
    uv run python scripts/m40_behavior_census.py --hash 9294d9d8 --min-score 1000
    # the training distribution: winner seats at 800+ (bc_m38_w9294's filter)
    uv run python scripts/m40_behavior_census.py --hash 9294d9d8 --min-score 800 --winners-only
"""
import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402
from cg.api import AreaType, OptionType, SelectContext, to_observation_class  # noqa: E402

from m39_live_mix import FAMILY_SIGNATURES, load_pokemon_names, opponent_deck  # noqa: E402
from rl.replay_bc import iter_replay_decisions  # noqa: E402
import rl.plan as rlplan  # noqa: E402

TELEPATH = 19
DUDUNSPARCE_IDS = set(getattr(rlplan, "DUDUNSPARCE_IDS", {1516, 1517}))
FEZ_ID = getattr(rlplan, "FEZANDIPITI_ID", 140)
ASH_ID = getattr(rlplan, "SACRED_ASH_ID", None)
POFFIN_ID = getattr(rlplan, "POFFIN_ID", None)


def family_of(pokemon_names: set) -> str:
    for fam, sigs in FAMILY_SIGNATURES:
        if any(s in pokemon_names for s in sigs):
            return fam
    return "other"


def load_cards():
    names, kinds, is_basic = {}, {}, set()
    with open(ROOT / "data/cards_features.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["card_id"])
            names[cid] = row["name"]
            for k in ("supporter", "item", "stadium", "tool"):
                if row.get(f"is_{k}") == "true":
                    kinds[cid] = k
            if row.get("is_basic") == "true":
                is_basic.add(cid)
    return names, kinds, is_basic


def seat_list(a) -> list[tuple[int, int, float, int]]:
    """[(episode_id, seat, score, won)] for the census."""
    if a.subs:
        df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet")
        out = []
        for r in df.iter_rows(named=True):
            for seat in (0, 1):
                if r[f"submission_id_{seat}"] in a.subs:
                    out.append((int(r["episode_id"]), seat,
                                r[f"updated_score_{seat}"] or 0.0, -1))
        return out
    df = pl.read_parquet(ROOT / "data/kaggle/opp_decks.parquet").filter(
        ~pl.col("is_ours").fill_null(False))
    out = []
    for r in df.iter_rows(named=True):
        if not r["deck_hash"].startswith(a.hash):
            continue
        if r["score"] is None or r["score"] < a.min_score:
            continue
        if a.winners_only and not r["won"]:
            continue
        out.append((int(r["episode_id"]), int(r["seat"]),
                    float(r["score"]), int(bool(r["won"]))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", type=int, nargs="+", default=None)
    ap.add_argument("--hash", default="9294d9d8")
    ap.add_argument("--min-score", type=float, default=1000.0)
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--max-seats", type=int, default=400)
    a = ap.parse_args()

    names, kinds, is_basic = load_cards()
    poke_names = load_pokemon_names()
    seats = seat_list(a)[: a.max_seats]
    label = (f"subs {a.subs}" if a.subs else
             f"hash {a.hash} score>={a.min_score:.0f}"
             + (" winners" if a.winners_only else ""))
    print(f"=== behavior census: {label}  ({len(seats)} seats) ===")

    fam_wl = defaultdict(lambda: [0, 0])
    n_games = wins = 0
    sup_turns, all_turns = set(), set()
    tele_opp = tele_att = 0
    bench_hist = Counter()
    dud_low = dud_low_used = fez_low = fez_low_used = 0
    end_n = end_item = 0
    offers = defaultdict(lambda: [0, 0])
    prizes_left_losses = []
    deck_end_losses = []
    turns_hist = []
    ash_deck_at_play = []

    for ep, seat, score, _ in seats:
        path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
        if not path.exists():
            continue
        raw = json.load(gzip.open(path))
        steps = raw["steps"]
        reward = raw["rewards"][seat] if raw.get("rewards") else None
        opp_deck = opponent_deck(steps, seat)
        fam = family_of({poke_names[c] for c in (opp_deck or [])
                         if c in poke_names})
        won = reward == 1
        if reward in (1, -1):
            n_games += 1
            wins += int(won)
            fam_wl[fam][0] += int(won)
            fam_wl[fam][1] += int(not won)
        drops = Counter()
        last = None
        for i, od, action in iter_replay_decisions(steps, seat, drops):
            obs = to_observation_class(od)
            st = obs.current
            if st is None or obs.select is None:
                continue
            last = st
            me = st.players[st.yourIndex]
            sel = obs.select
            if sel.context != SelectContext.MAIN:
                continue
            hand = me.hand or []
            nb = len([b for b in (me.bench or []) if b])
            bench_hist[nb] += 1
            all_turns.add((ep, seat, st.turn))
            opts = sel.option
            chosen = opts[action[0]]
            ct = OptionType(chosen.type)
            chosen_cid = None
            if ct in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
                idx = chosen.index
                if (chosen.area in (AreaType.HAND, None) and idx is not None
                        and idx < len(hand)):
                    chosen_cid = hand[idx].id
            offered = set()
            item_avail = False
            tele_here = False
            for o in opts:
                ot = OptionType(o.type)
                if ot in (OptionType.PLAY, OptionType.ATTACH) \
                        and o.index is not None \
                        and o.area in (AreaType.HAND, None) \
                        and o.index < len(hand):
                    cid = hand[o.index].id
                    if cid == TELEPATH:
                        tele_here = True
                    if ot == OptionType.PLAY and cid in kinds:
                        offered.add(cid)
                        if kinds[cid] == "item":
                            item_avail = True
            for cid in offered:
                offers[cid][0] += 1
            if chosen_cid in offered and ct == OptionType.PLAY:
                offers[chosen_cid][1] += 1
                if kinds.get(chosen_cid) == "supporter":
                    sup_turns.add((ep, seat, st.turn))
                if chosen_cid == ASH_ID:
                    ash_deck_at_play.append(me.deckCount)
            if tele_here:
                tele_opp += 1
                if chosen_cid == TELEPATH:
                    tele_att += 1
            if ct == OptionType.END:
                end_n += 1
                end_item += int(item_avail)
            if me.deckCount <= 6:
                for o in opts:
                    if OptionType(o.type) != OptionType.ABILITY:
                        continue
                    pok = None
                    if o.area == AreaType.ACTIVE and (me.active or []):
                        alive = [x for x in me.active if x]
                        pok = alive[0] if alive else None
                    elif o.area == AreaType.BENCH and o.index is not None \
                            and o.index < len(me.bench or []):
                        pok = (me.bench or [])[o.index]
                    if pok is None:
                        continue
                    hit = (ct == OptionType.ABILITY
                           and chosen.area == o.area
                           and chosen.index == o.index)
                    if pok.id in DUDUNSPARCE_IDS:
                        dud_low += 1
                        dud_low_used += int(hit)
                    elif pok.id == FEZ_ID:
                        fez_low += 1
                        fez_low_used += int(hit)
        if last is not None:
            turns_hist.append(last.turn)
            if reward == -1:
                me = last.players[last.yourIndex]
                opp = last.players[1 - last.yourIndex]
                prizes_left_losses.append((len(me.prize or []),
                                           len(opp.prize or [])))
                deck_end_losses.append(me.deckCount)

    print(f"\ngames {n_games}  WR {wins / max(n_games, 1):.3f}   "
          f"mean turns {sum(turns_hist) / max(len(turns_hist), 1):.1f}")
    print("\nby opponent family (W-L):")
    for fam, (w, l) in sorted(fam_wl.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {fam:12s} {w:3d}-{l:3d}  wr={w / max(w + l, 1):.2f}")
    tt = max(len(all_turns), 1)
    print(f"\nsupporter played on {len(sup_turns)}/{tt} turns "
          f"({len(sup_turns) / tt:.1%})")
    print(f"telepath attach rate {tele_att}/{tele_opp} "
          f"({tele_att / max(tele_opp, 1):.1%})")
    bt = sum(bench_hist.values())
    print("bench at MAIN prompts: "
          + "  ".join(f"{k}:{bench_hist[k] / max(bt, 1):.1%}"
                      for k in sorted(bench_hist)))
    print(f"END with item available: {end_item}/{end_n} "
          f"({end_item / max(end_n, 1):.1%})")
    print(f"dud ability at deck<=6: {dud_low_used}/{dud_low}   "
          f"fez at deck<=6: {fez_low_used}/{fez_low}")
    print(f"sacred ash deck-at-play: {sorted(ash_deck_at_play)[:20]}")
    if prizes_left_losses:
        mine = [m for m, _ in prizes_left_losses]
        print(f"\nLOSSES ({len(prizes_left_losses)}): own prizes left "
              f"mean {sum(mine) / len(mine):.1f}; "
              f"deck at end mean {sum(deck_end_losses) / len(deck_end_losses):.1f}; "
              f"deck-out losses {sum(1 for d in deck_end_losses if d == 0)}")
    print("\ntrainer offer/play rates (top 14 by offers):")
    for cid, (off, pl_) in sorted(offers.items(), key=lambda kv: -kv[1][0])[:14]:
        print(f"  {names.get(cid, cid):30s} {kinds.get(cid, '?'):9s} "
              f"{off:6d} offers  {pl_:5d} played  {pl_ / max(off, 1):6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
