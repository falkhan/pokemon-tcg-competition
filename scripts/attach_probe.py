"""Focused probes: telepath-vs-basic attach choice, low-deck draw discipline,
per-turn energy attach rate, and loss reasons.

Usage: uv run python attach_probe.py <submission_id> [max_games]
"""
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from cg.api import AreaType, LogType, OptionType, SelectContext, \
    to_observation_class
from rl.replay_bc import iter_replay_decisions

TELEPATH = 19


def scan_game_decisions(decisions, names, energy_ids, both_avail, low_deck_draw):
    """One game's MAIN decisions -> per-turn manual-attach map {turn: attached}.

    Mutates both_avail (what was chosen when Telepath AND another energy were
    both attachable) and low_deck_draw (draw ability offered/taken at deck<=6).
    decisions: iterable of (converted observation, action) pairs.
    """
    turn_attached = {}
    for obs, action in decisions:
        st = obs.current
        if st is None or obs.select is None or obs.select.context != SelectContext.MAIN:
            continue
        me = st.players[st.yourIndex]
        hand = me.hand or []
        opts = obs.select.option
        chosen = opts[action[0]]
        turn_attached.setdefault(st.turn, False)

        def hand_cid(o):
            if o.index is not None and o.area in (AreaType.HAND, None) \
                    and o.index < len(hand):
                return hand[o.index].id
            return None

        attachable = set()
        dud_ability_offered = False
        for o in opts:
            ot = OptionType(o.type)
            if ot == OptionType.ATTACH:
                c = hand_cid(o)
                if c in energy_ids:
                    attachable.add(c)
            if ot == OptionType.ABILITY:
                dud_ability_offered = True

        ct = OptionType(chosen.type)
        cid = hand_cid(chosen) if ct in (OptionType.ATTACH, OptionType.PLAY,
                                         OptionType.EVOLVE) else None
        if ct == OptionType.ATTACH and cid in energy_ids:
            turn_attached[st.turn] = True

        has_basic = any(c != TELEPATH for c in attachable)
        if TELEPATH in attachable and has_basic:
            label = (f"ATTACH {names.get(cid, cid)}" if ct == OptionType.ATTACH
                     else ct.name)
            both_avail[label] += 1

        if me.deckCount <= 6 and dud_ability_offered:
            low_deck_draw["offered"] += 1
            if ct == OptionType.ABILITY:
                low_deck_draw["took_ability"] += 1
    return turn_attached


def count_loss_reasons(steps, loss_reasons):
    """Find the last step carrying RESULT logs and count every (result, reason)
    code on it (both seats' log windows — a lost game's exit)."""
    for step in reversed(steps):
        found = False
        for s in step:
            if not isinstance(s, dict):
                continue
            obs = (s.get("observation") or {})
            for lg in (obs.get("logs") or []):
                if lg.get("type") == int(LogType.RESULT):
                    loss_reasons[(lg.get("result"), lg.get("reason"))] += 1
                    found = True
        if found:
            break


def main(argv):
    sub = int(argv[0])
    max_games = int(argv[1]) if len(argv) > 1 else 60

    names = {}
    energy_ids = set()
    with open(ROOT / "data/cards_features.csv") as f:
        for row in csv.DictReader(f):
            cid = int(row["card_id"])
            names[cid] = row["name"]
            if row["is_basic_energy"] == "true" or row["is_special_energy"] == "true":
                energy_ids.add(cid)

    df = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet").filter(
        (pl.col("submission_id_0") == sub) | (pl.col("submission_id_1") == sub)
    ).sort("episode_id", descending=True)

    both_avail = Counter()      # what was chosen when telepath AND basic P both attachable
    low_deck_draw = Counter()   # dudunsparce ability chosen/offered while deck <= 6
    turns_with_attach = 0
    our_turns = 0
    loss_reasons = Counter()
    n_done = 0

    for r in df.iter_rows(named=True):
        if n_done >= max_games:
            break
        ep = int(r["episode_id"])
        seat = 0 if r["submission_id_0"] == sub else 1
        path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
        if not path.exists():
            continue
        raw = json.load(gzip.open(path))
        n_done += 1
        decisions = ((to_observation_class(o), a) for _, o, a
                     in iter_replay_decisions(raw["steps"], seat, Counter()))
        turn_attached = scan_game_decisions(decisions, names, energy_ids,
                                            both_avail, low_deck_draw)

        our_turns += len(turn_attached)
        turns_with_attach += sum(1 for v in turn_attached.values() if v)

        if raw["rewards"][seat] == -1:
            count_loss_reasons(raw["steps"], loss_reasons)

    print(f"sub {sub}, {n_done} games")
    print(f"  energy attached on {turns_with_attach}/{our_turns} of our turns "
          f"({turns_with_attach/max(our_turns,1):.1%})")
    print("  when telepath AND another energy both attachable, chose:")
    for k, v in both_avail.most_common(12):
        print(f"    {k:40s} {v}")
    print(f"  low-deck (<=6) dudunsparce-draw: {dict(low_deck_draw)}")
    print(f"  loss reason (result,reason) codes: {dict(loss_reasons)}")


if __name__ == "__main__":
    main(sys.argv[1:])
