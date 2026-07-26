"""Live deck-race forensics for a shipped submission's cached kaggle replays.

Deck-out is the top live loss axis (M30 6/19, M31 9/26). `deck_drain.py` answers
"where do the cards go" on the OFFLINE engine loop; this answers the same on the
REAL replays, plus the question offline cannot see: are we losing the deck race
because we burn faster, or because the game goes long since we cannot close?

Per game it tracks our deckCount and the opponent's across our MAIN prompts,
attributes our own drain to the action taken between two consecutive
observations (turn-start draw vs the card/ability we chose vs opponent phase),
and reports prize progress at the deck-out losses (decking while level or ahead
on prizes = economy problem; while behind = strength problem).

Usage: uv run python scripts/live_deck_race.py <submission_id> [<submission_id> ...]
(refresh the cache first: uv run python -m rl.kaggle_ingest refresh --subs <id>)
"""
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
from cg.api import AreaType, OptionType, SelectContext, to_observation_class
from rl.replay_bc import iter_replay_decisions

names, is_pokemon = {}, set()
with open(ROOT / "data/cards_features.csv") as f:
    for row in csv.DictReader(f):
        cid = int(row["card_id"])
        names[cid] = row["name"]
        if row.get("is_pokemon") == "true":
            is_pokemon.add(cid)

EPISODES = pl.read_parquet(ROOT / "data/kaggle/episodes.parquet")


def alive(seq):
    return [x for x in (seq or []) if x]


def archetype(opp_deck):
    """Coarse archetype label from the opponent's 60-card deck list."""
    mons = sorted({names.get(c, str(c)) for c in (opp_deck or []) if c in is_pokemon})
    for key, label in (
        ("Alakazam", "alakazam(mirror)"), ("Mega Lucario", "lucario"),
        ("Dragapult ex", "dragapult"), ("Team Rocket'", "team_rocket"),
        ("Marnie's Gri", "marnie/froslass"), ("Cynthia's Ga", "cynthia_gabite"),
        ("Crustle", "crustle/tusk"), ("Archaludon e", "archaludon"),
        ("Hop's Phantu", "hop"), ("Mega Starmie", "starmie"),
        ("Mega Abomasn", "abomasnow"), ("Pikachu ex", "pikachu"),
    ):
        if any(m.startswith(key) for m in mons):
            return label
    return mons[0][:16] if mons else "unknown"


def analyse(sub):
    df = EPISODES.filter(
        (pl.col("submission_id_0") == sub) | (pl.col("submission_id_1") == sub)
    ).sort("episode_id")

    games = []
    drain_src = Counter()          # our deck drain attributed by action taken
    drain_src_loss = Counter()     # same, deck-out losses only

    for r in df.iter_rows(named=True):
        ep = int(r["episode_id"])
        seat = 0 if r["submission_id_0"] == sub else 1
        path = ROOT / f"data/kaggle/raw/episode_{ep}.json.gz"
        if not path.exists():
            continue
        raw = json.load(gzip.open(path))
        steps = raw["steps"]
        rew = raw.get("rewards") or [0, 0]
        won = rew[seat] > rew[1 - seat]

        opp_deck = None
        for step in steps[:4]:
            st = step[1 - seat] if 1 - seat < len(step) else None
            if isinstance(st, dict):
                act = st.get("action")
                if isinstance(act, list) and len(act) == 60:
                    opp_deck = [int(a) for a in act]
                    break

        drops = Counter()
        prev_deck = prev_label = None
        last = None
        first_deck = None
        turns = set()
        my_drain = Counter()
        for _i, obs_dict, action in iter_replay_decisions(steps, seat, drops):
            obs = to_observation_class(obs_dict)
            st = obs.current
            if st is None or obs.select is None:
                continue
            me = st.players[st.yourIndex]
            opp = st.players[1 - st.yourIndex]
            last = (me, opp, st.turn)
            turns.add(st.turn)
            if obs.select.context != SelectContext.MAIN:
                continue
            if first_deck is None:
                first_deck = me.deckCount

            # attribute the drain since the previous MAIN prompt to what we chose
            if prev_deck is not None and prev_label is not None:
                delta = prev_deck - me.deckCount
                if delta > 0:
                    my_drain[prev_label] += delta

            opts = obs.select.option
            chosen = opts[action[0]]
            ct = OptionType(chosen.type)
            hand = me.hand or []
            label = ct.name
            if ct in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
                idx = chosen.index
                if chosen.area in (AreaType.HAND, None) and idx is not None \
                        and idx < len(hand):
                    label = f"{ct.name} {names.get(hand[idx].id, hand[idx].id)}"
            elif ct == OptionType.ABILITY:
                pok = None
                if chosen.area == AreaType.ACTIVE and alive(me.active):
                    pok = alive(me.active)[0]
                elif chosen.area == AreaType.BENCH and chosen.index is not None \
                        and chosen.index < len(me.bench or []):
                    pok = (me.bench or [])[chosen.index]
                if pok is not None:
                    label = f"ABILITY {names.get(pok.id, pok.id)}"
            prev_deck, prev_label = me.deckCount, label

        if last is None:
            continue
        me, opp, turn = last
        n_turns = max(turns) if turns else turn
        deck_out = (not won) and me.deckCount == 0
        rec = dict(
            ep=ep, won=won, turns=n_turns, deck_out=deck_out,
            my_deck=me.deckCount, opp_deck_left=opp.deckCount,
            my_pz=len(me.prize or []), opp_pz=len(opp.prize or []),
            arch=archetype(opp_deck),
            my_burn=(first_deck - me.deckCount) / max(n_turns, 1) if first_deck else 0.0,
        )
        games.append(rec)
        for k, v in my_drain.items():
            drain_src[k] += v
            if deck_out:
                drain_src_loss[k] += v

    return games, drain_src, drain_src_loss


def report(sub):
    games, drain_src, drain_loss = analyse(sub)
    n = len(games)
    if not n:
        print(f"sub {sub}: no cached games")
        return
    wins = [g for g in games if g["won"]]
    losses = [g for g in games if not g["won"]]
    douts = [g for g in games if g["deck_out"]]
    print(f"\n===== sub {sub}: {n} games, {len(wins)}W-{len(losses)}L "
          f"| deck-out losses {len(douts)} "
          f"({len(douts)/n:.1%} of games, {len(douts)/max(len(losses),1):.1%} of losses) =====")

    def avg(rows, k):
        return sum(r[k] for r in rows) / len(rows) if rows else float("nan")

    print(f"  turns      win {avg(wins,'turns'):5.1f} | loss {avg(losses,'turns'):5.1f} "
          f"| deck-out {avg(douts,'turns'):5.1f}")
    print(f"  our burn/t win {avg(wins,'my_burn'):5.2f} | loss {avg(losses,'my_burn'):5.2f} "
          f"| deck-out {avg(douts,'my_burn'):5.2f}")
    print(f"  deck left at end: ours {avg(games,'my_deck'):5.1f} | opp {avg(games,'opp_deck_left'):5.1f}")

    print("\n  -- deck-out losses (prizes left: ours/opp; opp deck left) --")
    stalls = 0
    for g in sorted(douts, key=lambda x: -x["my_pz"]):
        tag = ""
        if g["my_pz"] >= g["opp_pz"]:
            tag = "  <- level-or-behind on prizes (economy, not tempo)"
        if g["my_pz"] == 6:
            tag = "  <- ZERO prizes taken (hard stall)"
            stalls += 1
        print(f"    ep{g['ep']} t={g['turns']:3d} pz {g['my_pz']}/{g['opp_pz']} "
              f"oppdeck={g['opp_deck_left']:2d} vs {g['arch']:18s}{tag}")
    print(f"  hard stalls (6 prizes left at deck-out): {stalls}/{len(douts)}")

    print("\n  -- W/L and deck-out by opponent archetype --")
    by = defaultdict(lambda: [0, 0, 0])
    for g in games:
        b = by[g["arch"]]
        b[0] += 1
        b[1] += g["won"]
        b[2] += g["deck_out"]
    for a, (tot, w, d) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"    {a:18s} n={tot:3d}  {w}W-{tot-w}L  wr={w/tot:.2f}  deck-outs={d}")

    print("\n  -- our deck drain by source (top 14; cards consumed) --")
    tot = sum(drain_src.values()) or 1
    tl = sum(drain_loss.values()) or 1
    print(f"    {'source':34s} {'all':>7s} {'%':>6s} {'deckout':>8s} {'%':>6s}")
    for k, v in drain_src.most_common(14):
        print(f"    {k[:34]:34s} {v:7d} {v/tot:6.1%} {drain_loss.get(k,0):8d} "
              f"{drain_loss.get(k,0)/tl:6.1%}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["54935640"]):
        report(int(s))
