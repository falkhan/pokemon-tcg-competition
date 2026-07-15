"""L3 archetype-inferred determinization (M8.4a; M7-plan §3b L3).

MCTS failed partly on Snorlax-placeholder determinization (docs/M2.md,
DECISIONS.md): the search reasoned about an opponent holding 20 Snorlax. This
module replaces the fillers with a meta-informed guess: infer the opponent's
archetype from their REVEALED cards (board Pokémon + tools + discard — all
observable) by cosine over the pooled Pokémon-core FEAT (the representation
the 98% archetype probe validated, rl/kaggle_ingest._core_vec), then sample
their hidden hand/deck/prizes from the matched harvested decklist minus what
is already revealed.

This is an INSTRUMENT for the L4 go/no-go (docs/M8-plan.md M8.4): the
ground-truth harness plays local games where both decks are known and scores
top-1 archetype accuracy + remaining-pool Jaccard by turn. Gate: top-1 >= 0.9
by turn 4 on the meta decks. Failing kills L4 (determinization quality was
half the measured MCTS failure).

  uv run python -m rl.determinize harness --games-per-deck 8
"""
import argparse
import json
import random
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

from rl.kaggle_ingest import _core_vec, _ft

ROOT = Path(__file__).resolve().parent.parent
META_DIR = ROOT / "data" / "kaggle"

# The M2 fillers — kept as the last-resort fallback (nothing revealed yet AND
# no meta prior wanted, or a meta deck runs out of Basics for a hidden active).
FILLER_POKEMON = 1072
FILLER_ENERGY = 1

MetaDeck = namedtuple("MetaDeck", "archetype ids vec weight")


def load_meta(version: str = "meta_v1") -> list[MetaDeck]:
    """Meta snapshot -> MetaDecks with pooled-core vectors, heaviest first."""
    snap = META_DIR / version
    manifest = json.loads((snap / "manifest.json").read_text())
    decks = []
    for entry in manifest["decks"]:
        ids = [int(x) for x in (snap / entry["csv"]).read_text().split() if x.strip()]
        decks.append(MetaDeck(entry["archetype"], ids, _core_vec(ids),
                              float(entry.get("weight", 1.0))))
    return sorted(decks, key=lambda d: -d.weight)


def revealed_ids(obs) -> list[int]:
    """Card ids the OPPONENT has revealed: board Pokémon + attached tools +
    the discard pile. (Attached energies are EnergyTypes, not card ids, and
    stadium ownership is ambiguous — both deliberately skipped.)"""
    st = obs.current
    op = st.players[1 - st.yourIndex]
    ids = [c.id for c in op.discard if c is not None]
    board = [p for p in list(op.active) + list(op.bench) if p is not None]
    for p in board:
        ids.append(p.id)
        ids.extend(t.id for t in p.tools if t is not None)
    return ids


def infer_deck(obs, meta: list[MetaDeck]) -> MetaDeck:
    """Best-matching meta deck for the opponent's revealed cards.

    Containment FIRST: the fraction of revealed ids (with multiplicity —
    trainers and energies in the discard count, they are highly
    discriminative) the candidate list actually contains. Pooled-core cosine
    is only the tiebreak for novel variants: measured 2026-07-13, a lone
    support basic (Makuhita) is cosine-CLOSER to the wrong archetype because
    the pooled representation is dominated by the big attackers. Before any
    reveal the field prior wins: the heaviest meta deck."""
    revealed = revealed_ids(obs)
    if not revealed:
        return meta[0]
    mons = [i for i in revealed if _ft[i]["is_pokemon"]]
    vec = _core_vec(mons) if mons else None

    def containment(d: MetaDeck) -> float:
        pool = Counter(d.ids)
        hits = 0
        for cid in revealed:
            if pool[cid] > 0:
                pool[cid] -= 1
                hits += 1
        return hits / len(revealed)

    return max(meta, key=lambda d: (containment(d),
                                    float(vec @ d.vec) if vec is not None else 0.0,
                                    d.weight))


def remaining_pool(deck_ids: list[int], revealed: list[int]) -> list[int]:
    """The inferred decklist minus the revealed multiset (clamped at zero —
    a variant mismatch must not create negative copies)."""
    pool = Counter(deck_ids)
    for cid in revealed:
        if pool[cid] > 0:
            pool[cid] -= 1
    return [cid for cid, n in pool.items() for _ in range(n)]


def determinize_kwargs(obs, meta: list[MetaDeck], rng: random.Random) -> dict:
    """search_begin's opponent_* kwargs from the archetype guess: the hidden
    hand/deck/prizes are one shuffle of the inferred remaining pool, padded
    with FILLER_ENERGY / trimmed if the variant's size mismatches. A hidden
    active gets the pool's most-common Basic (engine requires a Basic)."""
    st = obs.current
    op = st.players[1 - st.yourIndex]
    guess = infer_deck(obs, meta)
    pool = remaining_pool(guess.ids, revealed_ids(obs))
    rng.shuffle(pool)

    need_active = len(op.active) > 0 and op.active[0] is None
    active = []
    if need_active:
        basics = [cid for cid in pool if _ft[cid]["is_pokemon"]
                  and _ft[cid].get("is_basic", True)]
        pick = (Counter(basics).most_common(1)[0][0] if basics else FILLER_POKEMON)
        if pick in pool:
            pool.remove(pick)
        active = [pick]

    need = op.handCount + op.deckCount + len(op.prize)
    if len(pool) < need:
        pool += [FILLER_ENERGY] * (need - len(pool))
    hand = pool[:op.handCount]
    prize = pool[op.handCount:op.handCount + len(op.prize)]
    deck = pool[op.handCount + len(op.prize):need]
    return {"opponent_hand": hand, "opponent_prize": prize,
            "opponent_deck": deck, "opponent_active": active}


# ---------------------------------------------------------------------------
# Ground-truth harness (M8.4a gate): local games, both decks known.
# ---------------------------------------------------------------------------
def harness(games_per_deck: int = 8, my_deck: str = "lucario",
            version: str = "meta_v1", seed: int = 61) -> dict:
    """Play generic-pilot games vs each meta deck; at every one of MY decisions
    score (a) top-1 archetype accuracy and (b) remaining-pool multiset Jaccard
    vs ground truth, bucketed by turn. Returns {turn: (acc, jaccard, n)}."""
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from rl.generic_pilot import make_generic_pilot
    from rl.matchrunner import resolve_deck

    meta = load_meta(version)
    mine = resolve_deck(my_deck)
    rng = random.Random(seed)
    by_turn: dict[int, list] = {}

    for md in meta:
        for g in range(games_per_deck):
            me_first = (g % 2 == 0)                    # slot-fair
            decks = [mine, md.ids] if me_first else [md.ids, mine]
            my_seat = 0 if me_first else 1
            pilots = [make_generic_pilot(decks[0]), make_generic_pilot(decks[1])]
            obs_dict, start = battle_start(decks[0], decks[1])
            if start.errorPlayer >= 0:
                raise ValueError(f"battle_start rejected deck {md.archetype}")
            while obs_dict["current"]["result"] < 0:
                player = obs_dict["current"]["yourIndex"]
                if player == my_seat:
                    obs = to_observation_class(obs_dict)
                    turn = min(int(obs.current.turn), 12)
                    guess = infer_deck(obs, meta)
                    true_pool = Counter(remaining_pool(md.ids, revealed_ids(obs)))
                    got_pool = Counter(remaining_pool(guess.ids, revealed_ids(obs)))
                    inter = sum((true_pool & got_pool).values())
                    union = sum((true_pool | got_pool).values())
                    by_turn.setdefault(turn, []).append(
                        (guess.archetype == md.archetype,
                         inter / union if union else 1.0))
                obs_dict = battle_select(
                    [int(i) for i in pilots[player](obs_dict)])
            battle_finish()

    out = {}
    print(f"{'turn':>4} {'top1':>6} {'jaccard':>8} {'n':>5}")
    for turn in sorted(by_turn):
        rows = by_turn[turn]
        acc = sum(r[0] for r in rows) / len(rows)
        jac = sum(r[1] for r in rows) / len(rows)
        out[turn] = (acc, jac, len(rows))
        print(f"{turn:>4} {acc:>6.3f} {jac:>8.3f} {len(rows):>5}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harness", help="ground-truth accuracy/Jaccard by turn")
    h.add_argument("--games-per-deck", type=int, default=8)
    h.add_argument("--my-deck", type=str, default="lucario")
    h.add_argument("--version", type=str, default="meta_v1")
    h.add_argument("--seed", type=int, default=61)
    a = p.parse_args()
    harness(a.games_per_deck, a.my_deck, a.version, a.seed)
