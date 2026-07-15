"""Deck legality + mutation-bandit deck search (ARCHITECTURE.md §3.3, §5.3).

Phase 0 uses a fixed deck; this module already provides the legality checker
(mirrors the one validated in deck_analysis.ipynb) and the mutation primitive
for Phase 1.
"""
import random
from collections import Counter
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DECK_SIZE = 60
MAX_COPIES = 4

_cards = pl.read_parquet(DATA_DIR / "cards_features.parquet")
_ft = {r["card_id"]: r for r in _cards.iter_rows(named=True)}
ALL_IDS = list(_ft)


def validate_deck(ids: list[int]) -> tuple[bool, list[str]]:
    """Check a deck against the construction rules. Returns (is_legal, reasons)."""
    reasons = []
    if len(ids) != DECK_SIZE:
        reasons.append(f"deck has {len(ids)} cards, must be exactly {DECK_SIZE}")

    unknown = [i for i in ids if i not in _ft]
    if unknown:
        reasons.append(f"unknown card ids: {sorted(set(unknown))}")
    known = [i for i in ids if i in _ft]

    copies = Counter(_ft[i]["name"] for i in known if not _ft[i]["is_basic_energy"])
    over = {n: c for n, c in copies.items() if c > MAX_COPIES}
    if over:
        reasons.append(f">4 copies of: {over}")

    if not any(_ft[i]["is_basic"] and _ft[i]["is_pokemon"] for i in known):
        reasons.append("no Basic Pokémon (cannot legally start a game)")

    n_ace = sum(_ft[i]["is_ace_spec"] for i in known)
    if n_ace > 1:
        reasons.append(f"{n_ace} ACE SPEC cards (max 1)")

    return (len(reasons) == 0, reasons)


def mutate(deck: list[int], n_swaps: int | None = None,
           candidate_weights: dict[int, float] | None = None) -> list[int]:
    """Swap 1-4 random cards for new ones; always returns a *legal* deck.

    candidate_weights: optional card_id -> weight (e.g. learned card-impact
    scores) biasing which replacements get tried.
    """
    weights = None
    if candidate_weights:
        weights = [candidate_weights.get(i, 0.01) for i in ALL_IDS]

    for _ in range(200):  # retry until legal
        d = list(deck)
        for _ in range(n_swaps or random.randint(1, 4)):
            d[random.randrange(DECK_SIZE)] = random.choices(ALL_IDS, weights=weights)[0]
        if validate_deck(d)[0]:
            return d
    raise RuntimeError("could not produce a legal mutation in 200 tries")


def mutate_flex(deck: list[int], n_swaps: int | None = None) -> list[int]:
    """Archetype-preserving mutation: keep the Pokémon core fixed, swap only the
    non-Pokémon (trainer/energy) 'flex' slots for other legal non-Pokémon cards.
    This searches deck *builds* within an archetype (how real TCG deckbuilding works)
    instead of destroying the engine."""
    flex_pool = [i for i in ALL_IDS if not _ft[i]["is_pokemon"]]
    flex_idx = [k for k, c in enumerate(deck) if not _ft[c]["is_pokemon"]]
    if not flex_idx:
        return list(deck)
    for _ in range(200):
        d = list(deck)
        for _ in range(n_swaps or random.randint(1, 3)):
            d[random.choice(flex_idx)] = random.choice(flex_pool)
        if validate_deck(d)[0]:
            return d
    raise RuntimeError("could not produce a legal flex mutation in 200 tries")


# --- Diverse-field fitness (M7.1, docs/M7-plan.md §2.4) ---

# The OpponentSpec vocabulary and battle loops moved to rl/matchrunner.py (M7.2,
# the canonical home). These wrappers keep this module's historic signatures.
OpponentSpec = tuple


def _resolve_deck(deck) -> list[int]:
    from rl.matchrunner import resolve_deck
    return resolve_deck(deck)


def _spec_pilot(spec: OpponentSpec, instance: str):
    """Build (agent_callable, deck_ids) for an opponent spec. [ENGINE]"""
    from rl.matchrunner import make_pilot
    return make_pilot(spec, instance)


def _play_vs_spec(deck: list[int], spec: OpponentSpec, n_games: int,
                  pilot: str = "generic") -> list[int]:
    """Play `deck` (piloted by `pilot`) vs one opponent spec, slot-fair.
    Returns per-game results: 0 = deck won, 1 = opponent, 2 = draw. [ENGINE]"""
    from rl.matchrunner import play_series
    mine = ("generic", deck) if pilot == "generic" else ("rule", pilot, deck)
    return play_series(mine, spec, n_games)


def field_hill_climb(seed_deck: list[int], field: list[OpponentSpec],
                     proposals: int = 50, games_per_opp: int = 30,
                     min_gain: float = 0.02, pilot: str = "generic", seed: int = 0,
                     fitness_fn=None):
    """Archetype-preserving hill-climb on FIELD fitness — the M7.1 refinement
    pass (M7-plan §2.4). Unlike hill_climb below (candidate vs champion mirror,
    the documented overfit trap), every proposal is scored against the full
    diverse field, so improvements must generalize. Accepts only clear gains
    (min_gain over the current champion's fitness, same n). Returns
    (champion_deck, champion_fitness, history). [ENGINE] unless fitness_fn
    is injected."""
    fitness_fn = fitness_fn or field_fitness
    rng = random.Random(seed)
    champ = list(seed_deck)
    best, _ = fitness_fn(champ, field, games_per_opp=games_per_opp, pilot=pilot)
    print(f"seed fitness: {best:.3f} (field of {len(field)}, "
          f"{games_per_opp} games/opp)", flush=True)
    history = [("seed", round(best, 4), "SEED")]
    for p in range(proposals):
        cand = mutate_flex(champ, n_swaps=rng.randint(1, 2))
        fit, _ = fitness_fn(cand, field, games_per_opp=games_per_opp, pilot=pilot)
        if fit >= best + min_gain:
            champ, best = cand, fit
            history.append((p, round(fit, 4), "ACCEPT"))
            print(f"proposal {p}: fitness {fit:.3f} -> ACCEPTED", flush=True)
        else:
            history.append((p, round(fit, 4), "reject"))
    return champ, best, history


def field_fitness(deck: list[int], field: list[OpponentSpec], games_per_opp: int = 60,
                  pilot: str = "generic", weights: list[float] | None = None,
                  play_fn=None) -> tuple[float, dict]:
    """Fitness vs a diverse FIXED field — never mirror-only (the documented
    mirror-overfit failure, DECISIONS.md 2026-07-08). Slot-fair vs each member.

    Returns (weighted mean win rate, per-opponent breakdown). Draws count as
    half a win. `play_fn(deck, spec, n_games, pilot)` is injectable for tests;
    the default engine loop is [ENGINE].
    """
    if weights is not None and len(weights) != len(field):
        raise ValueError(f"{len(weights)} weights for {len(field)} field members")
    play = play_fn or _play_vs_spec
    per_opp, total, wsum = {}, 0.0, 0.0
    for k, spec in enumerate(field):
        r = play(deck, spec, games_per_opp, pilot)
        w = sum(1 for x in r if x == 0)
        d = sum(1 for x in r if x == 2)
        wr = (w + 0.5 * d) / len(r)
        per_opp[str(spec)] = {"wr": round(wr, 4), "wins": w, "draws": d, "n": len(r)}
        weight = weights[k] if weights else 1.0
        total += weight * wr
        wsum += weight
    return total / wsum if wsum else 0.0, per_opp


# --- Self-play deck evaluation + evolutionary search (M4) ---

def matchup(deckA: list[int], deckB: list[int], n_games: int = 12,
            agent: str = "lucario") -> list[int]:
    """Play deckA vs deckB, the SAME rule agent piloting both sides, slot-fair.
    Returns per-game results: 0 = A won, 1 = B won, 2 = draw. Direct engine loop (fast)."""
    from rl.matchrunner import play_series
    return play_series(("rule", agent, deckA), ("rule", agent, deckB), n_games)


def rate_population(decks, n_rounds: int = 4, games_per_pair: int = 8,
                    agent: str = "lucario", seed: int = 0):
    """Random-pairing tournament with openskill ratings. Returns list of ordinals."""
    from openskill.models import PlackettLuce
    rng = random.Random(seed)
    model = PlackettLuce()
    ratings = [model.rating(name=str(i)) for i in range(len(decks))]
    idx = list(range(len(decks)))
    for _ in range(n_rounds):
        rng.shuffle(idx)
        for a, b in zip(idx[::2], idx[1::2]):
            for r in matchup(decks[a], decks[b], games_per_pair, agent):
                if r == 2:
                    continue
                win, lose = (a, b) if r == 0 else (b, a)
                res = model.rate([[ratings[win]], [ratings[lose]]])  # winner first
                ratings[win], ratings[lose] = res[0][0], res[1][0]
    return [r.ordinal() for r in ratings]


def hill_climb(seed_deck: list[int], proposals: int = 50, games: int = 150,
               threshold: float = 0.57, agent: str = "lucario", seed: int = 0):
    """Robust local search that CANNOT regress: propose a flex mutation, play it vs the
    current champion over `games` (slot-fair), and accept only if it clears `threshold`
    (a clear win, not rating noise). Honest — reports if no improvement exists.
    Returns (champion_deck, n_accepted, log)."""
    rng = random.Random(seed)
    champ = list(seed_deck)
    accepted, log = 0, []
    for p in range(proposals):
        cand = mutate_flex(champ, n_swaps=rng.randint(1, 2))
        r = matchup(cand, champ, games, agent)
        a = sum(1 for x in r if x == 0); b = sum(1 for x in r if x == 1)
        wr = a / max(1, a + b)
        if wr >= threshold:
            champ, accepted = cand, accepted + 1
            log.append((p, round(wr, 3), "ACCEPT"))
            print(f"proposal {p}: wr {wr:.2f} vs champ -> ACCEPTED (#{accepted})", flush=True)
        else:
            log.append((p, round(wr, 3), "reject"))
    return champ, accepted, log


def evolve(seed_deck: list[int], pop_size: int = 12, generations: int = 6,
           agent: str = "lucario", seed: int = 0):
    """Mutation-bandit deck search: seed a population of flex-mutations, and each
    generation rate by self-play and replace the bottom half with mutations of the
    top half. Returns (best_deck, best_ordinal, history)."""
    rng = random.Random(seed)
    assert validate_deck(seed_deck)[0], "seed deck is illegal"
    pop = [list(seed_deck)] + [mutate_flex(seed_deck) for _ in range(pop_size - 1)]
    history = []
    for gen in range(generations):
        ords = rate_population(pop, agent=agent, seed=rng.randint(0, 1 << 30))
        order = sorted(range(len(pop)), key=lambda k: ords[k], reverse=True)
        best_i = order[0]
        history.append((gen, round(ords[best_i], 2)))
        print(f"gen {gen}: best ordinal {ords[best_i]:.2f} (deck {best_i})", flush=True)
        # keep top half, refill bottom half with mutations of the top
        keep = [pop[k] for k in order[: pop_size // 2]]
        pop = keep + [mutate_flex(rng.choice(keep)) for _ in range(pop_size - len(keep))]
    # final rating to pick the winner
    ords = rate_population(pop, n_rounds=6, agent=agent, seed=rng.randint(0, 1 << 30))
    best = max(range(len(pop)), key=lambda k: ords[k])
    return pop[best], round(ords[best], 2), history


def _main() -> None:
    """CLI for the refinement pass (M7-plan §6): hill-climb a deck on the
    league's anchor field under the generic pilot. [ENGINE]

    Usage:  python -m rl.deck_search climb --deck decks/gen/deck_<hash>.csv
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("climb", help="field-fitness hill-climb (anchor field, generic pilot)")
    s.add_argument("--deck", required=True, help="decks/ name, csv path, or league hash")
    s.add_argument("--proposals", type=int, default=50)
    s.add_argument("--games-per-opp", type=int, default=30)
    s.add_argument("--min-gain", type=float, default=0.02)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=None,
                   help="output csv (default decks/gen/refined_<hash12>.csv)")
    a = p.parse_args()

    # Lazy: league imports validate_deck from this module at its top level.
    from rl.kaggle_ingest import _write_deck_csv, deck_hash
    from rl.league import _load_or_init, anchor_field
    from rl.matchrunner import resolve_deck

    seed_deck = resolve_deck(a.deck)
    field = anchor_field(_load_or_init())
    champ, best, history = field_hill_climb(
        seed_deck, field, proposals=a.proposals, games_per_opp=a.games_per_opp,
        min_gain=a.min_gain, seed=a.seed)
    accepted = sum(1 for _, _, tag in history if tag == "ACCEPT")
    out = a.out or (Path(__file__).resolve().parent.parent / "decks" / "gen"
                    / f"refined_{deck_hash(champ)[:12]}.csv")
    _write_deck_csv(out, champ)
    print(f"{accepted}/{a.proposals} accepted; final fitness {best:.3f} -> {out}",
          flush=True)


if __name__ == "__main__":
    _main()
