"""Deck legality + mutation-bandit deck search (ARCHITECTURE.md §3.3, §5.3).

Same checks and search loops as the old ``rl/deck_search.py``. The legality
checker mirrors the one validated in deck_analysis.ipynb; the mutation
primitives and the self-play searches (M4) sit on top of it.

Seeded runs replay exactly: the ``random.*`` call sequences inside mutate /
mutate_flex / rate_population / hill_climb / evolve are part of the pinned
behavior — do not reorder them.
"""
import random
from collections import Counter
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DECK_SIZE = 60
MAX_COPIES = 4              # per card NAME, basic energy exempt
MUTATION_RETRY_LIMIT = 200  # attempts to hit a legal deck before giving up
MAX_SWAPS = 4               # mutate() swaps 1..MAX_SWAPS cards when n_swaps is None
MAX_FLEX_SWAPS = 3          # mutate_flex() swaps 1..MAX_FLEX_SWAPS flex slots

_cards = pl.read_parquet(DATA_DIR / "cards_features.parquet")
# One parquet row (as a plain dict) per card id: name/is_pokemon/is_basic/... flags.
CARD_ROWS: dict[int, dict] = {row["card_id"]: row
                              for row in _cards.iter_rows(named=True)}
ALL_CARD_IDS = list(CARD_ROWS)


def validate_deck(ids: list[int]) -> tuple[bool, list[str]]:
    """Check a deck against the construction rules. Returns (is_legal, reasons)."""
    reasons = []
    if len(ids) != DECK_SIZE:
        reasons.append(f"deck has {len(ids)} cards, must be exactly {DECK_SIZE}")

    unknown = [i for i in ids if i not in CARD_ROWS]
    if unknown:
        reasons.append(f"unknown card ids: {sorted(set(unknown))}")
    known = [i for i in ids if i in CARD_ROWS]

    copies = Counter(CARD_ROWS[i]["name"] for i in known
                     if not CARD_ROWS[i]["is_basic_energy"])
    over = {name: count for name, count in copies.items() if count > MAX_COPIES}
    if over:
        reasons.append(f">4 copies of: {over}")

    if not any(CARD_ROWS[i]["is_basic"] and CARD_ROWS[i]["is_pokemon"] for i in known):
        reasons.append("no Basic Pokémon (cannot legally start a game)")

    n_ace = sum(CARD_ROWS[i]["is_ace_spec"] for i in known)
    if n_ace > 1:
        reasons.append(f"{n_ace} ACE SPEC cards (max 1)")

    return (len(reasons) == 0, reasons)


def mutate(deck: list[int], n_swaps: int | None = None,
           candidate_weights: dict[int, float] | None = None) -> list[int]:
    """Swap 1-MAX_SWAPS random cards for new ones; always returns a *legal* deck.

    candidate_weights: optional card_id -> weight (e.g. learned card-impact
    scores) biasing which replacements get tried.
    """
    weights = None
    if candidate_weights:
        weights = [candidate_weights.get(i, 0.01) for i in ALL_CARD_IDS]

    for _ in range(MUTATION_RETRY_LIMIT):  # retry until legal
        candidate = list(deck)
        for _ in range(n_swaps or random.randint(1, MAX_SWAPS)):
            candidate[random.randrange(DECK_SIZE)] = \
                random.choices(ALL_CARD_IDS, weights=weights)[0]
        if validate_deck(candidate)[0]:
            return candidate
    raise RuntimeError(
        f"could not produce a legal mutation in {MUTATION_RETRY_LIMIT} tries")


def mutate_flex(deck: list[int], n_swaps: int | None = None) -> list[int]:
    """Archetype-preserving mutation: keep the Pokémon core fixed, swap only the
    non-Pokémon (trainer/energy) 'flex' slots for other legal non-Pokémon cards.
    This searches deck *builds* within an archetype (how real TCG deckbuilding works)
    instead of destroying the engine."""
    flex_pool = [i for i in ALL_CARD_IDS if not CARD_ROWS[i]["is_pokemon"]]
    flex_slots = [slot for slot, card_id in enumerate(deck)
                  if not CARD_ROWS[card_id]["is_pokemon"]]
    if not flex_slots:
        return list(deck)
    for _ in range(MUTATION_RETRY_LIMIT):
        candidate = list(deck)
        for _ in range(n_swaps or random.randint(1, MAX_FLEX_SWAPS)):
            candidate[random.choice(flex_slots)] = random.choice(flex_pool)
        if validate_deck(candidate)[0]:
            return candidate
    raise RuntimeError(
        f"could not produce a legal flex mutation in {MUTATION_RETRY_LIMIT} tries")


# --- Self-play deck evaluation + evolutionary search (M4) ---

def matchup(deck_a: list[int], deck_b: list[int], n_games: int = 12,
            agent: str = "lucario") -> list[int]:
    """Play deck_a vs deck_b, the SAME rule agent piloting both sides, slot-fair.
    Returns per-game results: 0 = A won, 1 = B won, 2 = draw. Direct engine loop (fast)."""
    from cg.game import battle_start, battle_select, battle_finish

    from tcg.teachers import load_teacher

    pilots = [load_teacher(f"dm_{agent}_0", agent=agent, deck=agent),
              load_teacher(f"dm_{agent}_1", agent=agent, deck=agent)]
    results = []
    for game in range(n_games):
        a_seat = game % 2                                # slot-fair
        deck_p0, deck_p1 = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        obs_dict, start = battle_start(deck_p0, deck_p1)
        if start.errorPlayer >= 0:
            battle_finish()
            raise ValueError(f"battle_start rejected a deck (errorType={start.errorType})")
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            obs_dict = battle_select([int(i) for i in pilots[seat](obs_dict)])
        result = obs_dict["current"]["result"]           # 0/1 winner seat, 2 draw
        battle_finish()
        if result == 2:
            results.append(2)
        else:                                            # map winner seat -> A/B
            a_won = (result == 0) if a_seat == 0 else (result == 1)
            results.append(0 if a_won else 1)
    return results


def rate_population(decks, n_rounds: int = 4, games_per_pair: int = 8,
                    agent: str = "lucario", seed: int = 0):
    """Random-pairing tournament with openskill ratings. Returns list of ordinals."""
    from openskill.models import PlackettLuce
    rng = random.Random(seed)
    model = PlackettLuce()
    ratings = [model.rating(name=str(i)) for i in range(len(decks))]
    indices = list(range(len(decks)))
    for _ in range(n_rounds):
        rng.shuffle(indices)
        for index_a, index_b in zip(indices[::2], indices[1::2]):
            for result in matchup(decks[index_a], decks[index_b],
                                  games_per_pair, agent):
                if result == 2:
                    continue
                win, lose = (index_a, index_b) if result == 0 else (index_b, index_a)
                rated = model.rate([[ratings[win]], [ratings[lose]]])  # winner first
                ratings[win], ratings[lose] = rated[0][0], rated[1][0]
    return [rating.ordinal() for rating in ratings]


def hill_climb(seed_deck: list[int], proposals: int = 50, games: int = 150,
               threshold: float = 0.57, agent: str = "lucario", seed: int = 0):
    """Robust local search that CANNOT regress: propose a flex mutation, play it vs the
    current champion over ``games`` (slot-fair), and accept only if it clears
    ``threshold`` (a clear win, not rating noise). Honest — reports if no improvement
    exists. Returns (champion_deck, n_accepted, log)."""
    rng = random.Random(seed)
    champion = list(seed_deck)
    accepted, log = 0, []
    for proposal_index in range(proposals):
        candidate = mutate_flex(champion, n_swaps=rng.randint(1, 2))
        results = matchup(candidate, champion, games, agent)
        wins_candidate = sum(1 for result in results if result == 0)
        wins_champion = sum(1 for result in results if result == 1)
        win_rate = wins_candidate / max(1, wins_candidate + wins_champion)
        if win_rate >= threshold:
            champion, accepted = candidate, accepted + 1
            log.append((proposal_index, round(win_rate, 3), "ACCEPT"))
            print(f"proposal {proposal_index}: wr {win_rate:.2f} vs champ "
                  f"-> ACCEPTED (#{accepted})", flush=True)
        else:
            log.append((proposal_index, round(win_rate, 3), "reject"))
    return champion, accepted, log


def evolve(seed_deck: list[int], pop_size: int = 12, generations: int = 6,
           agent: str = "lucario", seed: int = 0):
    """Mutation-bandit deck search: seed a population of flex-mutations, and each
    generation rate by self-play and replace the bottom half with mutations of the
    top half. Returns (best_deck, best_ordinal, history)."""
    rng = random.Random(seed)
    assert validate_deck(seed_deck)[0], "seed deck is illegal"
    population = [list(seed_deck)] + [mutate_flex(seed_deck)
                                      for _ in range(pop_size - 1)]
    history = []
    for generation in range(generations):
        ordinals = rate_population(population, agent=agent,
                                   seed=rng.randint(0, 1 << 30))
        order = sorted(range(len(population)), key=lambda k: ordinals[k], reverse=True)
        best_index = order[0]
        history.append((generation, round(ordinals[best_index], 2)))
        print(f"gen {generation}: best ordinal {ordinals[best_index]:.2f} "
              f"(deck {best_index})", flush=True)
        # keep top half, refill bottom half with mutations of the top
        keep = [population[k] for k in order[: pop_size // 2]]
        population = keep + [mutate_flex(rng.choice(keep))
                             for _ in range(pop_size - len(keep))]
    # final rating to pick the winner
    ordinals = rate_population(population, n_rounds=6, agent=agent,
                               seed=rng.randint(0, 1 << 30))
    best = max(range(len(population)), key=lambda k: ordinals[k])
    return population[best], round(ordinals[best], 2), history
