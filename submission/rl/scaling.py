"""Effective damage for attacks whose printed damage is not their real damage.

`rl/combat.py::_ATK` stores an attack's PRINTED damage, which for a scaling
attack is its base — the number before "…for each Energy attached" or "…for
each card in your hand". 174 of the pool's 1556 attacks scale and 142 of those
print <=30, so the damage model reads them as harmless. The starkest case is our
own list: Alakazam #743 `Powerful Hand` prints **0** and really places 2 damage
counters per card in hand — 340 at the 17-card hands that deck holds.

This module is a curated table, in the spirit of `combat.SPREAD_ATTACKS` and
`combat.CONDITIONAL_ATTACKS`: free-form English rules text cannot be parsed
reliably, so the handful of attacks that decide real matchups are encoded by
hand against the printed card, and everything else falls through to the printed
number.

OPT-IN ONLY. Nothing imports this unless a caller asks for it — the shipped
agent's features stay byte-identical, which matters because the live net was
BC-trained against the current (wrong) features and demonstrably learned around
them: over 617 live turns with Alakazam active it attacked 617 times while
`_best_damage` said "cannot attack" every single time
(`scripts/m41_scaling_probe.py`). Changing what the net is fed without
retraining would be a silent distribution shift for no measured gain.

The consumer today is the deck probe, whose rule pilot has no such learned
compensation and therefore ranks our own win condition below a 90-damage
Dudunsparce.

Each entry is `attackId -> (mode, per_unit, base)`; `effective_damage` resolves
the mode against the board:

    "hand"       per_unit x cards in our hand
    "opp_nrg"    per_unit x energy attached to the DEFENDER
    "my_nrg"     per_unit x energy attached to the ATTACKER
    "both_nrg"   per_unit x energy on BOTH actives
    "my_bench"   per_unit x our Pokemon IN PLAY (bench + active)
    "bench_only" per_unit x our BENCHED Pokemon (active excluded)
    "team_nrg"   per_unit x energy attached across ALL our Pokemon

Every id below was read off the engine's own rules text, not guessed — a first
draft of this table guessed two of three ids and pointed at Trapinch and
Magcargo ex instead of the cards it named. `tests/test_scaling.py` re-derives
each id from `cg.api` so a card-pool update cannot silently re-point an entry.
"""
import re

from cg.api import all_attack as _all_attack

from rl.combat import _ATK

# attackId -> (mode, per_unit, base). CURATED entries: hand-read against the
# printed card, and authoritative — `derive_scaling_table` merges UNDER these,
# never over them. Two of the six encode a judgement the parser must not make
# (see _HUMAN_OVERRIDES below); the other four the parser re-derives exactly,
# which is what `tests/test_scaling_derive.py` pins.
CURATED_ATTACKS: dict[int, tuple[str, int, int]] = {
    # Alakazam #743 "Powerful Hand" — 2 damage counters per card in hand. OUR deck's plan.
    1072: ("hand", 20, 0),
    # Teal Mask Ogerpon ex "Myriad Leaf Shower" — +30 per Energy on BOTH actives.
    120: ("both_nrg", 30, 30),
    # Team Rocket's Spidops "Rocket Rush" — 30 per Team Rocket's Pokemon in play.
    # Approximated by our bench+active count: these lists are ~all Team Rocket's.
    560: ("my_bench", 30, 0),
    # Team Rocket's Mewtwo ex "Erasure Ball" — +60 per Energy discarded from our
    # bench (optional, up to 2). Modelled at the 1-discard middle case, +60.
    608: ("flat", 0, 220),
    # Dipplin #93 "Do the Wave" — 20 per BENCHED Pokemon (bench only, unlike
    # Spidops which counts everything in play). Its ability `Festival Lead` lets
    # it attack TWICE while Festival Grounds is out, so a full bench is 100 x2
    # per turn off a single {G}. 3.7% of the top 250 and climbing.
    115: ("bench_only", 20, 0),
    # Hydrapple ex #150 "Syrup Storm" — +30 per {G} Energy attached to ALL of
    # your Pokemon, not just the attacker. The Dipplin line's Stage 2.
    195: ("team_nrg", 30, 30),
}

# The two curated entries the parser deliberately CANNOT reproduce, with the
# judgement each one encodes. Anything here must be human-owned forever: a
# parser that reproduced them would be guessing, which is the failure mode
# every damage instrument in M41 hit.
_HUMAN_OVERRIDES = {
    560: "counts a NAMED SUBSET ('your Team Rocket's Pokemon in play') that "
         "the curated entry approximates as the whole board, on the deck fact "
         "that these lists are ~all Team Rocket's. Not derivable from text.",
    608: "scales on 'each card you discarded in this way' — a cost paid INSIDE "
         "the attack, so the unit is a choice we have not made yet at scoring "
         "time. The curated entry pins the 1-discard middle case.",
}

# Damage assumed for a scaling attack we have NOT curated. Enough to beat the
# 10-30 chip attacks of an evolution line's lower stages, low enough not to
# outrank a genuine printed heavy hitter. Only used when `assume_unknown`.
UNKNOWN_SCALING_DAMAGE = 100


# ---------------------------------------------------------------------------
# M42: the table, derived from the engine's own attack text
# ---------------------------------------------------------------------------
# 6 curated entries covered 6 of the pool's ~174 scaling attacks. The rest read
# as their printed base, which is <=30 for 142 of them. But the text is
# REGULAR, and every number in it is printed on the card:
#
#     "This attack does {N} more damage for each {X}."   -> base = printed
#     "This attack does {N} damage for each {X}."        -> base = 0
#     "Place {N} damage counters on your opponent's Active
#      Pokemon for each {X}."                            -> per_unit = 10*N
#
# So mode, per-unit AND base come out mechanically. This is extraction, not
# estimation — the distinction that matters, because every damage GUESS in this
# lane has been falsified (docs/M41.md).
#
# ALLOW-LIST BY CONSTRUCTION. A unit phrase is only accepted when it FULLY
# matches one of _UNIT_MODES below; anything qualified, named, conditional or
# self-referential produces no entry and falls through to printed damage,
# exactly as today. That is the safe direction: a missing entry under-rates a
# scaling attacker, which is the status quo, while a wrong entry invents damage.
#
# The one approximation inherited from the curated table: a TYPED energy
# qualifier ("{G} Energy attached to all of your Pokemon") maps to the untyped
# mode, so the estimate over-counts when off-type energy is attached. Syrup
# Storm's curated entry already made that call; the parser matches it rather
# than diverging.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Redirects: the damage does not land on the defender we are scoring against,
# so the number is not this attack's damage vs the active.
# Every alternative must carry its "to ..." preposition: an early draft also
# listed a bare "of your opponent's benched pok", which caught the Blizzard
# Burst spread it was aimed at AND the legitimate opp_bench unit of Ogre
# Comeback, silently dropping a real entry.
_REDIRECT = re.compile(
    r"to itself"
    r"|also does"
    r"|to 1 of your"
    r"|to each of your")

# "{N} (more) damage for each ..." — the main grammar.
_DOES = re.compile(r"this attack does (\d+) (more )?damage for each ([^.]*)")
# "Place {N} damage counters on your opponent's Active Pokemon for each ..."
# Alakazam's Powerful Hand, and the reason it prints 0.
_COUNTERS = re.compile(
    r"place (\d+) damage counters? on your opponent.s active pok[^ ]* "
    r"for each ([^.]*)")

# Unit phrase (fully matched, after normalisation) -> mode. Optional typed
# energy qualifiers are absorbed by `(?:\{[a-z]\} )?` / `(?:basic )?`.
_UNIT_MODES: tuple[tuple[str, str], ...] = (
    (r"card in your hand", "hand"),
    (r"card in your opponent.s hand", "opp_hand"),
    (r"(?:basic )?(?:\{[a-z]\} )?energy (?:card )?attached to this pok\w*",
     "my_nrg"),
    (r"(?:basic )?(?:\{[a-z]\} )?energy (?:card )?attached to your opponent.s "
     r"active pok\w*", "opp_nrg"),
    (r"(?:basic )?(?:\{[a-z]\} )?energy (?:card )?attached to both active "
     r"pok\w*", "both_nrg"),
    (r"(?:basic )?(?:\{[a-z]\} )?energy (?:card )?attached to all of your "
     r"pok\w*", "team_nrg"),
    (r"of your benched pok\w*", "bench_only"),
    (r"of your pok\w* in play", "my_bench"),
    (r"of your opponent.s benched pok\w*", "opp_bench"),
    (r"benched pok\w* \(both yours and your opponent.s\)", "all_bench"),
    (r"damage counter on this pok\w*", "dmg_counters_self"),
    (r"damage counter on your opponent.s active pok\w*", "dmg_counters_opp"),
    (r"prize card you have taken", "prizes_taken_us"),
    (r"prize card your opponent has taken", "prizes_taken_opp"),
)
_UNIT_RE = tuple((re.compile(p + r"\.?$"), m) for p, m in _UNIT_MODES)


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, and fold the curly apostrophe the card
    data actually uses (U+2019) so patterns can be written with a plain one."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower().replace("’", "'")


def _unit_mode(unit: str) -> str | None:
    unit = unit.strip().rstrip(".")
    for rx, mode in _UNIT_RE:
        if rx.fullmatch(unit):
            return mode
    return None


def parse_attack_text(text: str, printed: int) -> tuple[str, int, int] | None:
    """(mode, per_unit, base) for one attack's rules text, or None.

    Sentence by sentence, because a single attack can carry both a real
    scaling clause and an unrelated one: Voltage Burst is "does 50 more damage
    for each Prize card your opponent has taken. This Pokemon also does 30
    damage to itself." — the first sentence is the attack, the second is not.
    """
    for sentence in _SENTENCE_SPLIT.split(_normalise(text)):
        if "for each" not in sentence:
            continue
        m = _COUNTERS.search(sentence)
        if m and not _REDIRECT.search(sentence.replace("on your opponent's "
                                                       "active", "")):
            mode = _unit_mode(m.group(2))
            if mode:                       # 10 HP per damage counter
                return (mode, 10 * int(m.group(1)), 0)
            continue
        if _REDIRECT.search(sentence):
            continue
        m = _DOES.search(sentence)
        if not m:
            continue
        mode = _unit_mode(m.group(3))
        if mode is None:
            continue
        per_unit = int(m.group(1))
        base = printed if m.group(2) else 0     # "more" = on top of printed
        return (mode, per_unit, base)
    return None


def derive_scaling_table() -> dict[int, tuple[str, int, int]]:
    """Every attack whose scaling is mechanically readable, CURATED WINS.

    Derived at import from `cg.api` rather than baked into a literal so `rl/`
    and `submission/rl/` cannot drift and a card-pool refresh cannot leave a
    stale copy behind. `tests/test_scaling_derive.py` pins a snapshot of the
    result, so a pool change fails CI loudly instead of silently re-pointing
    entries — the trap `tests/test_scaling.py` was written for.
    """
    table: dict[int, tuple[str, int, int]] = {}
    for atk in _all_attack():
        entry = parse_attack_text(getattr(atk, "text", ""), atk.damage)
        if entry is not None:
            table[atk.attackId] = entry
    table.update(CURATED_ATTACKS)
    return table


SCALING_ATTACKS: dict[int, tuple[str, int, int]] = derive_scaling_table()


def _count_energy(pokemon) -> int:
    return len(getattr(pokemon, "energies", None) or [])


def _damage_counters(pokemon) -> int:
    """Damage counters on a Pokemon: (maxHp - hp) / 10, floored at 0."""
    hp = getattr(pokemon, "hp", 0) or 0
    max_hp = getattr(pokemon, "maxHp", 0) or hp
    return max(0, (max_hp - hp) // 10)


def effective_damage(attack_id: int, attacker, defender, hand_size: int = 0,
                     bench_size: int = 0, team_energy: int = 0,
                     opp_bench_size: int = 0, opp_hand_size: int = 0,
                     prizes_taken_us: int = 0,
                     prizes_taken_opp: int = 0) -> int:
    """Printed damage, or the scaling estimate when we have one.

    Returns the PRE-weakness number, exactly like `_ATK[id][0]`, so callers keep
    applying weakness and resistance themselves.

    Every counting argument DEFAULTS TO 0, so a caller that does not supply a
    unit gets the conservative floor (`max(printed, ...)`) rather than a wrong
    number. That is deliberate: `rl/encoders.py`'s energy block passes only the
    energy-derived units and must not have hand-mode attacks invent damage.
    """
    printed = _ATK.get(attack_id, (0, ()))[0]
    entry = SCALING_ATTACKS.get(attack_id)
    if entry is None:
        return printed
    mode, per_unit, base = entry
    units = {
        "flat": 0,
        "hand": hand_size,
        "opp_hand": opp_hand_size,
        "opp_nrg": _count_energy(defender),
        "my_nrg": _count_energy(attacker),
        "both_nrg": _count_energy(attacker) + _count_energy(defender),
        "my_bench": bench_size + 1,        # "in play" — the active counts too
        "bench_only": bench_size,          # "your Benched Pokemon" — it does not
        "opp_bench": opp_bench_size,
        "all_bench": bench_size + opp_bench_size,
        "team_nrg": team_energy,
        "dmg_counters_self": _damage_counters(attacker),
        "dmg_counters_opp": _damage_counters(defender),
        "prizes_taken_us": prizes_taken_us,
        "prizes_taken_opp": prizes_taken_opp,
    }.get(mode)
    if units is None:
        raise ValueError(f"unknown scaling mode {mode!r} for attack {attack_id}")
    return max(printed, base + per_unit * units)


def is_scaling(attack_id: int) -> bool:
    return attack_id in SCALING_ATTACKS


# A typical board, per mode, for ranking a card with NO board in hand. The
# fetch / promote / discard ladders in generic_pilot ask "how good an attacker
# is this card?" about cards in the DECK, where hand size and bench count are
# not knowable. These are deliberately mid-game, conservative values: they only
# have to put a scaling attacker in the right ORDER against flat ones, not
# predict its damage.
NOMINAL_UNITS = {
    "flat": 0,
    # M42: was 8, now the MEASURED mean hand at attack time
    # (`scripts/m41_scaling_probe.py`). 8 was a guess made before that number
    # existed and it under-rated our own win condition by more than half. The
    # other entries here are deliberately conservative mid-game values; this
    # one is not a guess to be conservative about, and shading it downward
    # would re-introduce exactly the bias M41 diagnosed.
    "hand": 17,
    "opp_hand": 5,
    "opp_nrg": 3,
    "my_nrg": 3,
    "both_nrg": 4,
    "my_bench": 5,      # a full board
    "bench_only": 4,    # a full bench
    "opp_bench": 4,
    "all_bench": 8,
    "team_nrg": 5,
    # A mid-game body that has been hit once or twice, not a fresh one: these
    # only have to ORDER a scaling attacker against flat ones.
    "dmg_counters_self": 4,
    "dmg_counters_opp": 4,
    "prizes_taken_us": 2,
    "prizes_taken_opp": 2,
}


def nominal_damage(attack_id: int) -> int:
    """Context-free effective damage, for ranking cards you cannot see a board for.

    Without this the ladders read PRINTED damage and rank every scaling
    attacker last: Dipplin's `Do the Wave` at 0 loses to its own bench filler
    Grookey at 30, and Alakazam at 0 loses to Kadabra at 30 and to a 90-damage
    Dudunsparce. That is how the rule pilot came to fetch Thwackey over the
    Applin its win condition evolves from.
    """
    printed = _ATK.get(attack_id, (0, ()))[0]
    entry = SCALING_ATTACKS.get(attack_id)
    if entry is None:
        return printed
    mode, per_unit, base = entry
    if mode not in NOMINAL_UNITS:
        raise ValueError(f"unknown scaling mode {mode!r} for attack {attack_id}")
    return max(printed, base + per_unit * NOMINAL_UNITS[mode])
