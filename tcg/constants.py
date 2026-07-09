"""Every tuned constant of the rule-based pilot, named and in one place.

The score values encode carefully measured behavior (see docs/M6.md for the
experiments behind them) — renaming is fine, changing a value is a gameplay
change. The overall priority ladder, highest first:

    4000  attach that unblocks a KO by the active THIS turn
    3000  use an ability            (free, doesn't end the turn)
    2900  attach that unblocks a KO by a benched Pokémon / retreat into a lethal attacker
    2800  evolve                    (free, doesn't end the turn)
    2750  attach to my fastest closer in the active     (race math, M7.2b)
    2680  attach to my fastest closer on the bench      (above the KO tier: the
          attach doesn't end the turn — the KO fires on the re-prompt after it)
    2600  attach energy to the active attacker
    2500+ attack for a KO           (take the prize, then end the turn)
    2400  bench a Pokémon / attach energy on the bench
    2300+ chip attack in CLOSE MODE (opponent board is harmless: attack every
          turn instead of milling — the M6 floor-test self-deck fix, M7.2b)
    2200- play a trainer            (tapers with hand size — the anti-deck-out fix)
    1500  retreat to escape a KO
    1000+ chip attack               (develop first, attack last)
     ...  fallback priorities, ~0   pass / END

KO attacks sit deliberately BELOW free non-milling setup (ability / evolve /
attach-to-active) but ABOVE extra benching, bench attaches, and draw: once a
prize is on the table the pilot takes it and ends the turn instead of
over-developing, which draws cards and races its own deck to 0 (docs/M6.md,
"the self-deck bug").
"""
from cg.api import OptionType, SelectContext

# --- damage math -----------------------------------------------------------

COLORLESS = 0
"""EnergyType of a colorless cost slot — payable by any attached energy."""

WEAKNESS_MULTIPLIER = 2
RESISTANCE_REDUCTION = 30
UNREACHABLE_TURNS = 99
"""Race-math sentinel: this Pokémon can never KO (large int keeps min()/
comparisons in the scorers branch-free)."""

# --- free setup (never ends the turn) ---------------------------------------

SCORE_ABILITY = 3000
SCORE_EVOLVE = 2800

# --- attacking ---------------------------------------------------------------

SCORE_ATTACK_NO_TARGET = 1000  # opponent has no active: nothing to weigh, just attack
SCORE_KO_BASE = 2500           # a KO takes a prize NOW — close the game
KO_PRIZE_BONUS = 50            # ... and multi-prize KOs (ex / mega-ex) even more so
SCORE_CHIP_BASE = 1000         # no KO: chip damage stays low; develop first, attack last
CHIP_DAMAGE_DIVISOR = 10
SCORE_CHIP_CLOSE_BASE = 2300   # CLOSE MODE (M7.2b): opponent board is harmless -> attack
                               # every turn instead of milling; above trainers (<=2200),
                               # below play-pokemon/bench-attach 2400 (max chip ~2354)

# --- attaching energy --------------------------------------------------------

SCORE_ATTACH_UNBLOCKS_KO_ACTIVE = 4000  # active can cash the KO this turn: top priority
SCORE_ATTACH_UNBLOCKS_KO_BENCH = 2900
SCORE_ATTACH_RACE_CLOSER_ACTIVE = 2750  # M7.2b: my fastest closer (min turns-to-first-KO)
SCORE_ATTACH_RACE_CLOSER_BENCH = 2680   # ... on the bench: above attach-active AND the KO
                                        # tier — keep charging THE ONE attacker (the attach
                                        # doesn't end the turn; the KO fires on re-prompt)
SCORE_ATTACH_ACTIVE_BASE = 2600         # loading a real attacker that still needs energy
SCORE_ATTACH_BENCH_BASE = 2400
SCORE_ATTACH_ALREADY_LOADED = 600       # BEST damaging attack already charged (M7.2b: was
                                        # the cheapest — which stopped charging too early)
SCORE_ATTACH_NO_TARGET = 500            # couldn't resolve the target Pokémon
SCORE_ATTACH_NON_ATTACKER = 400         # don't waste energy on benchwarmers
ATTACH_DAMAGE_BONUS_CAP = 300           # tiny tiebreaker: prefer the harder hitter
ATTACH_DAMAGE_BONUS_DIVISOR = 100

# --- retreating --------------------------------------------------------------

SCORE_RETREAT_PROMOTE_LETHAL = 2900  # bench has a KO the active can't deliver
SCORE_RETREAT_ESCAPE_KO = 1500       # active would be KO'd; save it
SCORE_RETREAT_NEVER = -1             # healthy active or empty bench: stay put
SCORE_RETREAT_HURT_BASE = 100        # hurt active: mildly consider rotating out
RETREAT_BENCH_DAMAGE_DIVISOR = 20    # ... a bit more if a strong bench attacker wants in
HEALTHY_HP_FRACTION = 0.75

# --- playing cards from hand ---------------------------------------------------

SCORE_PLAY_UNRESOLVED_CARD = 2000  # can't tell what it is: playing is usually fine
SCORE_PLAY_POKEMON = 2400          # developing the board is always good
SCORE_PLAY_TRAINER_NO_BOARD = 300  # no Pokémon in play yet: trainers can't help
# Trainers are valued on NEED, not flat — the anti-deck-out fix (docs/M6.md).
# (Crude: can't yet tell a draw supporter from a gust/Switch — no effect-text
#  parsing — so it suppresses all trainers when the hand is full. Tech debt.)
DECKOUT_RESERVE_CARDS = 6          # near decking: stop thinning our own deck
SCORE_PLAY_NEAR_DECKOUT = 200
SCORE_TRAINER_BASE = 2200
TRAINER_HAND_TAPER = 200           # taper as the hand grows past COMFORTABLE_HAND_SIZE
COMFORTABLE_HAND_SIZE = 4
SCORE_TRAINER_FLOOR = 400

# --- card-selection contexts (score_card) --------------------------------------

USEFULNESS_POKEMON_BASE = 300  # attackers > energy > other, when fetching/keeping
ATTACKER_QUALITY_CAP = 300
USEFULNESS_ENERGY = 250
USEFULNESS_OTHER = 120
PROMOTE_READY_BONUS = 500      # promote a Pokémon that can damage the opponent NOW
PROMOTE_TURN_PENALTY = 50      # M7.2b race term: -50 per attach still needed ...
PROMOTE_TURNS_CAP = 4          # ... capped, so UNREACHABLE costs -200, not -4950
TARGET_PRIZE_WEIGHT = 100      # damage the highest-prize opponent Pokémon
SCORE_CARD_NEUTRAL = 50        # unknown card context: neutral

# Which card-selection contexts mean what to the pilot.
PROMOTE_CONTEXTS = frozenset({
    SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
    SelectContext.TO_ACTIVE, SelectContext.SWITCH, SelectContext.TO_FIELD,
    SelectContext.TO_BENCH, SelectContext.ATTACH_FROM,
})  # pick MY best Pokémon
KEEP_CONTEXTS = frozenset({
    SelectContext.TO_HAND, SelectContext.LOOK, SelectContext.NOT_MOVE,
})  # fetch/keep the most useful card
DISCARD_CONTEXTS = frozenset({
    SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
})  # lose the least useful card
TARGET_CONTEXTS = frozenset({
    SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
    SelectContext.DAMAGE_COUNTER_ANY, SelectContext.EFFECT_TARGET,
})  # hit the opponent where it hurts

# --- fallback ----------------------------------------------------------------

# Rough priority for option types with no dedicated scorer, so the pilot still
# develops, answers prompts, and doesn't just pass.
FALLBACK_PRIORITY: dict[OptionType, int] = {
    OptionType.ATTACK: 100, OptionType.ABILITY: 90, OptionType.EVOLVE: 80,
    OptionType.PLAY: 70, OptionType.ATTACH: 60, OptionType.CARD: 50,
    OptionType.YES: 40, OptionType.NUMBER: 30, OptionType.RETREAT: 20,
    OptionType.NO: 10, OptionType.END: 0,
}
