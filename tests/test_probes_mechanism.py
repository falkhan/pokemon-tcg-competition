"""M41b II.3a: golden fixtures for the four mechanism probes' measurement cores.

The probes play in-process games or walk live replays, which the suite cannot
do — so what gets pinned here is each probe's MEASUREMENT CORE: the
per-prompt / per-event classification it applies to one decision, extracted
into importable functions in the script files (CLI behavior unchanged). Every
fixture is true BY CONSTRUCTION: a `fires` case built so the core MUST count
it, and a `guards` near-miss one condition away that it must NOT.

Cores under test:
  scripts/conserve_probe.py   make_counting        (conserve fire = order diff)
  scripts/m39_race_probe.py   make_counting/on_game (per-rule fire attribution)
  scripts/m40_gust_probe.py   GustTally.feed       (Boss latch + prize classify)
  scripts/m40_vveto_probe.py  run_series/mechanism_fired/main (stub plumbing)
"""
import sys
from collections import Counter

import pytest

pytest.importorskip("numpy")
pytest.importorskip("polars")

import rl.matchrunner as mr
import rl.plan as rp
from tests import builders as b
from tests.fake_cg import AreaType, OptionType, SelectContext

from scripts import conserve_probe as cp
from scripts import m39_race_probe as racep
from scripts import m40_gust_probe as gp
from scripts import m40_vveto_probe as vp

DUDUN = next(iter(rp.DUDUNSPARCE_IDS))       # 66 — the deckguard/racemode body
BOSS = next(iter(rp.GUST_IDS))               # 1182 — Boss's Orders
WALL_ID = 58                                 # Great Tusk: blanket branch
PRESSURE_ID = 646                            # Marnie's Impidimp: margin branch


def _ability_prompt(own_active_id, own_deck, opp_active_id=4, opp_deck=30):
    """A MAIN prompt whose menu is [our active's ABILITY, END] with the
    ability as the model's top pick — the draw-ability shape every demote
    rule keys on. Card 4 is a fake-pool Pokémon in no racemode id set."""
    me = b.player(active=b.pokemon(own_active_id), deck_count=own_deck)
    opp = b.player(active=b.pokemon(opp_active_id), deck_count=opp_deck)
    opts = [b.option(OptionType.ABILITY, area=AreaType.ACTIVE),
            b.option(OptionType.END)]
    return b.observation(me=me, opponent=opp, options=opts)


# ---------------------------------------------------------------------------
# scripts/conserve_probe.py — make_counting
# ---------------------------------------------------------------------------

class TestConserveProbeCore:
    def test_fires_demote_zone_with_the_ability_on_top(self):
        """Deck AT the floor and Fezandipiti's ability ranked first: the
        conserve demote MUST reorder, and the diff MUST count as a fire."""
        stats = Counter()
        counting = cp.make_counting(stats, rp.apply_play_overrides)
        obs = _ability_prompt(rp.FEZANDIPITI_ID, own_deck=rp._CONSERVE_AT)
        out = counting(obs, [0, 1], frozenset({rp.PLAY_FIX_CONSERVE}))
        assert out == [1, 0]                  # the demote really moved it
        assert stats["main_prompts"] == 1
        assert stats["trigger_true"] == 1
        assert stats["conserve_fires"] == 1
        assert stats["min_deck_seen"] == rp._CONSERVE_AT

    def test_guards_near_misses_in_the_zone_and_one_card_above_it(self):
        stats = Counter()
        counting = cp.make_counting(stats, rp.apply_play_overrides)
        # one card above the floor: the demote zone is never entered
        above = _ability_prompt(rp.FEZANDIPITI_ID,
                                own_deck=rp._CONSERVE_AT + 1)
        assert counting(above, [0, 1],
                        frozenset({rp.PLAY_FIX_CONSERVE})) == [0, 1]
        assert stats["trigger_true"] == 0 and stats["conserve_fires"] == 0
        # in the zone but END already on top: trigger WITHOUT a fire (the
        # demote only acts when the targeted ability is the model's top pick)
        zone = _ability_prompt(rp.FEZANDIPITI_ID, own_deck=rp._CONSERVE_AT)
        assert counting(zone, [1, 0],
                        frozenset({rp.PLAY_FIX_CONSERVE})) == [1, 0]
        assert stats["trigger_true"] == 1 and stats["conserve_fires"] == 0
        # in the zone with the fix ABSENT: the zone is counted, no fire diff
        assert counting(zone, [0, 1], frozenset()) == [0, 1]
        assert stats["trigger_true"] == 2 and stats["conserve_fires"] == 0
        assert stats["main_prompts"] == 3

    def test_guards_prompts_outside_main_are_invisible(self):
        stats = Counter()
        counting = cp.make_counting(stats, rp.apply_play_overrides)
        obs = _ability_prompt(rp.FEZANDIPITI_ID, own_deck=rp._CONSERVE_AT)
        obs.select.context = SelectContext.TO_HAND
        assert counting(obs, [0, 1],
                        frozenset({rp.PLAY_FIX_CONSERVE})) == [0, 1]
        assert stats["main_prompts"] == 0 and stats["conserve_fires"] == 0


# ---------------------------------------------------------------------------
# scripts/m39_race_probe.py — make_counting / make_on_game
# ---------------------------------------------------------------------------

class TestRaceProbeCore:
    def test_fires_wall_blanket_demote_attributed_to_racemode2_alone(self):
        """Great Tusk on the opponent's board, our Dudunsparce ability on top,
        deck fat enough (20) that conserve is inert: removing racemode2 — and
        ONLY racemode2 — changes the order, so the fire attributes to it."""
        stats, live = Counter(), {"own_min": 60, "opp_min": 60}
        counting = racep.make_counting(stats, live, rp.apply_play_overrides)
        obs = _ability_prompt(DUDUN, own_deck=20,
                              opp_active_id=WALL_ID, opp_deck=20)
        out = counting(obs, [0, 1], racep.PKG)
        assert out == [1, 0]
        assert stats["main_prompts"] == 1
        assert stats["race_engaged"] == 1
        assert stats[f"fire_{rp.PLAY_FIX_RACEMODE2}"] == 1
        for rule in (rp.PLAY_FIX_CONSERVE, rp.PLAY_FIX_RACEMODE4,
                     rp.PLAY_FIX_RACEASH):
            assert stats[f"fire_{rule}"] == 0
        assert live == {"own_min": 20, "opp_min": 20}

    def test_guards_pressure_family_without_the_margin_is_a_near_miss(self):
        """Grim family visible but the deck race is EVEN: the margin gate must
        hold — no engagement, no fires. The identical board with the margin
        met must engage, proving the gate is the only difference."""
        stats, live = Counter(), {"own_min": 60, "opp_min": 60}
        counting = racep.make_counting(stats, live, rp.apply_play_overrides)
        even = _ability_prompt(DUDUN, own_deck=20,
                               opp_active_id=PRESSURE_ID, opp_deck=20)
        assert counting(even, [0, 1], racep.PKG) == [0, 1]
        assert stats["main_prompts"] == 1
        assert stats["race_engaged"] == 0
        for rule in racep.RULES:
            assert stats[f"fire_{rule}"] == 0
        # the margin met (behind by 6 > _RACEMODE_MARGIN, inside the window)
        stats2, live2 = Counter(), {"own_min": 60, "opp_min": 60}
        counting2 = racep.make_counting(stats2, live2, rp.apply_play_overrides)
        behind = _ability_prompt(DUDUN, own_deck=14,
                                 opp_active_id=PRESSURE_ID, opp_deck=20)
        assert counting2(behind, [0, 1], racep.PKG) == [1, 0]
        assert stats2["race_engaged"] == 1
        assert stats2[f"fire_{rp.PLAY_FIX_RACEMODE2}"] == 1

    def test_on_game_folds_the_deck_race_minima_and_resets_them(self):
        stats, live = Counter(), {"own_min": 60, "opp_min": 60}
        counting = racep.make_counting(stats, live, rp.apply_play_overrides)
        on_game = racep.make_on_game(stats, live)
        # our deck hit ZERO at a MAIN prompt (END on top: no reorder noise)
        obs = _ability_prompt(DUDUN, own_deck=0,
                              opp_active_id=WALL_ID, opp_deck=3)
        counting(obs, [1, 0], racep.PKG)
        on_game(0, 0, None)
        assert stats["deckout_risk"] == 1 and stats["opp_deckout"] == 0
        assert stats["own_min_sum"] == 0 and stats["opp_min_sum"] == 3
        assert live == {"own_min": 60, "opp_min": 60}   # ready for game 2


# ---------------------------------------------------------------------------
# scripts/m40_gust_probe.py — GustTally.feed
# ---------------------------------------------------------------------------

PRIZES = {2: 2, 5: 1}                        # synthetic prizes_on_ko table
NAMES = {2: "Stub Water ex", 5: "Stub Chaff"}


def _play_prompt(hand_card_id):
    """A MAIN decision playing hand[0] — the arming half of the latch."""
    me = b.player(active=b.pokemon(1), hand=[b.hand_card(hand_card_id)])
    opp = b.player(active=b.pokemon(4, hp=300),
                   bench=[b.pokemon(2), b.pokemon(5)])
    opts = [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
            b.option(OptionType.END)]
    return b.observation(me=me, opponent=opp, options=opts)


def _target_select():
    """A pure opponent-bench CARD menu (area 5): bench[0] is worth 2 prizes,
    bench[1] is worth 1 — the classification half of the latch."""
    opp = b.player(active=b.pokemon(4, hp=300),
                   bench=[b.pokemon(2), b.pokemon(5)])
    opts = [b.option(OptionType.CARD, area=gp.GUST_TARGET_AREA, index=i,
                     player_index=1) for i in range(2)]
    return b.observation(me=b.player(active=b.pokemon(1)), opponent=opp,
                         options=opts)


class TestGustProbeCore:
    def test_fires_boss_play_arms_and_the_next_select_is_classified(self):
        tally = gp.GustTally(PRIZES, NAMES)
        tally.feed(_play_prompt(BOSS), [0])
        assert tally.pending                  # the latch armed
        tally.feed(_target_select(), [0])     # took the 2-prize body
        assert tally.gusts == 1
        assert tally.chosen_pv == [2] and tally.best_pv == [2]
        assert tally.took_big == 1 and tally.missed_big == 0
        assert tally.picked == Counter({"Stub Water ex": 1})
        assert not tally.pending              # latch spent

    def test_fires_passing_over_the_two_prize_body_counts_a_miss(self):
        tally = gp.GustTally(PRIZES, NAMES)
        tally.feed(_play_prompt(BOSS), [0])
        tally.feed(_target_select(), [1])     # took the 1-prize body instead
        assert tally.gusts == 1
        assert tally.chosen_pv == [1] and tally.best_pv == [2]
        assert tally.missed_big == 1 and tally.took_big == 0

    def test_guards_a_non_gust_play_never_arms_the_latch(self):
        tally = gp.GustTally(PRIZES, NAMES)
        tally.feed(_play_prompt(7), [0])      # a plain trainer, id 7
        assert not tally.pending
        tally.feed(_target_select(), [0])     # a look-alike select right after
        assert tally.gusts == 0

    def test_guards_the_latch_is_one_step_and_the_menu_shape_is_checked(self):
        tally = gp.GustTally(PRIZES, NAMES)
        tally.feed(_play_prompt(BOSS), [0])
        # the NEXT decision is not a pure opp-bench menu (one option targets
        # the ACTIVE area) -> not a gust target select, nothing counted
        mixed = _target_select()
        mixed.select.option[0].area = int(AreaType.ACTIVE)
        tally.feed(mixed, [1])
        assert tally.gusts == 0 and not tally.pending
        # and the spent latch does NOT reach a later, well-shaped select
        tally.feed(_target_select(), [0])
        assert tally.gusts == 0


# ---------------------------------------------------------------------------
# scripts/m40_vveto_probe.py — run_series / mechanism_fired / main plumbing
# ---------------------------------------------------------------------------

class TestVvetoProbeCore:
    def test_run_series_scores_arm_a_wins_across_the_seat_swap(self):
        fn_a, fn_b = object(), object()
        seats = []

        def arm_a_always_wins(fn0, fn1, deck0, deck1):
            seats.append((fn0, fn1, deck0, deck1))
            return 0 if fn0 is fn_a else 1

        wins = vp.run_series(fn_a, fn_b, "da", "db", 4,
                             engine=arm_a_always_wins)
        assert wins == 4                      # every game credited to arm A
        assert seats == [(fn_a, fn_b, "da", "db"), (fn_b, fn_a, "db", "da"),
                         (fn_a, fn_b, "da", "db"), (fn_b, fn_a, "db", "da")]

        def arm_a_never_wins(fn0, fn1, deck0, deck1):
            return 1 if fn0 is fn_a else 0

        assert vp.run_series(fn_a, fn_b, "da", "db", 4,
                             engine=arm_a_never_wins) == 0

    def _run_main(self, monkeypatch, engine, games=2):
        """main() end-to-end with stub pilots and a stub engine: the probe's
        own plumbing (arg parsing, series, stats readout, verdict) with the
        real engine and checkpoints replaced by fixtures."""
        fresh = {"prompts": 0, "opened": 0, "stepped": 0, "vetoes": 0,
                 "time_max": 0.0, "time_sum": 0.0}
        monkeypatch.setattr(mr, "VVETO_STATS", fresh)
        monkeypatch.setattr(mr, "make_pilot",
                            lambda spec, instance: (object(), [0] * 60))
        monkeypatch.setattr(mr, "_engine_game", engine)
        monkeypatch.setattr(sys, "argv",
                            ["m40_vveto_probe.py", "-n", str(games)])
        return vp.main()

    def test_fires_a_simulated_veto_gives_exit_0_and_the_fires_verdict(
            self, monkeypatch, capsys):
        def one_veto_per_game(fn0, fn1, deck0, deck1):
            mr.VVETO_STATS["prompts"] += 3
            mr.VVETO_STATS["opened"] += 3
            mr.VVETO_STATS["stepped"] += 6
            mr.VVETO_STATS["vetoes"] += 1
            mr.VVETO_STATS["time_sum"] += 0.03
            mr.VVETO_STATS["time_max"] = max(mr.VVETO_STATS["time_max"], 0.02)
            return 0

        assert self._run_main(monkeypatch, one_veto_per_game) == 0
        out = capsys.readouterr().out
        assert "mechanism FIRES" in out
        assert "VETOES 2" in out              # exactly one per stub game

    def test_guards_opened_but_never_vetoed_is_did_not_fire(
            self, monkeypatch, capsys):
        def opens_never_vetoes(fn0, fn1, deck0, deck1):
            mr.VVETO_STATS["prompts"] += 3
            mr.VVETO_STATS["opened"] += 3
            mr.VVETO_STATS["stepped"] += 6
            mr.VVETO_STATS["time_sum"] += 0.03
            return 1

        assert self._run_main(monkeypatch, opens_never_vetoes) == 1
        assert "DID NOT FIRE" in capsys.readouterr().out
        # the stale-counter guard: vetoes without an opened search is NOT a
        # fire (the vacuous-pass rule cuts both ways)
        assert not vp.mechanism_fired({"vetoes": 1, "opened": 0})
        assert vp.mechanism_fired({"vetoes": 1, "opened": 1})
