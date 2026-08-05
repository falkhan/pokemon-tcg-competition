"""M41b § II.3a: golden fixtures for the four replay-analyzer probes.

Each probe's MEASUREMENT CORE — the per-decision classification it applies to
a replay decision stream — was extracted into importable functions (CLI and
printed output byte-identical, verified against captured runs) and is
exercised here on synthetic observations whose ground truth is true BY
CONSTRUCTION: a `fires` case the counter MUST count, and a `guards` near-miss
it must NOT. No engine, no parquet, no replay files; card pools are
monkeypatched module-bound, the test_postmortem way.

Cores under test:
  scripts/attach_probe.py        scan_game_decisions  (contested telepath
                                 attach, low-deck draw discipline, per-turn
                                 attach map), count_loss_reasons
  scripts/pm2_probe.py           scan_game_decisions  (Boss@opp-prizes rows,
                                 brick anatomy), boss_summary
  scripts/telepath_turn_probe.py scan_game, summarize_turns  (per-TURN
                                 offer/attach vs the diluted per-PROMPT rate)
  scripts/mirror_race_probe.py   is_mirror, board_id, scan_game, behind_uses,
                                 margin_bucket  (deck-race demote candidates)
"""
from collections import Counter, defaultdict

import pytest

from tests import builders as b
from tests.fake_cg import AreaType, LogType, OptionType, SelectContext

import scripts.attach_probe as attach_probe
import scripts.mirror_race_probe as mirror_probe
import scripts.pm2_probe as pm2_probe
import scripts.telepath_turn_probe as telepath_probe


# --- scripts/attach_probe.py — scan_game_decisions / count_loss_reasons -----

TELEPATH = attach_probe.TELEPATH        # 19, id-pinned card fact
WATER = 6                                # a second attachable energy
TOOL = 7                                 # ATTACH-able but NOT an energy
NAMES = {TELEPATH: "Telepath Psychic Energy", WATER: "Water Energy",
         TOOL: "Nest Ball"}
ENERGY_IDS = {TELEPATH, WATER}


def _attach_obs(hand_ids, options, deck=30, turn=2,
                context=SelectContext.MAIN):
    me = b.player(hand=[b.hand_card(i) for i in hand_ids], deck_count=deck)
    return b.observation(me=me, options=options, turn=turn, context=context)


def _attach_scan(decisions):
    both_avail, low_deck_draw = Counter(), Counter()
    turn_attached = attach_probe.scan_game_decisions(
        decisions, NAMES, ENERGY_IDS, both_avail, low_deck_draw)
    return turn_attached, both_avail, low_deck_draw


class TestAttachProbeContestedChoice:
    def test_fires_labels_the_choice_when_both_energies_attachable(self):
        # By construction both Telepath AND Water are ATTACH options from
        # hand, so BOTH decisions are contested: one labels the attached
        # card by name, the declined one labels the option type.
        menu = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                b.option(OptionType.ATTACH, area=AreaType.HAND, index=1),
                b.option(OptionType.END)]
        decisions = [(_attach_obs([TELEPATH, WATER], menu, turn=2), [1]),
                     (_attach_obs([TELEPATH, WATER], menu, turn=3), [2])]
        turn_attached, both_avail, _ = _attach_scan(decisions)
        assert both_avail == Counter({"ATTACH Water Energy": 1, "END": 1})
        assert turn_attached == {2: True, 3: False}

    def test_guards_solo_telepath_is_not_contested(self):
        # Telepath is the ONLY attachable energy -> no contested-choice row,
        # whatever gets chosen.
        menu = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                b.option(OptionType.PLAY, area=AreaType.HAND, index=1),
                b.option(OptionType.END)]
        _, both_avail, _ = _attach_scan(
            [(_attach_obs([TELEPATH, TOOL], menu), [0])])
        assert both_avail == Counter()

    def test_guards_attach_options_outside_hand_never_count(self):
        # An ATTACH pointing past the hand and one into DISCARD resolve to no
        # card -> Telepath stays uncontested.
        menu = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                b.option(OptionType.ATTACH, area=AreaType.HAND, index=5),
                b.option(OptionType.ATTACH, area=AreaType.DISCARD, index=0),
                b.option(OptionType.END)]
        _, both_avail, _ = _attach_scan([(_attach_obs([TELEPATH], menu), [3])])
        assert both_avail == Counter()


class TestAttachProbeTurnRateAndLowDeck:
    def test_fires_turn_counted_once_however_many_prompts(self):
        # Turn 2 has two MAIN prompts and the attach lands on the second ->
        # exactly one attached turn; turn 3 ends with no attach.
        menu = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                b.option(OptionType.END)]
        decisions = [(_attach_obs([WATER], menu, turn=2), [1]),
                     (_attach_obs([WATER], menu, turn=2), [0]),
                     (_attach_obs([WATER], menu, turn=3), [1])]
        turn_attached, _, _ = _attach_scan(decisions)
        assert turn_attached == {2: True, 3: False}

    def test_guards_tool_attach_and_non_main_prompts(self):
        # Attaching a non-energy is not a manual energy attach, and a
        # non-MAIN prompt contributes no turn at all.
        menu = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0)]
        decisions = [(_attach_obs([TOOL], menu, turn=4), [0]),
                     (_attach_obs([WATER], menu, turn=5,
                                  context=SelectContext.TO_HAND), [0])]
        turn_attached, _, _ = _attach_scan(decisions)
        assert turn_attached == {4: False}

    def test_fires_low_deck_draw_offered_and_taken(self):
        # deck <= 6 with the draw ability on the menu: both prompts count as
        # offered, only the first as taken.
        menu = [b.option(OptionType.ABILITY, area=AreaType.ACTIVE),
                b.option(OptionType.END)]
        decisions = [(_attach_obs([], menu, deck=6), [0]),
                     (_attach_obs([], menu, deck=6, turn=3), [1])]
        _, _, low_deck_draw = _attach_scan(decisions)
        assert low_deck_draw == Counter({"offered": 2, "took_ability": 1})

    def test_guards_deck_seven_or_no_ability_never_counts(self):
        ability_menu = [b.option(OptionType.ABILITY, area=AreaType.ACTIVE),
                        b.option(OptionType.END)]
        plain_menu = [b.option(OptionType.END)]
        decisions = [(_attach_obs([], ability_menu, deck=7), [0]),
                     (_attach_obs([], plain_menu, deck=6, turn=3), [0])]
        _, _, low_deck_draw = _attach_scan(decisions)
        assert low_deck_draw == Counter()


class TestAttachProbeLossReasons:
    def test_fires_counts_every_result_code_on_the_last_result_step(self):
        # Scans backwards: the FINAL step carrying RESULT logs is counted
        # (both seats' windows), the earlier RESULT is never reached.
        early = [{"observation": {"logs": [
            {"type": int(LogType.RESULT), "result": 1, "reason": 1}]}}, {}]
        quiet = [{}, {"observation": {"logs": []}}]
        last = [{"observation": {"logs": [
                    {"type": int(LogType.RESULT), "result": 2, "reason": 2}]}},
                {"observation": {"logs": [
                    {"type": int(LogType.RESULT), "result": 1, "reason": 2}]}}]
        loss_reasons = Counter()
        attach_probe.count_loss_reasons([early, quiet, last], loss_reasons)
        assert loss_reasons == Counter({(2, 2): 1, (1, 2): 1})

    def test_guards_other_log_types_and_malformed_steps(self):
        steps = [[None, 0,
                  {"observation": None},
                  {"observation": {"logs": [
                      {"type": int(LogType.DRAW)},
                      {"type": int(LogType.TURN_END)}]}}]]
        loss_reasons = Counter()
        attach_probe.count_loss_reasons(steps, loss_reasons)
        assert loss_reasons == Counter()


# --- scripts/pm2_probe.py — scan_game_decisions / boss_summary --------------

BOSS = pm2_probe.BOSS                    # 1182, Boss's Orders
ABRA, KADABRA = 60, 61


@pytest.fixture(autouse=True)
def _patch_pm2_cards(monkeypatch):
    monkeypatch.setattr(pm2_probe, "names",
                        {BOSS: "Boss's Orders", ABRA: "Abra",
                         KADABRA: "Kadabra"})
    monkeypatch.setattr(pm2_probe, "is_basic", {ABRA})


def _pm2_obs(hand_ids, options, opp_prizes=6, bench=(), turn=3,
             context=SelectContext.MAIN):
    me = b.player(hand=[b.hand_card(i) for i in hand_ids],
                  bench=[b.pokemon(i) if i else None for i in bench])
    opp = b.player(prizes_remaining=opp_prizes)
    return b.observation(me=me, opponent=opp, options=options, turn=turn,
                         context=context)


class TestPm2ProbeBossPlays:
    def test_fires_boss_play_records_opp_prizes_left(self):
        menu = [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                b.option(OptionType.END)]
        boss_rows, brick_rows = pm2_probe.scan_game_decisions(
            [(_pm2_obs([BOSS], menu, opp_prizes=1, turn=7), [0])],
            ep=101, reward=-1, log_brick=False)
        assert boss_rows == [(101, 7, 1, -1)]
        assert brick_rows == []

    def test_guards_near_misses_are_not_boss_plays(self):
        play_menu = [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                     b.option(OptionType.END)]
        cases = [
            # declined: END over the Boss in hand
            (_pm2_obs([BOSS], play_menu), [1]),
            # a different card played
            (_pm2_obs([ABRA], play_menu), [0]),
            # Boss id reached through an ATTACH option type
            (_pm2_obs([BOSS], [b.option(OptionType.ATTACH,
                                        area=AreaType.HAND, index=0)]), [0]),
            # PLAY whose index does not resolve inside the hand
            (_pm2_obs([BOSS], [b.option(OptionType.PLAY,
                                        area=AreaType.HAND, index=2)]), [0]),
            # a non-MAIN prompt is skipped outright
            (_pm2_obs([BOSS], play_menu, context=SelectContext.TO_HAND), [0]),
        ]
        boss_rows, _ = pm2_probe.scan_game_decisions(
            cases, ep=102, reward=1, log_brick=False)
        assert boss_rows == []


class TestPm2ProbeBrickAnatomy:
    def test_fires_brick_rows_capture_the_turn_anatomy(self):
        # bench_n counts only occupied slots; basics-in-hand resolves names
        # through is_basic; the chosen column is the option type's str tail
        # ("14"/"7" on 3.11+, where IntEnum str is the bare value).
        menu = [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                b.option(OptionType.END)]
        decisions = [(_pm2_obs([ABRA, KADABRA], menu, bench=(KADABRA, None),
                               turn=3), [1]),
                     (_pm2_obs([BOSS], menu, opp_prizes=4, turn=4), [0])]
        boss_rows, brick_rows = pm2_probe.scan_game_decisions(
            decisions, ep=103, reward=1, log_brick=True)
        assert brick_rows == [(3, 1, 2, ["Abra"], "14"),
                              (4, 0, 1, [], "7")]
        # brick logging must not suppress the Boss census on the same game
        assert boss_rows == [(103, 4, 4, 1)]

    def test_guards_no_brick_rows_for_unflagged_episodes(self):
        menu = [b.option(OptionType.END)]
        _, brick_rows = pm2_probe.scan_game_decisions(
            [(_pm2_obs([ABRA], menu), [0])], ep=104, reward=1,
            log_brick=False)
        assert brick_rows == []


class TestPm2ProbeBossSummary:
    def test_fires_wl_labels_and_low_prize_misfires(self):
        rows = [(1, 5, 2, 1), (2, 6, 1, -1), (3, 7, 0, 1), (4, 8, 6, -1)]
        c, low = pm2_probe.boss_summary(rows)
        assert c == Counter({(2, "W"): 1, (1, "L"): 1, (0, "W"): 1,
                             (6, "L"): 1})
        # only opp_pz <= 1 is a misfire candidate; pz=2 stays out
        assert low == [(2, 6, 1, -1), (3, 7, 0, 1)]


# --- scripts/telepath_turn_probe.py — scan_game / summarize_turns -----------

BASIC_MON = 11                           # a PLAY-able basic (benchfloor case)


@pytest.fixture(autouse=True)
def _patch_telepath_cards(monkeypatch):
    monkeypatch.setattr(telepath_probe, "TELEPATH_ID", TELEPATH)
    monkeypatch.setattr(telepath_probe, "_IS_ENERGY", {WATER})


def _tp_obs(hand_ids, options, turn=2, context=SelectContext.MAIN):
    me = b.player(hand=[b.hand_card(i) for i in hand_ids])
    return b.observation(me=me, options=options, turn=turn, context=context)


class TestTelepathTurnProbe:
    def test_fires_per_turn_undilutes_the_per_prompt_rate(self):
        # The M31 artifact by construction: the benchfloor PLAY prompt adds a
        # declined per-PROMPT offer, but the turn still attaches Telepath ->
        # per-PROMPT reads 1/2 while per-TURN reads 1/1.
        first = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                 b.option(OptionType.PLAY, area=AreaType.HAND, index=1),
                 b.option(OptionType.END)]
        second = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                  b.option(OptionType.END)]
        turns = defaultdict(telepath_probe._new_turn)
        opps, att = telepath_probe.scan_game(
            [(_tp_obs([TELEPATH, BASIC_MON], first, turn=2), [1]),
             (_tp_obs([TELEPATH], second, turn=2), [0])], ep=1, turns=turns)
        assert (opps, att) == (2, 1)
        assert telepath_probe.summarize_turns(turns) == (1, 1, 0, 0)

    def test_fires_other_energy_decline_and_true_no_attach(self):
        contested = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                     b.option(OptionType.ATTACH, area=AreaType.HAND, index=1),
                     b.option(OptionType.END)]
        solo = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                b.option(OptionType.END)]
        turns = defaultdict(telepath_probe._new_turn)
        opps, att = telepath_probe.scan_game(
            [(_tp_obs([TELEPATH, WATER], contested, turn=3), [1]),
             (_tp_obs([TELEPATH], solo, turn=4), [1])], ep=2, turns=turns)
        assert (opps, att) == (2, 0)
        # offered=2: one other-energy decline, one true no-attach failure
        assert telepath_probe.summarize_turns(turns) == (2, 0, 1, 1)

    def test_fires_telepath_wins_the_turn_over_an_earlier_other_energy(self):
        menu = [b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                b.option(OptionType.ATTACH, area=AreaType.HAND, index=1),
                b.option(OptionType.END)]
        turns = defaultdict(telepath_probe._new_turn)
        telepath_probe.scan_game(
            [(_tp_obs([TELEPATH, WATER], menu, turn=5), [1]),
             (_tp_obs([TELEPATH, WATER], menu, turn=5), [0])],
            ep=3, turns=turns)
        assert telepath_probe.summarize_turns(turns) == (1, 1, 0, 0)

    def test_guards_offer_requires_a_real_attach_from_hand(self):
        # Four near-miss menus, one per turn: Telepath PLAY-able but not
        # ATTACH-able, an index-less ATTACH, an ATTACH past the hand, and an
        # ATTACH from DISCARD. None is an offer.
        cases = [
            [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
             b.option(OptionType.END)],
            [b.option(OptionType.ATTACH, area=AreaType.HAND),
             b.option(OptionType.END)],
            [b.option(OptionType.ATTACH, area=AreaType.HAND, index=3),
             b.option(OptionType.END)],
            [b.option(OptionType.ATTACH, area=AreaType.DISCARD, index=0),
             b.option(OptionType.END)],
        ]
        turns = defaultdict(telepath_probe._new_turn)
        opps, att = telepath_probe.scan_game(
            [(_tp_obs([TELEPATH], menu, turn=t), [len(menu) - 1])
             for t, menu in enumerate(cases, start=2)], ep=4, turns=turns)
        assert (opps, att) == (0, 0)
        assert telepath_probe.summarize_turns(turns) == (0, 0, 0, 0)

    def test_guards_non_main_prompts_and_unoffered_turns_stay_out(self):
        attach_tele = [b.option(OptionType.ATTACH, area=AreaType.HAND,
                                index=0)]
        attach_water = [b.option(OptionType.ATTACH, area=AreaType.HAND,
                                 index=0)]
        turns = defaultdict(telepath_probe._new_turn)
        telepath_probe.scan_game(
            [  # a Telepath attach at a non-MAIN prompt is not this metric
             (_tp_obs([TELEPATH], attach_tele, turn=6,
                      context=SelectContext.ATTACH_FROM), [0]),
             # a water attach on a turn Telepath was never offered
             (_tp_obs([WATER], attach_water, turn=7), [0])],
            ep=5, turns=turns)
        assert telepath_probe.summarize_turns(turns) == (0, 0, 0, 0)


# --- scripts/mirror_race_probe.py — race trajectory and demote candidates ---

ALAKAZAM, RALTS = 50, 51
DUD = 66                                 # the draw-ability body in DRAW_IDS
FLOOR = 6                                # _CONSERVE_AT pinned for the fixture


@pytest.fixture(autouse=True)
def _patch_mirror_tables(monkeypatch):
    monkeypatch.setattr(mirror_probe, "_names",
                        {ALAKAZAM: "Alakazam ex", RALTS: "Ralts"})
    monkeypatch.setattr(mirror_probe, "DRAW_IDS", {DUD})
    monkeypatch.setattr(mirror_probe, "_CONSERVE_AT", FLOOR)


def _race_obs(my_deck, opp_deck, options, active=None, bench=(), turn=2,
              context=SelectContext.MAIN):
    me = b.player(active=active, bench=list(bench), deck_count=my_deck)
    opp = b.player(deck_count=opp_deck)
    return b.observation(me=me, opponent=opp, options=options, turn=turn,
                         context=context)


class TestMirrorRaceProbeMirrorAndBoard:
    def test_fires_any_alakazam_name_marks_the_mirror(self):
        assert mirror_probe.is_mirror([RALTS, ALAKAZAM])

    def test_guards_no_alakazam_or_no_deck_is_not_a_mirror(self):
        assert not mirror_probe.is_mirror([RALTS])
        assert not mirror_probe.is_mirror([])
        assert not mirror_probe.is_mirror(None)

    def test_fires_board_id_resolves_active_and_bench(self):
        me = b.player(active=b.pokemon(DUD),
                      bench=[b.pokemon(RALTS), b.pokemon(DUD)])
        assert mirror_probe.board_id(
            b.option(OptionType.ABILITY, area=AreaType.ACTIVE), me) == DUD
        assert mirror_probe.board_id(
            b.option(OptionType.ABILITY, area=AreaType.BENCH, index=1),
            me) == DUD

    def test_guards_board_id_misses_resolve_to_none(self):
        empty = b.player(bench=[None])
        assert mirror_probe.board_id(
            b.option(OptionType.ABILITY, area=AreaType.ACTIVE), empty) is None
        assert mirror_probe.board_id(
            b.option(OptionType.ABILITY, area=AreaType.BENCH, index=0),
            empty) is None
        assert mirror_probe.board_id(
            b.option(OptionType.ABILITY, area=AreaType.BENCH, index=4),
            empty) is None
        assert mirror_probe.board_id(
            b.option(OptionType.ABILITY, area=AreaType.HAND, index=0),
            empty) is None


class TestMirrorRaceProbeScanAndFilter:
    def test_fires_draw_ability_use_lands_with_its_race_position(self):
        menu = [b.option(OptionType.ABILITY, area=AreaType.ACTIVE),
                b.option(OptionType.END)]
        traj, uses = mirror_probe.scan_game(
            [(_race_obs(20, 22, menu, active=b.pokemon(DUD), turn=2), [0]),
             (_race_obs(18, 21, menu, active=b.pokemon(DUD), turn=3), [1])])
        assert traj == [(2, 20, 22), (3, 18, 21)]
        assert uses == [(2, 20, 22, DUD)]

    def test_guards_non_draw_bodies_and_non_main_prompts(self):
        menu = [b.option(OptionType.ABILITY, area=AreaType.ACTIVE),
                b.option(OptionType.END)]
        traj, uses = mirror_probe.scan_game(
            [  # ability on a body outside DRAW_IDS: trajectory, no use
             (_race_obs(20, 22, menu, active=b.pokemon(RALTS), turn=2), [0]),
             # non-MAIN prompt: not even a trajectory point
             (_race_obs(19, 22, menu, active=b.pokemon(DUD), turn=3,
                        context=SelectContext.TO_HAND), [0])])
        assert traj == [(2, 20, 22)]
        assert uses == []

    def test_fires_behind_and_above_floor_bucketed_by_margin(self):
        uses = [(1, FLOOR + 1, FLOOR + 3, DUD),    # margin 2  -> margin1-2
                (2, FLOOR + 1, FLOOR + 6, DUD),    # margin 5  -> margin3-5
                (3, FLOOR + 1, FLOOR + 7, DUD)]    # margin 6  -> margin6+
        assert mirror_probe.behind_uses(uses) == uses
        assert [mirror_probe.margin_bucket(theirs - mine)
                for _, mine, theirs, _ in uses] == \
            ["margin1-2", "margin3-5", "margin6+"]

    def test_guards_ahead_tied_or_at_floor_is_never_a_demote_candidate(self):
        uses = [(1, 9, 7, DUD),                    # ahead on the race
                (2, 10, 10, DUD),                  # tied
                (3, FLOOR, FLOOR + 6, DUD),        # AT the floor
                (4, FLOOR - 1, FLOOR + 6, DUD)]    # below it
        assert mirror_probe.behind_uses(uses) == []
