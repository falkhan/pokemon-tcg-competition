"""M41b § II.3a: golden fixtures for the four G-11 rule-mechanism probes.

Each probe's MEASUREMENT CORE — the per-prompt classification it applies to a
single decision — was extracted into an importable function (CLI and printed
output unchanged) and is exercised here on synthetic observations whose ground
truth is true BY CONSTRUCTION: a `fires` case the counter MUST count, and a
`guards` near-miss it must NOT. No games are played and no checkpoints load;
the racemode fixtures run the REAL rl.plan.apply_play_overrides on the
synthetic boards, so the fire/diff isolation is measured against the actual
rule, not a stub of it.

Cores under test:
  scripts/m40_rule_probe.py       record_pick          (fires / END / ash@deck
                                                        / hammer / tempo census)
  scripts/m40_planzero_probe.py   plan_is_nonzero, claim_failures,
                                  _instrumented_pilot  (the make_pilot-scoped
                                                        seam binding)
  scripts/m41_gustsnipe_probe.py  record_decision      (prompts -> trigger_true
                                                        -> gust_offered -> fires)
  scripts/racemode_fire_probe.py  record_decision      (trigger / margin / the
                                                        strip-and-diff fire)
"""
from collections import Counter

import pytest

from tests import builders as b
from tests.fake_cg import AreaType, OptionType, SelectContext

import rl.plan as rp

import scripts.m40_rule_probe as rule_probe
import scripts.m41_gustsnipe_probe as gustsnipe_probe
import scripts.racemode_fire_probe as racemode_probe

GUST_ID = next(iter(rp.GUST_IDS))               # Boss's Orders
DUDUNSPARCE_ID = next(iter(rp.DUDUNSPARCE_IDS))  # the demoted draw ability
GREAT_TUSK = 58                                  # wall-family trigger id


# --- scripts/m40_rule_probe.py — record_pick --------------------------------

def _rule_obs(hand_ids, options, deck_count=30):
    me = b.player(active=b.pokemon(1), deck_count=deck_count,
                  hand=[b.hand_card(i) for i in hand_ids])
    return b.observation(me=me, options=options)


class TestM40RuleProbeRecordPick:
    def test_fires_when_override_promotes_ash_and_deck_is_recorded(self):
        # By construction: out[0] != ranked[0] (a fire), and the new top pick
        # is a PLAY of Sacred Ash at deck 8 -> ash_deck must read [8].
        obs = _rule_obs([rp.SACRED_ASH_ID],
                        [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                         b.option(OptionType.END)],
                        deck_count=8)
        rec = rule_probe.new_census()
        rule_probe.record_pick(rec, obs, ranked=[1, 0], out=[0, 1])
        assert rec["prompts"] == 1
        assert rec["fires"] == 1
        assert rec["ash_deck"] == [8]
        assert rec["end_picks"] == 0
        assert rec["hammer_plays"] == 0 and rec["tempo_plays"] == 0

    def test_guards_unchanged_end_pick_is_passivity_not_a_fire(self):
        # Same menu, but the override left the model's END on top: an END
        # pick is counted, a fire is not.
        obs = _rule_obs([rp.SACRED_ASH_ID],
                        [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                         b.option(OptionType.END)],
                        deck_count=8)
        rec = rule_probe.new_census()
        rule_probe.record_pick(rec, obs, ranked=[1, 0], out=[1, 0])
        assert rec["prompts"] == 1
        assert rec["fires"] == 0
        assert rec["end_picks"] == 1
        assert rec["ash_deck"] == []

    def test_fires_hammer_and_tempo_census_key_on_the_hand_card(self):
        obs = _rule_obs([rp.ENHANCED_HAMMER_ID, rp.POFFIN_ID],
                        [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                         b.option(OptionType.PLAY, area=AreaType.HAND, index=1),
                         b.option(OptionType.END)])
        rec = rule_probe.new_census()
        rule_probe.record_pick(rec, obs, ranked=[2, 0, 1], out=[0, 2, 1])
        assert rec["hammer_plays"] == 1 and rec["tempo_plays"] == 0
        rule_probe.record_pick(rec, obs, ranked=[2, 1, 0], out=[1, 2, 0])
        assert rec["hammer_plays"] == 1 and rec["tempo_plays"] == 1
        assert rec["prompts"] == 2 and rec["fires"] == 2

    def test_guards_untracked_play_and_stale_index_count_nothing(self):
        # A PLAY of a card outside the tracked ids moves no census column,
        # and a PLAY whose index runs past the hand must not be resolved.
        obs = _rule_obs([7],                     # plain trainer, untracked
                        [b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                         b.option(OptionType.PLAY, area=AreaType.HAND, index=5),
                         b.option(OptionType.END)])
        rec = rule_probe.new_census()
        rule_probe.record_pick(rec, obs, ranked=[2, 0, 1], out=[0, 2, 1])
        rule_probe.record_pick(rec, obs, ranked=[2, 0, 1], out=[1, 2, 0])
        assert rec["prompts"] == 2 and rec["fires"] == 2
        assert rec["ash_deck"] == []
        assert rec["hammer_plays"] == 0 and rec["tempo_plays"] == 0
        assert rec["end_picks"] == 0


# --- scripts/m40_planzero_probe.py — the zero-plan claim --------------------

@pytest.fixture(scope="module")
def pz():
    pytest.importorskip("torch")     # the probe imports rl.policy at module top
    import scripts.m40_planzero_probe as mod
    return mod


class TestPlanzeroProbe:
    def test_fires_nonzero_plan_vector_is_flagged(self, pz):
        import numpy as np
        assert pz.plan_is_nonzero(np.array([0.0, 0.0, 1.0]))
        assert pz.plan_is_nonzero([0, 2, 0])

    def test_guards_zero_plan_vector_is_not(self, pz):
        import numpy as np
        assert not pz.plan_is_nonzero(np.zeros(8, np.float32))
        assert not pz.plan_is_nonzero([0.0, 0.0])

    def test_guards_clean_arm_with_live_control_passes(self, pz):
        ctl = {"plan_enumerations": 5, "prompts": 10, "nonzero_plan_prompts": 4}
        arm = {"plan_enumerations": 0, "prompts": 10, "nonzero_plan_prompts": 0}
        assert pz.claim_failures(ctl, arm) == []

    def test_fires_arm_plan_head_activity_fails_both_halves(self, pz):
        ctl = {"plan_enumerations": 5, "prompts": 10, "nonzero_plan_prompts": 4}
        arm = {"plan_enumerations": 3, "prompts": 10, "nonzero_plan_prompts": 2}
        failures = pz.claim_failures(ctl, arm)
        assert len(failures) == 2
        assert "ran the plan head 3x" in failures[0]
        assert "2/10 prompts" in failures[1]

    def test_fires_silent_control_is_a_harness_defect_not_a_pass(self, pz):
        # The non-vacuity half: a clean arm against a control that never
        # planned proves nothing and must FAIL loudly.
        ctl = {"plan_enumerations": 0, "prompts": 10, "nonzero_plan_prompts": 0}
        arm = {"plan_enumerations": 0, "prompts": 10, "nonzero_plan_prompts": 0}
        failures = pz.claim_failures(ctl, arm)
        assert len(failures) == 1 and "not measuring" in failures[0]

    def test_instrumentation_binds_to_the_single_make_pilot_call(
            self, pz, monkeypatch):
        # The wrapper plumbing (the probe's own docstring: a global patch once
        # counted the OPPONENT's plan head). Stub seams stand in for the real
        # net; fn3's contract — both seams resolved AT make_pilot time — is
        # played by the fake make_pilot, and the counters must read exactly
        # what it did inside the patch window and nothing after.
        import rl.matchrunner as mr
        import rl.policy

        calls = Counter()

        def stub_enum(obs):
            calls["enum"] += 1
            return []

        class StubScorer:
            def act(self, state_ctx, plan, state_ids, options, option_ids, k,
                    greedy=True):
                calls["act"] += 1
                return [0]

        monkeypatch.setattr(rp, "enumerate_plans", stub_enum)
        monkeypatch.setattr(rl.policy, "OptionScorerV3", StubScorer)
        rec = {"plan_enumerations": 0, "prompts": 0, "nonzero_plan_prompts": 0}

        def fake_make_pilot(spec, instance=""):
            rp.enumerate_plans(None)                       # the plan head
            scorer = rl.policy.OptionScorerV3()
            scorer.act(None, [0.0, 1.0], None, None, None, 1)  # non-zero plan
            scorer.act(None, [0.0, 0.0], None, None, None, 1)  # zero plan
            return "fn", "deck"

        monkeypatch.setattr(mr, "make_pilot", fake_make_pilot)
        assert pz._instrumented_pilot(("model",), "t", rec) == ("fn", "deck")
        assert rec == {"plan_enumerations": 1, "prompts": 2,
                       "nonzero_plan_prompts": 1}
        # Outside the window = the other pilot's side: it must NOT count.
        rp.enumerate_plans(None)
        rl.policy.OptionScorerV3().act(None, [1.0], None, None, None, 1)
        assert rec == {"plan_enumerations": 1, "prompts": 2,
                       "nonzero_plan_prompts": 1}
        assert calls == Counter({"enum": 2, "act": 3})


# --- scripts/m41_gustsnipe_probe.py — record_decision -----------------------

def _snipe_obs(bench_target, hand_ids=(GUST_ID,), context=SelectContext.MAIN,
               opp_prizes=6):
    me = b.player(active=b.pokemon(1),
                  hand=[b.hand_card(i) for i in hand_ids])
    opp = b.player(active=b.pokemon(4, hp=300), bench=[bench_target],
                   prizes_remaining=opp_prizes)
    return b.observation(
        me=me, opponent=opp, context=context,
        options=[b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                 b.option(OptionType.END)])


class TestGustsnipeProbeRecordDecision:
    # fake_cg card 2 is the ex (2 prizes on KO); hp 100 <= the 120 frailty
    # line and softer than the 300-hp active — _gustsnipe_target MUST be true.
    def test_fires_full_funnel_when_pick_changed(self):
        obs = _snipe_obs(b.pokemon(2, hp=100, max_hp=250))
        stats, targets = Counter(), Counter()
        gustsnipe_probe.record_decision(stats, targets, obs,
                                        ranked=[1, 0], out=[0, 1])
        assert stats == Counter({"prompts": 1, "trigger_true": 1,
                                 "gust_offered": 1, "fires": 1})
        assert sum(targets.values()) == 1     # the one benched 2-prize body

    def test_guards_unchanged_pick_is_offered_not_fired(self):
        obs = _snipe_obs(b.pokemon(2, hp=100, max_hp=250))
        stats, targets = Counter(), Counter()
        gustsnipe_probe.record_decision(stats, targets, obs,
                                        ranked=[1, 0], out=[1, 0])
        assert stats == Counter({"prompts": 1, "trigger_true": 1,
                                 "gust_offered": 1})
        assert stats["fires"] == 0 and not targets

    def test_guards_healthy_bench_is_no_trigger(self):
        # 250/250 hp: above the flat line AND above half max — a near-miss
        # the predicate must reject, so nothing past `prompts` may count.
        obs = _snipe_obs(b.pokemon(2, hp=250))
        stats, targets = Counter(), Counter()
        gustsnipe_probe.record_decision(stats, targets, obs,
                                        ranked=[1, 0], out=[1, 0])
        assert stats == Counter({"prompts": 1})

    def test_guards_dead_lever_frail_target_but_no_gust_in_hand(self):
        # The gap the probe exists to expose: board condition true, card not
        # held — trigger_true counts, gust_offered must not.
        obs = _snipe_obs(b.pokemon(2, hp=100, max_hp=250), hand_ids=(7,))
        stats, targets = Counter(), Counter()
        gustsnipe_probe.record_decision(stats, targets, obs,
                                        ranked=[1, 0], out=[1, 0])
        assert stats == Counter({"prompts": 1, "trigger_true": 1})
        assert stats["gust_offered"] == 0

    def test_guards_match_point_and_non_main_prompts_not_counted(self):
        stats, targets = Counter(), Counter()
        # Opponent at match point: gustveto's finding wins, no trigger.
        obs = _snipe_obs(b.pokemon(2, hp=100, max_hp=250), opp_prizes=1)
        gustsnipe_probe.record_decision(stats, targets, obs,
                                        ranked=[1, 0], out=[0, 1])
        assert stats == Counter({"prompts": 1})
        # A submenu is not an own MAIN prompt at all.
        obs = _snipe_obs(b.pokemon(2, hp=100, max_hp=250),
                         context=SelectContext.EFFECT_TARGET)
        gustsnipe_probe.record_decision(stats, targets, obs,
                                        ranked=[1, 0], out=[0, 1])
        assert stats == Counter({"prompts": 1})


# --- scripts/racemode_fire_probe.py — record_decision -----------------------

def _race_obs(opp_active_id, me_deck=10, opp_deck=30,
              context=SelectContext.MAIN):
    # Our active IS the demote target (Dudunsparce), so option 0 — its
    # ABILITY — is exactly what the O12 demote acts on when it engages.
    me = b.player(active=b.pokemon(DUDUNSPARCE_ID), deck_count=me_deck)
    opp = b.player(active=b.pokemon(opp_active_id), deck_count=opp_deck)
    return b.observation(
        me=me, opponent=opp, context=context,
        options=[b.option(OptionType.ABILITY, area=AreaType.ACTIVE),
                 b.option(OptionType.END)])


RACEMODE3 = frozenset({rp.PLAY_FIX_RACEMODE3})


class TestRacemodeFireProbeRecordDecision:
    def test_wall_id_is_still_a_wall_trigger(self):
        # The fixtures below construct truth from this membership.
        assert GREAT_TUSK in rp._RACEMODE_WALL_IDS

    def test_fires_wall_demote_changes_the_order_and_is_diffed(self):
        # Real apply_play_overrides: Great Tusk on the opponent's board fires
        # the racemode3 blanket demote of our top-pick Dudunsparce ABILITY,
        # so the stripped-fixes re-run MUST differ -> exactly one o12 fire.
        # Deck 10 vs 30 also opens the margin gate (behind by >5, 6<10<=25).
        stats = Counter()
        obs = _race_obs(GREAT_TUSK, me_deck=10, opp_deck=30)
        out = racemode_probe.record_decision(stats, obs, [0, 1], RACEMODE3,
                                             rp.apply_play_overrides)
        assert out == [1, 0]
        assert stats == Counter({"main_prompts": 1, "trigger_true": 1,
                                 "margin_open": 1, "o12_fires": 1})

    def test_guards_no_race_family_on_board(self):
        stats = Counter()
        obs = _race_obs(4)                    # plain attacker, no trigger id
        out = racemode_probe.record_decision(stats, obs, [0, 1], RACEMODE3,
                                             rp.apply_play_overrides)
        assert out == [0, 1]
        assert stats == Counter({"main_prompts": 1})

    def test_guards_trigger_true_but_demote_target_not_on_top(self):
        # The probe's own NOTE branch: END on top means the demote cannot
        # reorder anything — trigger true, zero fires. Equal decks also keep
        # the margin gate closed.
        stats = Counter()
        obs = _race_obs(GREAT_TUSK, me_deck=30, opp_deck=30)
        out = racemode_probe.record_decision(stats, obs, [1, 0], RACEMODE3,
                                             rp.apply_play_overrides)
        assert out == [1, 0]
        assert stats == Counter({"main_prompts": 1, "trigger_true": 1})
        assert stats["margin_open"] == 0 and stats["o12_fires"] == 0

    def test_guards_no_race_fix_in_the_set_counts_nothing(self):
        stats = Counter()
        obs = _race_obs(GREAT_TUSK, me_deck=10, opp_deck=30)
        out = racemode_probe.record_decision(
            stats, obs, [0, 1], frozenset({rp.PLAY_FIX_CONSERVE}),
            rp.apply_play_overrides)
        assert out == [0, 1]
        assert stats == Counter()

    def test_guards_non_main_prompt_counts_nothing(self):
        stats = Counter()
        obs = _race_obs(GREAT_TUSK, context=SelectContext.EFFECT_TARGET)
        out = racemode_probe.record_decision(stats, obs, [0, 1], RACEMODE3,
                                             rp.apply_play_overrides)
        assert out == [0, 1]
        assert stats == Counter()
