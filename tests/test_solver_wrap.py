"""M22c: wrapping any pilot with the within-turn combo solver.

The neural bundle has shipped a greedy argmax since M11 while turn_solver.py —
written to fix exactly that — shipped only in the rules bundle. These tests pin
the wrapper's two load-bearing properties: inner state stays live, and the
solver can never cost the crash gate.
"""
import pytest

from tests import builders as b
from tests.fake_cg import OptionType, SelectContext

import rl.turn_solver as ts


@pytest.fixture
def obs_dict():
    me = b.player(active=b.pokemon(1, energies=[6]))
    opp = b.player(active=b.pokemon(4, hp=300))
    return b.observation(me=me, opponent=opp,
                         options=[b.option(OptionType.ATTACK),
                                  b.option(OptionType.END)])


class TestInnerStaysLive:
    def test_inner_is_called_on_every_prompt_even_when_solver_fires(self, obs_dict,
                                                                   monkeypatch):
        """The v4 pilot runs memory.observe() once per own prompt and caches a
        per-turn plan. Short-circuiting it would silently desync both."""
        calls = []
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T1_ko_one_attach")
        monkeypatch.setattr(ts, "solve_turn", lambda *a, **k: [1])
        agent = ts.wrap_with_solver(lambda od: calls.append(od) or [0], [1, 2, 3])
        assert agent(obs_dict) == [1], "solver pick must win"
        assert len(calls) == 1, "inner must STILL have been called for its side effects"

    def test_inner_result_used_when_solver_declines(self, obs_dict, monkeypatch):
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T1_ko_one_attach")
        monkeypatch.setattr(ts, "solve_turn", lambda *a, **k: None)
        agent = ts.wrap_with_solver(lambda od: [7], [1, 2, 3])
        assert agent(obs_dict) == [7]

    def test_inner_result_used_when_trigger_does_not_fire(self, obs_dict, monkeypatch):
        monkeypatch.setattr(ts, "solve_trigger", lambda o: None)
        monkeypatch.setattr(ts, "solve_turn",
                            lambda *a, **k: pytest.fail("must not solve"))
        agent = ts.wrap_with_solver(lambda od: [7], [1, 2, 3])
        assert agent(obs_dict) == [7]


class TestNeverCostsTheCrashGate:
    def test_solver_exception_falls_back_to_inner(self, obs_dict, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("solver blew up")
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T1_ko_one_attach")
        monkeypatch.setattr(ts, "solve_turn", boom)
        agent = ts.wrap_with_solver(lambda od: [7], [1, 2, 3])
        assert agent(obs_dict) == [7], "a solver crash must never propagate"

    def test_trigger_exception_is_not_swallowed_into_a_wrong_action(self, obs_dict,
                                                                   monkeypatch):
        """should_solve raising is a bug we want visible, not a silent greedy."""
        monkeypatch.setattr(ts, "solve_trigger",
                            lambda o: (_ for _ in ()).throw(RuntimeError("x")))
        agent = ts.wrap_with_solver(lambda od: [7], [1, 2, 3])
        with pytest.raises(RuntimeError):
            agent(obs_dict)


class TestDeckReturn:
    def test_select_none_defers_to_inner(self):
        agent = ts.wrap_with_solver(lambda od: [9, 9], [1, 2, 3])
        assert agent(b.observation(select=False)) == [9, 9]


class TestStats:
    def test_stats_count_fires_and_changes(self, obs_dict, monkeypatch):
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T1_ko_one_attach")
        monkeypatch.setattr(ts, "solve_turn", lambda *a, **k: [1])
        stats = {}
        agent = ts.wrap_with_solver(lambda od: [0], [1, 2, 3], stats=stats)
        agent(obs_dict)
        assert stats["prompts"] == 1 and stats["solver_fired"] == 1
        assert stats["changed"] == 1, "solver pick differs from inner -> changed"

    def test_agreement_is_not_counted_as_a_change(self, obs_dict, monkeypatch):
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T1_ko_one_attach")
        monkeypatch.setattr(ts, "solve_turn", lambda *a, **k: [5])
        stats = {}
        agent = ts.wrap_with_solver(lambda od: [5], [1, 2, 3], stats=stats)
        agent(obs_dict)
        assert stats["solver_fired"] == 1 and stats["changed"] == 0


class TestTierGating:
    def test_allow_list_suppresses_untrusted_tiers(self, obs_dict, monkeypatch):
        """Firing deeper requires firing less often — the allow-list is how we
        buy depth without breaching the 50ms G6 gate."""
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T4_closing")
        monkeypatch.setattr(ts, "solve_turn",
                            lambda *a, **k: pytest.fail("suppressed tier must not solve"))
        agent = ts.wrap_with_solver(lambda od: [7], [1, 2, 3],
                                    allow=frozenset({"T1_ko_one_attach"}))
        assert agent(obs_dict) == [7]

    def test_allow_list_permits_listed_tiers(self, obs_dict, monkeypatch):
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T1_ko_one_attach")
        monkeypatch.setattr(ts, "solve_turn", lambda *a, **k: [1])
        agent = ts.wrap_with_solver(lambda od: [7], [1, 2, 3],
                                    allow=frozenset({"T1_ko_one_attach"}))
        assert agent(obs_dict) == [1]

    def test_stats_separate_triggered_from_fired(self, obs_dict, monkeypatch):
        """trig_* counts the cheap trigger; fire_* counts solves that overrode.
        The ratio is what tells us which tier is worth its latency."""
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T2_trainer_gap")
        monkeypatch.setattr(ts, "solve_turn", lambda *a, **k: None)   # declined
        stats = {}
        agent = ts.wrap_with_solver(lambda od: [7], [1, 2, 3], stats=stats)
        agent(obs_dict)
        assert stats["trig_T2_trainer_gap"] == 1
        assert "fire_T2_trainer_gap" not in stats


class TestShouldSolveCompat:
    def test_should_solve_is_the_bool_view_of_solve_trigger(self, monkeypatch):
        monkeypatch.setattr(ts, "solve_trigger", lambda o: "T3_multiprize")
        assert ts.should_solve(object()) is True
        monkeypatch.setattr(ts, "solve_trigger", lambda o: None)
        assert ts.should_solve(object()) is False
