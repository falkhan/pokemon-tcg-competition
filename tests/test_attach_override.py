"""M26 attach-override arms (rl/plan.apply_attach_overrides).

Both pilots (rl/matchrunner fn3/fn4 and submission/main.py) import THIS ONE
function and feed it the model's full greedy order — the predicate has a
single source of truth, so parity reduces to (a) these behavioral cases and
(b) the twin files staying byte-identical (test_bundle_twins below).
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

from cg.api import AreaType, OptionType, SelectContext
from rl.plan import (ATTACH_FIX_BACKSTOP, ATTACH_FIX_TELEPATH, TELEPATH_ID,
                     apply_attach_overrides)

BASIC_P = 4          # Basic {P} Energy — any basic energy id works for tests
NON_ENERGY = 741     # Abra


@pytest.fixture(autouse=True)
def _energy_ids(monkeypatch):
    """The suite runs on tests/fake_cg, whose card table doesn't carry the
    real energy ids — pin the card-fact set the predicate reads."""
    import rl.plan as rp
    monkeypatch.setattr(rp, "_IS_ENERGY", {BASIC_P, TELEPATH_ID})


def _card(cid):
    return SimpleNamespace(id=cid)


def _opt(otype, index=None, area=AreaType.HAND, card_id=None):
    return SimpleNamespace(type=otype, index=index, area=area, cardId=card_id)


def _obs(options, hand, bench=(None,) * 5, energy_attached=False,
         context=SelectContext.MAIN, bench_max=5):
    me = SimpleNamespace(hand=[_card(c) for c in hand],
                         bench=list(bench), benchMax=bench_max)
    current = SimpleNamespace(players=[me], yourIndex=0,
                              energyAttached=energy_attached)
    select = SimpleNamespace(context=context, option=options)
    return SimpleNamespace(current=current, select=select)


BOTH = frozenset({ATTACH_FIX_TELEPATH, ATTACH_FIX_BACKSTOP})


def test_o1_moves_telepath_attach_to_front():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0),
            _opt(OptionType.ATTACH, 1), _opt(OptionType.END)]
    obs = _obs(opts, hand=[BASIC_P, TELEPATH_ID])
    out = apply_attach_overrides(obs, [0, 1, 2, 3],
                                 frozenset({ATTACH_FIX_TELEPATH}))
    assert out == [2, 0, 1, 3]          # Telepath attach (idx 2) forced first


def test_o1_respects_full_bench_and_energy_attached():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0)]
    full = _obs(opts, hand=[TELEPATH_ID], bench=[_card(NON_ENERGY)] * 5)
    assert apply_attach_overrides(
        full, [0, 1], frozenset({ATTACH_FIX_TELEPATH})) == [0, 1]
    attached = _obs(opts, hand=[TELEPATH_ID], energy_attached=True)
    assert apply_attach_overrides(attached, [0, 1], BOTH) == [0, 1]


def test_o2_backstop_fires_on_turn_ending_choice():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0),
            _opt(OptionType.END)]
    obs = _obs(opts, hand=[BASIC_P])
    fixes = frozenset({ATTACH_FIX_BACKSTOP})
    # model wants ATTACK -> attach (model's best attach) goes first
    assert apply_attach_overrides(obs, [0, 1, 2], fixes) == [1, 0, 2]
    # model wants END -> same
    assert apply_attach_overrides(obs, [2, 0, 1], fixes) == [1, 2, 0]
    # model already picked a non-terminal action -> untouched
    play = [_opt(OptionType.PLAY, 0)] + opts
    obs2 = _obs(play, hand=[NON_ENERGY, BASIC_P])
    assert apply_attach_overrides(obs2, [0, 2, 1, 3], fixes) == [0, 2, 1, 3]


def test_o2_ignores_non_energy_attach():
    # ATTACH of a TOOL (non-energy card) must not satisfy the backstop
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0)]
    obs = _obs(opts, hand=[NON_ENERGY])
    assert apply_attach_overrides(
        obs, [0, 1], frozenset({ATTACH_FIX_BACKSTOP})) == [0, 1]


def test_off_by_default_and_non_main_untouched():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0)]
    obs = _obs(opts, hand=[TELEPATH_ID])
    assert apply_attach_overrides(obs, [0, 1], frozenset()) == [0, 1]
    sub = _obs(opts, hand=[TELEPATH_ID], context=SelectContext.DISCARD)
    assert apply_attach_overrides(sub, [0, 1], BOTH) == [0, 1]


def test_o1_wins_over_o2_when_both_on():
    opts = [_opt(OptionType.ATTACK), _opt(OptionType.ATTACH, 0),
            _opt(OptionType.ATTACH, 1)]
    obs = _obs(opts, hand=[BASIC_P, TELEPATH_ID])
    # composed arm: telepath (idx 2) preferred over the model's best attach (1)
    assert apply_attach_overrides(obs, [0, 1, 2], BOTH) == [2, 0, 1]


def test_bundle_twins():
    """submission/rl/plan.py is a build-time copy of rl/plan.py — the
    override predicate must never diverge between screen and ship."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert (root / "rl/plan.py").read_bytes() == \
        (root / "submission/rl/plan.py").read_bytes()


def test_matchrunner_spec_kinds_parse():
    from rl.matchrunner import _MODEL_FIX_KINDS, parse_spec, spec_deck
    for kind in _MODEL_FIX_KINDS:
        spec = parse_spec(f"{kind}:checkpoints/x.pt:clone54618168")
        assert spec == (kind, "checkpoints/x.pt", "clone54618168")
        assert spec_deck(spec) == "clone54618168"
