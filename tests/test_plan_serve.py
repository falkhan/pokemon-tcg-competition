"""M40 S6: the serve-time plan state machine, and the `planzero` token.

WHY THIS FILE EXISTS. The plan head has been a train/serve mismatch since M24 —
every replay-derived corpus loads with plans=all-zeros (rl/plan_iter.py's
BCDatasetV3 shim) while every pilot builds a NON-zero plan at each own MAIN
prompt. It survived sixteen milestones because `tests/test_matchrunner.py` had
ZERO coverage of fn3/fn4's plan state machine: nothing anywhere asserted what
the trunk is actually fed. So these tests pin BOTH regimes — the as-shipped
behaviour and the zeroed one — and the as-shipped assertions are the ones that
would have made the regression visible.

The seam throughout: fn3/fn4 resolve `enumerate_plans` and `OptionScorerV3` at
make_pilot time (both are function-local imports inside rl/matchrunner.py's
branches), so patching around a single make_pilot call instruments exactly one
pilot and leaves any opponent pristine. Getting this wrong is not hypothetical
— scripts/m40_planzero_probe.py's first version patched globally, counted the
opponent bed's plan head, and reported a false failure.
"""
import numpy as np
import pytest

import rl.matchrunner as mr
import rl.plan as rp
from tests.builders import observation, option, player, pokemon

torch = pytest.importorskip("torch")

ALAKAZAM = "alakazam_v2_h4"


# --- fixtures ---------------------------------------------------------------

def _v3_ckpt(tmp_path):
    """A plan-conditioned v3 checkpoint — the fn3 branch, and the one both
    M40 battery nets (cont3, retain_b) actually are."""
    from rl.encoders import N_STATE_IDS_V3, OPTION_M28_DIM
    from rl.policy import OptionScorerV3
    ck = tmp_path / "v3.pt"
    torch.save(OptionScorerV3(n_state_ids=N_STATE_IDS_V3,
                              option_dim=OPTION_M28_DIM).state_dict(), ck)
    return str(ck)


def _v4_ckpt(tmp_path):
    """An encoder-v4 checkpoint — the fn4 branch. The M40 battery never
    exercises it (both nets are v3), so this test is its only coverage."""
    from rl.encoders import N_STATE_IDS_V4, OPTION_M28_DIM, V4_EXTRA_DIM
    from rl.policy import OptionScorerV3
    ck = tmp_path / "v4.pt"
    net = OptionScorerV3(n_state_ids=N_STATE_IDS_V4,
                         option_dim=OPTION_M28_DIM, extra_dim=V4_EXTRA_DIM)
    torch.save(net.state_dict(), ck)
    return str(ck)


def _main_obs(turn=1, context=None, logs=()):
    """A MAIN prompt that yields REAL non-null plan candidates.

    Card ids and the fighting energy come from tests/test_plan.py's
    TestEnumeratePlans fixtures: a card-1 attacker holding F against a
    high-HP opponent active is the shape enumerate_plans is known to expand.
    A board with no legal attack collapses to [None], encode_plan(None) is
    all-zeros, and every plan assertion below would pass vacuously.
    """
    from tests.fake_cg import EnergyType, OptionType, SelectContext
    fighting = EnergyType.FIGHTING     # card 1's attacks cost F (tests/fake_cg.py)
    me = player(active=pokemon(1, hp=100, energies=[fighting]),
                bench=[pokemon(2, hp=60)], hand=[])
    opp = player(active=pokemon(4, hp=300), bench=[pokemon(5, hp=50)])
    obs = observation(
        me=me, opponent=opp, turn=turn,
        context=SelectContext.MAIN if context is None else context,
        options=[option(OptionType.END), option(OptionType.ATTACK, attack_id=1)])
    obs.logs = list(logs)
    return obs


class _Spy:
    """Counts plan-head entries for ONE pilot (patched around make_pilot)."""

    def __init__(self, force_nonnull=False):
        self.enumerations = 0
        self.plans_seen = []
        self.picked = []
        # A freshly-initialised net argmaxes the plan head onto the NULL
        # candidate on a synthetic board, and encode_plan(None) is all-zeros —
        # so "as-shipped feeds a non-zero plan" is not observable from an
        # untrained net. force_nonnull makes the CHOICE deterministic so the
        # test pins the hold semantics on a real vector instead of on the
        # accident of an untrained argmax. (With real weights on real boards
        # the choice is non-null ~35% of prompts — scripts/m40_planzero_probe.py.)
        self.force_nonnull = force_nonnull

    def pilot(self, spec, instance="t"):
        import rl.policy
        real_enum, real_cls = rp.enumerate_plans, rl.policy.OptionScorerV3
        spy = self

        def counting_enum(obs):
            spy.enumerations += 1
            return real_enum(obs)

        class _Recording(real_cls):
            def act(self, sc, plan, sids, opts, oids, k, greedy=True):
                spy.plans_seen.append(np.asarray(plan).copy())
                return real_cls.act(self, sc, plan, sids, opts, oids, k,
                                    greedy=greedy)

            def act_plan(self, sc, sids, mat):
                idx = real_cls.act_plan(self, sc, sids, mat)
                if spy.force_nonnull:
                    nz = [i for i, v in enumerate(mat) if np.asarray(v).any()]
                    idx = nz[-1] if nz else idx
                spy.picked.append(np.asarray(mat[idx]).copy())
                return idx

        rp.enumerate_plans = counting_enum
        rl.policy.OptionScorerV3 = _Recording
        try:
            return mr.make_pilot(spec, instance=instance)
        finally:
            rp.enumerate_plans = real_enum
            rl.policy.OptionScorerV3 = real_cls


# --- the fix-set algebra (what makes the battery single-variable) -----------

def test_planzero_arm_is_the_control_plus_exactly_one_token():
    # The M40 S6 battery's whole attribution rests on this and nothing else
    # asserted it. If a rule ever leaks into the arm, the gate silently
    # measures two variables.
    assert (mr._MODEL_FIX_KINDS["model-c-pkgz"]
            == mr._MODEL_FIX_KINDS["model-c-pkg"] | {rp.SERVE_FIX_PLANZERO})
    assert mr._MODEL_FIX_KINDS["model-pz"] == {rp.SERVE_FIX_PLANZERO}
    # ...and the control is untouched by the M40 edit: it is the live Ship B
    # string, which is what makes the control arm the *as-shipped* agent.
    assert mr._MODEL_FIX_KINDS["model-c-pkg"] == {
        "conserve", "racemode2", "racemode4"}


def test_planzero_is_inert_in_both_override_predicates():
    # `planzero` rides in the same frozenset as the rerankers but acts
    # upstream of them. Both predicates must ignore it, or the token would
    # silently change rule behaviour too and the battery would measure two
    # things.
    obs = _main_obs()
    ranked = [0, 1]
    fixes = frozenset({rp.SERVE_FIX_PLANZERO})
    assert rp.apply_attach_overrides(obs, list(ranked), fixes) == ranked
    assert rp.apply_play_overrides(obs, list(ranked), fixes) == ranked


# --- the as-shipped state machine (the M24 coverage gap) --------------------

def test_fn3_plans_at_first_main_and_holds_it_for_the_turn(tmp_path):
    """As-shipped behaviour, pinned. M11's fix was to plan ONCE per turn and
    hold it across submenus; nothing has asserted that since."""
    from tests.fake_cg import SelectContext
    spy = _Spy(force_nonnull=True)
    fn, _deck = spy.pilot(("model", _v3_ckpt(tmp_path), ALAKAZAM))

    fn(_main_obs(turn=1))                       # first MAIN of turn 1
    assert spy.enumerations == 1
    fn(_main_obs(turn=1))                       # second MAIN, same turn
    assert spy.enumerations == 1, "replanned mid-turn (the M11 defect)"
    fn(_main_obs(turn=1, context=SelectContext.EFFECT_TARGET))
    assert spy.enumerations == 1, "planned on a submenu"
    fn(_main_obs(turn=2))                       # new turn -> replan
    assert spy.enumerations == 2

    # The chosen plan is HELD: every prompt of turn 1 is fed the same vector
    # the head picked at that turn's first MAIN. This is the input no training
    # row has ever carried, and it persists across the whole turn — which is
    # why the defect surface is a third of served prompts, not a rare edge.
    # Guard against the assertion going vacuous: a board with no legal attack
    # collapses to [None], encode_plan(None) is all-zeros, and everything
    # below would pass while testing nothing. This bit me once already.
    assert spy.picked and spy.picked[0].any(), \
        "fixture board yields no non-null plan - the hold assertion is vacuous"
    held = spy.picked[0]
    assert all(np.array_equal(p, held) for p in spy.plans_seen[:3]), \
        "the plan was not held across the turn's submenus"


def test_fn3_planzero_never_touches_the_plan_head(tmp_path):
    spy = _Spy()
    fn, _deck = spy.pilot(("model-c-pkgz", _v3_ckpt(tmp_path), ALAKAZAM))
    for turn in (1, 1, 2, 3):
        fn(_main_obs(turn=turn))
    assert spy.enumerations == 0, "planzero ran the plan head"
    assert spy.plans_seen, "test is vacuous - the scorer was never called"
    assert not any(p.any() for p in spy.plans_seen), \
        "planzero fed a non-zero plan to the trunk"


def test_fn4_planzero_zeroes_the_plan_and_keeps_memory_bookkeeping(tmp_path):
    """fn4 is NOT exercised by the M40 battery (both nets are v3), so this is
    the only thing standing between a future v4 ship and an untested serve
    path. The trap in the diff is that an over-eager guard could skip
    memory.observe()/reset() along with the plan head — and v4's whole value
    is that memory."""
    from rl.memory import OppMemory
    spy = _Spy()
    observed = {"n": 0, "resets": 0}
    real_observe, real_reset = OppMemory.observe, OppMemory.reset

    def counting_observe(self, obs):
        observed["n"] += 1
        return real_observe(self, obs)

    def counting_reset(self):
        observed["resets"] += 1
        return real_reset(self)

    OppMemory.observe, OppMemory.reset = counting_observe, counting_reset
    try:
        fn, _deck = spy.pilot(("model-c-pkgz", _v4_ckpt(tmp_path), ALAKAZAM))
        base = observed["resets"]          # the constructor's own reset()
        for turn in (1, 1, 2):
            fn(_main_obs(turn=turn))
        assert observed["n"] == 3, "memory must be observed once per own prompt"
        fn(observation(select=False))      # deck-return: reset, no observe
        assert observed["resets"] > base
        assert observed["n"] == 3
    finally:
        OppMemory.observe, OppMemory.reset = real_observe, real_reset

    assert spy.enumerations == 0
    assert spy.plans_seen and not any(p.any() for p in spy.plans_seen)


def test_fn4_as_shipped_still_plans(tmp_path):
    # Non-vacuity twin of the above: without the token, fn4 does plan.
    spy = _Spy()
    fn, _deck = spy.pilot(("model", _v4_ckpt(tmp_path), ALAKAZAM))
    fn(_main_obs(turn=1))
    assert spy.enumerations == 1


def test_planzero_matches_feeding_zeros_explicitly(tmp_path):
    """Turns the equivalence ARGUMENT (score_plans/act_plan have no side
    effects, so 'run and discard' == 'never run') into a tested fact."""
    from rl.encoders import (N_CONTEXTS, encode_context, encode_option_v2,
                             encode_state_v3)
    from rl.plan import PLAN_DIM
    from rl.policy import OptionScorerV3, option_dim_of
    ck = _v3_ckpt(tmp_path)
    fn, deck = mr.make_pilot(("model-pz", ck, ALAKAZAM), "eq")

    sd = torch.load(ck, map_location="cpu")
    net = OptionScorerV3(n_state_ids=20, option_dim=option_dim_of(sd))
    net.load_state_dict(sd)
    net.eval()

    obs = _main_obs(turn=1)
    num, sids = encode_state_v3(obs.current, deck)
    sc = np.concatenate([num, encode_context(obs.select.context)]).astype(np.float32)
    pairs = [encode_option_v2(o, obs) for o in obs.select.option]
    opts = np.stack([n for n, _ in pairs]).astype(np.float32)
    oids = np.stack([i for _, i in pairs])
    expect = net.act(sc, np.zeros(PLAN_DIM, np.float32), sids, opts, oids,
                     obs.select.maxCount, greedy=True)
    assert fn(obs) == expect
    assert N_CONTEXTS  # encode_context sanity, keeps the import honest
