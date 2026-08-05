"""Golden fixtures for the engine-side probe scripts (M41b § II.3a).

Each probe's MEASUREMENT CORE — the diff / classification / aggregation logic
it applies to what it observed — is pinned on synthetic inputs whose ground
truth is true by construction: a `fires` case that must flag/count, and a
`guards` near-miss that must not. The engine-bound halves (battle loops,
make_pilot, matchrunner subprocess cells) stay out of scope: those probes
exist precisely to interrogate the LIVE engine, and CI has no engine.

Instruments stay independent (§ II.3b): only the tests/builders.py observation
factories are shared here, never any probe logic. Where a fake-pool table
lacks a shape (stadium ids, hand-scaler attacks, retreat costs), the PROBE
module's own binding is patched module-bound, the fake_cg way — rl.combat's
tables are never touched.
"""
from collections import Counter
from types import SimpleNamespace

import pytest

import scripts.ability_probe as ability_probe
import scripts.damage_probe as damage_probe
import scripts.m42_perception_probe as m42_probe
import scripts.prize_semantics_probe as prize_probe
from tests import builders
from tests.fake_cg import AreaType, OptionType, SelectContext

FIGHTING = 6  # EnergyType.FIGHTING — fake card 1's own type


# ===========================================================================
# scripts/ability_probe.py — the forced-use board diff (ForceTap)
# ===========================================================================

def test_ability_delta_is_multiset_math():
    added, removed = ability_probe._delta([5, 5, 1], [5, 5, 5, 2])
    assert added == Counter({5: 1, 2: 1})
    assert removed == Counter({1: 1})
    agg = ability_probe._agg_counter(
        [{"k": Counter({7: 1})}, {"k": Counter({7: 2, 8: 1})}], "k")
    assert agg == Counter({7: 3, 8: 1})


def test_ability_force_fires_and_diffs_at_next_main():
    """The M30 P0 lesson by construction: the sub-prompt in between must
    record nothing, and the diff lands only at the NEXT MAIN prompt."""
    tap = ability_probe.ForceTap(lambda od: [1], "ability", target_id=5)
    before = builders.observation(
        me=builders.player(active=builders.pokemon(1),
                           bench=[builders.pokemon(5)],
                           hand=[builders.hand_card(7)], deck_count=10),
        options=[builders.option(OptionType.ABILITY,
                                 area=AreaType.BENCH, index=0),
                 builders.option(OptionType.END)])
    assert tap(before) == [0]                    # target forced to the front
    assert (tap.offered, tap.forced) == (1, 1)

    sub = builders.observation(                  # the ability's own sub-prompt
        me=builders.player(), context=SelectContext.TO_HAND,
        options=[builders.option(OptionType.CARD, area=AreaType.DECK, index=0)])
    tap(sub)
    assert tap.samples == []                     # naive next-decision read = 0

    after = builders.observation(
        me=builders.player(active=builders.pokemon(1),
                           bench=[builders.pokemon(5), builders.pokemon(11)],
                           hand=[builders.hand_card(7), builders.hand_card(6)],
                           deck_count=9),
        options=[builders.option(OptionType.END)])
    tap(after)
    sample, = tap.samples
    assert sample["same_turn"]
    assert sample["ddeck"] == -1 and sample["dhand"] == 1
    assert sample["dbench"] == 1 and sample["ddiscard"] == 0
    assert sample["bench_added"] == Counter({11: 1})
    assert sample["hand_added"] == Counter({6: 1})
    assert not sample["active_change"]
    assert tap.pending is None


def test_ability_guard_wrong_pokemon_not_forced():
    tap = ability_probe.ForceTap(lambda od: [1], "ability", target_id=5)
    obs = builders.observation(
        me=builders.player(active=builders.pokemon(1),
                           bench=[builders.pokemon(4)]),
        options=[builders.option(OptionType.ABILITY,
                                 area=AreaType.BENCH, index=0),
                 builders.option(OptionType.END)])
    assert tap(obs) == [1]                       # untouched pilot answer
    assert tap.offered == 0 and tap.pending is None
    assert tap.main_prompts == 1                 # diagnostics still saw it
    assert tap.ability_pokemon == Counter({4: 1})


def test_ability_play_kind_targets_the_exact_hand_card():
    tap = ability_probe.ForceTap(lambda od: [], "play", target_id=7)
    obs = builders.observation(
        me=builders.player(active=builders.pokemon(1),
                           hand=[builders.hand_card(6), builders.hand_card(7)]),
        options=[builders.option(OptionType.PLAY, index=0),
                 builders.option(OptionType.PLAY, index=1),
                 builders.option(OptionType.END)])
    assert tap(obs) == [1]        # index 1 holds card 7; index 0 (card 6) not
    assert tap.offered == 1


def test_ability_cross_turn_diff_marked_unclean():
    tap = ability_probe.ForceTap(lambda od: [1], "ability", target_id=5)
    tap(builders.observation(
        me=builders.player(bench=[builders.pokemon(5)]), turn=1,
        options=[builders.option(OptionType.ABILITY,
                                 area=AreaType.BENCH, index=0),
                 builders.option(OptionType.END)]))
    tap(builders.observation(
        me=builders.player(bench=[builders.pokemon(5)]), turn=2,
        options=[builders.option(OptionType.END)]))
    sample, = tap.samples
    assert not sample["same_turn"]               # confounded, not clean


# ===========================================================================
# scripts/damage_probe.py — printed-vs-engine damage measurement (AttackTap)
# ===========================================================================

def _dmg_obs(opp_active, options, hand=(), me_active=None,
             context=SelectContext.MAIN):
    me = builders.player(
        active=me_active if me_active is not None
        else builders.pokemon(1, energies=[FIGHTING]),
        hand=list(hand))
    return builders.observation(me=me, opponent=builders.player(opp_active),
                                context=context, options=options)


def test_damage_fires_on_next_view_delta():
    tap = damage_probe.AttackTap(lambda od: [0])
    first = _dmg_obs(builders.pokemon(2, hp=100),
                     options=[builders.option(OptionType.ATTACK, index=0),
                              builders.option(OptionType.END)],
                     hand=[builders.hand_card(7)] * 3)
    assert tap(first) == [0]
    # attack id resolved through the card table: fake card 1, slot 0 -> 101
    assert tap.pending[:4] == (101, 3, 2, 100)
    tap(_dmg_obs(builders.pokemon(2, hp=80),
                 options=[builders.option(OptionType.END)]))
    assert tap.events == [(101, 3, 2, 20, ((), ()))]
    assert tap.censored == Counter() and tap.healed == 0


def test_damage_guard_swap_censored_heal_dropped():
    attack = [builders.option(OptionType.ATTACK, index=0),
              builders.option(OptionType.END)]
    end = [builders.option(OptionType.END)]
    # target swapped/KO'd: censored, never an event
    tap = damage_probe.AttackTap(lambda od: [0])
    tap(_dmg_obs(builders.pokemon(2, hp=100), options=attack))
    tap(_dmg_obs(builders.pokemon(3, hp=200), options=end))
    assert tap.events == [] and tap.censored == Counter({101: 1})
    # target healed above the recorded hp: dropped, never an event
    tap = damage_probe.AttackTap(lambda od: [0])
    tap(_dmg_obs(builders.pokemon(2, hp=100), options=attack))
    tap(_dmg_obs(builders.pokemon(2, hp=120), options=end))
    assert tap.events == [] and tap.healed == 1


def test_damage_guard_non_attack_or_non_main_sets_no_pending():
    # END chosen: an hp drop on the next view is attributed to nothing
    script = [[1], [0]]
    tap = damage_probe.AttackTap(lambda od: script.pop(0))
    tap(_dmg_obs(builders.pokemon(2, hp=100),
                 options=[builders.option(OptionType.ATTACK, index=0),
                          builders.option(OptionType.END)]))
    assert tap.pending is None
    tap(_dmg_obs(builders.pokemon(2, hp=60),
                 options=[builders.option(OptionType.END)]))
    assert tap.events == [] and tap.censored == Counter()
    # not a MAIN prompt: same option shape, no measurement
    tap = damage_probe.AttackTap(lambda od: [0])
    tap(_dmg_obs(builders.pokemon(2, hp=100),
                 options=[builders.option(OptionType.ATTACK, index=0)],
                 context=SelectContext.TO_HAND))
    assert tap.pending is None


def test_damage_guard_out_of_range_attack_slot():
    tap = damage_probe.AttackTap(lambda od: [0])
    tap(_dmg_obs(builders.pokemon(2, hp=100),
                 options=[builders.option(OptionType.ATTACK, index=7)]))
    assert tap.pending is None                   # fake card 1 has 3 slots


# ===========================================================================
# scripts/deck_probe.py — census shares + share-weighted aggregation
# ===========================================================================

def _deck_probe():
    pytest.importorskip("polars")
    import scripts.deck_probe as dp
    return dp


def test_deck_weighted_summary_fires():
    dp = _deck_probe()
    import math
    rows = [
        {"candidate": "x", "opponent": "grim", "share": 0.5, "wr": 0.6, "n": 100},
        {"candidate": "x", "opponent": "wall", "share": 0.25, "wr": 0.4, "n": 100},
        # decoy candidate: must not leak into x's row
        {"candidate": "decoy", "opponent": "grim", "share": 0.5, "wr": 0.9, "n": 10},
    ]
    (weighted, name, ci, wsum), = dp.weighted_summary(rows, ["x"])
    assert name == "x" and wsum == pytest.approx(0.75)
    assert weighted == pytest.approx((0.5 * 0.6 + 0.25 * 0.4) / 0.75)
    var = ((0.5 / 0.75) ** 2 * 0.6 * 0.4 / 100
           + (0.25 / 0.75) ** 2 * 0.4 * 0.6 / 100)
    assert ci == pytest.approx(1.96 * math.sqrt(var))


def test_deck_weighted_summary_guards():
    dp = _deck_probe()
    # a candidate whose every cell carries zero share, and one with no cells
    # at all, are both dropped — never a divide-by-zero, never a 0.0 row
    rows = [{"candidate": "x", "opponent": "grim",
             "share": 0.0, "wr": 1.0, "n": 10}]
    assert dp.weighted_summary(rows, ["x", "missing"]) == []


def test_deck_ladder_shares_fires_and_missing_file_guard(tmp_path, monkeypatch):
    pl = pytest.importorskip("polars")
    dp = _deck_probe()
    pq = tmp_path / "census.parquet"
    pl.DataFrame({
        "family": ["grim", "grim", "grim", "ogerpon", "grim", "grim"],
        "label": ["a", "b", "c", "d", None, None],
    }).write_parquet(pq)
    monkeypatch.setattr(dp, "CENSUS_PQ", pq)
    # unlabeled rows are excluded from numerator AND denominator: 3/4 not 5/6
    assert dp.ladder_shares() == {"grim": 0.75, "ogerpon": 0.25}
    monkeypatch.setattr(dp, "CENSUS_PQ",
                        dp.ROOT / "data/kaggle/_does_not_exist.parquet")
    with pytest.raises(SystemExit):
        dp.ladder_shares()


def test_deck_ci95_floors():
    dp = _deck_probe()
    assert dp.ci95(0.5, 100) == pytest.approx(0.098)
    assert dp.ci95(0.0, 100) > 0                 # variance floored, never 0
    dp.ci95(0.5, 0)                              # n floored: no ZeroDivision


# ===========================================================================
# scripts/prize_semantics_probe.py — the semantics asserted about the engine
# ===========================================================================

def test_prize_transition_keys_fires_and_guards():
    # fires: the NON-acting seat's array drained — the sample-agent inversion
    assert prize_probe.prize_transition_keys(0, (6, 6), (6, 5)) == \
        ["other_drained"]
    assert prize_probe.prize_transition_keys(1, (6, 6), (5, 5)) == \
        ["other_drained", "actor_drained"]
    # guards: the pinned convention, and non-drain changes
    assert prize_probe.prize_transition_keys(0, (6, 6), (5, 6)) == \
        ["actor_drained"]
    assert prize_probe.prize_transition_keys(0, (6, 6), (6, 6)) == []
    assert prize_probe.prize_transition_keys(0, (5, 6), (6, 6)) == []  # grew


def test_prize_end_state_key_fires_and_guards():
    # fires: the LOSER's array reached zero — inverted semantics
    assert prize_probe.end_state_key(0, (3, 0)) == "loser_array_zero"
    # the pinned convention: the winner's own array reaches zero
    assert prize_probe.end_state_key(0, (0, 4)) == "winner_array_zero"
    assert prize_probe.end_state_key(1, (2, 0)) == "winner_array_zero"
    # guards: draws, unfinished games, and wins with no empty array
    assert prize_probe.end_state_key(2, (0, 4)) is None
    assert prize_probe.end_state_key(-1, (0, 4)) is None
    assert prize_probe.end_state_key(0, (2, 4)) is None


def test_prize_endstate_verdict():
    assert prize_probe.endstate_verdict(
        Counter(loser_array_zero=1, actor_drained=9)) == "mismatch"
    assert prize_probe.endstate_verdict(
        Counter(other_drained=2, actor_drained=5)) == "mismatch"
    assert prize_probe.endstate_verdict(
        Counter(winner_array_zero=3, actor_drained=7)) == "ok"
    # guard: an all-zero tally is inconclusive, never a pass
    assert prize_probe.endstate_verdict(Counter()) == "inconclusive"


def test_prize_leaf_keys_and_verdict():
    # fires: a leaf where WE took prizes scored negative — the score_leaf
    # inversion that made every solver knockout read -150,000
    assert prize_probe.leaf_keys(-150000.0, -1, 2) == \
        ["prize_leaves", "negative"]
    assert prize_probe.leaf_verdict(
        Counter(prize_leaves=3, negative=1, positive=2)) == "regressed"
    # guards: positive prize leaves, terminal leaves, no-prize leaves
    assert prize_probe.leaf_keys(500.0, -1, 1) == ["prize_leaves", "positive"]
    assert prize_probe.leaf_keys(-5.0, 0, 2) == []     # game already over
    assert prize_probe.leaf_keys(-5.0, -1, 0) == []    # we took nothing
    assert prize_probe.leaf_verdict(Counter(prize_leaves=2, positive=2)) == "ok"
    assert prize_probe.leaf_verdict(Counter()) == "inconclusive"


# ===========================================================================
# scripts/m42_perception_probe.py — the five per-defect ladders + the
# foreign-target counter, driven through instrument() exactly as live
# ===========================================================================

def _ladder_run(lad, obs_picks):
    """Feed (obs, picks) pairs through one instrumented scripted pilot."""
    script = [list(p) for _, p in obs_picks]
    wrapped = m42_probe.instrument(lambda od: script.pop(0), lad)
    for obs, _ in obs_picks:
        wrapped(obs)
    assert lad.c["probe_errors"] == 0            # nothing swallowed


def test_m42_over_attach_fires_on_dead_target():
    lad = m42_probe.Ladders()
    # fake card 5: only attack costs nothing, retreat 0 -> energy is dead
    me = builders.player(active=builders.pokemon(1),
                         bench=[builders.pokemon(5)])
    obs = builders.observation(me=me, options=[
        builders.option(OptionType.ATTACH, in_play_area=5, in_play_index=0),
        builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["over_attach.situation"] == 1
    assert lad.c["over_attach.offered"] == 1
    assert lad.c["over_attach.chose_bad"] == 1
    assert lad.detail["over_attach"]["id5"] == 1


def test_m42_over_attach_guards():
    # near-miss 1: the target still has an unpaid attack (card 1, attack 102)
    lad = m42_probe.Ladders()
    me = builders.player(active=builders.pokemon(5),
                         bench=[builders.pokemon(1)])
    obs = builders.observation(me=me, options=[
        builders.option(OptionType.ATTACH, in_play_area=5, in_play_index=0),
        builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["over_attach.situation"] == 0
    # near-miss 2: dead target offered but DECLINED
    lad = m42_probe.Ladders()
    me = builders.player(active=builders.pokemon(1),
                         bench=[builders.pokemon(5)])
    obs = builders.observation(me=me, options=[
        builders.option(OptionType.ATTACH, in_play_area=5, in_play_index=0),
        builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [1])])
    assert lad.c["over_attach.situation"] == 1
    assert lad.c["over_attach.chose_bad"] == 0


def test_m42_over_attach_damage_forgone_counts_hand_scaler(monkeypatch):
    """Each surplus attach spends a card out of a hand-scaling formula:
    per-unit comes from rl.scaling's curated Powerful Hand entry (20/card)."""
    monkeypatch.setattr(m42_probe, "_CARD",
                        {**m42_probe._CARD, 1: (None, None, 6, (1072,), 1)})
    lad = m42_probe.Ladders()
    me = builders.player(active=builders.pokemon(1),
                         bench=[builders.pokemon(5)])
    obs = builders.observation(me=me, options=[
        builders.option(OptionType.ATTACH, in_play_area=5, in_play_index=0),
        builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["over_attach.chose_bad"] == 1
    assert lad.c["over_attach.damage_forgone"] == 20


def test_m42_hand_scaler_blind_fires(monkeypatch):
    """Active carries a hand scaler, an ATTACK is on the menu, and
    _best_damage genuinely reads 0 (no energy attached, real combat math)."""
    monkeypatch.setattr(m42_probe, "_CARD",
                        {**m42_probe._CARD, 1: (None, None, 6, (101, 1072), 1)})
    lad = m42_probe.Ladders()
    obs = builders.observation(
        me=builders.player(active=builders.pokemon(1, energies=[])),
        opponent=builders.player(active=builders.pokemon(4, hp=100)),
        options=[builders.option(OptionType.ATTACK, index=0),
                 builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["hand_scaler_blind.situation"] == 1
    assert lad.c["hand_scaler_blind.offered"] == 1
    assert lad.c["hand_scaler_blind.chose_bad"] == 1
    assert lad.c["hand_scaler_blind.attacked_anyway"] == 1  # the 617/617 read


def test_m42_hand_scaler_blind_guards(monkeypatch):
    monkeypatch.setattr(m42_probe, "_CARD",
                        {**m42_probe._CARD, 1: (None, None, 6, (101, 1072), 1)})
    # near-miss 1: charged — _best_damage reads 50, the model is not blind
    lad = m42_probe.Ladders()
    obs = builders.observation(
        me=builders.player(active=builders.pokemon(1, energies=[FIGHTING])),
        opponent=builders.player(active=builders.pokemon(4, hp=100)),
        options=[builders.option(OptionType.ATTACK, index=0),
                 builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["hand_scaler_blind.situation"] == 1
    assert lad.c["hand_scaler_blind.offered"] == 0
    # near-miss 2: no hand scaler on the active at all (card 10)
    lad = m42_probe.Ladders()
    obs = builders.observation(
        me=builders.player(active=builders.pokemon(10, energies=[])),
        opponent=builders.player(active=builders.pokemon(4, hp=100)),
        options=[builders.option(OptionType.ATTACK, index=0),
                 builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["hand_scaler_blind.situation"] == 0


def _stranded_board(active_energies, bench_energies):
    return (builders.player(
                active=builders.pokemon(1, energies=list(active_energies)),
                bench=[builders.pokemon(1, energies=list(bench_energies))]),
            builders.player(active=builders.pokemon(4, hp=100)))


def test_m42_retreat_stranded_fires(monkeypatch):
    monkeypatch.setattr(m42_probe, "_RETREAT", {1: 2})
    lad = m42_probe.Ladders()
    me, opp = _stranded_board([FIGHTING], [FIGHTING, FIGHTING])
    obs = builders.observation(me=me, opponent=opp, options=[
        builders.option(OptionType.ATTACH, in_play_area=4, in_play_index=0),
        builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [1])])               # ended the turn stranded
    assert lad.c["retreat_stranded.situation"] == 1
    assert lad.c["retreat_stranded.offered"] == 1
    assert lad.c["retreat_stranded.chose_bad"] == 1
    assert lad.detail["retreat_stranded"]["id1"] == 1


def test_m42_retreat_stranded_guards(monkeypatch):
    monkeypatch.setattr(m42_probe, "_RETREAT", {1: 2})
    attach_end = [builders.option(OptionType.ATTACH,
                                  in_play_area=4, in_play_index=0),
                  builders.option(OptionType.END)]
    # near-miss 1 (the plan's fixture): NO legal way out — the board's fault,
    # not the pilot's; ending the turn must not count
    lad = m42_probe.Ladders()
    me, opp = _stranded_board([FIGHTING], [FIGHTING, FIGHTING])
    obs = builders.observation(me=me, opponent=opp,
                               options=[builders.option(OptionType.END)])
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["retreat_stranded.situation"] == 1
    assert lad.c["retreat_stranded.offered"] == 0
    assert lad.c["retreat_stranded.chose_bad"] == 0
    # near-miss 2: fixed it (chose the ATTACH) instead of ending
    lad = m42_probe.Ladders()
    me, opp = _stranded_board([FIGHTING], [FIGHTING, FIGHTING])
    obs = builders.observation(me=me, opponent=opp, options=attach_end)
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["retreat_stranded.offered"] == 1
    assert lad.c["retreat_stranded.chose_bad"] == 0
    # near-miss 3: retreat is payable — not stranded
    lad = m42_probe.Ladders()
    me, opp = _stranded_board([FIGHTING, FIGHTING], [FIGHTING, FIGHTING])
    obs = builders.observation(me=me, opponent=opp, options=attach_end)
    _ladder_run(lad, [(obs, [1])])
    assert lad.c["retreat_stranded.situation"] == 0
    # near-miss 4: no ready bench member waiting
    lad = m42_probe.Ladders()
    me, opp = _stranded_board([FIGHTING], [])
    obs = builders.observation(me=me, opponent=opp, options=attach_end)
    _ladder_run(lad, [(obs, [1])])
    assert lad.c["retreat_stranded.situation"] == 0


def _fetch_obs(deck_ids, context, me=None):
    """A fetch/promote menu over `select.deck` (fake pool: 12 evolves from 11
    BY NAME; 11 is the basic)."""
    return builders.observation(
        me=me if me is not None else builders.player(
            active=builders.pokemon(5)),
        context=context,
        select_deck=[builders.hand_card(i) for i in deck_ids],
        options=[builders.option(OptionType.CARD, area=int(AreaType.DECK),
                                 index=k) for k in range(len(deck_ids))])


def test_m42_dead_basis_fetch_fires_in_keep_and_promote():
    # KEEP: took the dead evolution while a live basic sat on the same menu
    lad = m42_probe.Ladders()
    _ladder_run(lad, [(_fetch_obs([12, 11], SelectContext.TO_HAND), [0])])
    assert lad.c["fetch_prompts"] == 1
    assert lad.c["dead_basis_fetch.situation"] == 1
    assert lad.c["dead_basis_fetch.offered"] == 1
    assert lad.c["dead_basis_fetch.chose_bad"] == 1
    assert lad.c["dead_basis_fetch.in_KEEP"] == 1
    # Piotr's strict trigger: bench empty AND no basic in hand
    assert lad.c["dead_basis_fetch.strict_situation"] == 1
    assert lad.c["dead_basis_fetch.strict_chose_bad"] == 1
    assert lad.detail["dead_basis_fetch"]["Stub Evolution [KEEP]"] == 1
    # PROMOTE: same pick under a promote context lands in the other bucket
    lad = m42_probe.Ladders()
    _ladder_run(lad, [(_fetch_obs([12, 11], SelectContext.TO_BENCH), [0])])
    assert lad.c["dead_basis_fetch.chose_bad"] == 1
    assert lad.c["dead_basis_fetch.in_PROMOTE"] == 1


def test_m42_dead_basis_fetch_guards():
    # near-miss 1: the basis IS on the bench — the evolution is live
    lad = m42_probe.Ladders()
    me = builders.player(active=builders.pokemon(5),
                         bench=[builders.pokemon(11)])
    _ladder_run(lad, [(_fetch_obs([12, 11], SelectContext.TO_HAND, me=me),
                       [0])])
    assert lad.c["dead_basis_fetch.situation"] == 0
    # near-miss 2: took the live basic AND the dead evolution — declined
    # nothing, so neither offered nor chose_bad may count
    lad = m42_probe.Ladders()
    _ladder_run(lad, [(_fetch_obs([12, 11], SelectContext.TO_HAND), [0, 1])])
    assert lad.c["dead_basis_fetch.situation"] == 1
    assert lad.c["dead_basis_fetch.offered"] == 0
    assert lad.c["dead_basis_fetch.chose_bad"] == 0
    # near-miss 3 (the Kadabra false positive): promoting an ALREADY-BENCHED
    # evolution needs no basis anywhere
    lad = m42_probe.Ladders()
    me = builders.player(active=builders.pokemon(5),
                         bench=[builders.pokemon(12)])
    obs = builders.observation(
        me=me, context=SelectContext.TO_ACTIVE,
        options=[builders.option(OptionType.CARD,
                                 area=int(AreaType.BENCH), index=0)])
    _ladder_run(lad, [(obs, [0])])
    assert lad.c["fetch_prompts"] == 1
    assert lad.c["dead_basis_fetch.situation"] == 0


def _stadium_obs(hand_id, in_play, turn=4):
    me = builders.player(active=builders.pokemon(1),
                         hand=[builders.hand_card(hand_id)])
    return builders.observation(
        me=me, turn=turn, stadium=list(in_play),
        options=[builders.option(OptionType.PLAY, index=0),
                 builders.option(OptionType.END)])


def test_m42_stadium_ignored_fires_per_turn(monkeypatch):
    monkeypatch.setattr(m42_probe, "_IS_STADIUM", {31})
    lad = m42_probe.Ladders()
    theirs = SimpleNamespace(id=32, playerIndex=1)
    _ladder_run(lad, [(_stadium_obs(31, [theirs]), [1])])   # declined
    assert lad.c["stadium_ignored.offered"] == 0    # banked per TURN
    lad.end_game()
    assert lad.c["stadium_ignored.situation"] == 1
    assert lad.c["stadium_ignored.offered"] == 1
    assert lad.c["stadium_ignored.chose_bad"] == 1
    assert lad.detail["stadium_ignored"]["id32"] == 1


def test_m42_stadium_ignored_guards(monkeypatch):
    monkeypatch.setattr(m42_probe, "_IS_STADIUM", {31})
    theirs = SimpleNamespace(id=32, playerIndex=1)
    # near-miss 1 (the plan's prompt-5 fixture): declined at the first MAIN
    # prompt, played at a LATER prompt of the same turn — not a miss
    lad = m42_probe.Ladders()
    _ladder_run(lad, [(_stadium_obs(31, [theirs]), [1]),
                      (_stadium_obs(31, [theirs]), [0])])
    lad.end_game()
    assert lad.c["stadium_ignored.offered"] == 1
    assert lad.c["stadium_ignored.chose_bad"] == 0
    # near-miss 2: the stadium out is OURS — displacing it is not the defect
    lad = m42_probe.Ladders()
    ours = SimpleNamespace(id=32, playerIndex=0)
    _ladder_run(lad, [(_stadium_obs(31, [ours]), [1])])
    lad.end_game()
    assert lad.c["stadium_ignored.ours_already_out"] == 1
    assert lad.c["stadium_ignored.offered"] == 0
    # near-miss 3: a same-id offer only feeds the void-framing tripwire
    lad = m42_probe.Ladders()
    same = SimpleNamespace(id=31, playerIndex=1)
    _ladder_run(lad, [(_stadium_obs(31, [same]), [1])])
    lad.end_game()
    assert lad.c["same_id_offered"] == 1
    assert lad.c["stadium_ignored.offered"] == 0
    # near-miss 4: empty stadium slot — declining displaces nothing
    lad = m42_probe.Ladders()
    _ladder_run(lad, [(_stadium_obs(31, []), [1])])
    lad.end_game()
    assert lad.c["stadium_ignored.situation"] == 1
    assert lad.c["stadium_ignored.offered"] == 0


def test_m42_foreign_target_opts_fires_and_guards():
    lad = m42_probe.Ladders()
    obs = builders.observation(me=builders.player(), options=[
        # fires: addresses a board that is NOT ours through inPlayArea
        builders.option(OptionType.CARD, in_play_area=4, in_play_index=0,
                        player_index=1),
        # guards: our own board, and a foreign option with no inPlayArea
        builders.option(OptionType.CARD, in_play_area=4, in_play_index=0,
                        player_index=0),
        builders.option(OptionType.CARD, player_index=1)])
    _ladder_run(lad, [(obs, [])])
    assert lad.c["foreign_target_opts"] == 1


def test_m42_report_flags_situation_never_occurs(capsys):
    args = SimpleNamespace(games=2, arm="a", opp="b")
    lad = m42_probe.Ladders()
    for name in m42_probe.LADDERS:
        lad.c[f"{name}.situation"] = 1
        lad.c[f"{name}.offered"] = 1
    assert m42_probe.report(lad, args) == 0
    lad.c["retreat_stranded.situation"] = 0
    assert m42_probe.report(lad, args) == 1      # a dead ladder measured nothing
    assert "SITUATION NEVER OCCURS: retreat_stranded" in capsys.readouterr().out
