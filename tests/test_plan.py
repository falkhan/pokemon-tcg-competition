"""M11 rl/plan.py: encode / enumerate / derive on fake-cg card data.

Fake table (tests/fake_cg.py): card 1 = Fighting attacker (attacks 101: 50dmg/1F,
102: 120dmg/F+C), card 2 = Water ex resisting Fighting, card 4 = 30-dmg
attacker, card 5 = no damaging attack, cards 6/8 = energy, 7 = trainer.
"""
import numpy as np
import pytest

from tests import builders as b
from tests.fake_cg import AreaType, EnergyType, OptionType

import rl.plan as rp

F = EnergyType.FIGHTING


def _plan(**kw):
    base = dict(attacker_slot=0, target_slot=0, attack_idx=0, attack_id=101,
                needs_attach=False, needs_gust=False, damage=50,
                target_prize=1, lethal=False, wins=False, attacker_prize=1,
                return_ko=False, concedes=False, opp_ttk=2)
    base.update(kw)
    return rp.Plan(**base)


class TestEncodePlan:
    def test_null_plan_is_all_zeros(self):
        assert not rp.encode_plan(None).any()
        assert rp.encode_plan(None).shape == (rp.PLAN_DIM,)

    def test_layout(self):
        v = rp.encode_plan(_plan(attacker_slot=2, target_slot=1, attack_idx=1,
                                 needs_attach=True, needs_gust=True,
                                 damage=170, target_prize=3, lethal=True,
                                 wins=True, attacker_prize=3, return_ko=True,
                                 concedes=True, opp_ttk=2))
        assert v[0] == 1.0
        assert v[1 + 2] == 1.0 and v[1:7].sum() == 1.0     # attacker one-hot
        assert v[7 + 1] == 1.0 and v[7:13].sum() == 1.0    # target one-hot
        assert v[13 + 1] == 1.0 and v[13:17].sum() == 1.0  # attack one-hot
        assert v[17] == 1.0 and v[18] == 1.0
        assert v[19] == pytest.approx(170 / 340)
        assert v[20] == pytest.approx(1.0)                  # 3 prizes / 3
        assert v[21] == 1.0 and v[22] == 1.0
        assert v[23] == pytest.approx(1.0)                  # attacker prize 3/3
        assert v[24] == 1.0 and v[25] == 1.0
        assert v[26] == pytest.approx(2 / 4)


class TestAttackDamage:
    def test_specific_attack_with_resistance(self):
        # attack 101 (50, Fighting) into card 2 which resists Fighting: 20.
        attacker = b.pokemon(1, energies=[F])
        assert rp._attack_damage(attacker, b.pokemon(2), 101) == 20

    def test_unaffordable_without_extra_energy(self):
        attacker = b.pokemon(1, energies=[F])            # 102 needs F+C
        assert rp._attack_damage(attacker, b.pokemon(5), 102) == 0
        assert rp._attack_damage(attacker, b.pokemon(5), 102,
                                 extra_energy=1) == 120


class TestEnumeratePlans:
    def test_null_first_and_no_opponent_means_null_only(self):
        obs = b.observation(me=b.player(active=b.pokemon(1, energies=[F])))
        # default opponent has no active
        assert rp.enumerate_plans(obs) == [None]

    def test_active_target_only_without_gust(self):
        me = b.player(active=b.pokemon(1, energies=[F]))
        opp = b.player(active=b.pokemon(4, hp=300), bench=[b.pokemon(5, hp=50)])
        cands = rp.enumerate_plans(b.observation(me=me, opponent=opp))
        assert cands[0] is None
        assert all(c.target_slot == 0 for c in cands[1:])
        assert len(cands) > 1

    def test_gust_in_hand_unlocks_bench_targets(self, monkeypatch):
        monkeypatch.setattr(rp, "GUST_IDS", frozenset({7}))
        me = b.player(active=b.pokemon(1, energies=[F]),
                      hand=[b.hand_card(7)])
        opp = b.player(active=b.pokemon(4, hp=300), bench=[b.pokemon(5, hp=50)])
        cands = rp.enumerate_plans(b.observation(me=me, opponent=opp))
        gusts = [c for c in cands[1:] if c.target_slot > 0]
        assert gusts and all(c.needs_gust for c in gusts)

    def test_needs_attach_requires_hand_energy(self):
        # attacker holds 1 F: attack 102 (F+C) reachable only via +1 attach.
        opp = b.player(active=b.pokemon(5, hp=200))
        no_energy = b.observation(
            me=b.player(active=b.pokemon(1, energies=[F])), opponent=opp)
        with_energy = b.observation(
            me=b.player(active=b.pokemon(1, energies=[F]),
                        hand=[b.hand_card(6)]), opponent=opp)
        ids_no = {(c.attack_id, c.needs_attach)
                  for c in rp.enumerate_plans(no_energy)[1:]}
        ids_with = {(c.attack_id, c.needs_attach)
                    for c in rp.enumerate_plans(with_energy)[1:]}
        assert (102, True) not in ids_no
        assert (102, True) in ids_with

    def test_risk_flags_on_sacrificial_plan(self):
        # My 3-prize mega (card 3) attacks while opp active can return-KO it
        # and the OPPONENT has <= 3 prizes left to take (their array drains
        # as THEY take prizes — M38 P2 fix): concedes_game must flag.
        # Pre-fix this fixture set MY prizes low, encoding the inversion.
        me = b.player(active=b.pokemon(3, hp=50, energies=[F, F]))
        opp = b.player(active=b.pokemon(1, hp=400, energies=[F, F]),
                       prizes_remaining=2)
        cands = rp.enumerate_plans(b.observation(me=me, opponent=opp))
        atk = [c for c in cands[1:] if c.attacker_slot == 0]
        assert atk and all(c.return_ko and c.concedes for c in atk)


class TestDerivePlan:
    def _root(self):
        self.op_active = b.pokemon(4, hp=300)
        self.op_bench = b.pokemon(5, hp=50)
        me = b.player(active=b.pokemon(1, energies=[F]))
        opp = b.player(active=self.op_active, bench=[self.op_bench])
        return b.observation(me=me, opponent=opp)

    def test_plain_attack_line(self):
        root = self._root()
        atk_obs = b.observation(
            me=b.player(active=b.pokemon(1, energies=[F])),
            opponent=b.player(active=self.op_active),
            options=[b.option(OptionType.ATTACK, attack_id=101),
                     b.option(OptionType.END)])
        plan = rp.derive_plan([[0]], [atk_obs], root)
        assert plan is not None
        assert (plan.attacker_slot, plan.target_slot, plan.attack_id) == (0, 0, 101)
        assert not plan.needs_attach and not plan.needs_gust

    def test_gust_line_maps_bench_target(self):
        root = self._root()
        play_obs = b.observation(
            me=b.player(active=b.pokemon(1, energies=[F]),
                        hand=[b.hand_card(7)]),
            opponent=b.player(active=self.op_active, bench=[self.op_bench]),
            options=[b.option(OptionType.PLAY, area=AreaType.HAND, index=0),
                     b.option(OptionType.END)])
        atk_obs = b.observation(
            me=b.player(active=b.pokemon(1, energies=[F])),
            opponent=b.player(active=self.op_bench),   # bench got promoted
            options=[b.option(OptionType.ATTACK, attack_id=101),
                     b.option(OptionType.END)])
        plan = rp.derive_plan([[0], [0]], [play_obs, atk_obs], root)
        assert plan is not None
        assert plan.target_slot == 1 and plan.needs_gust
        assert plan.lethal                                # 50 dmg vs 50 hp

    def test_attach_before_attack_sets_needs_attach(self):
        root = self._root()
        attach_obs = b.observation(
            me=b.player(active=b.pokemon(1, energies=[F]),
                        hand=[b.hand_card(6)]),
            opponent=b.player(active=self.op_active),
            options=[b.option(OptionType.ATTACH, area=AreaType.HAND, index=0),
                     b.option(OptionType.END)])
        atk_obs = b.observation(
            me=b.player(active=b.pokemon(1, energies=[F, F])),
            opponent=b.player(active=self.op_active),
            options=[b.option(OptionType.ATTACK, attack_id=102),
                     b.option(OptionType.END)])
        plan = rp.derive_plan([[0], [0]], [attach_obs, atk_obs], root)
        assert plan is not None and plan.needs_attach

    def test_no_attack_means_null_plan(self):
        root = self._root()
        obs = b.observation(options=[b.option(OptionType.PLAY,
                                              area=AreaType.HAND, index=0),
                                     b.option(OptionType.END)])
        assert rp.derive_plan([[1]], [obs], root) is None


class TestMatchCandidate:
    def test_null_matches_index_zero_and_slots_match(self):
        me = b.player(active=b.pokemon(1, energies=[F]))
        opp = b.player(active=b.pokemon(4, hp=300))
        cands = rp.enumerate_plans(b.observation(me=me, opponent=opp))
        assert rp.match_candidate(None, cands) == 0
        assert rp.match_candidate(cands[1], cands) == 1
        missing = _plan(attacker_slot=5, target_slot=5, attack_idx=3)
        assert rp.match_candidate(missing, cands) == -1
