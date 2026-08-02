"""M40 S2: the resume contract and the sampler.

An untested resume is worse than no resume — it invites you to trust a restart
that silently drops or duplicates work. These cover the three pure pieces that
carry the contract: manifest durability, the config fingerprint that stops two
corpora being blended into one dir, and the sampler that the pre-registered
collection kill is measured against.
"""
import json
from argparse import Namespace

import pytest

from scripts.m40_s2_collect import (BEDS, config_fingerprint, load_manifest,
                                    make_sampler, save_manifest)


def _args(**kw):
    base = dict(arm="model:ck.pt:deck", deck="alakazam_v2_h4", tau=0.0,
                eps=0.0, top_k=3, chunk_games=50)
    base.update(kw)
    return Namespace(**base)


# --- manifest durability ----------------------------------------------------

def test_manifest_roundtrips_and_defaults_empty(tmp_path):
    assert load_manifest(tmp_path) == {"config": None, "chunks": {}}
    man = {"config": {"tau": 0.6}, "chunks": {"grim_d1:1:0": {"rows": 12}}}
    save_manifest(tmp_path, man)
    assert load_manifest(tmp_path) == man


def test_manifest_write_leaves_no_partial_file(tmp_path):
    """save_manifest writes a tmp then os.replace's it. A crash mid-write must
    never leave an unreadable manifest, because that would turn a recoverable
    pause into total loss of the run."""
    save_manifest(tmp_path, {"config": None, "chunks": {}})
    for i in range(5):
        save_manifest(tmp_path, {"config": None,
                                 "chunks": {f"c{j}": {} for j in range(i)}})
        json.loads((tmp_path / "s2_manifest.json").read_text())   # always valid
    assert not list(tmp_path.glob("*.tmp")), "tmp file survived the replace"


# --- the anti-blending guard ------------------------------------------------

def test_config_fingerprint_separates_corpora_that_must_not_mix():
    base = config_fingerprint(_args(tau=0.6))
    assert base == config_fingerprint(_args(tau=0.6))
    # Everything that changes what a ROW MEANS must change the fingerprint.
    for field, val in (("tau", 0.9), ("eps", 0.1), ("top_k", 5),
                       ("arm", "model:other.pt:deck"), ("deck", "lucario")):
        assert config_fingerprint(_args(**{field: val})) != base, field


def test_fingerprint_ignores_nothing_that_matters_to_a_row():
    # chunk_games is included deliberately: it sets game_id bucketing, so
    # resuming across a change to it can collide ids between chunks.
    assert config_fingerprint(_args(chunk_games=50)) != \
        config_fingerprint(_args(chunk_games=25))


# --- the sampler, which the pre-registered kill is measured against ---------

def test_greedy_by_default_is_pure_agreement():
    """tau=0, eps=0 must never deviate. This is the agreement FLOOR used to
    measure an unexplored corpus before choosing a rate."""
    import random
    pick = make_sampler(_args(), random.Random(0))
    for _ in range(200):
        idx, explored = pick([0.1, 5.0, 0.2], base=1)
        assert idx == 1 and not explored


def test_epsilon_stays_inside_the_top_k():
    """M39's bestresp drew UNIFORMLY over all legal options, so an explored
    action was usually nonsense the net would never consider — which is why
    the corpus came out 93.5% agreeing. Top-k keeps a deviation plausible."""
    import random
    logits = [9.0, 8.0, 7.0, -50.0, -60.0]
    pick = make_sampler(_args(eps=1.0, top_k=2), random.Random(1))
    seen = {pick(logits, base=0)[0] for _ in range(200)}
    assert seen <= {0, 1}, f"escaped the top-2: {seen}"


def test_temperature_explores_and_lower_temperature_explores_less():
    import random
    logits = [3.0, 2.5, 2.0, 1.0]

    def rate(tau, n=3000):
        pick = make_sampler(_args(tau=tau), random.Random(7))
        return sum(pick(logits, base=0)[0] != 0 for _ in range(n)) / n

    hot, cold = rate(1.5), rate(0.15)
    assert hot > cold, f"temperature is inverted: hot={hot} cold={cold}"
    assert cold < 0.2, "a near-zero temperature should be nearly greedy"
    assert hot > 0.3, "a high temperature should deviate often"


def test_explored_flag_is_deviation_not_draw():
    """`explored` must count actions that DIFFER from the arm's own pick, not
    sampler invocations — otherwise the reported exploration rate is inflated
    by draws that happened to land on the greedy action, and the agreement
    number stops meaning what the kill gate reads it as."""
    import random
    pick = make_sampler(_args(tau=0.001), random.Random(3))
    idx, explored = pick([10.0, -10.0, -10.0], base=0)
    assert idx == 0 and not explored


# --- the opponent pool ------------------------------------------------------

def test_bed_pool_spans_draws_and_includes_composites():
    import rl.matchrunner as mr
    for name, spec in BEDS.items():
        mr.parse_spec(spec)                      # every entry must parse
    families = {n.rsplit("_d", 1)[0] for n in BEDS if "_d" in n}
    for fam in ("wall", "grim", "arch"):
        draws = [n for n in BEDS if n.startswith(fam + "_d")]
        assert len(draws) >= 3, f"{fam} is not a panel: {draws}"
    assert families
    comps = [n for n in BEDS if n.startswith("comp_")]
    assert comps, "no composite beds — the only legitimate route to a bed above the clone ceiling"
    for c in comps:
        assert BEDS[c].startswith("solved:")
        # per-spec budget must be pinned, not left to the module default
        assert len(BEDS[c].split(":")) == 5, f"{c} has no explicit budget"


@pytest.mark.parametrize("bed", sorted(BEDS))
def test_every_bed_spec_parses_and_names_a_deck(bed):
    import rl.matchrunner as mr
    spec = mr.parse_spec(BEDS[bed])
    assert mr.spec_deck(spec)
