"""Golden fixtures for the two Stage B probes (M41b § II.3a, extended).

`scripts/m41b_divergence_probe.py` produced the number that shaped the corpus
design, and `scripts/m41b_wall_probe.py` produced the swap table that led to
the width census. Both were written after § II.3a landed, and the coverage
registry caught them missing fixtures on the next `ci_gate` run — which is
the registry doing its job, so they get the same treatment as the other 20.

Ground truth is arithmetic and card-table facts, both true by construction.
"""
import pytest

pytest.importorskip("numpy")

import scripts.m41b_divergence_probe as dp  # noqa: E402
import scripts.m41b_wall_probe as wp  # noqa: E402
from tests import builders  # noqa: E402
from tests.fake_cg import AreaType, OptionType  # noqa: E402


# --- divergence probe: the rates that decide corpus design ------------------

def test_divergence_summarize_fires_on_the_measured_shape():
    """The real M41b reading: 2,815 prompts, 151 fires, 77 changes."""
    s = dp.summarize({"prompts": 2815, "solver_fired": 151, "changed": 77,
                      "trig_T2_trainer_gap": 651, "fire_T2_trainer_gap": 131,
                      "trig_T1_ko_one_attach": 1, "fire_T1_ko_one_attach": 1})
    assert s["change_rate"] == pytest.approx(77 / 2815)
    assert s["fire_rate"] == pytest.approx(151 / 2815)
    assert s["change_given_fire"] == pytest.approx(77 / 151)
    assert s["rows_per_1000"] == pytest.approx(27.35, abs=0.01)
    assert s["tiers"]["T2_trainer_gap"] == (651, 131)
    assert s["tiers"]["T1_ko_one_attach"] == (1, 1)
    assert s["sparse"] is False          # 2.7% clears the 2% bar


def test_divergence_summarize_calls_a_thin_corpus_sparse():
    """The branch that changes the corpus design: below 2% of prompts,
    uniform cross-entropy is mostly the student learning itself."""
    s = dp.summarize({"prompts": 10_000, "solver_fired": 300, "changed": 100})
    assert s["change_rate"] == pytest.approx(0.01)
    assert s["sparse"] is True


def test_divergence_summarize_guards_an_empty_run():
    """No prompts must read as zeros, never a ZeroDivisionError and never a
    verdict — the probe's main() turns this into an explicit FAIL."""
    s = dp.summarize({})
    assert s["prompts"] == 0 and s["changed"] == 0
    assert s["change_rate"] == 0.0 and s["change_given_fire"] == 0.0
    assert s["tiers"] == {}


def test_divergence_summarize_guards_fires_with_no_changes():
    """The solver can fire and agree — that is the `fired_but_same` half, and
    it must not be counted as distillable signal."""
    s = dp.summarize({"prompts": 100, "solver_fired": 40, "changed": 0})
    assert s["change_given_fire"] == 0.0
    assert s["sparse"] is True


# --- wall probe: the option labels the swap table is built from -------------

def test_wall_probe_labels_an_attack_with_its_PRINTED_damage():
    """The label reads `_ATK`'s printed number on purpose: `ATTACK(dmg=0)` in
    the swap table is the SIGNAL that the net is looking at a scaling attack
    whose printed damage is a lie. fake_cg attack 103 prints 0."""
    obs = builders.observation(me=builders.player(active=builders.pokemon(1)),
                               opponent=builders.player(
                                   active=builders.pokemon(2)))
    zero = builders.option(OptionType.ATTACK, attack_id=103)
    real = builders.option(OptionType.ATTACK, attack_id=102)   # prints 120
    assert wp.option_label(zero, obs) == "ATTACK(dmg=0)"
    assert wp.option_label(real, obs) == "ATTACK(dmg=120)"


def test_wall_probe_labels_attach_by_destination():
    """Active vs bench is the distinction the over-attach reading turns on."""
    obs = builders.observation(
        me=builders.player(active=builders.pokemon(1),
                           bench=[builders.pokemon(3)]),
        opponent=builders.player(active=builders.pokemon(2)))
    to_active = builders.option(OptionType.ATTACH,
                                in_play_area=AreaType.ACTIVE, in_play_index=0)
    to_bench = builders.option(OptionType.ATTACH,
                               in_play_area=AreaType.BENCH, in_play_index=0)
    assert wp.option_label(to_active, obs) == "ATTACH->ACTIVE"
    assert wp.option_label(to_bench, obs) == "ATTACH->BENCH0"


def test_wall_probe_labels_a_play_by_the_hand_card():
    """PLAY options carry only a hand index (the M16/engine quirk), so the
    label has to resolve it or every trainer reads the same."""
    obs = builders.observation(
        me=builders.player(active=builders.pokemon(1),
                           hand=[builders.hand_card(7)]),
        opponent=builders.player(active=builders.pokemon(2)))
    play = builders.option(OptionType.PLAY, index=0)
    assert wp.option_label(play, obs).startswith("PLAY(")


def test_wall_probe_guards_types_it_has_no_detail_for():
    """Anything else falls back to the bare type name rather than inventing
    a detail — a wrong detail in the swap table would misdirect the reading."""
    obs = builders.observation(me=builders.player(active=builders.pokemon(1)),
                               opponent=builders.player(
                                   active=builders.pokemon(2)))
    for opt_type in (OptionType.END, OptionType.RETREAT, OptionType.EVOLVE):
        assert wp.option_label(builders.option(opt_type), obs) == opt_type.name


def test_wall_probe_guards_an_attack_with_no_id():
    """A malformed ATTACK option must not crash the probe mid-battery."""
    obs = builders.observation(me=builders.player(active=builders.pokemon(1)),
                               opponent=builders.player(
                                   active=builders.pokemon(2)))
    assert wp.option_label(
        builders.option(OptionType.ATTACK), obs) == "ATTACK(dmg=0)"


# --- the corpus slicer (M41b R2) --------------------------------------------

def test_slicing_is_exact_and_keeps_every_other_column(tmp_path):
    """The control corpus must differ from the arm's in EXACTLY one thing.
    Slicing is the append-and-slice law applied at training time: [0, N) of
    the wide encoding IS the narrow encoding."""
    import numpy as np

    import scripts.m41b_slice_corpus as sl

    src, dst = tmp_path / "wide", tmp_path / "narrow"
    src.mkdir()
    rng = np.random.default_rng(0)
    opts = rng.standard_normal((7, 143)).astype(np.float32)
    labels = np.arange(3, dtype=np.int32)
    np.savez(src / "shard_0000.npz", options=opts, labels=labels,
             n_options=np.array([3, 2, 2], dtype=np.int32))
    (src / "deck_registry.json").write_text('{"base": 1000}', encoding="utf-8")

    out = sl.slice_dir(src, dst, 100)
    assert out == {"shards": 1, "rows": 7, "from": [143], "to": 100}
    with np.load(dst / "shard_0000.npz") as z:
        assert z["options"].shape == (7, 100)
        assert np.array_equal(z["options"], opts[:, :100])   # exact prefix
        assert np.array_equal(z["labels"], labels)           # untouched
        assert np.array_equal(z["n_options"], np.array([3, 2, 2]))
    # provenance travels with the slice
    assert (dst / "deck_registry.json").read_text(encoding="utf-8")


def test_slicing_refuses_to_widen(tmp_path):
    """Widening by slicing is impossible; asking for it must fail loudly
    rather than silently producing a narrower corpus than requested."""
    import numpy as np

    import scripts.m41b_slice_corpus as sl

    src, dst = tmp_path / "narrow", tmp_path / "wider"
    src.mkdir()
    np.savez(src / "shard_0000.npz",
             options=np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(SystemExit, match="cannot widen"):
        sl.slice_dir(src, dst, 143)


def test_slicing_an_empty_corpus_is_an_error(tmp_path):
    import scripts.m41b_slice_corpus as sl
    src = tmp_path / "empty"
    src.mkdir()
    with pytest.raises(SystemExit, match="no .npz shards"):
        sl.slice_dir(src, tmp_path / "out", 100)
