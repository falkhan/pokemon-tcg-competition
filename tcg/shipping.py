"""Ship a submission: export the bundle, then gate it before uploading.

Merges the old ``rl/export.py`` (bundle building) and ``rl/gate.py``
(pre-submission checks) — one module for everything between "trained/tested
agent" and "tarball on Kaggle". The old CLIs stay untouched in ``rl/``; this
module's CLI uses subcommands:

  python -m tcg.shipping export --agent rules --deck lucario
  python -m tcg.shipping export --checkpoint bc_lucario.pt --deck lucario
  python -m tcg.shipping gate  [--agent rules]

Two agent kinds:
  neural (default): a (checkpoint, deck) pair -> numpy-only net bundle in submission/.
      A policy and the deck it was trained on are ONE artifact — a policy trained on
      Lucario plays Kyogre badly and vice-versa (M1 lesson) — so the deck travels
      with the weights.
  rules: the deck-agnostic generic pilot + a deck -> a torch-free bundle in
      submission_rules/ (bundles cg/ and a minimal rl/ package; ships the ACTUAL
      pilot module we test — no drift-prone hand-copy). See docs/M6.md.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"
SUBMISSION_RULES = ROOT / "submission_rules"
SUBMISSION_MAIN = "submission/main.py"
SUBMISSION_RULES_MAIN = "submission_rules/main.py"

# Current shipping pair = our best agent so far. bc_lucario was WORSE (BC can't
# capture the Lucario expert's hidden-state lookahead — see docs/M2 findings), so
# bc_v1+Kyogre remains the champion until PPO/search beats it.
DEFAULT_CHECKPOINT = "bc_v1.pt"
DEFAULT_DECK = "kyogre"
# Rule agent ships our strongest tested deck (M6.0 legibility work).
DEFAULT_RULES_DECK = "lucario"


# ---------------------------------------------------------------------------
# Export — build the bundles
# ---------------------------------------------------------------------------

def copy_engine(dest: Path) -> None:
    """Kaggle provides numpy but NOT the cg engine bindings -- bundle cg/ with the agent."""
    shutil.copytree(ROOT / "cg", dest / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)


def deck_source(deck: str) -> Path:
    """M24: a named deck that does not resolve is a hard error. The silent
    root-deck fallback survived the M18.1 --deck-required guard (the guard
    checks presence, not resolution) and nearly shipped the kyogre fossil a
    third time via a typo'd deck name."""
    src = ROOT / "decks" / f"{deck}.csv"
    if not src.exists():
        raise SystemExit(
            f"deck '{deck}' not found at {src} — refusing the root-deck "
            f"fallback (M18.1/M22c/M24 lesson: verify deck identity)")
    return src


def export_rules(deck: str = DEFAULT_RULES_DECK) -> None:
    """Bundle the rule-based generic pilot: cg/, a minimal rl/ package (only the pure-Python
    combat core + pilot — no torch/polars), and the deck. main.py is committed source.

    The bundle still ships the OLD rl/ modules on purpose: switching the shipped
    artifact to tcg/ is gated on real-engine `gate --agent rules` games and stays
    a follow-up (docs/M6.md) — this refactor must not change what ships.
    """
    SUBMISSION_RULES.mkdir(exist_ok=True)
    copy_engine(SUBMISSION_RULES)

    rl_pkg = SUBMISSION_RULES / "rl"
    rl_pkg.mkdir(exist_ok=True)
    for name in ("__init__.py", "combat.py", "generic_pilot.py", "turn_solver.py"):
        shutil.copy(str(ROOT / "rl" / name), str(rl_pkg / name))

    shutil.copy(str(deck_source(deck)), str(SUBMISSION_RULES / "deck.csv"))
    print(f"exported RULE agent (generic pilot) paired with deck '{deck}' -> {SUBMISSION_RULES}")


def export(checkpoint: str = DEFAULT_CHECKPOINT, deck: str = DEFAULT_DECK) -> None:
    """Neural bundle. Arch is detected from the checkpoint: `embedding.weight`
    means OptionScorerV2 (M7.3+), whose bundle also ships the ACTUAL encoder
    modules (rl/encoders.py + rl/combat.py) and the feature matrix they load —
    submission/main.py replays v2 only, so v1 exports are legacy artifacts."""
    import torch

    from tcg.encoders import FEAT
    from tcg.network import (OptionScorer, OptionScorerV2, OptionScorerV3,
                             save_npz)

    ckpt_path = ROOT / "checkpoints" / checkpoint
    if not ckpt_path.exists() and Path(checkpoint).exists():
        ckpt_path = Path(checkpoint)        # M16: accept full/relative paths too
    if checkpoint and checkpoint != DEFAULT_CHECKPOINT and not ckpt_path.exists():
        # M16 guard: an explicit --checkpoint that doesn't resolve used to
        # fall through to the seeded-RANDOM export — a random agent one
        # inattentive gate away from being shipped. Fail loudly instead.
        raise FileNotFoundError(
            f"--checkpoint {checkpoint!r} not found at {ckpt_path} — refusing "
            "the seeded-random fallback (only the default name may fall back)")
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "plan_enc.0.weight" in state_dict:      # M11 plan-conditioned v3
            from tcg.encoders import (EMBED_DIM, N_CONTEXTS, N_OPTION_IDS,
                                      STATE_V2_DIM)
            from tcg.network import PLAN_DIM
            extra = 0
            if "enc_ver" in state_dict:            # M21 encoder-v4 checkpoint
                from rl.encoders import V4_EXTRA_DIM as extra
            n_ids = (state_dict["state_enc.0.weight"].shape[1]
                     - STATE_V2_DIM - N_CONTEXTS - extra - PLAN_DIM) // EMBED_DIM
            opt_dim = (state_dict["option_enc.0.weight"].shape[1]
                       - N_OPTION_IDS * EMBED_DIM)   # M16: legacy | identity
            model = OptionScorerV3(n_state_ids=n_ids, option_dim=opt_dim,
                                   extra_dim=extra)
        elif "embedding.weight" in state_dict:
            model = OptionScorerV2()
        else:
            model = OptionScorer()
        model.load_state_dict(state_dict)
        print(f"exporting checkpoint {checkpoint} ({type(model).__name__}) "
              f"paired with deck '{deck}'")
    else:
        torch.manual_seed(0)
        model = OptionScorer()
        print(f"checkpoint {checkpoint} not found -- exporting seeded random weights")

    save_npz(model, str(SUBMISSION / "policy_weights.npz"))
    np.save(str(SUBMISSION / "card_features.npy"), FEAT)
    shutil.copy(str(deck_source(deck)), str(SUBMISSION / "deck.csv"))
    copy_engine(SUBMISSION)

    # v2: the bundled rl/ package main.py imports (encoders fall back to the
    # .npy matrix when the training parquet is absent — no polars on Kaggle).
    rl_pkg = SUBMISSION / "rl"
    rl_pkg.mkdir(exist_ok=True)
    names = ["__init__.py", "combat.py", "encoders.py"]
    if isinstance(model, OptionScorerV3):
        names.append("plan.py")                    # v3: plan enumeration ships
    if getattr(model, "extra_dim", 0) > 0:
        names.append("memory.py")                  # M21 v4: OppMemory ships
    for name in names:
        shutil.copy(str(ROOT / "rl" / name), str(rl_pkg / name))
    np.save(str(rl_pkg / "card_features.npy"), FEAT)


# ---------------------------------------------------------------------------
# Gates — everything that must be true before a package ships
# (each check is importable on its own: notebook, tests, future CI)
# ---------------------------------------------------------------------------

def load_submission_module(main_path: str = SUBMISSION_MAIN):
    """Import a submission main.py as a module (defines __file__, unlike Kaggle's exec)."""
    spec = importlib.util.spec_from_file_location("sub_main", main_path)
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    return sub


def parity_check(main_path: str = SUBMISSION_MAIN) -> float:
    """Shipped npz through the submission's numpy forward pass vs torch. Returns max diff.

    Arch-aware (M7.5): `embedding.weight` in the npz means OptionScorerV2 and
    the submission's ``score_options_v2``; otherwise the legacy v1 pair. The
    torch reference is ``tcg.network`` (proven equivalent to the old
    ``rl.policy`` classes by tests/test_network.py).
    """
    import torch

    from tcg import encoders
    from tcg.network import OptionScorer, OptionScorerV2, OptionScorerV3

    sub = load_submission_module(main_path)
    weights = np.load(Path(main_path).parent / "policy_weights.npz")
    torch_weights = {name: torch.from_numpy(array) for name, array in weights.items()}
    rng = np.random.default_rng(0)

    if "plan_enc.0.weight" in weights:             # M11 v3: forward AND plan head
        from tcg.network import PLAN_DIM
        extra = 0
        if "enc_ver" in weights:                   # M21 v4 export
            from rl.encoders import V4_EXTRA_DIM as extra
        n_ids = (weights["state_enc.0.weight"].shape[1]
                 - encoders.STATE_V2_DIM - encoders.N_CONTEXTS
                 - extra - PLAN_DIM) // encoders.EMBED_DIM  # M15: 12|20, M21: 25
        opt_dim = (weights["option_enc.0.weight"].shape[1]
                   - encoders.N_OPTION_IDS * encoders.EMBED_DIM)  # M16 width
        model = OptionScorerV3(n_state_ids=n_ids, option_dim=opt_dim,
                               extra_dim=extra)
        model.load_state_dict(torch_weights)
        state_ctx = rng.random(encoders.STATE_V2_DIM + encoders.N_CONTEXTS
                               + extra, dtype=np.float32)
        plan = rng.random(PLAN_DIM, dtype=np.float32)
        state_ids = rng.integers(0, encoders.N_CARD_IDS, size=n_ids)
        options = rng.random((9, opt_dim), dtype=np.float32)
        option_ids = rng.integers(0, encoders.N_CARD_IDS,
                                  size=(9, encoders.N_OPTION_IDS))
        cands = rng.random((7, PLAN_DIM), dtype=np.float32)
        ours = sub.score_options_v3(state_ctx, plan, state_ids, options,
                                    option_ids)
        ours_p = sub.score_plans(state_ctx, state_ids, cands)
        with torch.no_grad():
            ref, _ = model(torch.from_numpy(state_ctx).unsqueeze(0),
                           torch.from_numpy(plan).unsqueeze(0),
                           torch.from_numpy(state_ids).long().unsqueeze(0),
                           torch.from_numpy(options).unsqueeze(0),
                           torch.from_numpy(option_ids).long().unsqueeze(0))
            ref_p = model.plan_logits(
                torch.from_numpy(state_ctx).unsqueeze(0),
                torch.from_numpy(state_ids).long().unsqueeze(0),
                torch.from_numpy(cands).unsqueeze(0)).squeeze(0).numpy()
        assert np.allclose(ours_p, ref_p, rtol=1e-4, atol=1e-4), \
            f"plan-head parity MISMATCH\nnumpy: {ours_p}\ntorch: {ref_p}"
    elif "embedding.weight" in weights:
        model = OptionScorerV2()
        model.load_state_dict(torch_weights)
        state_ctx = rng.random(encoders.STATE_V2_DIM + encoders.N_CONTEXTS,
                               dtype=np.float32)
        state_ids = rng.integers(0, encoders.N_CARD_IDS,
                                 size=encoders.N_STATE_IDS)
        options = rng.random((9, encoders.OPTION_V2_DIM), dtype=np.float32)
        option_ids = rng.integers(0, encoders.N_CARD_IDS,
                                  size=(9, encoders.N_OPTION_IDS))
        ours = sub.score_options_v2(state_ctx, state_ids, options, option_ids)
        with torch.no_grad():
            ref, _ = model(torch.from_numpy(state_ctx).unsqueeze(0),
                           torch.from_numpy(state_ids).long().unsqueeze(0),
                           torch.from_numpy(options).unsqueeze(0),
                           torch.from_numpy(option_ids).long().unsqueeze(0))
    else:
        model = OptionScorer()
        model.load_state_dict(torch_weights)
        state_ctx = rng.random(sub.STATE_DIM + sub.N_CONTEXTS, dtype=np.float32)
        options = rng.random((9, sub.OPTION_DIM), dtype=np.float32)
        ours = sub.score_options(state_ctx, options)
        with torch.no_grad():
            ref, _ = model(torch.from_numpy(state_ctx).unsqueeze(0),
                           torch.from_numpy(options).unsqueeze(0))
    ref = ref.squeeze(0).numpy()

    assert np.allclose(ours, ref, rtol=1e-4, atol=1e-4), \
        f"parity MISMATCH\nnumpy: {ours}\ntorch: {ref}"
    return float(np.abs(ours - ref).max())


def encoder_parity_check(steps: int = 40) -> int:
    """Guard against train/serve ENCODER drift (M0 lesson #5): walk a real game and
    assert tcg.encoders and the submission module encode every observation and
    option identically. The weight-parity test can't catch this — it feeds random
    vectors, never real observations. Returns the number of decisions compared."""
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish

    from tcg import encoders

    sub = load_submission_module()
    assert (encoders.STATE_DIM, encoders.OPTION_DIM) == (sub.STATE_DIM, sub.OPTION_DIM), \
        (f"dim mismatch: tcg {(encoders.STATE_DIM, encoders.OPTION_DIM)} "
         f"vs sub {(sub.STATE_DIM, sub.OPTION_DIM)}")

    deck = [int(x) for x in open("submission/deck.csv") if x.strip()]
    obs_dict, _ = battle_start(deck, deck)
    compared = 0
    try:
        for _ in range(steps):
            if obs_dict["current"]["result"] >= 0:
                break
            observation = to_observation_class(obs_dict)
            ours = np.concatenate([encoders.encode_state(observation.current),
                                   encoders.encode_context(observation.select.context)])
            theirs = np.concatenate([sub.encode_state(observation.current),
                                     sub.encode_context(observation.select.context)])
            assert np.allclose(ours, theirs), \
                "encode_state/context drift between tcg.encoders and submission"
            for option in observation.select.option:
                assert np.allclose(encoders.encode_option(option, observation),
                                   sub.encode_option(option, observation)), \
                    "encode_option drift between tcg.encoders and submission"
            compared += 1
            obs_dict = battle_select(list(range(obs_dict["select"]["maxCount"])))
    finally:
        battle_finish()
    return compared


def memory_parity_check(steps: int = 60) -> int:
    """M21 v4 bundles: drive a real game with the SHIPPED agent on seat 0 and
    assert its accumulated OppMemory encoding equals a training-side reference
    (rl.memory + rl.encoders) fed the exact same per-seat log stream. Random
    weight-parity can't catch memory-accumulation drift — it has no history.
    No-ops (returns 0) for pre-v4 bundles."""
    from cg.api import to_observation_class
    from cg.game import battle_start, battle_select, battle_finish

    sub = load_submission_module()
    if not getattr(sub, "_IS_V4", False):
        return 0
    from rl.encoders import encode_ctx_v4
    from rl.memory import OppMemory

    ref_mem = OppMemory()
    deck = sub.DECK
    obs_dict, _ = battle_start(list(deck), list(deck))
    compared = 0
    try:
        for _ in range(steps):
            if obs_dict["current"]["result"] >= 0:
                break
            seat = obs_dict["current"]["yourIndex"]
            if seat == 0:
                picks = sub.agent(obs_dict)      # observes sub._MEM internally
                obs = to_observation_class(obs_dict)
                ref_mem.observe(obs)
                ref_ctx, ref_ids = encode_ctx_v4(obs, list(deck), ref_mem)
                sub_ctx, sub_ids = sub.encode_ctx_v4(obs, list(deck), sub._MEM)
                assert np.array_equal(ref_ctx, sub_ctx) and \
                    np.array_equal(ref_ids, sub_ids), \
                    "OppMemory accumulation drift: bundle vs training encoders"
                compared += 1
            else:
                n = obs_dict["select"]["maxCount"]
                picks = list(range(n))
            obs_dict = battle_select([int(i) for i in picks])
    finally:
        battle_finish()
    return compared


def deck_check(base: str = "submission") -> int:
    """Shipped deck.csv is a legal deck. Returns deck size."""
    # MUST be the tcg twin: by gate time the bundle's main.py has put
    # submission/ first on sys.path, so ``rl`` is the bundle's stripped
    # package and rl.deck_search is unreachable in this process.
    from tcg.deck_search import validate_deck

    deck = [int(x) for x in open(f"{base}/deck.csv") if x.strip()]
    legal, reasons = validate_deck(deck)
    assert legal, f"illegal deck: {reasons}"
    return len(deck)


def gate_game(main_path: str = SUBMISSION_MAIN) -> dict:
    """Full self-play game with the agent loaded Kaggle-style (file path -> exec)."""
    from kaggle_environments import make

    env = make("cabt")
    env.run([main_path, main_path])
    statuses = [s.status for s in env.state]
    assert statuses == ["DONE", "DONE"], f"gate game failed: {statuses}"
    return {"rewards": [s.reward for s in env.state], "decisions": len(env.steps)}


# The bundle-isolation driver, run with `python -c` in a stripped-down
# subprocess. {bundle_posix} / {project_root_posix} are forward-slash paths
# (safe inside a source string on all OSes). It asserts the bundled rl/ wins
# over the project's — the bundle ships the OLD rl modules (see export_rules).
ISOLATION_DRIVER_TEMPLATE = (
    "import sys,os\n"
    "nc=lambda p: os.path.normcase(os.path.abspath(p))\n"
    "here=nc('{bundle_posix}');root=nc('{project_root_posix}')\n"
    "sys.path[:]=[p for p in sys.path if nc(p or os.getcwd())!=root]\n"
    "sys.path.insert(0,'{bundle_posix}')\n"  # bundled cg/rl take precedence; stdlib+site kept
    "import main\n"
    "from cg.game import battle_start,battle_select,battle_finish\n"
    "od,_=battle_start(main.DECK,main.DECK);n=0\n"
    "while od['current']['result']<0 and n<5000:\n"
    "    od=battle_select([int(i) for i in main.agent(od)]);n+=1\n"
    "battle_finish()\n"
    "assert od['current']['result'] in (0,1,2),'game did not finish'\n"
    "assert 'polars' not in sys.modules and 'torch' not in sys.modules,'bundle pulled a heavy dep'\n"
    "rlf=nc(sys.modules['rl'].__file__)\n"
    "assert rlf.startswith(here),'imported project rl (%s), not bundled rl'%rlf\n"
    "print('ISO_OK',n)"
)


def bundle_isolation_check(base: str = "submission_rules") -> int:
    """Prove the bundle is self-contained: run main.py + a full game in a subprocess whose
    sys.path is ONLY the bundle dir (no project root), and assert it never imports polars/
    torch. This is what the in-process gate can't catch — the real Kaggle isolation. Rule
    bundle only (the neural bundle ships flat numpy files, not a package)."""
    bundle_dir = Path(base).resolve()
    driver = ISOLATION_DRIVER_TEMPLATE.format(
        bundle_posix=bundle_dir.as_posix(),
        project_root_posix=bundle_dir.parent.as_posix())
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, "-c", driver], cwd=str(bundle_dir),
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0 and "ISO_OK" in proc.stdout, \
        f"bundle isolation FAILED:\n{proc.stdout}\n{proc.stderr}"
    return int(proc.stdout.strip().split("ISO_OK")[-1])


def main_neural() -> None:
    v2 = "embedding.weight" in np.load(SUBMISSION / "policy_weights.npz")
    diff = parity_check()
    print(f"parity OK ({'v2' if v2 else 'v1'}, max diff {diff:.2e})")
    if v2:
        # v2 ships the ACTUAL rl/encoders.py — no hand-copy to drift. The
        # isolation game proves the bundle self-contained AND exercises the
        # bundled encoders end-to-end (the encoder_parity_check equivalent).
        decisions = bundle_isolation_check("submission")
        print(f"bundle isolation OK (self-contained; {decisions} decisions, "
              f"no polars/torch)")
        n_mem = memory_parity_check()
        if n_mem:
            print(f"memory parity OK (v4 OppMemory, {n_mem} prompts compared)")
    else:
        n_compared = encoder_parity_check()
        print(f"encoder parity OK ({n_compared} decisions compared)")
    deck_size = deck_check()
    print(f"deck legal ({deck_size} cards)")
    game = gate_game()
    print(f"gate game OK: rewards {game['rewards']}, {game['decisions']} decisions")


def main_rules() -> None:
    deck_size = deck_check("submission_rules")
    print(f"deck legal ({deck_size} cards)")
    decisions = bundle_isolation_check("submission_rules")
    print(f"bundle isolation OK (self-contained; {decisions} decisions, no polars/torch)")
    game = gate_game(SUBMISSION_RULES_MAIN)
    print(f"gate game OK: rewards {game['rewards']}, {game['decisions']} decisions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    export_parser = sub.add_parser("export", help="build a submission bundle")
    export_parser.add_argument("--agent", choices=["neural", "rules"], default="neural")
    export_parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    export_parser.add_argument("--deck", default=None,
                               help="deck name in decks/ (defaults: neural=kyogre, rules=lucario)")
    gate_parser = sub.add_parser("gate", help="run the pre-submission gates")
    gate_parser.add_argument("--agent", choices=["neural", "rules"], default="neural")
    args = parser.parse_args()

    if args.cmd == "export":
        if args.agent == "rules":
            export_rules(args.deck or DEFAULT_RULES_DECK)
        else:
            export(args.checkpoint, args.deck or DEFAULT_DECK)
    else:
        (main_rules if args.agent == "rules" else main_neural)()
