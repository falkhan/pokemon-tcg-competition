"""Shared direct-engine match runner — ONE battle loop for every evaluator (M7.2).

Before this module, three separate battle-loop implementations drifted apart:
rl/collector.py's worker, rl/deck_search.py's matchup (hardcoded Lucario pilots),
and rl/eval.py. This is now the canonical home of the OpponentSpec vocabulary and
the slot-fair game loop; deck_search delegates here, the collector migrates in
M7.3 when it gains per-game deck sampling (it records training tensors mid-game —
a recording concern layered on top of match running), and eval.play_games stays
on kaggle_environments because it exercises the SHIPPED file agents (the deploy
surface, cross-checked against this fast path in M7-plan verification item 5).

Engine constraint (cg/sim.py holds one live battle per process): a worker runs
many battles sequentially (battle_start -> loop -> battle_finish per game);
parallelism is process-level via a spawn Pool, with only picklable str/int job
tuples crossing the boundary — the rl/collector.py pattern.

Opponent specs (picklable tuples; deck = decks/ name | csv path | list of ids):
  ("rule",  agent, deck)    rule expert brain ("lucario"/"iono"/"tuned") on a deck
  ("model", ckpt, deck)     neural pilot (greedy OptionScorer) from a checkpoint
  ("generic", deck)         the deck-agnostic rule pilot
  ("solver", deck)          generic pilot + within-turn combo solver (M7.4a)
  ("solver-dev", deck)      solver + the development tier (M8.1: setup search)
  ("random", deck)          uniform-random legal moves
  ("generic2"/"solver2", deck)   pilot v2: base pilot + M9 Leg 1 fixes; the
                            a/b variants carry one fix each for attribution
  ("ext", path, deck)       external kaggle-style main.py agent (M9 Leg 0 probe)

Usage:
  python -m rl.matchrunner play --a generic:lucario --b random:kyogre -n 60
"""
import argparse
import inspect
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# M22c-C1 search budget for the `solved:` spec — see the G6 note in make_pilot.
SOLVED_DEADLINE_S = 0.05
SOLVED_MAX_NODES = 120
SOLVED_ALLOW = None        # None = every trigger tier; frozenset to gate tiers
SOLVED_STATS = None        # set to a dict to collect trig_*/fire_* counters
DECK_DIR = ROOT / "decks"

OpponentSpec = tuple

# M9 Leg 1 pilot-v2 spec kinds -> (base kind, fixes). Separable a/b variants
# keep the two fixes individually measurable for attribution.
_FIXED_KINDS = {
    "generic2":  ("generic", frozenset({"handdiscard", "gust"})),
    "generic2a": ("generic", frozenset({"handdiscard"})),
    "generic2b": ("generic", frozenset({"gust"})),
    "solver2":   ("solver",  frozenset({"handdiscard", "gust"})),
    "solver2a":  ("solver",  frozenset({"handdiscard"})),
    "solver2b":  ("solver",  frozenset({"gust"})),
    # M41 deck probe: the rule pilot reads an attack's PRINTED damage, so it
    # ranks Alakazam's Powerful Hand (prints 0, really 20/card in hand) below a
    # 90-damage Dudunsparce and plays our own list as a Dudunsparce deck. These
    # kinds add rl/scaling.py's curated effective damage, so a deck-vs-deck
    # comparison is not decided by which decks happen to print flat numbers.
    "generic-scale": ("generic", frozenset({"scaling"})),
    "solver-scale":  ("solver",  frozenset({"scaling"})),
}

# M26 attach-override arms on the NEURAL pilot (rl/plan.apply_attach_overrides;
# docs/M26-plan.md Phase 4). Separate kinds — not env vars — so jsonl run keys
# never alias an override arm with the plain model pilot.
_MODEL_FIX_KINDS = {
    "modelt":  frozenset({"telepath"}),               # O1
    "modela":  frozenset({"backstop"}),               # O2
    "modelta": frozenset({"telepath", "backstop"}),   # O1+O2 composed
    # M30 deck-economy arms (rl/plan.apply_play_overrides), each on O1:
    "modelt-tempo": frozenset({"telepath", "tempo"}),         # O1+O3
    "modelt-guard": frozenset({"telepath", "deckguard"}),     # O1+O4
    "modelt-ash":   frozenset({"telepath", "ash"}),           # O1+O5
    "modelt-eco":   frozenset({"telepath", "tempo",           # O1+O3+O4+O5
                               "deckguard", "ash"}),
    # P3 survivor composition: tempo dropped (deck-burn harm vs grind decks)
    "modelt-guardash": frozenset({"telepath", "deckguard", "ash"}),
    # P5 follow-up (Piotr's QC no-go): Fezandipiti low-deck conserve, solo
    # for attribution and composed as the revised ship candidate:
    "modelt-con": frozenset({"telepath", "conserve"}),
    "modelt-gac": frozenset({"telepath", "deckguard", "ash", "conserve"}),
    # M31 bench-economy / supporter arms: the live gac ship + one new rule,
    # screened separately (ship <= 1). gacb = + O7 poffinfloor, gacd = + O8
    # drawfloor (docs/M31-plan.md).
    "modelt-gacb": frozenset({"telepath", "deckguard", "ash", "conserve",
                              "poffinfloor"}),
    "modelt-gacd": frozenset({"telepath", "deckguard", "ash", "conserve",
                              "drawfloor"}),
    # M35 bench-floor arms (rl/plan O10 benchfloor). gacf = gac + benchfloor;
    # gacbf = gacb + benchfloor. The 4-arm matrix {gac, gacb, gacf, gacbf}
    # tests the new rule AND whether poffinfloor still earns its slot
    # (docs/M35-plan.md).
    "modelt-gacf": frozenset({"telepath", "deckguard", "ash", "conserve",
                              "benchfloor"}),
    "modelt-gacbf": frozenset({"telepath", "deckguard", "ash", "conserve",
                               "poffinfloor", "benchfloor"}),
    # M36 gust-veto arm (rl/plan O11 gustveto): the gacf live ship + no
    # Boss's Orders PLAY while the opponent needs <= 1 prize
    # (docs/M36-plan.md W2; the 5/5-in-losses live pattern).
    "modelt-gacfv": frozenset({"telepath", "deckguard", "ash", "conserve",
                               "benchfloor", "gustveto"}),
    # M37 race-mode arms (rl/plan O12): the gacf live ship + archetype-
    # detected card-economy racing vs stall/grim boards (docs/M37-plan.md W1).
    # gacfr = margin-gated (the m36 raceconserve gate), gacfrr = blanket;
    # the m37 battery picks between them (bar B2). No gustveto composition —
    # gustveto is not in the ship default, composing would be two-variable.
    "modelt-gacfr": frozenset({"telepath", "deckguard", "ash", "conserve",
                               "benchfloor", "racemode"}),
    "modelt-gacfrr": frozenset({"telepath", "deckguard", "ash", "conserve",
                                "benchfloor", "racemoder"}),
    # O12c synthesis arm: blanket conserve vs the no-pressure wall family,
    # margin-gated conserve vs pressure-stall (hop/garchomp/grim) — the
    # per-family split the v1 battery demanded (docs/M37-plan.md P3b).
    "modelt-gacfr2": frozenset({"telepath", "deckguard", "ash", "conserve",
                                "benchfloor", "racemode2"}),
    # O12d final: wall-family blanket only — the pressure-family margin
    # branch was killed by the garchomp 5-seed confirm (z -2.13).
    "modelt-gacfr3": frozenset({"telepath", "deckguard", "ash", "conserve",
                                "benchfloor", "racemode3"}),
    # M39 P0.7 leave-one-out ablation of the SHIPPED stack (gacfr3). G5
    # measured the stack only in aggregate (plain .323 > gacf .278 > gacfr3
    # .254 on the wall bed) and its diary states per-rule attribution beyond
    # racemode3 is INCOMPLETE — so "strip everything" was an inference from
    # an aggregate. These five cells + the existing `modelt-gacf`
    # (= gacfr3 minus racemode3) + `model` (plain) decompose it.
    "abl-no-telepath": frozenset({"deckguard", "ash", "conserve",
                                  "benchfloor", "racemode3"}),
    "abl-no-deckguard": frozenset({"telepath", "ash", "conserve",
                                   "benchfloor", "racemode3"}),
    "abl-no-ash": frozenset({"telepath", "deckguard", "conserve",
                             "benchfloor", "racemode3"}),
    "abl-no-conserve": frozenset({"telepath", "deckguard", "ash",
                                  "benchfloor", "racemode3"}),
    "abl-no-benchfloor": frozenset({"telepath", "deckguard", "ash",
                                    "conserve", "racemode3"}),
    # M39 P1-inv: the third Ship A candidate. Decision 1 settled on a full
    # strip, but `conserve` was the one rule whose removal showed a
    # directional cost (-2.75pp, z=-1.91 at n=2400 on the 900+ bed) and it
    # is the only rule whose mechanism (M30 low-deck draw protection) maps
    # onto our largest loss mode (deck-out races, 29% of losses). Rather
    # than settle it by judgement, it becomes a gate cell: strip everything
    # EXCEPT conserve.
    "model-conserve": frozenset({"conserve"}),
    # M39 P2 — the anti-deck-out package, measured one rule at a time on top
    # of the SHIPPED Ship A config (`conserve`) so every cell is a single
    # variable against the live agent rather than against a hypothetical.
    # r2  = racemode2 (P2a): the m37 wall-blanket/pressure-margin split,
    #       never shipped live, now carrying the P2a id additions (Mega
    #       Kangaskhan ex on the wall side, Fan Rotom on the pressure side).
    # rm4 = racemode4 (P2b): demote OUR measured burn sources in a race.
    # ash = raceash (P2b): recycle Sacred Ash early in a race (a PROMOTE —
    #       opposite sign to rm4, which is why it is its own cell).
    "model-c-r2": frozenset({"conserve", "racemode2"}),
    "model-c-rm4": frozenset({"conserve", "racemode4"}),
    "model-c-ash": frozenset({"conserve", "raceash"}),
    "model-c-pkg": frozenset({"conserve", "racemode2", "racemode4"}),
    "model-c-pkga": frozenset({"conserve", "racemode2", "racemode4",
                               "raceash"}),
    # M40 S6 — the plan-head train/serve mismatch, as a config token.
    # `planzero` is a SERVE fix (rl/plan.py O14): it is inert in both
    # apply_*_overrides and acts upstream, on what the trunk is fed. Two kinds:
    #   model-pz     isolated — the mechanism cell, nothing else moving
    #   model-c-pkgz the gate arm — Ship B's live package + planzero, so the
    #                battery's single variable vs `model-c-pkg` is the plan
    #                vector and nothing else.
    "model-pz": frozenset({"planzero"}),
    # M42 O19 `deadenergy`, single-variable over the M40 Ship B package
    # (conserve,racemode2,racemode4) — that is the Alakazam-lineage config the
    # probe measured over-attaching 29 of 297 offers (9.8%, 440 damage
    # forgone) while the rule pilot on the same deck did it 0.0% of the time.
    # Pairs with `model-c-pkg` as the control.
    "model-c-pkg-de": frozenset({"conserve", "racemode2", "racemode4",
                                 "deadenergy"}),
    # isolated mechanism cell, nothing else moving
    "model-de": frozenset({"deadenergy"}),
    # M41 O18 `gustsnipe`, single-variable over model-pz — the ogerpon ship
    # candidate's config plus the one new rule, so the battery attributes it.
    "model-pz-snipe": frozenset({"planzero", "gustsnipe"}),
    # M45 lucario fix probes, single-variable over model-pz (the lucario
    # ship base): benchfloor targets the measured 14% bench-out losses vs
    # tuned; gustveto is the M36 anti-scaling demote, probed vs iono's
    # Voltaic Chain (docs/M45-plan.md A3).
    "model-pz-bf": frozenset({"planzero", "benchfloor"}),
    "model-pz-gv": frozenset({"planzero", "gustveto"}),
    "model-c-pkgz": frozenset({"conserve", "racemode2", "racemode4",
                               "planzero"}),
    # M40 deep-dive candidates (2026-08-03). Base = conserve+planzero: the
    # live evidence ranks conserve-only above the racemode package (Ship A
    # 773.6 > Ship B 726.3 > floor 659.7) and planzero is the adopted S6
    # serve fix. Each candidate is single-variable vs `model-cz`.
    "model-cz": frozenset({"conserve", "planzero"}),
    "model-cz-ashw": frozenset({"conserve", "planzero", "ash", "ashguard"}),
    "model-cz-ham": frozenset({"conserve", "planzero", "hammer"}),
    "model-cz-tempo": frozenset({"conserve", "planzero", "tempo"}),
    # M40b Track C — value-guided veto on top of the cz base.
    "model-cz-vv": frozenset({"conserve", "planzero", "vveto"}),
}

# M40b Track C `vveto` knobs (module-level like SOLVED_* so probes can read
# and tests can monkeypatch). Critic = the ONLY E0-passing value head
# (m39_retain_b, matched-pair 0.642); cont3's own head FAILED E0 at 0.469.
# A discredited policy carrying the only validated critic is fine — the
# critic is frozen and only ranks afterstates.
VVETO_CRITIC = "checkpoints/m39_retain_b.pt"
VVETO_TOPK = 3
VVETO_DELTA = 0.10        # min V-margin to override the policy pick
VVETO_DEADLINE_S = 0.12   # per-prompt wall cap across all candidate steps
VVETO_STATS = {"prompts": 0, "opened": 0, "stepped": 0, "vetoes": 0,
               "time_max": 0.0, "time_sum": 0.0}


def resolve_deck(deck) -> list[int]:
    """A decks/ name, a csv path (absolute or repo-ROOT-relative), or an id
    list -> 60 card ids. League specs store ROOT-relative POSIX paths so
    data/league/league.json is portable across machines/OSes."""
    if isinstance(deck, (list, tuple)):
        return list(deck)
    path = Path(deck)
    if not path.suffix:
        path = DECK_DIR / f"{deck}.csv"
    elif not path.is_absolute() and not path.exists():
        path = ROOT / path
    return [int(x) for x in path.read_text().split() if x.strip()]


def spec_deck(spec: OpponentSpec):
    """The deck slot of a spec (unresolved)."""
    return spec[2] if spec[0] in ("rule", "model", "solved", "ext", "rank",
                                  "vsolver", "mcts", *_MODEL_FIX_KINDS) \
        else spec[1]


def parse_spec(s: str) -> OpponentSpec:
    """CLI shorthand -> spec tuple: "generic:lucario", "rule:iono[:deck]",
    "model:checkpoints/bc_v1.pt:kyogre", "random:kyogre"."""
    parts = s.split(":")
    kind = parts[0]
    if kind in ("generic", "random", "solver", "solver-dev", *_FIXED_KINDS) \
            and len(parts) == 2:
        return (kind, parts[1])
    if kind == "ext" and len(parts) == 3:
        return ("ext", parts[1], parts[2])
    if kind == "rank" and len(parts) == 3:
        return ("rank", parts[1], parts[2])
    if kind == "vsolver" and len(parts) == 3:
        return ("vsolver", parts[1], parts[2])
    if kind == "mcts" and len(parts) == 4:
        return ("mcts", parts[1], parts[2], int(parts[3]))
    if kind == "rule" and len(parts) in (2, 3):
        return ("rule", parts[1], parts[2] if len(parts) == 3 else parts[1])
    if kind in ("model", *_MODEL_FIX_KINDS) and len(parts) == 3:
        return (kind, parts[1], parts[2])
    if kind == "solved" and len(parts) in (3, 4, 5):
        # M40 S3: optional PER-SPEC search budget —
        #     solved:<ckpt>:<deck>[:<max_nodes>[:<deadline_ms>]]
        # The budget has been a per-CALL dict since M11 (wrap_with_solver ->
        # solve_turn -> solve_turn_line -> _dfs all take it and resolve None to
        # the module constant), deliberately so data-gen never mutates globals
        # shared across seats. Only this parser and the `solved` branch pinned
        # it to the module constants, which meant every `solved:` bed in a run
        # had to share one budget — unusable for S3, where the whole point is
        # composites at DIFFERENT strengths in the same battery. Specs are
        # picklable str/int tuples, so this crosses the spawn-Pool boundary
        # unchanged.
        return ("solved", parts[1], parts[2],
                *(int(p) for p in parts[3:]))
    raise ValueError(f"cannot parse opponent spec {s!r} "
                     "(want kind:deck or rule:agent[:deck] or model:ckpt:deck)")


def make_pilot(spec: OpponentSpec, instance: str):
    """Build (agent_callable, deck_ids) for a spec. [ENGINE] for rule/model.

    `instance` must be unique per live rule pilot — teacher modules keep
    module-level mutable state (rl/teacher.py docstring).
    """
    kind = spec[0]
    if kind == "rule":
        from rl.teacher import load_teacher
        # deck=spec[1]: load_teacher's deck param only sets module.my_deck (the
        # kaggle-env deck return, unused in direct loops) and accepts names only;
        # the battle deck is resolved from the spec's own deck slot below.
        return load_teacher(instance, agent=spec[1], deck=spec[1]), resolve_deck(spec[2])
    if kind in _FIXED_KINDS:
        base, fixes = _FIXED_KINDS[kind]
        ids = resolve_deck(spec[1])
        if base == "generic":
            from rl.generic_pilot import make_generic_pilot
            return make_generic_pilot(ids, fixes=fixes), ids
        from rl.turn_solver import make_solver_pilot
        return make_solver_pilot(ids, instance=instance, fixes=fixes), ids
    if kind == "ext":
        # External kaggle-style bundle: import OUR engine first so cg is pinned
        # in sys.modules, then exec the bundle's main.py and grab its agent.
        import importlib.util
        import cg.api  # noqa: F401
        main_py = Path(spec[1])
        if main_py.is_dir():
            main_py = main_py / "main.py"
        mspec = importlib.util.spec_from_file_location(f"ext_{instance}", main_py)
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
        fn = getattr(mod, "agent", None)
        if fn is None:
            candidates = [v for v in vars(mod).values() if callable(v)]
            if not candidates:
                raise ValueError(f"no callable agent found in {main_py}")
            fn = candidates[-1]
        return fn, resolve_deck(spec[2])
    if kind == "vsolver":
        # M13 Rung 1: the solver pilot with the OUTCOME-GROUNDED setup value
        # as the search leaf on non-lethal turns. Lethal tier unchanged
        # (heuristic prize hunting stays exact); on other own MAIN prompts a
        # dev-mode solve runs with leaf_value replacing the heuristic tail,
        # overriding greedy only past DEV_OVERRIDE_MARGIN. Turn-passed leaves
        # are encoded from the flipped perspective and NEGATED (zero-sum) —
        # exact in the mirror, approximate cross-deck (logged limitation).
        import numpy as np
        import torch
        from cg.api import SelectContext, to_observation_class
        from rl.encoders import encode_context, encode_state_v2
        from rl.plan import PLAN_DIM
        from rl.policy import OptionScorerV3
        from rl.generic_pilot import make_generic_pilot
        from rl.turn_solver import should_solve, solve_turn
        LAMBDA = 3000.0
        # Calibrated 2026-07-18 on osv3_setupval2's held-out pairs (scratchpad
        # calibrate_margin.py): smallest LAMBDA*|dV| gap with ordering acc
        # >= 0.80 (measured 0.804 at coverage 0.84). One value, no sweep.
        VS_MARGIN = 200.0
        # v3 (S1 re-gate 2, user-approved): trust the value only where its own
        # held-out per-bucket report says it can rank — t4-15 buckets 0.79-0.80
        # vs near-chance past turn ~32. Greedy handles the long grind.
        VS_MAX_TURN = 32
        ckpt = Path(spec[1])
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = ROOT / ckpt
        net = OptionScorerV3()
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net.eval()
        ids = resolve_deck(spec[2])
        inner = make_generic_pilot(ids)
        ctx_main = encode_context(SelectContext.MAIN)
        cell = {"me": 0}

        def leaf_value(obs):
            num, sids = encode_state_v2(obs.current, ids)
            sc = np.concatenate([num, ctx_main]).astype(np.float32)
            with torch.no_grad():
                se = net.embedding(
                    torch.from_numpy(sids).long().unsqueeze(0)).flatten(-2)
                zeros = torch.zeros(1, PLAN_DIM)
                s = net.state_enc(torch.cat(
                    [torch.from_numpy(sc).unsqueeze(0), zeros, se], dim=-1))
                v = float(net.value_head(s).squeeze())
            sign = 1.0 if obs.current.yourIndex == cell["me"] else -1.0
            return LAMBDA * sign * v

        def fnv(od):
            obs = to_observation_class(od)
            if obs.select is None:
                return ids
            if should_solve(obs):
                try:
                    pick = solve_turn(obs, ids)
                except Exception:
                    pick = None
                if pick is not None:
                    return pick
            elif obs.select.context == SelectContext.MAIN \
                    and obs.current.turn < VS_MAX_TURN:
                cell["me"] = obs.current.yourIndex
                try:
                    pick = solve_turn(obs, ids, dev=True,
                                      leaf_value=leaf_value,
                                      dev_margin=VS_MARGIN)
                except Exception:
                    pick = None
                if pick is not None:
                    return pick
            return inner(od)
        return fnv, ids
    if kind == "rank":
        # M12 consumer: the solver pilot, plus a CONFIDENT ranker override on
        # single-pick MAIN prompts the solver tiers pass on. The net was
        # trained to predict widened-search sibling preferences from the root
        # prompt — inference is one forward pass, no engine calls. Overrides
        # only when top1-top2 logit gap >= CONF_MARGIN (no sweep), restricted
        # to the greedy beam (unbeamed options were never labeled).
        import numpy as np
        import torch
        from cg.api import SelectContext, to_observation_class
        from rl.encoders import (OPTION_V3_DIM, encode_context,
                                 encode_option_v2, encode_option_v2_legacy,
                                 encode_state_v2)
        from rl.generic_pilot import make_generic_pilot
        from rl.plan import PLAN_DIM
        from rl.policy import OptionScorerV3, option_dim_of
        from rl.turn_solver import _candidate_actions, should_solve, solve_turn
        CONF_MARGIN = 1.0
        ckpt = Path(spec[1])
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = ROOT / ckpt
        rsd = torch.load(ckpt, map_location="cpu")
        net = OptionScorerV3(option_dim=option_dim_of(rsd))
        net.load_state_dict(rsd)
        net.eval()
        # M16: pre-option-identity checkpoints get their exact encoding
        encode_option_v2 = (encode_option_v2
                            if net.option_dim >= OPTION_V3_DIM
                            else encode_option_v2_legacy)
        ids = resolve_deck(spec[2])
        inner = make_generic_pilot(ids)

        def fnr(od):
            obs = to_observation_class(od)
            if obs.select is None:
                return ids
            if should_solve(obs):
                try:
                    pick = solve_turn(obs, ids)
                except Exception:
                    pick = None
                if pick is not None:
                    return pick
            if obs.select.context == SelectContext.MAIN \
                    and obs.select.maxCount == 1 \
                    and len(obs.select.option) >= 2:
                beam = [a[0] for a in _candidate_actions(obs) if len(a) == 1]
                if len(beam) >= 2:
                    num, sids = encode_state_v2(obs.current, ids)
                    sc = np.concatenate(
                        [num, encode_context(obs.select.context)]
                    ).astype(np.float32)
                    pairs = [encode_option_v2(o, obs)
                             for o in obs.select.option]
                    opts = np.stack([x for x, _ in pairs]).astype(np.float32)
                    oids = np.stack([i for _, i in pairs])
                    with torch.no_grad():
                        logits, _ = net(
                            torch.from_numpy(sc).unsqueeze(0),
                            torch.zeros(1, PLAN_DIM),
                            torch.from_numpy(sids).long().unsqueeze(0),
                            torch.from_numpy(opts).unsqueeze(0),
                            torch.from_numpy(oids).long().unsqueeze(0))
                    l = logits.squeeze(0)
                    ranked = sorted(beam, key=lambda i: float(l[i]),
                                    reverse=True)
                    if float(l[ranked[0]]) - float(l[ranked[1]]) \
                            >= CONF_MARGIN:
                        return [int(ranked[0])]
            return inner(od)
        return fnr, ids
    if kind == "solved":
        # M22c: the model pilot + the within-turn combo solver. The neural
        # bundle has shipped a greedy argmax since M11 while turn_solver.py —
        # written to fix exactly that — shipped only in the rules bundle.
        from rl.turn_solver import wrap_with_solver
        inner_fn, ids = make_pilot(("model", spec[1], spec[2]), instance)
        # G6 budget: mean move < 50ms (rl/league.py:353-357). Plain model pilot
        # measures 11.5ms mean, so search has ~38ms of headroom. The solver's
        # stock 800 nodes / 0.4s deadline took the mean to 116.6ms — 2.3x over.
        # Tightened here rather than globally so make_solver_pilot (the rules
        # bundle, which passes G6 today) keeps its measured behaviour.
        stats = SOLVED_STATS
        # M40 S3: per-spec budget override, module constants as the default.
        max_nodes = int(spec[3]) if len(spec) > 3 else SOLVED_MAX_NODES
        deadline_s = (int(spec[4]) / 1000.0 if len(spec) > 4
                      else SOLVED_DEADLINE_S)
        return wrap_with_solver(inner_fn, ids, deadline_s=deadline_s,
                                max_nodes=max_nodes,
                                allow=SOLVED_ALLOW, stats=stats), ids
    if kind == "generic":
        from rl.generic_pilot import make_generic_pilot
        ids = resolve_deck(spec[1])
        return make_generic_pilot(ids), ids
    if kind == "solver":
        from rl.turn_solver import make_solver_pilot
        ids = resolve_deck(spec[1])
        return make_solver_pilot(ids, instance=instance), ids
    if kind == "solver-dev":
        from rl.turn_solver import make_solver_pilot
        ids = resolve_deck(spec[1])
        return make_solver_pilot(ids, instance=instance, dev=True), ids
    if kind == "mcts":
        # ("mcts", ckpt, deck, n_sims) — the M8.4(b) sims-ladder instrument:
        # MCTS over the checkpoint's own policy/value with L3 archetype
        # determinization (rl/determinize.py). Dimension-aware like "model".
        import torch
        from rl.determinize import load_meta
        from rl.mcts import V3Evaluator, make_mcts_agent
        from rl.policy import OptionScorer, OptionScorerV3, option_dim_of
        ckpt = Path(spec[1])
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = ROOT / ckpt
        sd = torch.load(ckpt, map_location="cpu")
        ids = resolve_deck(spec[2])
        if "plan_enc.0.weight" in sd:
            # M28: v3 checkpoints drive the search through V3Evaluator, which
            # carries the decklist the v3 state encoder needs. The M8.4 kill
            # was measured on the v1 value head only (docs/M22.md).
            from rl.plan_iter import _n_ids_of
            m3 = OptionScorerV3(n_state_ids=_n_ids_of(sd),
                                option_dim=option_dim_of(sd))
            m3.load_state_dict(sd)
            m3.eval()
            model = V3Evaluator(m3, ids)
        else:
            model = OptionScorer(state_ctx_dim=sd["state_enc.0.weight"].shape[1])
            model.load_state_dict(sd)
            model.eval()
        return make_mcts_agent(model, ids, n_sims=int(spec[3]), meta=load_meta()), ids
    if kind == "random":
        rng = random.Random(hash(instance) & 0xFFFF)
        fn = lambda od: rng.sample(range(len(od["select"]["option"])),  # noqa: E731
                                   od["select"]["maxCount"])
        return fn, resolve_deck(spec[1])
    if kind in ("model", *_MODEL_FIX_KINDS):
        import numpy as np
        import torch
        from cg.api import to_observation_class
        from rl.encoders import (COMBAT_SLICE, N_COMBAT, N_CONTEXTS, STATE_DIM,
                                 encode_context, encode_option, encode_state)
        from rl.plan import apply_attach_overrides, apply_play_overrides
        from rl.policy import OptionScorer

        attach_fixes = _MODEL_FIX_KINDS.get(kind, frozenset())

        ckpt = Path(spec[1])
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = ROOT / ckpt          # league specs store ROOT-relative paths
        sd = torch.load(ckpt, map_location="cpu")

        if "enc_ver" in sd:
            # M21 encoder-v4 checkpoint: explicit version buffer (width
            # sniffing is ambiguous once the v4 block + memory ids are in).
            # Same plan protocol as the v3 branch below, plus a per-game
            # OppMemory observed once per own prompt BEFORE encoding.
            from cg.api import SelectContext
            from rl.encoders import (OPTION_V3_DIM, N_STATE_IDS_V4,
                                     V4_EXTRA_DIM, encode_ctx_v4,
                                     encode_option_v2)
            from rl.memory import OppMemory
            from rl.plan import (PLAN_DIM, SERVE_FIX_PLANZERO, encode_plan,
                                 enumerate_plans)
            from rl.policy import OptionScorerV3, option_dim_of
            plan_zero = SERVE_FIX_PLANZERO in attach_fixes
            m4 = OptionScorerV3(n_state_ids=N_STATE_IDS_V4,
                                option_dim=option_dim_of(sd),
                                extra_dim=V4_EXTRA_DIM)
            m4.load_state_dict(sd)
            m4.eval()
            deck_ids = resolve_deck(spec[2])
            memory = OppMemory()
            pstate = {"key": None, "vec": np.zeros(PLAN_DIM, np.float32),
                      "last_turn": -1}

            def fn4(od):
                obs = to_observation_class(od)
                if obs.select is None:
                    pstate.update(key=None, last_turn=-1)
                    memory.reset()
                    return deck_ids
                t = obs.current.turn
                if t < pstate["last_turn"]:          # new game in this series
                    pstate.update(key=None)
                    memory.reset()
                pstate["last_turn"] = t
                key = (t, obs.current.yourIndex)
                memory.observe(obs)                  # once per own prompt
                sc, sids = encode_ctx_v4(obs, deck_ids, memory)
                if plan_zero:
                    # M40 S6 `planzero`: the plan head never runs, so the trunk
                    # sees the zero vector every corpus row was trained on.
                    plan = np.zeros(PLAN_DIM, np.float32)
                else:
                    if obs.select.context == SelectContext.MAIN \
                            and pstate["key"] != key:
                        cands = enumerate_plans(obs)
                        mat = np.stack([encode_plan(c) for c in cands]
                                       ).astype(np.float32)
                        idx = m4.act_plan(sc, sids, mat)
                        pstate.update(key=key, vec=mat[idx].copy())
                    plan = (pstate["vec"] if pstate["key"] == key
                            else np.zeros(PLAN_DIM, np.float32))
                pairs = [encode_option_v2(o, obs) for o in obs.select.option]
                opts = np.stack([n for n, _ in pairs]).astype(np.float32)
                oids = np.stack([i for _, i in pairs])
                if attach_fixes:      # M26 arms: full order -> override -> k
                    ranked = m4.act(sc, plan, sids, opts, oids,
                                    len(obs.select.option), greedy=True)
                    ranked = apply_attach_overrides(obs, ranked, attach_fixes)
                    ranked = apply_play_overrides(obs, ranked, attach_fixes)
                    return ranked[:obs.select.maxCount]
                return m4.act(sc, plan, sids, opts, oids,
                              obs.select.maxCount, greedy=True)
            return fn4, deck_ids

        if "plan_enc.0.weight" in sd:
            # Plan-conditioned v3 checkpoint (OptionScorerV3, M11): replan at
            # every own MAIN prompt, hold the plan for submenus keyed on
            # (turn, yourIndex). Closure state persists across a whole series
            # (pilots are built once, line ~318) — reset on a turn-counter
            # drop (new game, direct loop) and on select-None (kaggle path).
            from cg.api import SelectContext
            from rl.encoders import (EMBED_DIM, OPTION_V3_DIM,
                                     encode_option_v2, encode_option_v2_legacy,
                                     encode_state_v2, encode_state_v3)
            from rl.plan import (PLAN_DIM, SERVE_FIX_PLANZERO, encode_plan,
                                 enumerate_plans)
            from rl.policy import OptionScorerV3, option_dim_of
            from rl.encoders import N_CONTEXTS as _NC, STATE_V2_DIM as _SV2
            plan_zero = SERVE_FIX_PLANZERO in attach_fixes
            n_ids = (sd["state_enc.0.weight"].shape[1] - _SV2 - _NC
                     - PLAN_DIM) // EMBED_DIM
            opt_dim = option_dim_of(sd)
            m3 = OptionScorerV3(n_state_ids=n_ids, option_dim=opt_dim)
            m3.load_state_dict(sd)
            m3.eval()
            enc_state = encode_state_v3 if n_ids > 12 else encode_state_v2
            # M16: pre-option-identity checkpoints (pinned baselines) get the
            # exact encoding they trained on — sniffed width picks the encoder.
            enc_opt = (encode_option_v2 if opt_dim >= OPTION_V3_DIM
                       else encode_option_v2_legacy)
            deck_ids = resolve_deck(spec[2])
            pstate = {"key": None, "vec": np.zeros(PLAN_DIM, np.float32),
                      "last_turn": -1}

            vveto = None
            if "vveto" in attach_fixes:
                # M40b Track C (O17): 1-ply afterstate veto. score_siblings'
                # skeleton — open ONE determinized search per prompt, step
                # each of the policy's top-k picks once, score the child with
                # the frozen E0-validated critic, override only on a clear
                # V-margin. Sound within our own turn (the opponent never
                # acts); skipped whenever search_begin_input is absent
                # (logged replays) or a child leaves our perspective.
                from time import perf_counter
                from cg.api import search_end, search_step
                from rl.turn_solver import _open_search
                csd = torch.load(ROOT / VVETO_CRITIC, map_location="cpu")
                c_ids = (csd["state_enc.0.weight"].shape[1] - _SV2 - _NC
                         - PLAN_DIM) // EMBED_DIM
                critic = OptionScorerV3(n_state_ids=c_ids,
                                        option_dim=option_dim_of(csd))
                critic.load_state_dict(csd)
                critic.eval()
                c_zero_plan = torch.zeros(1, critic.plan_dim)

                def _critic_v(cur, ctx) -> float:
                    num2, sids2 = enc_state(cur, deck_ids)
                    sc2 = np.concatenate(
                        [num2, encode_context(ctx)]).astype(np.float32)
                    with torch.no_grad():
                        v = critic.value_head(critic._trunk(
                            torch.from_numpy(sc2).unsqueeze(0), c_zero_plan,
                            torch.from_numpy(
                                sids2.astype(np.int64)).unsqueeze(0)))
                    return float(v.squeeze())

                def vveto(obs, ranked):
                    sel = obs.select
                    if (sel.context != SelectContext.MAIN
                            or sel.maxCount != 1 or len(sel.option) < 2
                            or getattr(obs, "search_begin_input", None)
                            is None):
                        return ranked
                    t0 = perf_counter()
                    VVETO_STATS["prompts"] += 1
                    me_idx = obs.current.yourIndex
                    vals = {}
                    try:
                        root = _open_search(obs, deck_ids)
                    except Exception:
                        return ranked      # odd states can fail determinize
                    VVETO_STATS["opened"] += 1
                    try:
                        for i in ranked[:VVETO_TOPK]:
                            if perf_counter() - t0 > VVETO_DEADLINE_S:
                                break
                            try:
                                child = search_step(root.searchId, [int(i)])
                            except Exception:
                                continue
                            VVETO_STATS["stepped"] += 1
                            cur = child.observation.current
                            if cur.result >= 0:      # action ended the game
                                vals[i] = (2.0 if cur.result == me_idx
                                           else -2.0 if cur.result
                                           == 1 - me_idx else 0.0)
                                continue
                            if (cur.yourIndex != me_idx
                                    or child.observation.select is None):
                                continue             # turn passed: V is from
                            vals[i] = _critic_v(     # the wrong side — skip
                                cur, child.observation.select.context)
                    finally:
                        search_end()
                    dt = perf_counter() - t0
                    VVETO_STATS["time_sum"] += dt
                    VVETO_STATS["time_max"] = max(VVETO_STATS["time_max"], dt)
                    if ranked[0] not in vals or len(vals) < 2:
                        return ranked    # no comparison basis -> no veto
                    best = max(vals, key=vals.get)
                    if best != ranked[0] \
                            and vals[best] - vals[ranked[0]] >= VVETO_DELTA:
                        VVETO_STATS["vetoes"] += 1
                        return [best] + [i for i in ranked if i != best]
                    return ranked

            def fn3(od):
                obs = to_observation_class(od)
                if obs.select is None:
                    pstate.update(key=None, last_turn=-1)
                    return deck_ids
                t = obs.current.turn
                if t < pstate["last_turn"]:          # new game in this series
                    pstate.update(key=None)
                pstate["last_turn"] = t
                key = (t, obs.current.yourIndex)
                num, sids = enc_state(obs.current, deck_ids)
                sc = np.concatenate(
                    [num, encode_context(obs.select.context)]
                ).astype(np.float32)
                if plan_zero:
                    # M40 S6 `planzero`: the plan head never runs, so the trunk
                    # sees the zero vector every corpus row was trained on.
                    plan = np.zeros(PLAN_DIM, np.float32)
                else:
                    if obs.select.context == SelectContext.MAIN \
                            and pstate["key"] != key:
                        # Plan ONCE at the turn's first MAIN and hold it (M11 fix
                        # 2026-07-17): training plan rows exist only at first-MAIN
                        # states — replanning every MAIN is off-distribution for
                        # the head AND flip-flops the plan mid-turn (measured:
                        # 0.278 vs 0.345 base at Rung 0 before this fix).
                        cands = enumerate_plans(obs)
                        mat = np.stack([encode_plan(c) for c in cands]
                                       ).astype(np.float32)
                        idx = m3.act_plan(sc, sids, mat)      # argmax at eval
                        pstate.update(key=key, vec=mat[idx].copy())
                    plan = (pstate["vec"] if pstate["key"] == key
                            else np.zeros(PLAN_DIM, np.float32))
                pairs = [enc_opt(o, obs) for o in obs.select.option]
                opts = np.stack([n for n, _ in pairs]).astype(np.float32)
                oids = np.stack([i for _, i in pairs])
                if attach_fixes:      # M26 arms: full order -> override -> k
                    ranked = m3.act(sc, plan, sids, opts, oids,
                                    len(obs.select.option), greedy=True)
                    ranked = apply_attach_overrides(obs, ranked, attach_fixes)
                    ranked = apply_play_overrides(obs, ranked, attach_fixes)
                    if vveto is not None:
                        ranked = vveto(obs, ranked)
                    return ranked[:obs.select.maxCount]
                return m3.act(sc, plan, sids, opts, oids,
                              obs.select.maxCount, greedy=True)
            return fn3, deck_ids

        if "embedding.weight" in sd:
            # Encoders-v2 checkpoint (OptionScorerV2, M7.3): id embeddings +
            # deck-context pools — the pilot closes over its own deck list.
            # Option width is sniffed from option_enc.0.weight: pinned pre-M16
            # checkpoints are OPTION_V2_DIM (legacy encoding, no option-identity
            # block); M23 replay-clones are OPTION_V3_DIM (modern encoding).
            from rl.encoders import N_OPTION_IDS, OPTION_V3_DIM, encode_state_v2
            from rl.policy import OptionScorerV2
            embed = sd["embedding.weight"].shape[1]
            option_dim = sd["option_enc.0.weight"].shape[1] - N_OPTION_IDS * embed
            if option_dim == OPTION_V3_DIM:
                from rl.encoders import encode_option_v2
            else:
                from rl.encoders import (encode_option_v2_legacy as
                                         encode_option_v2)
            m2 = OptionScorerV2(embed=embed, option_dim=option_dim)
            m2.load_state_dict(sd)
            m2.eval()
            deck_ids = resolve_deck(spec[2])

            def fn2(od):
                obs = to_observation_class(od)
                num, sids = encode_state_v2(obs.current, deck_ids)
                sc = np.concatenate([num, encode_context(obs.select.context)]).astype(np.float32)
                pairs = [encode_option_v2(o, obs) for o in obs.select.option]
                opts = np.stack([n for n, _ in pairs]).astype(np.float32)
                oids = np.stack([i for _, i in pairs])
                return m2.act(sc, sids, opts, oids, obs.select.maxCount, greedy=True)
            return fn2, deck_ids

        # Dimension-aware v1 load: bc_v1 predates the M3 combat features. Its
        # state input is exactly N_COMBAT narrower, and the combat block is a
        # contiguous slice of the current encoding — slicing it out
        # reconstructs the encoder the checkpoint was trained on.
        in_dim = sd["state_enc.0.weight"].shape[1]
        expected = STATE_DIM + N_CONTEXTS
        if in_dim == expected:
            cut = None
        elif in_dim == expected - N_COMBAT:
            cut = COMBAT_SLICE
        else:
            raise ValueError(
                f"{spec[1]}: state input dim {in_dim} matches neither the current "
                f"encoder ({expected}) nor the pre-M3 one ({expected - N_COMBAT})")
        m = OptionScorer(state_ctx_dim=in_dim)
        m.load_state_dict(sd)
        m.eval()

        def fn(od):
            obs = to_observation_class(od)
            sc = np.concatenate([encode_state(obs.current),
                                 encode_context(obs.select.context)]).astype(np.float32)
            if cut is not None:
                sc = np.delete(sc, np.s_[cut[0]:cut[1]])
            opts = np.stack([encode_option(o, obs) for o in obs.select.option]).astype(np.float32)
            with torch.no_grad():
                logits, _ = m(torch.from_numpy(sc).unsqueeze(0), torch.from_numpy(opts).unsqueeze(0))
            order = torch.argsort(logits.squeeze(0), descending=True).tolist()
            return [int(i) for i in order[: obs.select.maxCount]]
        return fn, resolve_deck(spec[2])
    raise ValueError(f"unknown opponent spec kind: {spec!r}")


def _engine_game(fn0, fn1, deck0: list[int], deck1: list[int],
                 stats: dict | None = None) -> int:
    """One battle on the direct engine loop. Returns the winner seat (0/1) or 2
    for a draw. `stats`, if given, accumulates per-seat move counts, wall time,
    and agent exceptions under stats[0] / stats[1] — the G1/G6 gate inputs —
    and captures the end state under stats["final"] (the loss-forensics data:
    per-seat deck counts and prizes remaining). [ENGINE]"""
    from cg.game import battle_start, battle_select, battle_finish

    obs_dict, start = battle_start(deck0, deck1)
    if start.errorPlayer >= 0:
        battle_finish()
        raise ValueError(f"battle_start rejected a deck (errorType={start.errorType})")
    try:
        while obs_dict["current"]["result"] < 0:
            seat = obs_dict["current"]["yourIndex"]
            fn = fn0 if seat == 0 else fn1
            if stats is not None:
                s = stats.setdefault(seat, {"moves": 0, "time_s": 0.0, "errors": 0})
                t0 = time.perf_counter()
                try:
                    picks = fn(obs_dict)
                except Exception:  # noqa: BLE001 — G1 counts crashes, then re-raises
                    s["errors"] += 1
                    raise
                dt = time.perf_counter() - t0
                s["time_s"] += dt
                s["moves"] += 1
                if stats.get("collect_samples"):   # per-move p99 (M7.4a G6)
                    s.setdefault("samples", []).append(dt)
            else:
                picks = fn(obs_dict)
            obs_dict = battle_select([int(i) for i in picks])
        if stats is not None:
            players = obs_dict["current"]["players"]
            stats["final"] = {
                "decks": [p.get("deckCount") for p in players],
                "prizes": [len(p.get("prize", [])) for p in players],
            }
            # M44 3g: the engine's own end reason (RESULT log, type 23 —
            # 1=prizes 2=deckout 3=benchout 4=card effect), None when the log
            # landed in the other seat's per-seat window. A SIBLING of
            # "final": _from_a_view re-keys every "final" value by seat and
            # would choke on a scalar.
            stats["reason"] = next(
                (lg.get("reason") for lg in obs_dict.get("logs") or []
                 if lg.get("type") == 23), None)
        return obs_dict["current"]["result"]
    finally:
        battle_finish()


def play_series(spec_a: OpponentSpec, spec_b: OpponentSpec, n_games: int,
                seed: int = 0, game_fn=None, stats: dict | None = None,
                on_game=None) -> list[int]:
    """Slot-fair series: a takes seat g%2. Returns 0 = a won, 1 = b won, 2 = draw
    per game. `game_fn(fn0, fn1, deck0, deck1, stats)` is the test seam (defaults
    to the [ENGINE] loop). A game_fn that declares an `a_seat` parameter also
    receives side a's seat for this game — per-seat instrumentation MUST use it:
    fn0/deck0 are seat-0's, which is side a only on even games, so hardcoding
    `seat == 0` records the OPPONENT half the time (the M22 from_series defect).
    `stats`, if given, accumulates per-SIDE ("a"/"b")
    move/time/error totals across the series; pre-seed it with
    {"collect_samples": True} to also keep every per-move latency under
    side["samples"] (the M7.4a p99 input). `on_game(g, result, seat_stats)`
    is called after each game with a's result and that game's raw stats — the
    loss-forensics hook (the CLI's --diag)."""
    game = game_fn or _engine_game
    try:
        wants_a_seat = "a_seat" in inspect.signature(game).parameters
    except (TypeError, ValueError):   # builtins / C callables
        wants_a_seat = False
    fn_a, deck_a = make_pilot(spec_a, instance=f"mr{seed}_a")
    fn_b, deck_b = make_pilot(spec_b, instance=f"mr{seed}_b")
    collect = stats is not None and bool(stats.get("collect_samples"))
    out = []
    for g in range(n_games):
        a_seat = g % 2
        fns = (fn_a, fn_b) if a_seat == 0 else (fn_b, fn_a)
        decks = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        seat_stats: dict | None = None
        if stats is not None or on_game:
            seat_stats = {"collect_samples": True} if collect else {}
        res = (game(fns[0], fns[1], decks[0], decks[1], seat_stats, a_seat=a_seat)
               if wants_a_seat else
               game(fns[0], fns[1], decks[0], decks[1], seat_stats))
        if stats is not None:
            for seat in (0, 1):
                if seat not in seat_stats:
                    continue
                side = stats.setdefault("a" if seat == a_seat else "b",
                                        {"moves": 0, "time_s": 0.0, "errors": 0})
                for k in ("moves", "time_s", "errors"):
                    side[k] += seat_stats[seat][k]
                if collect:
                    side.setdefault("samples", []).extend(
                        seat_stats[seat].get("samples", ()))
        result = 2 if res == 2 else (0 if res == a_seat else 1)
        if on_game is not None:
            on_game(g, result, _from_a_view(seat_stats, a_seat))
        out.append(result)
    return out


def _from_a_view(seat_stats: dict, a_seat: int) -> dict:
    """Re-key one game's stats from seat indices to side-a's perspective."""
    view = {"a": seat_stats.get(a_seat), "b": seat_stats.get(1 - a_seat)}
    final = seat_stats.get("final")
    if final:
        view["final"] = {k: [v[a_seat], v[1 - a_seat]] for k, v in final.items()}
    view["reason"] = seat_stats.get("reason")   # M44: seat-independent scalar
    return view


def series_wr(results: list[int]) -> float:
    """Win rate for side a, draws counting half."""
    if not results:
        return 0.0
    return (sum(1 for r in results if r == 0) + 0.5 * sum(1 for r in results if r == 2)) / len(results)


def percentile(xs: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); pure Python, no numpy."""
    if not xs:
        return 0.0
    ys = sorted(xs)
    rank = -(-q * len(ys) // 100)                     # ceil without math
    return ys[min(len(ys) - 1, max(0, int(rank) - 1))]


# ---------------------------------------------------------------------------
# Multiprocessing across pairs (the rl/collector.py Pool pattern)
# ---------------------------------------------------------------------------
def _make_jobs(pairs: list[tuple], workers: int, seed_base: int = 1000) -> list[tuple]:
    """Split (spec_a, spec_b, n) pairs into even game chunks, ~2 jobs per worker.
    Chunks keep even sizes so each stays slot-fair. Pure — unit-tested offline.
    Job = (pair_idx, spec_a, spec_b, n_chunk, seed)."""
    total = sum(n for _, _, n in pairs)
    if total == 0:
        return []
    # target chunk size: fill ~2*workers jobs, rounded to even, min 2
    target = max(2, (total // max(1, 2 * workers) + 1) // 2 * 2)
    jobs = []
    for pair_idx, (a, b, n) in enumerate(pairs):
        done = 0
        while done < n:
            chunk = min(target, n - done)
            jobs.append((pair_idx, a, b, chunk, seed_base + len(jobs)))
            done += chunk
    return jobs


#: One game's MARGIN, side-a-relative, as persisted per game (M41b § II.3c):
#: [a prizes left, b prizes left, a deck count, b deck count]. `None` when the
#: engine ended without a final state (an errored game).
MARGIN_FIELDS = ("a_prizes", "b_prizes", "a_deck", "b_deck")


def _pair_worker(arg: tuple) -> tuple[int, int, list[int], list, list]:
    """One chunk: the per-game results AND their margins.

    M41b § II.3c: `_engine_game` has always captured the end state, and
    `_from_a_view` has always re-keyed it to side a, but the margin died here
    — the worker returned bare win/loss ints, so every `--workers` battery
    ever run discarded it (verified across 3,245 run files: zero carry it).
    The `on_game` closure is built INSIDE the worker, so it never has to cross
    the spawn boundary that forbids callables in `run_pairs`' job tuples.
    """
    job_idx, (pair_idx, spec_a, spec_b, n, seed) = arg
    margins: list = []
    reasons: list = []                      # M44: engine end-reason per game

    def on_game(_g, _result, view):
        final = view.get("final") or {}
        prizes, decks = final.get("prizes"), final.get("decks")
        margins.append([prizes[0], prizes[1], decks[0], decks[1]]
                       if prizes and decks else None)
        reasons.append(view.get("reason"))

    chunk = play_series(spec_a, spec_b, n, seed=seed, on_game=on_game)
    # Index alignment is the whole contract: margins[i] MUST describe the game
    # results[i] scored, or the analysis pairs an outcome with someone else's
    # margin and reports a confident wrong correlation. A game_fn that never
    # fires on_game (test seams) would otherwise short the list.
    margins += [None] * (len(chunk) - len(margins))
    reasons += [None] * (len(chunk) - len(reasons))
    return job_idx, pair_idx, chunk, margins[:len(chunk)], reasons[:len(chunk)]


def _run_key(pairs: list[tuple], workers: int, seed: int,
             extra: dict | None = None) -> dict:
    """The checkpoint header — json-normalized so tuple/list mismatch can't
    false-negative the resume validation.

    `extra` is stamped in verbatim when given (M41b § II.3d: the gate spec's
    hash rides here, so a battery file carries the bar it was run under and
    the existing header-mismatch refusal becomes the tamper check). Omitted
    entirely when None, so every pre-M41b checkpoint still validates.
    """
    key = {"pairs": pairs, "workers": workers, "seed": seed}
    if extra:
        key["extra"] = extra
    return json.loads(json.dumps(key))


def run_pairs(pairs: list[tuple], workers: int = 4, game_fn=None,
              seed: int = 0, checkpoint: str | None = None,
              key_extra: dict | None = None) -> list[list[int]]:
    """Run [(spec_a, spec_b, n_games), ...]; returns per-pair result lists.

    workers <= 1 runs in-process (required for an injected game_fn — callables
    don't cross the spawn boundary). Otherwise a spawn Pool over even game
    chunks; only str/int tuples are pickled. [ENGINE] unless game_fn given.

    seed offsets every chunk's instance seed (before M8.1 the CLI --seed was
    silently dropped on this path; note the engine's own RNG drives game
    variance either way — repeated runs are independent samples).

    Each persisted chunk also carries `margins` — one MARGIN_FIELDS row per
    game (M41b § II.3c) — and `reasons` (M44): the engine's end reason per
    game (1=prizes 2=deckout 3=benchout 4=card effect, None when absent).
    Old checkpoints have neither key and still resume.

    checkpoint (M8.1): a jsonl path. Line 1 pins the run key
    (pairs/workers/seed); each completed chunk appends one line as it
    finishes, and a rerun with the SAME key resumes, skipping completed
    chunks — long measurements survive crashes and pauses. A key mismatch
    raises instead of silently mixing two different runs."""
    if workers <= 1 or game_fn is not None:
        if workers > 1:
            raise ValueError("game_fn requires workers<=1 (not picklable)")
        if checkpoint:
            print("warning: --checkpoint ignored on the in-process path "
                  "(no resume)", flush=True)
        return [play_series(a, b, n, seed=1000 + seed + k, game_fn=game_fn)
                for k, (a, b, n) in enumerate(pairs)]

    jobs = _make_jobs(pairs, workers, seed_base=1000 + seed)
    results: list[list[int]] = [[] for _ in pairs]
    # Per-game margins ride alongside results (M41b § II.3c). They are not
    # returned — the return type is load-bearing for every caller — but they
    # ARE persisted to the checkpoint, which is the analysis artifact.
    margins: list[list] = [[] for _ in pairs]
    done: set[int] = set()
    fh = None
    if checkpoint:
        path = Path(checkpoint)
        key = _run_key(pairs, workers, seed, key_extra)
        if path.exists() and path.read_text().strip():
            lines = [json.loads(line) for line in path.read_text().splitlines()
                     if line.strip()]
            if lines[0] != key:
                raise ValueError(f"{checkpoint} belongs to a different run "
                                 "(header mismatch) — delete it or use a new path")
            for row in lines[1:]:
                done.add(row["job"])
                results[row["pair"]].extend(row["results"])
                # `margins` is absent in every pre-M41b checkpoint; resuming
                # one must still work, so the results list stays the authority
                # and margins are read only where they exist.
                margins[row["pair"]].extend(
                    row.get("margins") or [None] * len(row["results"]))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(key) + "\n")
        fh = path.open("a")

    pending = [(i, job) for i, job in enumerate(jobs) if i not in done]
    try:
        if pending:
            ctx = mp.get_context("spawn")
            with ctx.Pool(min(workers, len(pending))) as pool:
                for job_idx, pair_idx, chunk, chunk_margins, chunk_reasons in \
                        pool.imap_unordered(_pair_worker, pending):
                    results[pair_idx].extend(chunk)
                    margins[pair_idx].extend(chunk_margins)
                    if fh is not None:
                        fh.write(json.dumps({"job": job_idx, "pair": pair_idx,
                                             "results": chunk,
                                             "margins": chunk_margins,
                                             "reasons": chunk_reasons}) + "\n")
                        fh.flush()
    finally:
        if fh is not None:
            fh.close()
    return results


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("play", help="one spec-vs-spec series, slot-fair")
    s.add_argument("--a", required=True, help="e.g. generic:lucario, rule:iono, model:<ckpt>:<deck>")
    s.add_argument("--b", required=True)
    s.add_argument("-n", "--games", type=int, default=60)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--diag", action="store_true",
                   help="per-game end-state lines (loss forensics; forces workers=1)")
    s.add_argument("--latency", action="store_true",
                   help="per-move latency mean/p50/p99 ms per side (M7.4a G6; "
                        "forces workers=1)")
    s.add_argument("--checkpoint", default=None, metavar="FILE",
                   help="jsonl chunk checkpoint: appends per completed chunk; "
                        "rerunning the same command resumes (M8.1)")
    a = p.parse_args()

    spec_a, spec_b = parse_spec(a.a), parse_spec(a.b)
    on_game = None
    if a.diag:
        def on_game(g, result, view):
            f = view.get("final") or {}
            decks = f.get("decks", ["?", "?"])
            prizes = f.get("prizes", ["?", "?"])
            moves = (view.get("a") or {}).get("moves", "?")
            print(f"game {g:>3}: {'WLD'[result]}  moves={moves:>3}  "
                  f"deck a/b={decks[0]}/{decks[1]}  prizes-left a/b={prizes[0]}/{prizes[1]}",
                  flush=True)
    stats = {"collect_samples": True} if a.latency else None
    if a.diag or a.latency or a.workers <= 1:
        if a.checkpoint:
            print("warning: --checkpoint ignored on the in-process path "
                  "(no resume)", flush=True)
        results = play_series(spec_a, spec_b, a.games, seed=a.seed,
                              stats=stats, on_game=on_game)
    else:
        results = run_pairs([(spec_a, spec_b, a.games)], workers=a.workers,
                            seed=a.seed, checkpoint=a.checkpoint)[0]
    w = sum(1 for r in results if r == 0)
    d = sum(1 for r in results if r == 2)
    print(f"{a.a} vs {a.b}: {w}W {len(results) - w - d}L {d}D over {len(results)} "
          f"(wr={series_wr(results):.3f})", flush=True)
    if a.latency:
        for side in ("a", "b"):
            ms = [1000 * x for x in (stats.get(side) or {}).get("samples", [])]
            if ms:
                print(f"side {side}: mean={sum(ms) / len(ms):.1f}ms  "
                      f"p50={percentile(ms, 50):.1f}ms  p99={percentile(ms, 99):.1f}ms  "
                      f"over {len(ms)} moves", flush=True)


if __name__ == "__main__":
    _main()
