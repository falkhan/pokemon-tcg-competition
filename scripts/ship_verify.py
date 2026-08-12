"""Tier-1 artifact correctness for a ship candidate (docs/VALIDATION.md §1).

Runs against the EXPORTED bundle, not the checkpoint — gate the artifact that
ships, not the one that trained. Previously these checks were done ad hoc per
milestone and one of them (from-tarball verify) only ever happened in M37.

Includes the M37-audit's "silent fix-name ignore" hardening as a PRE-SHIP
check rather than runtime code: an unrecognised name in the fix string is
silently dropped by `rl.plan`, so a typo would ship an agent with its rules
quietly disabled. The obvious fix — raise at runtime — trades a silent-typo
risk for a live-crash risk, which is a bad trade near the end of a campaign.
Catching it here gets the protection with no production risk.

Usage:
    uv run python scripts/ship_verify.py --checkpoint m38_w9294_cont3.pt \
        --deck alakazam_v2_h4
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUBMISSION = ROOT / "submission"
# M42: scaling.py added. It has been bundled since M41 -- encode_option_v2
# reaches rl.scaling on every ATTACH option, so without it the agent crashes on
# Kaggle at the first energy card -- but it was never twin-checked, so the one
# newly-bundled file could have shipped stale.
TWIN_FILES = ("encoders.py", "combat.py", "plan.py", "memory.py", "scaling.py")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# A plan-CARRYING corpus does not measure 1.0 non-zero plans: encode_plan(None)
# is all-zeros and the null plan is a legitimate, frequent choice. Measured on
# the three plan-carrying corpora we have (data/plan_m16, plan_m18,
# plan_m19_rw2x): 0.159 / 0.180 / 0.193. Replay-derived corpora measure exactly
# 0.000. The two regimes are three orders of magnitude apart, so the thresholds
# below are not close calls.
PLAN_SERVED_MIN = 0.05     # a bundle that serves plans needs a corpus with them
PLAN_ZEROED_MAX = 0.01     # a bundle that zeroes them needs a corpus without


def _corpus_stats(d: Path) -> dict | None:
    """Per-shard-dir widths and plan occupancy. Never pools across dirs — the
    BCDatasetV3 pad shim pools them at TRAIN time (zero-padding narrow states
    to the widest), and that pooling is precisely what hides an encoder-version
    mismatch. Reporting per dir is the whole point."""
    import numpy as np
    shards = sorted(d.glob("*.npz"))
    if not shards:
        return None
    ctx_w, id_w, opt_w = set(), set(), set()
    rows = plan_rows = 0
    has_plans = False
    for s in shards:
        z = np.load(s)
        ctx_w.add(int(z["states"].shape[1]))
        id_w.add(int(z["state_ids"].shape[1]))
        opt_w.add(int(z["options"].shape[1]))
        n = int(z["states"].shape[0])
        rows += n
        if "plans" in z.files:
            has_plans = True
            plan_rows += int(np.count_nonzero(z["plans"].any(axis=1)))
    return dict(dir=str(d), rows=rows, ctx_w=ctx_w, id_w=id_w, opt_w=opt_w,
                has_plans=has_plans,
                plan_frac=(plan_rows / rows) if rows else 0.0)


def input_parity_checks(npz_path: Path, main_src: str,
                        corpus_dirs) -> list[tuple[str, bool, str]]:
    """G-14. Returns (label, ok, detail) triples. No engine dependency."""
    import numpy as np
    from rl.encoders import N_CONTEXTS, STATE_V2_DIM, V4_EXTRA_DIM
    from rl.plan import PLAN_DIM, SERVE_FIX_PLANZERO

    out: list[tuple[str, bool, str]] = []
    if not npz_path.exists():
        return [("6. serve/train input parity", False, "no policy_weights.npz")]
    z = np.load(npz_path)
    is_v3 = "plan_enc.0.weight" in z
    is_v4 = "enc_ver" in z
    embed_dim = int(z["embedding.weight"].shape[1])
    w_in = int(z["state_enc.0.weight"].shape[1])
    serve_ctx_w = STATE_V2_DIM + N_CONTEXTS + (V4_EXTRA_DIM if is_v4 else 0)
    serve_n_ids = (w_in - serve_ctx_w - (PLAN_DIM if is_v3 else 0)) // embed_dim

    m = re.search(r'"PKM_ATTACH_FIXES",\s*\n?\s*"([^"]*)"', main_src)
    names = [x for x in (m.group(1).split(",") if m else []) if x]
    serves_plan = is_v3 and SERVE_FIX_PLANZERO not in names

    stats = []
    for d in corpus_dirs:
        st = _corpus_stats(Path(d))
        if st is None:
            out.append(("6. corpus readable", False, f"no shards in {d}"))
        else:
            stats.append(st)
    if not stats:
        return out or [("6. serve/train input parity", False, "no corpus")]

    # 6a — the S6 defect: what the trunk is CONDITIONED on.
    tot = sum(s["rows"] for s in stats)
    frac = sum(s["plan_frac"] * s["rows"] for s in stats) / tot
    if serves_plan:
        ok = frac >= PLAN_SERVED_MIN
        detail = (f"bundle serves a plan-head argmax every MAIN "
                  f"(fixes: {','.join(names) or 'none'}); corpus has "
                  f"{frac:.4f} non-zero-plan rows over {tot}")
        if not ok:
            detail += (f"\n       every served MAIN prompt is OUT OF "
                       f"DISTRIBUTION for the trunk (want >= "
                       f"{PLAN_SERVED_MIN}; plan-conditioned corpora "
                       f"measure 0.16-0.19)\n       -> add "
                       f"'{SERVE_FIX_PLANZERO}' to PKM_ATTACH_FIXES, or "
                       f"train on a plan-carrying corpus")
            for s in stats:
                detail += (f"\n         {s['dir']:<34} {s['rows']:>7} rows  "
                           + ("plans present" if s["has_plans"]
                              else "NO `plans` column (-> zeros)"))
    else:
        ok = frac <= PLAN_ZEROED_MAX
        detail = (f"bundle serves plan=0 ('{SERVE_FIX_PLANZERO}'); corpus "
                  f"{frac:.4f} non-zero-plan rows -- matched")
    out.append(("6a. serve/train PLAN parity", ok, detail))

    # 6b — the S5 defect: encoder version, per dir, never pooled.
    bad = [s for s in stats
           if s["ctx_w"] != {serve_ctx_w} or s["id_w"] != {serve_n_ids}]
    out.append((
        "6b. serve/train ENCODER width parity", not bad,
        f"serve ctx {serve_ctx_w} ids {serve_n_ids} "
        f"(v{'4' if is_v4 else '3'}) vs {len(stats)} corpus dir(s)"
        + ("" if not bad else "\n       MISMATCH: " + "; ".join(
            f"{s['dir']} ctx {sorted(s['ctx_w'])} ids {sorted(s['id_w'])}"
            for s in bad)
            + "\n       BCDatasetV3 zero-pads this at train time, so it "
              "loads silently and trains on a vector\n       the encoder can "
              "never emit at serve time")))

    # 6c — option width. main.py truncates wide options, so wider is fine;
    # NARROWER means the net's tail option columns saw nothing in training.
    serve_opt = int(z["option_enc.0.weight"].shape[1]) - 2 * embed_dim
    narrow = [s for s in stats if min(s["opt_w"]) < serve_opt]
    out.append(("6c. corpus option width >= serve width", not narrow,
                f"serve {serve_opt}, corpus min "
                f"{min(min(s['opt_w']) for s in stats)}"))

    # 6d — v4 integrity. OppMemory.features writes v[3 + hot] = 1.0
    # unconditionally, where hot in 0..6 (rl/memory.py:153-165) — so the
    # attach-target ONE-HOT at v4-block offsets 3..9 sums to exactly 1.0 in
    # every vector encode_ctx_v4 can emit, and to 0.0 in a zero-padded v3 row.
    #
    # NB the invariant is the one-hot's SUM, not any single offset: offset 3 is
    # the "no attach seen yet" slot and goes cold the moment the opponent
    # attaches. An earlier version of this check asserted offset 3 == 1.0 and
    # was contradicted by the first real v4 corpus (3629/5075 rows "failed" a
    # corpus that is in fact clean). The unit-test fixture had encoded the same
    # misreading, so the test agreed with the bug — real data caught it.
    if is_v4:
        lo = STATE_V2_DIM + N_CONTEXTS + 3
        padded = []
        for d in corpus_dirs:
            for s in sorted(Path(d).glob("*.npz")):
                arr = np.load(s)["states"]
                if arr.shape[1] >= lo + 7:
                    hot = arr[:, lo:lo + 7].sum(axis=1)
                    n0 = int(np.count_nonzero(~np.isclose(hot, 1.0)))
                    if n0:
                        padded.append(f"{s.name}:{n0}")
        out.append(("6d. v4 corpus has no zero-padded rows", not padded,
                    "; ".join(padded[:4]) if padded
                    else "attach-target one-hot sums to 1 on every row"))
    return out


def gate_fix_package(arm: str) -> frozenset | None:
    """M43 review finding #4: the fix package a gated arm spec token ran
    with. Kind prefix -> rl/matchrunner's _MODEL_FIX_KINDS package; bare
    'model' = no fixes; None = unknown kind (a verification failure, not a
    silent pass)."""
    from rl.matchrunner import _MODEL_FIX_KINDS
    kind = arm.split(":", 1)[0]
    if kind == "model":
        return frozenset()
    return _MODEL_FIX_KINDS.get(kind)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--deck", required=True)
    # G-14 (M40): REQUIRED, not optional. A skippable serve/train parity check
    # is not a guardrail — the two regressions it exists to catch both survived
    # because nothing forced anyone to look. Pass the same --data list that
    # `plan_iter train` was given for this checkpoint.
    ap.add_argument("--corpus", required=True, nargs="+", type=Path,
                    help="shard dirs this checkpoint was trained on")
    # M43 review finding #4: the gate measures a matchrunner kind token's fix
    # package while the bundle reads main.py's hardcoded default — unlinked
    # copies that have already diverged once. Pass the gated arm spec (e.g.
    # 'model-c-pkgz:checkpoints/x.pt:alakazam_v2_h4') to require EQUALITY.
    ap.add_argument("--gate-arm", default=None,
                    help="gated arm spec token; its kind's fix package must "
                         "equal the bundle's PKM_ATTACH_FIXES set")
    args = ap.parse_args()
    fails: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  {'OK  ' if ok else 'FAIL'} {label}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(label)

    print("=== Tier 1: artifact correctness ===")

    # 1. twin parity — submission/rl/* must be byte-identical to rl/*
    for f in TWIN_FILES:
        sub = SUBMISSION / "rl" / f
        src = ROOT / "rl" / f
        if not sub.exists():
            check(False, f"twin parity rl/{f}", "MISSING from bundle")
            continue
        check(sub.read_bytes() == src.read_bytes(), f"twin parity rl/{f}")

    # 2. deck identity — the bundle's deck must BE the deck we gated on
    src_deck = ROOT / "decks" / f"{args.deck}.csv"
    bundle_deck = SUBMISSION / "deck.csv"
    check(src_deck.exists() and bundle_deck.exists(), "deck files present")
    if src_deck.exists() and bundle_deck.exists():
        check(md5(src_deck) == md5(bundle_deck), "deck md5 matches source",
              md5(bundle_deck))
        n = len([ln for ln in bundle_deck.read_text().splitlines() if ln.strip()])
        check(n == 60, "deck is 60 cards", str(n))

    # 3. fix-name resolution — the silent-typo guard
    import rl.plan as rp
    main_src = (SUBMISSION / "main.py").read_text(encoding="utf-8")
    m = re.search(r'"PKM_ATTACH_FIXES",\s*\n?\s*"([^"]*)"', main_src)
    names = None
    if not m:
        check(False, "fix string found in submission/main.py")
    else:
        names = [x for x in m.group(1).split(",") if x]
        known = {v for k, v in vars(rp).items()
                 if k.startswith(("PLAY_FIX_", "ATTACH_FIX_", "SERVE_FIX_"))
                 and isinstance(v, str)}
        unknown = [n for n in names if n not in known]
        check(not unknown, "every fix name resolves",
              f"[{', '.join(names)}]" if not unknown
              else f"UNRECOGNISED: {unknown} (would be SILENTLY dropped)")

    # 3b. gate/ship fix-set EQUALITY (M43 review finding #4). Check 3 proves
    #     the names parse; this proves they are the SAME SET the gate ran.
    if args.gate_arm:
        gated = gate_fix_package(args.gate_arm)
        if gated is None:
            check(False, "3b. gate/ship fix-set equality",
                  f"unknown spec kind '{args.gate_arm.split(':', 1)[0]}' "
                  "(not in _MODEL_FIX_KINDS)")
        elif names is None:
            check(False, "3b. gate/ship fix-set equality",
                  "no fix string in the bundle to compare")
        else:
            kind = args.gate_arm.split(":", 1)[0]
            check(set(names) == set(gated), "3b. gate/ship fix-set equality",
                  f"gate[{kind}]={sorted(gated)} bundle={sorted(names)}")

    # 4. weights are the gated checkpoint's, not a stale or random export
    import numpy as np
    import torch
    npz = SUBMISSION / "policy_weights.npz"
    ckpt = ROOT / "checkpoints" / args.checkpoint
    check(npz.exists(), "policy_weights.npz present")
    if npz.exists() and ckpt.exists():
        z = np.load(npz)
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        key = "embedding.weight"
        ok = key in z and key in sd and np.allclose(
            z[key], sd[key].numpy(), atol=1e-6)
        check(ok, f"bundle weights match {args.checkpoint}")
    else:
        check(False, f"checkpoint {args.checkpoint} present")

    # 5. the bundled RULES ACTUALLY ACT (M39). "Every fix name resolves" only
    #    proves the string parses — it does not prove the bundled predicate
    #    fires, which is the failure the M37 audit was really about. This runs
    #    the BUNDLE's own fix set through the BUNDLE's own plan module on a
    #    two-case fixture: a board that should trigger a demote, and one that
    #    should not. It is deliberately generic — a config with no
    #    board-triggered rule simply reports "no board-triggered rule in the
    #    string" rather than failing.
    #    It runs in a SUBPROCESS with the bundle first on sys.path, because
    #    this process has already imported the project's own rl.plan (step 3)
    #    and would otherwise silently test that copy instead of the shipped one.
    import subprocess
    import textwrap

    probe = textwrap.dedent("""
        import importlib.util, json, sys
        from types import SimpleNamespace
        sub = sys.argv[1]
        sys.path.insert(0, sub)
        import cg.api
        from cg.api import AreaType, OptionType, SelectContext
        spec = importlib.util.spec_from_file_location("bmain", sub + "/main.py")
        bmain = importlib.util.module_from_spec(spec); spec.loader.exec_module(bmain)
        import rl.plan as bp
        def fire(opp_id):
            card = lambda c: SimpleNamespace(id=c)
            opts = [SimpleNamespace(type=OptionType.ATTACH, index=0,
                                    area=AreaType.HAND, cardId=None),
                    SimpleNamespace(type=OptionType.PLAY, index=1,
                                    area=AreaType.HAND, cardId=None),
                    SimpleNamespace(type=OptionType.END, index=None,
                                    area=None, cardId=None)]
            me = SimpleNamespace(hand=[card(bp.ENRICHING_ENERGY_ID),
                                       card(bp.POFFIN_ID)],
                                 bench=[None]*5, benchMax=5, deckCount=20,
                                 active=[card(741)])
            op = SimpleNamespace(prize=[0]*6, active=[card(opp_id)], bench=[],
                                 deckCount=40)
            obs = SimpleNamespace(
                current=SimpleNamespace(players=[me, op], yourIndex=0,
                                        energyAttached=False),
                select=SimpleNamespace(context=SelectContext.MAIN, option=opts))
            return bp.apply_play_overrides(obs, [0, 1, 2], bmain._ATTACH_FIXES)
        print(json.dumps({"plan_file": bp.__file__,
                          "fixes": sorted(bmain._ATTACH_FIXES),
                          "wall": fire(345), "mirror": fire(741)}))
    """)
    out = subprocess.run([sys.executable, "-c", probe, str(SUBMISSION)],
                         capture_output=True, text=True, cwd=str(ROOT))
    if out.returncode != 0:
        check(False, "bundle behavioural probe ran", out.stderr.strip()[-300:])
    else:
        import json as _json
        r = _json.loads(out.stdout.strip().splitlines()[-1])
        check(Path(r["plan_file"]).is_relative_to(SUBMISSION),
              "probe imported the BUNDLE's rl.plan", r["plan_file"])
        if "racemode4" in r["fixes"]:
            check(r["wall"] == [1, 2, 0],
                  "bundled racemode4 DEMOTES vs a wall board", str(r["wall"]))
            check(r["mirror"] == [0, 1, 2],
                  "bundled racemode4 INERT vs a mirror board", str(r["mirror"]))
        else:
            print("  --   no board-triggered rule in the fix string; "
                  "behavioural check skipped")

    # 6. G-14 (M40): SERVE/TRAIN INPUT PARITY.
    #    Checks 1-5 all pass while the net is fed an input distribution it was
    #    never trained on — that is not hypothetical, it is the live state of
    #    Ship A and Ship B, and it has been true since M24. Two independent
    #    regressions of exactly this shape landed in one milestone (S5's
    #    encoder-version drift, S6's plan-head mismatch), so the check covers
    #    the general form: does the bundle SERVE what the corpus TRAINED?
    #    Static on purpose — numpy + a regex, no engine, no game.
    for label, ok, detail in input_parity_checks(
            SUBMISSION / "policy_weights.npz", main_src, args.corpus):
        check(ok, label, detail)

    print(f"\n{'ALL TIER-1 CHECKS PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
