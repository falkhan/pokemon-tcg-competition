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
TWIN_FILES = ("encoders.py", "combat.py", "plan.py", "memory.py")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--deck", required=True)
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
    if not m:
        check(False, "fix string found in submission/main.py")
    else:
        names = [x for x in m.group(1).split(",") if x]
        known = {v for k, v in vars(rp).items()
                 if k.startswith(("PLAY_FIX_", "ATTACH_FIX_")) and isinstance(v, str)}
        unknown = [n for n in names if n not in known]
        check(not unknown, "every fix name resolves",
              f"[{', '.join(names)}]" if not unknown
              else f"UNRECOGNISED: {unknown} (would be SILENTLY dropped)")

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

    print(f"\n{'ALL TIER-1 CHECKS PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
