"""Build the M39 QC bed opponents as standalone kaggle-agent bundles.

Why this exists (docs/M39-plan.md finding #4): M38's QC went 11W-1L against
a roster covering ZERO of the top-3 live loss families. The multi-deck QC
rule was followed to the letter and still measured the wrong thing, so
Piotr's mandatory replay review had nothing relevant to look at. QC's job is
a smoke test plus REPLAYS OF THE MATCHUPS THAT ACTUALLY LOSE — which needs
the loss-family beds playable as kaggle agents, not just as matchrunner
specs.

`tcg.shipping.export()` writes weights/deck/engine into the module-global
SUBMISSION directory and does NOT author main.py. So per bed we: copy
`submission/main.py`, neutralise its fix string, then export into the bed's
own directory with SUBMISSION redirected.

**Beds run PLAIN.** The shipped fix stack (telepath/deckguard/ash/conserve/
benchfloor/racemode3) is tuned against OUR alakazam list and references our
cards; applying it to a wall or grim clone would model an opponent nobody
plays. matchrunner's `model:` kind applies no fixes either, so the QC bed
and the gate bed are the same agent.

Usage:
    uv run python scripts/m39_build_qc_beds.py
    uv run python scripts/m39_build_qc_beds.py --only wall
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "dist/qc_beds"

# name -> (checkpoint under checkpoints/, deck name resolvable by deck_source)
BEDS = {
    "wall": ("m38_bc_wall.pt", "greattusk_wall"),
    # G-13 (2026-08-02): the wall panel's three draws score .296 / .188 / .197
    # against the live config — d1, the bed every wall claim in this campaign
    # was measured on, is the SOFTEST by ~10pp. QC is a smoke test and not a
    # measurement, so it does not need the whole panel; it does need the
    # HARDEST draw, because the replays Piotr reviews should come from the
    # opponent most likely to expose a defect.
    "wall_hard": ("m39_bed_wall_d2.pt", "greattusk_wall"),
    "grim": ("m39_bc_grim.pt", "grim_live"),
    "archaludon": ("m39_bc_archaludon.pt", "archaludon"),
    "top": ("m39_bc_top.pt", "clone54618168"),
}

# Kept in sync with submission/main.py's _ATTACH_FIXES default. The assert
# below is a deliberate tripwire: when the ship config changes, this script
# must fail rather than quietly build beds carrying a stale rule stack. It
# fired as designed on the M39 Ship A change (gacfr3 -> conserve).
# M42: refreshed to the M41 ogerpon ship's string. The tripwire had been stale
# since that ship, so the next bed rebuild would have SystemExit'd -- which is
# the design working, but the beds on disk were built under the older config
# and any rebuild must re-read this line rather than bump it reflexively.
SHIP_FIXES = '"conserve,planzero,ash,ashguard"'


def build_one(name: str, checkpoint: str, deck: str) -> Path:
    import tcg.shipping as shipping

    dest = OUT_ROOT / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # 1. main.py, with the fix string neutralised (assert, never silently pass)
    src = (ROOT / "submission/main.py").read_text(encoding="utf-8")
    if SHIP_FIXES not in src:
        raise SystemExit(
            f"submission/main.py no longer contains {SHIP_FIXES} — the ship "
            "fix string changed; update SHIP_FIXES in this script rather than "
            "shipping a bed that silently carries our rule stack")
    (dest / "main.py").write_text(src.replace(SHIP_FIXES, '""'), encoding="utf-8")

    # 2. weights / deck / engine / rl package, into the bed dir
    original = shipping.SUBMISSION
    try:
        shipping.SUBMISSION = dest
        shipping.export(checkpoint=checkpoint, deck=deck)
    finally:
        shipping.SUBMISSION = original
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="build just this bed")
    args = ap.parse_args()

    items = {args.only: BEDS[args.only]} if args.only else BEDS
    for name, (ckpt, deck) in items.items():
        dest = build_one(name, ckpt, deck)
        n = len(list(dest.rglob("*")))
        print(f"built qc bed {name:12} -> {dest.relative_to(ROOT)} "
              f"({n} files, {ckpt}, deck {deck})")
    print("\nNOTE: verify `git status submission/` is clean — these builds "
          "redirect the export and must never touch the ship directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
