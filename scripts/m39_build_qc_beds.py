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
    # M46: the first stall bed ever (live stall wr 0.00 and it was never in
    # QC). d2 = the hardest of the three G-13 draws on the null-panel read.
    "stall": ("m46_bed_stall_d2.pt", "fanrotom_stall"),
}

# M46 note on fix neutralisation: this script used to blank a SHIP_FIXES
# literal copied from main.py by hand — a tripwire that went stale twice
# (M42, M44) and, against the M44 K string, would have matched only the
# `_PLAN_ZERO = "planzero" in ...` line and built beds SILENTLY CARRYING
# our rule stack. Beds are now neutralised through the same regex the
# export itself uses (`_set_bundle_fixes` via `export(fixes="")`): the
# anchor is structural, and a moved anchor raises BundleError instead of
# passing quietly. An empty default also turns planzero off — matchrunner's
# `model:` kind applies no fixes and serves plans, so QC bed == gate bed.


def build_one(name: str, checkpoint: str, deck: str) -> Path:
    import tcg.shipping as shipping

    dest = OUT_ROOT / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # 1. main.py copied verbatim; export(fixes="") neutralises the default
    #    through the structural regex (BundleError if the anchor moved).
    src = (ROOT / "submission/main.py").read_text(encoding="utf-8")
    (dest / "main.py").write_text(src, encoding="utf-8")

    # 2. weights / deck / engine / rl package, into the bed dir
    original = shipping.SUBMISSION
    try:
        shipping.SUBMISSION = dest
        shipping.export(checkpoint=checkpoint, deck=deck, fixes="")
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
