"""Play any two matchrunner arms and get games you can actually WATCH.

`rl.matchrunner` drives `cg.game` directly for speed, so it never builds the
`visualize` payload the official viewer needs — you get a win rate and nothing
to look at. `rl.eval.play_games` runs through `kaggle_environments`, which does
build it, but was written for shipped file agents.

They join cleanly and nobody had joined them: `play_games` accepts CALLABLES,
and `make_pilot` returns `(agent(obs_dict), deck_ids)` — exactly the kaggle
agent contract, including returning the deck when `obs.select is None`. So any
spec matchrunner understands can be watched:

    generic:decks/lucario.csv        generic-scale:decks/custom/my_grass.csv
    solver:decks/grim_live.csv       model:checkpoints/m39_retain_b.pt:alakazam_v2_h4
    solved:checkpoints/m39_bc_grim.pt:grim_live:800:400

Each run produces BOTH halves of a review loop:

  replays/<tag>_NNN.html      a self-contained page per game with a button that
                              POSTs to the official ptcgvis viewer — the "watch
                              it and spot the obvious mistake" half
  replays/<tag>/ep_gNNN_aS.json   the same games in cached-episode shape, which
                              `python -m rl.postmortem --batch` reads for the
                              7-flag taxonomy — the "is it systematic" half

This is the kaggle_environments path, so it is ~10x slower per game than
matchrunner and single-process (the engine keeps one global Battle). It is a
review instrument, not a measurement one: use `scripts/deck_probe.py` or the
panel gate for win rates you intend to act on.

Usage:
    uv run python scripts/watch_games.py --a generic:lucario --b generic:grim_live -n 4
    uv run python scripts/watch_games.py --a model-cz:checkpoints/m39_retain_b.pt:alakazam_v2_h4 \\
        --b model:checkpoints/m39_bc_grim.pt:grim_live -n 6 --tag m41_grim_review
"""
import argparse
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPLAY_DIR = ROOT / "replays"

# Card names and this script's own glyphs are non-ASCII; Windows defaults stdout
# to cp1252, where a bare print raises UnicodeEncodeError mid-run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def short_name(spec: str) -> str:
    """A filesystem-safe label for a spec: kind + the deck/checkpoint stem."""
    parts = spec.split(":")
    stem = Path(parts[-1]).stem if len(parts) > 1 else spec
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{parts[0]}_{stem}").strip("_")[:40]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="side A matchrunner spec")
    ap.add_argument("--b", required=True, help="side B matchrunner spec")
    ap.add_argument("-n", "--games", type=int, default=4)
    ap.add_argument("--tag", default=None,
                    help="replay prefix; defaults to <a>_vs_<b>")
    ap.add_argument("--no-json", action="store_true",
                    help="skip the postmortem-readable dumps")
    ap.add_argument("--postmortem", action="store_true",
                    help="run the batch flag taxonomy when the games finish")
    a = ap.parse_args()

    # Imported late: kaggle_environments pulls in litellm and is slow+noisy, and
    # argument errors should not pay for it. It also logs its whole OpenSpiel
    # game registry at INFO on import, which buries our output.
    logging.getLogger("kaggle_environments").setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("kaggle_environments") or name.startswith("LiteLLM"):
            logging.getLogger(name).setLevel(logging.WARNING)

    from rl.eval import play_games
    from rl.matchrunner import make_pilot, parse_spec

    for name in list(logging.root.manager.loggerDict):
        if name.startswith("kaggle_environments") or name.startswith("LiteLLM"):
            logging.getLogger(name).setLevel(logging.WARNING)

    tag = a.tag or f"{short_name(a.a)}_vs_{short_name(a.b)}"
    fn_a, deck_a = make_pilot(parse_spec(a.a), "watch_a")
    fn_b, deck_b = make_pilot(parse_spec(a.b), "watch_b")
    print(f"A {a.a}  ({len(deck_a)} cards)\nB {a.b}  ({len(deck_b)} cards)\n"
          f"{a.games} games -> replays/{tag}_*.html", flush=True)

    json_prefix = None if a.no_json else f"{(REPLAY_DIR / tag).as_posix()}/ep"
    wr, results = play_games(
        fn_a, fn_b, a.games, replay_prefix=tag,
        names=(short_name(a.a), short_name(a.b)), json_prefix=json_prefix)

    wins = sum(1 for r in results if (r[0] or 0) > (r[1] or 0))
    draws = sum(1 for r in results if (r[0] or 0) == (r[1] or 0))
    print(f"\nside A: {wins}W {a.games - wins - draws}L {draws}D "
          f"over {a.games} (wr={wr:.3f})", flush=True)
    print("  slots are swapped every other game, so A plays each seat equally.",
          flush=True)

    pages = sorted(REPLAY_DIR.glob(f"{tag}_*.html"))
    print(f"\nwatch ({len(pages)} pages) — open one and press ▶ Watch replay:")
    for p in pages[:8]:
        print(f"  {p.relative_to(ROOT).as_posix()}")
    if len(pages) > 8:
        print(f"  ... and {len(pages) - 8} more")
    print(f"  index: {(REPLAY_DIR / 'index.html').relative_to(ROOT).as_posix()}")

    if json_prefix:
        batch_dir = (REPLAY_DIR / tag).relative_to(ROOT).as_posix()
        print(f"\nflag taxonomy over the same games:\n"
              f"  uv run python -m rl.postmortem --batch {batch_dir}")
        if a.postmortem:
            from rl.postmortem import batch
            print(flush=True)
            batch(str(REPLAY_DIR / tag))

    print(f"\nn={a.games} is a review sample, not a measurement — "
          "use scripts/deck_probe.py for win rates you intend to act on.",
          flush=True)


if __name__ == "__main__":
    main()
