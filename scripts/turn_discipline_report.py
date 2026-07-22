"""Turn-discipline report: does the clone end its turn earlier than the teacher?

Answers the M26/M27 thesis question directly. Two sections:

1. **Contested decisions** — rows where BOTH a turn-ending option (ATTACK/END)
   and a free action (ATTACH/PLAY/EVOLVE/ABILITY) are legal. Compares the
   teacher's rate of ending the turn with the model's argmax rate, and lists the
   premature-end confusions (teacher took a free action, model ended the turn).
   The teacher side is reported over the FULL corpus, the model side over the
   val split (the model has trained on everything else).
2. **Where fidelity breaks** — val accuracy sliced by our deck count, turn
   bucket, and whether the teacher won the demonstration game.

The val split is reconstructed EXACTLY as rl.plan_iter.train does it
(default_rng(0), 10% of unique game_ids) by reusing scripts/class_report.py's
`val_indices` — pass --data dirs in the SAME ORDER as the training invocation.

Usage:
    uv run python scripts/turn_discipline_report.py \
        --ckpt checkpoints/m25_bc_alakazam_v3h.pt --data data/bc_m25_alakazam_v3h

Acceptance pin (m25_bc_alakazam_v3h + data/bc_m25_alakazam_v3h, M27 Phase A):
contested rows 16086 = 56.1% of corpus; teacher ends 12.9% (full) / 12.1% (val);
model ends 12.9% (val); model keeps a free action on 92.3% of teacher-free rows;
top confusion PLAY -> ATTACK x63. Slices: deck 30+ 0.738, deck 7-15 0.601,
teacher-WON 0.667 vs teacher-LOST 0.668. Headline val_acc 0.668.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import OptionType                             # noqa: E402
from rl.plan_iter import BCDatasetV3, collate_v3          # noqa: E402
from class_report import load_model, val_indices          # noqa: E402

N_OPTION_TYPES = 17          # OptionType one-hot block at the head of an option
TURN_COL, DECK_COL = 0, 3    # encode_state globals: turn/30, my deckCount/60

TURN_ENDING = frozenset({int(OptionType.ATTACK), int(OptionType.END)})
FREE = frozenset({int(OptionType.PLAY), int(OptionType.ATTACH),
                  int(OptionType.EVOLVE), int(OptionType.ABILITY)})


def option_types(ds: BCDatasetV3, row: int) -> np.ndarray:
    """OptionType int of every legal option of `row`."""
    start, n = ds.starts[row], ds.n_options[row]
    return ds.options[start:start + n, :N_OPTION_TYPES].argmax(axis=1)


def predict(ds: BCDatasetV3, model, rows: np.ndarray,
            batch_size: int = 256) -> dict[int, int]:
    """{row -> argmax option index} over the legal (unpadded) menu."""
    out = {}
    with torch.no_grad():
        for b0 in range(0, len(rows), batch_size):
            idxs = rows[b0:b0 + batch_size]
            (states, plans, sids, options, oids, valid,
             *_rest) = collate_v3([ds[int(i)] for i in idxs])
            logits, _ = model(states, plans, sids, options, oids)
            preds = logits.masked_fill(~valid, -1e9).argmax(dim=1)
            out.update(zip((int(i) for i in idxs), (int(p) for p in preds)))
    return out


def contested(ds: BCDatasetV3, row: int) -> np.ndarray | None:
    """Option types of `row`, or None if it isn't a contested decision."""
    types = option_types(ds, row)
    present = set(types.tolist())
    if not (present & TURN_ENDING) or not (present & FREE):
        return None
    return types


def report(ckpt: Path, data_dirs: list[Path]) -> dict:
    ds = BCDatasetV3(data_dirs)
    model = load_model(ckpt)
    val_idx = val_indices(ds)
    preds = predict(ds, model, val_idx)

    print(f"ckpt {ckpt.name}  data {[d.name for d in data_dirs]}")
    print(f"rows {len(ds)}  val {len(val_idx)}\n")

    # --- 1. contested decisions -------------------------------------------
    corpus_end = corpus_total = 0
    for row in range(len(ds)):
        types = contested(ds, row)
        if types is None:
            continue
        corpus_total += 1
        corpus_end += int(types[ds.labels[row]]) in TURN_ENDING

    print(f"contested rows (turn-ending AND free both legal): {corpus_total}"
          f"  ({corpus_total / len(ds):.1%} of all decisions)")
    print(f"  TEACHER ends the turn (full corpus): "
          f"{corpus_end}/{corpus_total} = {corpus_end / max(1, corpus_total):.1%}\n")

    tea_end = mod_end = tea_free = kept_free = 0
    confusion = Counter()
    for row in val_idx:
        row = int(row)
        types = contested(ds, row)
        if types is None:
            continue
        teacher = int(types[ds.labels[row]])
        model_pick = int(types[preds[row]])
        tea_end += teacher in TURN_ENDING
        mod_end += model_pick in TURN_ENDING
        if teacher not in TURN_ENDING:
            tea_free += 1
            if model_pick in TURN_ENDING:
                confusion[f"{OptionType(teacher).name} -> "
                          f"{OptionType(model_pick).name}"] += 1
            else:
                kept_free += 1

    n = tea_end + tea_free
    print(f"val contested rows: {n}")
    print(f"  teacher ends {tea_end / max(1, n):.1%}   "
          f"model ends {mod_end / max(1, n):.1%}   "
          f"(gap {(mod_end - tea_end) / max(1, n):+.1%})")
    print(f"  teacher took a free action on {tea_free} rows; the model kept a "
          f"free action on {kept_free / max(1, tea_free):.1%} of them")
    print("  premature-end confusions (teacher free -> model ends):")
    for label, count in confusion.most_common(10):
        print(f"    {label:26s} {count}")

    # --- 2. where fidelity breaks -----------------------------------------
    correct = np.array([preds[int(r)] == ds.labels[int(r)] for r in val_idx])
    print(f"\nheadline val_acc {correct.mean():.3f}\n")

    def slice_report(title, keys, order=None):
        print(title)
        buckets = defaultdict(list)
        for key, ok in zip(keys, correct):
            buckets[key].append(ok)
        for key in (order or sorted(buckets)):
            if key in buckets:
                hits = buckets[key]
                print(f"  {key:>14s}  n={len(hits):5d}  acc={np.mean(hits):.3f}")
        print()

    results = ds.results[val_idx]
    turns = ds.states[val_idx, TURN_COL] * 30.0
    decks = ds.states[val_idx, DECK_COL] * 60.0

    slice_report(
        "val_acc by demonstration outcome (did the teacher win that game?):",
        ["WON" if r > 0 else "LOST" if r < 0 else "DREW" for r in results],
        order=["WON", "LOST", "DREW"])
    slice_report(
        "val_acc by turn bucket:",
        [f"t{int(t) // 5 * 5}-{int(t) // 5 * 5 + 4}" for t in turns])
    slice_report(
        "val_acc by our deck count:",
        ["deck 0-6" if d <= 6 else "deck 7-15" if d <= 15 else
         "deck 16-29" if d <= 29 else "deck 30+" for d in decks],
        order=["deck 0-6", "deck 7-15", "deck 16-29", "deck 30+"])

    all_decks = ds.states[:, DECK_COL] * 60.0
    print("corpus decision mass by our deck count (ALL rows):")
    for lo, hi, name in [(0, 6, "0-6"), (7, 15, "7-15"),
                         (16, 29, "16-29"), (30, 60, "30+")]:
        mask = (all_decks >= lo) & (all_decks <= hi)
        print(f"  deck {name:>5s}: {mask.sum():6d} rows = {mask.mean():.1%}")

    return {"headline": float(correct.mean()),
            "contested_corpus": (corpus_end, corpus_total),
            "contested_val": {"teacher_ends": tea_end, "model_ends": mod_end,
                              "n": n, "kept_free": kept_free,
                              "teacher_free": tea_free}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data", required=True, nargs="+", type=Path,
                    help="corpus dirs, SAME ORDER as the training invocation")
    args = ap.parse_args()
    report(args.ckpt, args.data)


if __name__ == "__main__":
    main()
