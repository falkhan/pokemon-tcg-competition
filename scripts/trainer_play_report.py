"""Trainer-play report: does the clone play trainer cards at all?

Trainer = item | supporter | stadium | tool (data/cards_features.csv). Over the
val decisions that OFFER at least one trainer PLAY option, compares the
teacher's take rate with the model's argmax take rate, broken down by trainer
subtype and by acted card, plus what the model reaches for when it declines.

This is the M27 instrument: the M26 per-class report says PLAY val_acc is 0.486
but not *which way* the errors run. They run one way — the model under-plays
supporters and stadiums ~2x and 16x while matching or over-playing items.

Denominator note: the engine re-prompts within a turn, so one turn can expose
the same trainer at several prompts. The denominator is identical for teacher
and model, so the COMPARISON is exact; the absolute rate is a per-decision rate,
not a per-turn rate. The teacher's per-turn rate is reported separately at the
end over the full corpus.

The val split is reconstructed EXACTLY as rl.plan_iter.train does it
(default_rng(0), 10% of unique game_ids) by reusing scripts/class_report.py's
`val_indices` — pass --data dirs in the SAME ORDER as the training invocation.

Usage:
    uv run python scripts/trainer_play_report.py \
        --ckpt checkpoints/m25_bc_alakazam_v3h.pt --data data/bc_m25_alakazam_v3h

Acceptance pin (m25_bc_alakazam_v3h + data/bc_m25_alakazam_v3h, M27 Phase A):
1359 val decisions offer a trainer (48.5%); teacher takes 31.9%, model 24.1%.
By subtype — supporter 7.5% vs 3.8% (2211 offers), stadium 6.6% vs 0.4% (272),
item 13.8% vs 13.4% (1807). Boss's Orders 6.5% vs 0.8% of 478; Hilda 5.6% vs
1.2% of 571. Declines go to EVOLVE 39.8% / ABILITY 18.0%, ATTACK+END only 16.9%.
Teacher plays >=1 trainer on 48.0% of its 4354 turns.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cg.api import OptionType                                    # noqa: E402
from rl.plan_iter import BCDatasetV3                             # noqa: E402
from class_report import card_names, load_model, val_indices     # noqa: E402
from turn_discipline_report import (N_OPTION_TYPES, TURN_COL,    # noqa: E402
                                    option_types, predict)

OT_PLAY = int(OptionType.PLAY)
TRAINER_KINDS = ("supporter", "item", "stadium", "tool")
MIN_CARD_OFFERS = 15          # per-card table cutoff; below this it is noise


def trainer_kinds() -> dict[int, str]:
    """{card id -> 'supporter'|'item'|'stadium'|'tool'} for every trainer."""
    kinds = {}
    with open(ROOT / "data/cards_features.csv") as f:
        for row in csv.DictReader(f):
            for kind in TRAINER_KINDS:
                if row[f"is_{kind}"] == "true":
                    kinds[int(row["card_id"])] = kind
                    break
    return kinds


def acted_ids(ds: BCDatasetV3, row: int) -> np.ndarray:
    """Acted card id of every legal option of `row`."""
    start, n = ds.starts[row], ds.n_options[row]
    return ds.option_ids[start:start + n, 0].astype(int)


def report(ckpt: Path, data_dirs: list[Path]) -> dict:
    ds = BCDatasetV3(data_dirs)
    model = load_model(ckpt)
    val_idx = val_indices(ds)
    preds = predict(ds, model, val_idx)
    kinds, names = trainer_kinds(), card_names()

    print(f"ckpt {ckpt.name}  data {[d.name for d in data_dirs]}")
    print(f"rows {len(ds)}  val {len(val_idx)}  trainers in card table "
          f"{len(kinds)} ({dict(Counter(kinds.values()))})\n")

    offers = teacher_takes = model_takes = 0
    by_card = defaultdict(lambda: [0, 0, 0])      # cid -> [offered, teacher, model]
    by_kind = defaultdict(lambda: [0, 0, 0])
    declined_to = Counter()

    for row in val_idx:
        row = int(row)
        types, cids = option_types(ds, row), acted_ids(ds, row)
        slots = [j for j in range(len(types))
                 if types[j] == OT_PLAY and cids[j] in kinds]
        if not slots:
            continue
        offers += 1
        teacher_j, model_j = int(ds.labels[row]), preds[row]
        # One offer per DISTINCT card, so duplicate copies in hand don't
        # inflate the denominator.
        for cid in {cids[j] for j in slots}:
            by_card[cid][0] += 1
            by_kind[kinds[cid]][0] += 1
        if teacher_j in slots:
            teacher_takes += 1
            by_card[cids[teacher_j]][1] += 1
            by_kind[kinds[cids[teacher_j]]][1] += 1
        if model_j in slots:
            model_takes += 1
            by_card[cids[model_j]][2] += 1
            by_kind[kinds[cids[model_j]]][2] += 1
        else:
            declined_to[int(types[model_j])] += 1

    print(f"val decisions offering >=1 trainer PLAY: {offers} "
          f"({offers / max(1, len(val_idx)):.1%} of val decisions)")
    print(f"  teacher plays a trainer: {teacher_takes:5d} = "
          f"{teacher_takes / max(1, offers):.1%}")
    print(f"  model   plays a trainer: {model_takes:5d} = "
          f"{model_takes / max(1, offers):.1%}  "
          f"({model_takes / max(1, teacher_takes):.2f}x the teacher)\n")

    total_declines = max(1, sum(declined_to.values()))
    print("  when the model declined a trainer it chose instead:")
    for opt_type, count in declined_to.most_common(8):
        print(f"    {OptionType(opt_type).name:12s} {count:5d} = "
              f"{count / total_declines:.1%}")

    print("\n  by trainer subtype (offers | teacher | model):")
    for kind, (off, tea, mod) in sorted(by_kind.items(), key=lambda kv: -kv[1][0]):
        print(f"    {kind:10s} {off:5d} | teacher {tea:4d} = {tea / max(1, off):5.1%}"
              f" | model {mod:4d} = {mod / max(1, off):5.1%}")

    print(f"\n  per card (offered >= {MIN_CARD_OFFERS}):")
    print(f"    {'card':30s} {'offers':>7s} {'teacher':>16s} {'model':>16s}")
    for cid, (off, tea, mod) in sorted(by_card.items(), key=lambda kv: -kv[1][0]):
        if off < MIN_CARD_OFFERS:
            continue
        print(f"    {names.get(cid, str(cid)):30s} {off:7d} "
              f"{tea:7d} = {tea / max(1, off):5.1%} "
              f"{mod:7d} = {mod / max(1, off):5.1%}")

    # --- teacher per-turn baseline over the full corpus --------------------
    turns = ds.states[:, TURN_COL] * 30.0
    seen, played = set(), set()
    for row in range(len(ds)):
        key = (int(ds.game_ids[row]), int(round(turns[row])))
        seen.add(key)
        types, cids = option_types(ds, row), acted_ids(ds, row)
        j = int(ds.labels[row])
        if types[j] == OT_PLAY and cids[j] in kinds:
            played.add(key)
    print(f"\n[TEACHER, full corpus] our-turns {len(seen)}; turns with >=1 "
          f"trainer played {len(played)} = {len(played) / max(1, len(seen)):.1%}")

    return {"offers": offers, "teacher_takes": teacher_takes,
            "model_takes": model_takes,
            "by_kind": {k: tuple(v) for k, v in by_kind.items()},
            "by_card": {k: tuple(v) for k, v in by_card.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data", required=True, nargs="+", type=Path,
                    help="corpus dirs, SAME ORDER as the training invocation")
    args = ap.parse_args()
    report(args.ckpt, args.data)


if __name__ == "__main__":
    main()
