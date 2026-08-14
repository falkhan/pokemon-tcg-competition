"""Emit the M46 band-gate specs from ONE in-code table (docs/M46-plan.md A0).

Three specs share one roster and one weights table; generating all of them
from here is what keeps them from drifting apart (the m44_make_gate_spec.py
precedent). Kinds:

    null  -> docs/specs/m46_null_panel.json   arm == control (m41b bundle),
             n=200 x seed [1] — calibrates bars.family_z
    a0v   -> docs/specs/m46_A0v.json          K vs m41b, the acceptance A/B
             with the KNOWN live answer (810 vs 735); doubles as the ship
             path's candidate net A/B
    gate  -> --name/--arm/--control required — the 08-15 ship gate against
             the 810 incumbent under the same weights and bars

Weights are INTEGER live-game counts from the m44 post-mortem's 780+ band
table (n=83; docs/m44-live-postmortem.md §2): grim 32, mirror 19, wall 5,
lucario 5, stall 4, dragapult 3, remainder 15 split 5/5/5 over
garchomp/archaludon/rocket (equal split DECIDED at plan approval). Each
net in an arm/control carries its OWN live ship string (K = model-c-pkgz,
m41b = model-cz-ashw) — A0v validates against the live outcome of the FULL
bundles, not a shared string.

    uv run python scripts/m46_make_band_specs.py --kind null --kind a0v
    uv run python scripts/m46_make_band_specs.py --kind gate \
        --name m46_ship_gate --arm <spec> --control <spec> [--family-z -2.5]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SPECS_DIR = ROOT / "docs" / "specs"

M41B_BUNDLE = "model-cz-ashw:checkpoints/m41b_wide_prod.pt:alakazam_v2_h4"
K_BUNDLE = "model-c-pkgz:checkpoints/ppo_best_m44_K_r1.pt:alakazam_v2_h4"

WEIGHTS = {
    "n_games": 83,
    "source": ("docs/m44-live-postmortem.md §2 — pooled 780+-band opponents "
               "over six subs; remainder split 5/5/5 over garchomp/"
               "archaludon/rocket (equal split, plan-approval decision)"),
    "games": {"grim": 32, "mirror": 19, "wall": 5, "lucario": 5, "stall": 4,
              "dragapult": 3, "garchomp": 5, "archaludon": 5, "rocket": 5},
}

# family -> [(bed_name, matchrunner spec)]. The m46_* beds are tonight's
# G-13 draws from the 1100+/1000/900 winners hand-aware corpora
# (scripts/m46_build_beds.sh); garchomp/archaludon/rocket ride verbatim from
# docs/specs/m43_laneA_base.json. Deck choices: grim_live == 3121746f;
# mirror clones pilot OUR deck (what live mirrors play); drag/wall pilot
# their higher-volume list's exported CSV; lucario is the m45 single-draw
# bed (G-13 caveat printed at decode).
BEDS = {
    "grim": [(f"grim_d{i}",
              f"model:checkpoints/m46_bed_grim_d{i}.pt:grim_live")
             for i in (1, 2, 3)],
    "mirror": [(f"mirror_d{i}",
                f"model:checkpoints/m46_bed_mirror_d{i}.pt:alakazam_v2_h4")
               for i in (1, 2, 3)],
    "drag": [],   # placeholder — filled below (name 'dragapult' in weights)
    "wall": [],
    "stall": [],
    "lucario": [("lucario",
                 "model:checkpoints/m45_bc_lucario_d3.pt:lucario_solrock")],
    "garchomp": [(f"garchomp_d{i}",
                  f"model:checkpoints/m40_bed_garchomp_d{i}.pt:"
                  "data/kaggle/garchomp_c7b3253f_deck.csv")
                 for i in (1, 2, 3)],
    "archaludon": [("arch_d1",
                    "model:checkpoints/m39_bc_archaludon.pt:archaludon"),
                   ("arch_d2",
                    "model:checkpoints/m39_bed_arch_d2.pt:archaludon"),
                   ("arch_d3",
                    "model:checkpoints/m39_bed_arch_d3.pt:archaludon")],
}
BEDS["dragapult"] = [
    (f"dragapult_d{i}",
     f"model:checkpoints/m46_bed_drag_d{i}.pt:"
     "data/kaggle/dragapult_f2b4039a_deck.csv") for i in (1, 2, 3)]
BEDS["wall"] = [
    (f"wall_d{i}",
     f"model:checkpoints/m46_bed_wall_d{i}.pt:"
     "data/kaggle/wall_6f87e9d4_deck.csv") for i in (1, 2, 3)]
BEDS["stall"] = [
    (f"stall_d{i}",
     f"model:checkpoints/m46_bed_stall_d{i}.pt:"
     "data/kaggle/stall_e234578d_deck.csv") for i in (1, 2, 3)]
BEDS["rocket"] = [
    (f"rocket_d{i}",
     f"model:checkpoints/m40_bed_rocket_d{i}.pt:"
     "data/kaggle/rocket_59e27a5e_deck.csv") for i in (1, 2, 3)]
del BEDS["drag"]

ADVISORY = [("tuned", "rule:tuned:lucario"), ("iono", "rule:iono")]


def bed_list() -> list[dict]:
    beds = []
    for fam in WEIGHTS["games"]:
        for name, spec in BEDS[fam]:
            beds.append({"name": name, "spec": spec, "family": fam})
    for name, spec in ADVISORY:
        beds.append({"name": name, "spec": spec, "family": None,
                     "advisory": True})
    return beds


def base_spec(family_z: float) -> dict:
    return {
        "beds": bed_list(),
        "weights": WEIGHTS,
        "bars": {"pass": 0.02, "kill": 0.0, "family_z": family_z},
        "bootstrap": {"B": 2000, "seed": 46, "stability_floor": 0.95},
        "coverage_floor": 0.80,
    }


def make(kind: str, family_z: float, name: str | None,
         arm: str | None, control: str | None) -> Path:
    spec = base_spec(family_z)
    if kind == "null":
        spec.update({
            "name": "m46_null_panel",
            "question": ("control-vs-control null: what does the per-family "
                         "z spread look like when the true delta is 0? "
                         "Calibrates bars.family_z (the M26 lesson)."),
            "arm": M41B_BUNDLE, "control": M41B_BUNDLE,
            "n_per_cell": 200, "seeds": [1],
            "null_calibration": True,
        })
        out = SPECS_DIR / "m46_null_panel.json"
    elif kind == "a0v":
        spec.update({
            "name": "m46_A0v_acceptance",
            "question": ("does the 780+-band-weighted panel rank "
                         "m41b_wide_prod (live 810) above ppo_best_m44_K_r1 "
                         "(live 735)? A0v: if not, the instrument is "
                         "invalid."),
            "arm": K_BUNDLE, "control": M41B_BUNDLE,
            "n_per_cell": 800, "seeds": [1, 2],
            "acceptance": {
                "expect": "KILL", "max_z": -1.96,
                "reason": ("live truth 810 vs 735; PASS or delta>0 = "
                           "instrument invalid -> advisory demotion + "
                           "guards-only fallback + escalate to Piotr")},
        })
        out = SPECS_DIR / "m46_A0v.json"
    else:
        if not (name and arm and control):
            raise SystemExit("gate kind needs --name --arm --control")
        spec.update({
            "name": name,
            "question": ("does the candidate beat the 810 incumbent on the "
                         "780+-band-weighted panel under the A0 rule?"),
            "arm": arm, "control": control,
            "n_per_cell": 800, "seeds": [1, 2],
        })
        out = SPECS_DIR / f"{name}.json"
    out.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    import scripts.m46_band_gate as bg
    loaded = bg.load_band_spec(out)          # self-check through the validator
    print(f"{out.relative_to(ROOT)}  band {bg.band_hash(loaded)}")
    missing = [p for bed in loaded["beds"]
               for p in bed["spec"].split(":")
               if (p.endswith(".pt") or p.endswith(".csv"))
               and not (ROOT / p).exists()]
    if missing:
        print(f"  NOTE: {len(missing)} referenced artifact(s) not on disk "
              f"yet (A1/A2 still building?): {sorted(set(missing))}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", action="append", required=True,
                    choices=("null", "a0v", "gate"))
    ap.add_argument("--family-z", type=float, default=-2.0,
                    help="the family-kill bar (re-set from the null panel)")
    ap.add_argument("--name")
    ap.add_argument("--arm")
    ap.add_argument("--control")
    a = ap.parse_args()
    for kind in a.kind:
        make(kind, a.family_z, a.name, a.arm, a.control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
