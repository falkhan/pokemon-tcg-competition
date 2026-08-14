"""M44 Step 6 — generate a ship-candidate's gate-diagnostic spec.

Arm = the pair's CURRENT league net under its fix token (read from the live
roster); control = the m43a incumbent (pre-registered); beds VERBATIM from
docs/specs/m43_laneA_base.json; n=400/cell, seed 1, bars {pass 0.02,
kill 0.0} read ADVISORILY (the gate is a diagnostic informing Piotr's go —
it no longer selects; a kill triggers Kill-criterion-4 escalation).

Usage: uv run python scripts/m44_make_gate_spec.py K
Then:  gate_spec.py hash/run/decode per the normal flow.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

pair = sys.argv[1]
roster = json.loads((ROOT / "docs/specs/m44_roster.json").read_text())
me = next(p for p in roster["players"] if p["id"] == pair)
base = json.loads((ROOT / "docs/specs/m43_laneA_base.json").read_text())

spec = {
    "name": f"m44_{pair}_ship_diag",
    "question": (f"is pair {pair}'s final league net better than the m43a "
                 "incumbent (55464234's control) on the 25-bed ladder proxy? "
                 "ADVISORY — informs Piotr's go, does not select"),
    "arm": me["spec"],
    "control": "model-c-pkgz:checkpoints/ppo_best_m43a_base.pt:alakazam_v2_h4",
    "beds": base["beds"],
    "n_per_cell": 400,
    "seed": 1,
    "bars": {"pass": 0.02, "kill": 0.0},
    "note": (f"M44 r4 Step 6. Arm net md5 {me.get('md5')}. Beds verbatim from "
             "m43_laneA_base.json (r1 box-migration manifest applies). A kill "
             "(<= 0.0) escalates per Kill criterion 4, never silent no-ship."),
}
out = ROOT / f"docs/specs/m44_{pair}.json"
out.write_text(json.dumps(spec, indent=1) + "\n")
print(f"wrote {out} (arm={me['spec']})")
