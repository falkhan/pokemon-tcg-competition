"""Generates notebooks/pipeline.ipynb. Edit here, re-run, re-open the notebook.

Kept as a .py because hand-editing .ipynb JSON is how cells silently break
(same convention as build_m22_ab_monitor.py). tests/test_notebooks.py pins the
committed .ipynb to this generator's output.

The notebook is the M43-era pipeline surface: Kaggle ingest -> BC shards ->
arena (OpenSkill/implied Elo, PPO checkpoints as entrants) with an automated
gate + durable state log + Hermes notification after every stage. It never
ships or submits — the ship path stays human (CLAUDE.md mandatory QC stop).
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "pipeline.ipynb"


def md(*src):
    return {"cell_type": "markdown", "metadata": {}, "source": list(src)}


def code(*src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": list(src)}


def _with_ids(cells):
    for i, c in enumerate(cells):
        c["id"] = f"pipe-{i:02d}"
    return cells


CELLS = [
    # ── 0 ── title + laws ────────────────────────────────────────────────────
    md("""# Pipeline — Kaggle → BC shards → Arena (OpenSkill / implied Elo) → Gates

One notebook per pipeline run: pull fresh live replays, rebuild the BC corpus,
play the arena, and stamp a machine-readable verdict per stage into
`runs/pipeline_state.json`. Every stage cell is guarded by the previous stage's
gate and every gate cell notifies Hermes (non-fatal when the CLI is absent).

**How to use:** run cells top to bottom. Re-running any cell is safe — every
stage is idempotent or resumable. Change the config in cell 2, re-run from
cell 2. A config change flips `config_hash`, which marks earlier verdicts
stale (the state file records the hash they were computed under).

## Laws this notebook enforces (CLAUDE.md)

- **`--workers` ≤ 8, everywhere.** 12 workers deadlocks `mp.Pool` via a native
  `libcg.so` corruption (M17, m19b). Cell 2 clamps once; nothing downstream may
  raise it.
- **jsonl `results`: `0` = side-a WIN, `1` = loss, `2` = draw.** Summing the
  array counts *losses*. Decode with `rl.matchrunner.series_wr`, never by hand.
- **`plan_iter collect` has NO resume** — a relaunch into the same `--out`
  clobbers shards. This notebook only drives the resumable producers
  (`kaggle_ingest refresh`, `replay_bc build`, `league run`).
- **This notebook never ships or submits.** The final cell is instructions for
  the human ship path: offline gate → export → ship_verify → qc_battery →
  **STOP for manual replay review** — and the submission itself stays manual.
"""),

    # ── 1 ── bootstrap + helpers ─────────────────────────────────────────────
    code("""# 1 — bootstrap + shared helpers. Run once per kernel.
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path.cwd()
while not (_ROOT / "rl").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
ROOT = _ROOT
print("repo root:", ROOT)


def hermes(text):
    \"\"\"Stage-transition ping (CLAUDE.md). Never fatal: the CLI is absent on
    some boxes (docs/M38.md) — print the fallback and move on.\"\"\"
    try:
        subprocess.run(["hermes", "send", "-t", "telegram", f"[pipeline] {text}"],
                       check=False, timeout=30, capture_output=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        print(f"(hermes unavailable) [pipeline] {text}", file=sys.stderr)


def sh(cmd):
    \"\"\"Run a stage command, streaming output; returns the exit code (the
    gates read it). Heavy stages shell out so they keep their CLIs' own
    validation and resume semantics.\"\"\"
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=ROOT).returncode


def _utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_state(stage, status, reason=None, **metrics):
    \"\"\"Atomically merge one stage verdict into runs/pipeline_state.json —
    the durable handoff the next session (or the heartbeat) reads.\"\"\"
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state = (json.loads(STATE.read_text()) if STATE.exists()
             else {"version": 1, "stages": {}})
    state["config_hash"] = CONFIG_HASH
    state["updated_at"] = _utcnow()
    state["stages"][stage] = {"status": status, "at": _utcnow(),
                              "config_hash": CONFIG_HASH,
                              "reason": reason, "metrics": metrics}
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)
    print(f"state: {stage} = {status}" + (f" ({reason})" if reason else ""))
    return state


def stage_status(stage):
    if not STATE.exists():
        return "pending"
    entry = json.loads(STATE.read_text())["stages"].get(stage)
    if entry is None:
        return "pending"
    if entry.get("config_hash") != CONFIG_HASH:
        return "stale"
    return entry["status"]


def stop_banner(stage):
    print("=" * 66)
    print(f"STOP — {stage} did not pass. Do not run the later stages against")
    print("this state; fix the input (or the config) and re-run the stage.")
    print("=" * 66)
"""),

    # ── 2 ── config ──────────────────────────────────────────────────────────
    code("""# 2 — config. Edit and re-run from here. A change flips CONFIG_HASH,
# which marks verdicts computed under the old config as stale.
import hashlib

CFG = {
    # (a) Kaggle ingest — our live submission ids (see notebooks/model_monitor.ipynb
    # MODELS dict for the ledger) + optional opponent subs to snowball-harvest.
    "SUBS": [55265099, 55265105],          # M41b: alakazam-w143 + ogerpon
    "OPP_SUBS": [],
    "MAX_NEW": 500,
    "EPISODE_FLOOR": 30,                    # rl.live_monitor MIN_N_FOR_ANY_READ

    # (b) BC shards from live replays (rl.replay_bc build)
    "BC_OUT": "data/bc_pipeline",
    "MIN_SCORE": 550.0,
    "SHARD_SIZE": 5000,
    "BUILD_EXTRA": [],                      # e.g. ["--v4"], ["--deck-hash", "20dcd313..."]

    # (c) Arena — PPO checkpoints entering the OpenSkill league as model: pilots.
    # This checkout tops out ~M30; point at the live nets on the box.
    "PPO_CKPTS": ["checkpoints/ppo_best_m20legB.pt"],
    "ARENA_DECK": "lucario",                # deck the PPO entrants pilot
    "GAMES_PER_ANCHOR": 40,
    "MIN_GAMES_FOR_READ": 30,               # below this the arena gate is a NO READ
    "BASE_ELO": 600.0,                      # presentational zero for implied Elo

    "FORCE": False,                         # True = re-run stages even if state says pass
}

WORKERS = min(8, os.cpu_count() or 8)       # HARD ceiling — see the laws cell
assert WORKERS <= 8
STATE = ROOT / "runs" / "pipeline_state.json"
CONFIG_HASH = hashlib.sha256(
    json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:16]
print(f"config {CONFIG_HASH}  workers {WORKERS}  state {STATE.relative_to(ROOT)}")
"""),

    # ── 3 ── preflight ───────────────────────────────────────────────────────
    code("""# 3 — preflight. Print-and-flag, never raise: on a box without creds or
# checkpoints the later cells degrade to SKIP instead of crashing.
FLAGS = {
    "kaggle_creds": (Path.home() / ".kaggle" / "kaggle.json").exists(),
    "kaggle_cache": (ROOT / "data" / "kaggle").exists(),
    "engine": (ROOT / "cg" / "libcg.so").exists(),
    "hermes": shutil.which("hermes") is not None,
    "league": (ROOT / "data" / "league" / "league.json").exists(),
    "ckpts": {c: (ROOT / c).exists() for c in CFG["PPO_CKPTS"]},
}
HAVE_KAGGLE = FLAGS["kaggle_creds"]
HAVE_CKPTS = all(FLAGS["ckpts"].values()) and bool(CFG["PPO_CKPTS"])
for name, val in FLAGS.items():
    print(f"  {'ok ' if val and val != {} else 'NO '} {name}: {val}")
if not HAVE_KAGGLE:
    print("-> Stage A will SKIP (no ~/.kaggle/kaggle.json; see README setup)")
if not HAVE_CKPTS:
    print("-> Arena will SKIP missing checkpoints (M38+ nets live on the box)")
write_state("preflight", "pass", flags={k: v for k, v in FLAGS.items() if k != "ckpts"},
            ckpts_present=sum(FLAGS["ckpts"].values()), ckpts_total=len(CFG["PPO_CKPTS"]))
"""),

    # ── 4 ── stage A header ──────────────────────────────────────────────────
    md("""## Stage A — Kaggle ingest

`rl.kaggle_ingest.refresh` lists episodes for our submissions, fetches + parses
the new ones into `data/kaggle/raw/` (immutable cache) and merges
`data/kaggle/episodes.parquet`; `harvest_decks` derives `opp_decks.parquet`.
Resumable and throttled by design (1 s between calls, 429 backoff) — re-running
the cell only fetches what's new. Requires `~/.kaggle/kaggle.json`."""),

    # ── 5 ── stage A run ─────────────────────────────────────────────────────
    code("""# 5 — Stage A: refresh live episodes + harvest opponent decks.
if not HAVE_KAGGLE:
    write_state("kaggle", "skip", reason="no ~/.kaggle/kaggle.json")
else:
    from rl.kaggle_ingest import refresh, harvest_decks
    refreshed = refresh(CFG["SUBS"], opp_subs=tuple(CFG["OPP_SUBS"]),
                        max_new=CFG["MAX_NEW"])
    print(f"episodes.parquet rows: {refreshed.height}")
    harvested = harvest_decks(min_games=3)
    print(f"opp_decks rows: {harvested.height}")
"""),

    # ── 6 ── gate A ──────────────────────────────────────────────────────────
    code("""# 6 — Gate A: enough fresh evidence to trust the corpus stages?
from rl.kaggle_ingest import EPISODES_PQ

if stage_status("kaggle") == "skip":
    print("Gate A: SKIP (no credentials) — corpus stages may still run on the",
          "existing data/kaggle cache if present.")
    verdict = "skip"
    n_episodes = 0
else:
    import polars as pl
    n_episodes = pl.read_parquet(EPISODES_PQ).height if EPISODES_PQ.exists() else 0
    verdict = "pass" if n_episodes >= CFG["EPISODE_FLOOR"] else "kill"
    write_state("kaggle", verdict, episodes=n_episodes,
                floor=CFG["EPISODE_FLOOR"])
hermes(f"stage A (kaggle ingest): {verdict.upper()} — {n_episodes} episodes")
if verdict == "kill":
    stop_banner("Stage A")
"""),

    # ── 7 ── stage B header ──────────────────────────────────────────────────
    md("""## Stage B — BC shards from live replays

`rl.replay_bc build` encodes qualifying replay seats into ragged-option npz
shards. **A corpus is untrusted until Gate B passes**: `audit` (G0, integrity
hard-fail) and `roundtrip` (G1, ≥ 98 % same-code alignment) — conversion bugs
produce silently-wrong labels, which train fine and lose games."""),

    # ── 8 ── stage B run ─────────────────────────────────────────────────────
    code("""# 8 — Stage B: build the BC corpus (idempotent: skips when shards exist
# for this config; set CFG["FORCE"]=True to rebuild).
bc_out = ROOT / CFG["BC_OUT"]
have_shards = bc_out.exists() and any(bc_out.glob("shard_*.npz"))
if stage_status("kaggle") == "kill" and not CFG["FORCE"]:
    print("SKIP: Gate A killed — fix the ingest first (or FORCE).")
elif have_shards and stage_status("bc_shards") == "pass" and not CFG["FORCE"]:
    print(f"SKIP (exists): {len(list(bc_out.glob('shard_*.npz')))} shards in",
          bc_out.relative_to(ROOT))
else:
    rc = sh(["uv", "run", "python", "-m", "rl.replay_bc", "build",
             "--out", bc_out, "--min-score", CFG["MIN_SCORE"],
             "--shard-size", CFG["SHARD_SIZE"], *CFG["BUILD_EXTRA"]])
    print(f"build rc={rc}")
"""),

    # ── 9 ── gate B ──────────────────────────────────────────────────────────
    code("""# 9 — Gate B: audit (G0) + roundtrip (G1 >= 98% alignment).
bc_out = ROOT / CFG["BC_OUT"]
n_shards = len(list(bc_out.glob("shard_*.npz"))) if bc_out.exists() else 0
if n_shards == 0:
    write_state("bc_shards", "skip", reason="no shards built", shards=0)
    verdict = "skip"
else:
    rc_audit = sh(["uv", "run", "python", "-m", "rl.replay_bc", "audit"])
    rc_roundtrip = sh(["uv", "run", "python", "-m", "rl.replay_bc",
                       "roundtrip", "-n", 6])
    verdict = "pass" if rc_audit == 0 and rc_roundtrip == 0 else "kill"
    write_state("bc_shards", verdict, shards=n_shards,
                rc_audit=rc_audit, rc_roundtrip=rc_roundtrip)
hermes(f"stage B (bc shards): {verdict.upper()} — {n_shards} shards")
if verdict == "kill":
    stop_banner("Stage B")
"""),

    # ── 10 ── stage C header ─────────────────────────────────────────────────
    md("""## Stage C — Arena: OpenSkill league + implied Elo

The rating system is `rl/league.py`: a persistent **OpenSkill PlackettLuce**
table (`data/league/league.json`) over `(deck, pilot)` entries, with frozen
anchors pinning the scale. PPO checkpoints enter as ordinary `model:<ckpt>`
pilots; `league run` schedules candidates against their nearest-ordinal peers
through the one battle loop (`rl.matchrunner`, resumable jsonl checkpoints).

Two footguns, pre-answered:

- **`rl/rank.py` is NOT Elo** — despite the name it's the M12 value-as-ranker
  (pairwise loss over solver sibling scores). Rankings come from `rl/league.py`.
- **Implied Elo is presentational.** It's the `scripts/live_ci.py` transform
  (`ELO_PER_WR = 700`) applied to each entry's predicted win rate against the
  frozen anchor field, zeroed at `BASE_ELO`. The load-bearing number is the
  OpenSkill ordinal (μ − 3σ); the gate reads ordinals, not Elo."""),

    # ── 11 ── stage C run ────────────────────────────────────────────────────
    code("""# 11 — Stage C: register PPO entrants, play the schedule.
league_json = ROOT / "data" / "league" / "league.json"
missing = [c for c, ok in FLAGS["ckpts"].items() if not ok]
if missing:
    print("SKIP missing checkpoints:", ", ".join(missing))
entrants = [c for c, ok in FLAGS["ckpts"].items() if ok]
if entrants:
    if not league_json.exists():
        rc = sh(["uv", "run", "python", "-m", "rl.league", "init"])
        print(f"league init rc={rc}")
    existing = {tuple(e["pilot"]) for e in
                json.loads(league_json.read_text())["entries"].values()}
    for ckpt in entrants:                    # check-before-add = idempotent
        if any(p[0] == "model" and p[1] == ckpt for p in existing):
            print(f"already entered: {ckpt}")
            continue
        rc = sh(["uv", "run", "python", "-m", "rl.league", "add",
                 "--deck", CFG["ARENA_DECK"], "--pilot", f"model:{ckpt}",
                 "--origin", "checkpoint"])
        print(f"add {ckpt} rc={rc}")
    rc_run = sh(["uv", "run", "python", "-m", "rl.league", "run",
                 "--games-per-anchor", CFG["GAMES_PER_ANCHOR"],
                 "--workers", WORKERS])
    print(f"league run rc={rc_run}")
else:
    print("SKIP: no PPO checkpoints present — standings below reflect the",
          "existing league state, if any.")
"""),

    # ── 12 ── standings + gate C ─────────────────────────────────────────────
    code("""# 12 — standings, implied Elo, Gate C.
league_json = ROOT / "data" / "league" / "league.json"
if not league_json.exists():
    write_state("arena", "skip", reason="no league.json")
    verdict, top = "skip", []
else:
    import pandas as pd
    raw = json.loads(league_json.read_text())["entries"]

    def _implied_elo(mu, sigma, anchors):
        # scripts/live_ci.py transform: elo = base + 700*(wr - 0.5), with wr =
        # PlackettLuce predicted win prob vs the mean frozen anchor. Presentational.
        try:
            from openskill.models import PlackettLuce
            m = PlackettLuce()
            a_mu = sum(a["mu"] for a in anchors) / len(anchors)
            a_sg = sum(a["sigma"] for a in anchors) / len(anchors)
            wr = m.predict_win([[m.rating(mu=mu, sigma=sigma)],
                                [m.rating(mu=a_mu, sigma=a_sg)]])[0]
            return CFG["BASE_ELO"] + 700.0 * (wr - 0.5)
        except Exception as exc:            # openskill API drift — degrade, don't die
            print(f"(implied elo unavailable: {exc})", file=sys.stderr)
            return float("nan")

    anchors = [e for e in raw.values() if e["frozen"]]
    df = pd.DataFrame([{
        "entry": eid, "origin": e["origin"], "games": e["games"],
        "mu": round(e["mu"], 2), "sigma": round(e["sigma"], 2),
        "ordinal": round(e["mu"] - 3 * e["sigma"], 2),
        "implied_elo": round(_implied_elo(e["mu"], e["sigma"], anchors), 0)
        if anchors else float("nan"),
    } for eid, e in raw.items()]).sort_values("ordinal", ascending=False)
    display(df.head(20))

    ours = df[df.entry.str.startswith("model") | (df.origin == "checkpoint")]
    if ours.empty:
        verdict, top = "skip", []
        write_state("arena", verdict, reason="no model entrants in league")
    elif int(ours.games.min()) < CFG["MIN_GAMES_FOR_READ"]:
        # the m22 lesson: refuse a verdict the sample size cannot support
        verdict, top = "no-read", []
        write_state("arena", verdict,
                    reason=f"min games {int(ours.games.min())} < "
                           f"{CFG['MIN_GAMES_FOR_READ']} — accrue more, re-run",
                    entrants=len(ours))
        print("NO READ — not a result; run more games (cell 11).")
    else:
        anchor_ord = df[df.origin == "anchor"].ordinal.max()
        best = ours.iloc[0]
        verdict = "pass" if best.ordinal >= anchor_ord else "kill"
        top = df.head(5).to_dict("records")
        write_state("arena", verdict, best_entry=best.entry,
                    best_ordinal=float(best.ordinal),
                    top_anchor_ordinal=float(anchor_ord), standings=top)
hermes(f"stage C (arena): {verdict.upper()}"
       + (f" — top: {top[0]['entry']} ord {top[0]['ordinal']}" if top else ""))
if verdict == "kill":
    stop_banner("Stage C")
"""),

    # ── 13 ── ship instructions (human) ──────────────────────────────────────
    md("""## Ship path — HUMAN ONLY, this notebook stops here

The pipeline's verdicts are **screening only** (three consecutive offline→live
inversions — docs/M43-plan.md). Shipping is the standing CLAUDE.md sequence,
run by a person, with a hard stop for review:

1. Offline gate on the hashed spec:
   `uv run python scripts/gate_spec.py run docs/specs/<spec>.json --out runs/<name> --workers 8`
   then `... decode` — verdict must be PASS.
2. Export the exact ship state:
   `./build_submission.sh --checkpoint checkpoints/<ckpt>.pt --deck <deck>`
   (runs `tcg.shipping export` + the bundle self-gate).
3. `uv run python scripts/ship_verify.py --checkpoint <ckpt> --deck <deck> --corpus <shard dirs>`
4. `uv run python scripts/qc_battery.py --prefix mXX_qc_<arm>` — all working
   sample agents + the previous ship tarball mirror leg.
5. **STOP.** Piotr reviews the replays in `replays/` and gives the explicit go.
   Which deck ships is always confirmed with him first.
6. On submit (manual): add the submission id to the `MODELS` dict in
   `notebooks/model_monitor.ipynb` **in the ship commit**, and diary the gate
   numbers in the milestone doc as they land.

Next session: read `runs/pipeline_state.json` for where this run stopped —
each stage carries its verdict, metrics, and the config hash it was computed
under."""),
]


def build() -> dict:
    return {
        "cells": _with_ids(CELLS),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.8"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
