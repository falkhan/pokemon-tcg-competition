"""Generates notebooks/m22_ab_monitor.ipynb. Edit here, re-run, re-open the notebook.

Kept as a .py because hand-editing .ipynb JSON is how cells silently break.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "m22_ab_monitor.ipynb"


def md(*src):
    return {"cell_type": "markdown", "metadata": {}, "source": list(src)}


def _with_ids(cells):
    for i, c in enumerate(cells):
        c["id"] = f"m22a-{i:02d}"
    return cells


def code(*src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": list(src)}


CELLS = [
    md("""# M22a — SHIP → FIGHT → MEASURE → REPLAN

The durable artifact for the M22 live A/B. Replaces the session cron (which died
with its Claude session and expired in 7 days, against a 4–22 day accrual).

**How to use:** run cell 1 once, then **re-run cell 2 (REFRESH) whenever you want a
fresh read**. Cells 3–6 recompute from it. Cell 7 writes the handoff file that the
next milestone session reads.

## The one rule this notebook enforces

It **refuses to name a winner** unless the observed gap exceeds the minimum
detectable difference at the current sample size. M21 shipped twice on differences
its instruments could not resolve (B2 z=0.68, B3 z=0.80), and on 2026-07-20 an
early read of "B3 601 vs B2 529" got quoted directionally while B3's score was
still the μ=600 opening prior decaying — it went 600 → 511 → 663 → 601 inside four
hours. Both errors were avoidable by computing power first, so cell 3 computes it
first and prints `NO READ` rather than a number you can over-interpret.

Throughput is ~60–100 episodes/day across **all** active arms, so a 15pp gap needs
~4–6 days, 10pp ~8–13 days, and 5pp ~5–10 weeks. Expect `NO READ` for a while.
That is the instrument working."""),

    code("""# 1 — setup. Run once.
import sys
from pathlib import Path

_ROOT = Path.cwd()
while not (_ROOT / "rl").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from rl.kaggle_ingest import forensics, list_episodes, refresh
from rl.live_monitor import compare, mde, n_needed, read_arms

# The two arms under test. Champion is context only — NOT a third series
# (see docs/M22.md: this A/B is B2 vs B3).
ARMS = {
    54849475: "B3 — KL→0 self-play",
    54846434: "B2 — encoder-v4 + plan-PPO",
}
CHAMPION = 54836093          # M20 legB, for reference in the table only

# Repo house palette (notebooks/model_monitor.ipynb). Validated for 2 categorical
# slots: normal ΔE 29.0, protan 26.5, contrast ≥3:1 both. Tritan ΔE 7.6 sits in the
# 6–8 floor band, so direct labels are REQUIRED as secondary encoding — never rely
# on hue alone here.
COLOR = {54849475: "#2a78d6", 54846434: "#008300"}

INK, INK2, GRID, SURFACE = "#1b1b1b", "#5b5b5b", "#e3e3e0", "#fcfcfb"
pio.templates["pokedex"] = go.layout.Template(layout=dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
))
pio.templates.default = "pokedex"

TARGET_N = 200          # accrual target per arm (~a 14pp read)
DESIGN_MDE_PP = 10.0    # the gap we would like to be able to resolve

print(f"arms: {list(ARMS)}   target n/arm: {TARGET_N}")"""),

    code("""# 2 — REFRESH. ⟳ Re-run this cell to pull new episodes and update everything below.
# Resumable and idempotent: the raw episode cache is immutable, so re-running is cheap.
_ = refresh(our_subs=list(ARMS) + [CHAMPION], max_new=300)


def fetch_arm(sub_id: int, label: str) -> pd.DataFrame:
    \"\"\"One row per episode, from OUR seat's perspective.\"\"\"
    rows = list_episodes(submission_id=sub_id).to_dicts()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ours0 = df["submission_id_0"] == sub_id
    df["our_reward"] = df["reward_0"].where(ours0, df["reward_1"])
    df["opp_reward"] = df["reward_1"].where(ours0, df["reward_0"])
    df["our_score"] = df["updated_score_0"].where(ours0, df["updated_score_1"])
    df["time"] = pd.to_datetime(df["end_time"].fillna(df["create_time"]),
                                format="mixed", utc=True)
    df["win"] = (df["our_reward"] > df["opp_reward"]).astype(float)
    df.loc[df["our_reward"] == df["opp_reward"], "win"] = 0.5
    df["model"], df["submission"] = label, sub_id
    return df.sort_values("time").reset_index(drop=True)


episodes = pd.concat([fetch_arm(s, l) for s, l in ARMS.items()], ignore_index=True)
print(f"{len(episodes)} episodes across {episodes['submission'].nunique()} arms")
episodes.groupby("model")["episode_id"].count()"""),

    code("""# 3 — THE POWER GATE. This is the cell that is allowed to say who is winning.
# It will print NO READ until the gap clears the MDE. That is not a failure.
arms = read_arms(list(ARMS))
decidable, report = compare(arms, DESIGN_MDE_PP)
print(report)
print()
if decidable:
    print("→ A real read exists. Proceed to the post-mortem (cell 6) and record it.")
else:
    short = [a for a in arms if a.n < TARGET_N]
    if short:
        eta_lo = max((TARGET_N - a.n) / 50 for a in short)   # ~50 eps/arm/day, 2 arms
        eta_hi = max((TARGET_N - a.n) / 30 for a in short)   # ~30 on a slow day
        print(f"→ Still accruing. ETA to n={TARGET_N}/arm: ~{eta_lo:.1f}–{eta_hi:.1f} days.")
    print("→ Do NOT interpret the gap above. Re-run cell 2 tomorrow.")"""),

    code("""# 4 — Leaderboard score over time. Single axis, one line per arm, direct-labelled.
# Direct labels are not decoration here: they are the required secondary encoding
# for the tritan pair (ΔE 7.6), so identity never rests on hue alone.
fig = go.Figure()
for sub, label in ARMS.items():
    d = episodes[episodes["submission"] == sub].dropna(subset=["our_score"])
    if d.empty:
        continue
    fig.add_trace(go.Scatter(
        x=d["time"], y=d["our_score"], name=label, mode="lines",
        line=dict(color=COLOR[sub], width=2),
        hovertemplate=f"<b>{label}</b><br>%{{x|%b %d %H:%M}}<br>"
                      f"score %{{y:.1f}}<extra></extra>"))
    last = d.iloc[-1]
    fig.add_annotation(x=last["time"], y=last["our_score"], text=f"  {label.split(' — ')[0]}",
                       showarrow=False, xanchor="left", font=dict(color=INK, size=12))

fig.add_hline(y=600, line=dict(color=GRID, width=1, dash="dot"))
fig.add_annotation(x=0, xref="paper", y=600, text="μ=600 opening prior", showarrow=False,
                   xanchor="left", yanchor="bottom", font=dict(color=INK2, size=11))
fig.update_layout(
    title="Live leaderboard score — the two M22 arms",
    xaxis_title=None, yaxis_title="Kaggle score",
    hovermode="x unified", height=420,
    margin=dict(l=60, r=110, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
fig.show()

print("Early movement is the μ=600 prior decaying, not play. Read cell 3, not this chart.")"""),

    code("""# 5 — Table view. The accessible counterpart to chart 4, and the record the
# handoff file is built from. Champion included as context only.
rows = []
for a in read_arms(list(ARMS) + [CHAMPION]):
    lo, hi = a.ci95() if a.n else (float("nan"), float("nan"))
    rows.append({
        "submission": a.sub,
        "arm": ARMS.get(a.sub, "champion (context)"),
        "n": a.n, "W": a.wins, "L": a.losses, "D": a.draws,
        "live_wr": round(a.wr, 3),
        "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3),
        "score": round(a.score, 1) if a.n else None,
        "last_seen": a.last_seen,
    })
table = pd.DataFrame(rows)
display(table)

_a, _b = read_arms(list(ARMS))[:2]
if _a.n and _b.n:
    print(f"MDE at current n: {mde(_a.n, _b.n) * 100:.1f}pp  ·  "
          f"n needed for a {DESIGN_MDE_PP:.0f}pp read: {n_needed(DESIGN_MDE_PP / 100)}/arm")"""),

    code("""# 6 — POST-MORTEM. Per-archetype W/L for each arm.
# NOTE: forensics reads archetypes from opp_decks.parquet, so `harvest` must have
# run since the last refresh or every new episode reports as `unknown` — silently.
# Uncomment to (re)build it; it rescans the whole raw cache, so it takes a few minutes.
# !cd {_ROOT} && uv run python -m rl.kaggle_ingest harvest --min-games 3

for sub, label in ARMS.items():
    print(f"\\n=== {label} ({sub}) ===")
    try:
        forensics(submission_id=sub)
    except Exception as exc:                      # noqa: BLE001 — notebook ergonomics
        print(f"  (unavailable: {exc})")

print(\"\"\"
Reading these: the field is ~90% mega_lucario_ex+solrock, so only that row will
carry usable n. The tail rows are decoration until they reach double digits.
And per-archetype splits divide an already-underpowered sample — treat them as
hypothesis-generating, never as a result.\"\"\")"""),

    code("""# 7 — DURABLE ARTIFACT. Writes the handoff the next milestone session reads.
import json
from datetime import datetime, timezone

state = {
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "milestone": "M22a",
    "arms": {str(s): l for s, l in ARMS.items()},
    "champion_context": CHAMPION,
    "target_n_per_arm": TARGET_N,
    "design_mde_pp": DESIGN_MDE_PP,
    "decidable": bool(decidable),
    "verdict": report,
    "table": table.to_dict(orient="records"),
}

out_json = _ROOT / "data" / "kaggle" / "m22_ab_state.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(state, indent=2, default=str))

lines = [
    "# M22 live A/B — current state",
    "",
    f"_Generated {state['generated_at']} by `notebooks/m22_ab_monitor.ipynb`._",
    "",
    f"**Decidable: {'YES' if decidable else 'NO'}**",
    "",
    "```",
    report,
    "```",
    "",
    "| submission | arm | n | W–L | live WR | 95% CI | score |",
    "|---|---|---|---|---|---|---|",
]
for r in state["table"]:
    ci = (f"[{r['ci95_lo']}, {r['ci95_hi']}]"
          if r["n"] else "—")
    lines.append(f"| {r['submission']} | {r['arm']} | {r['n']} | "
                 f"{r['W']}–{r['L']} | {r['live_wr']} | {ci} | {r['score']} |")
lines += [
    "",
    "## For the next session",
    "",
    "- Re-run `notebooks/m22_ab_monitor.ipynb` cell 2, then this cell, before quoting any number.",
    "- If **Decidable: NO**, the arms are not separable yet — do not report a direction.",
    "- B2 vs B3 differ by 2.0pp mirror / 3.5pp meta offline, which is below what this",
    "  pipeline can resolve in reasonable time. This A/B may never resolve; if so that",
    "  is the finding, and future pairs must be chosen to differ by ≥10pp.",
    "- Context: `docs/M22.md` (diary), `docs/M22-plan.md` (plan).",
]
out_md = _ROOT / "docs" / "M22-live-state.md"
out_md.write_text("\\n".join(lines) + "\\n")

print(f"wrote {out_json.relative_to(_ROOT)}")
print(f"wrote {out_md.relative_to(_ROOT)}")"""),

    md("""## 8 — Shipping (SHIP)

Deliberately **not** automated: a submission is irreversible, consumes one of 4
daily slots, and competes for the episode throughput the arms above need.

```bash
./build_submission.sh --checkpoint <ckpt>.pt --deck <deck>
uv run kaggle competitions submit -c pokemon-tcg-ai-battle \\
    -f submission.tar.gz -m "<milestone> <what changed>"
```

After shipping, add the new id to `ARMS` in cell 1 and re-run.

**Pair-selection rule.** Ship pairs simultaneously so both face the same field over
the same window — that controls for field drift and for the ship-order confound
(mirror correlates with calendar order at ρ=+0.991, which is what made it look
live-predictive). And pick pairs that differ by **≥10pp** on some offline measure;
anything closer cannot be adjudicated here before the competition ends."""),
]

nb = {
    "cells": _with_ids(CELLS),
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(CELLS)} cells)")
