# M43 PPO Pipeline — Architectural Review (2026-08-11)

Reviewed against the latest milestone, **M43** (`docs/M43-plan.md`, pre-registered
2026-08-05; last shipped = M41b, subs 55265099 + 55265105). Companion to the M37
code audit (`docs/m37-code-audit.md`). Produced on branch
`claude/environment-preparation-yx40dw` together with a repo cleanup and the
pipeline notebook (`notebooks/pipeline.ipynb`); diaried in `docs/DECISIONS.md`.

## Scope & method

Every module on the PPO path was read end-to-end and cross-checked against the
M43 plan's recipes, the hashed gate specs in `docs/specs/`, and the standing
rules in `CLAUDE.md`. Constraint that drives every disposition below: **this
checkout has none of the M38+ artifacts** (`checkpoints/` tops out ~M30; the 25
Lane-A beds, `m41b_wide_prod.pt`, `m41_ogerpon.pt`, `m39_retain_b.pt` live on
the box), so no fix can be validated by a training run here — only by the
offline test suite (`uv run pytest`, 1259 passed at review time). Fixes are
therefore limited to parse/CLI-layer changes with unit tests; anything touching
reward math or the ship artifact is documented with its exact remediation and
deferred to the box.

## Pipeline map (as of M43)

```
Kaggle live replays
  └─ rl/kaggle_ingest.py  (refresh/harvest/meta/forensics — resumable, throttled)
       └─ rl/replay_bc.py build          → BC shards from leaderboard replays
          scripts/m40_s2_collect.py      → BC shards from self-play vs beds (Lane B,
                                           resumable, agreement kill at collection)
            └─ rl/plan_iter.py train     → the supervised trunk (BCDatasetV3,
                                           --init-wide widening, --outcome-weight)
                 └─ rl/ppo.py            → PPO loop: GAE (compute_gae), clipped
                    + rl/collector.py      update (ppo_update), KL anchor (M20),
                                           plan surrogate (M21), vs_teacher
                                           promotion (M23), value shaping (M43);
                                           collector owns rollouts, reward
                                           shaping, opponent pool
                      └─ eval/ranking    → rl/matchrunner.py play (single battle
                                           loop, spec grammar, resumable jsonl);
                                           rl/league.py (OpenSkill PlackettLuce)
                           └─ gates      → scripts/gate_spec.py run/decode vs
                                           hashed docs/specs/*.json;
                                           scripts/strength_gate.sh;
                                           scripts/ci_gate.py
                                └─ ship  → tcg/shipping.py export+gate →
                                           scripts/ship_verify.py →
                                           scripts/qc_battery.py →
                                           MANDATORY human replay review →
                                           human submits
```

Not the pipeline, deliberately kept: `tcg/ppo.py` + `tcg/selfplay.py` are
frozen M8-era parity twins (pure functions pinned by `tests/test_ppo.py`; the
`train()` loops have diverged — the twin has no post-M20 flag). `rl/bc.py
collect` and `rl/plan_iter.py collect --mode ei` / `relabel` are dead recipes
kept wired for reproducibility (`ARCHITECTURE.md` §"dead recipes").

All three M43 spec hashes verify against `scripts/gate_spec.py hash`:
`c21d3351e44de0e9` (laneA_base), `ad5930925ddf024b` (laneA_phi),
`9da112f64aaa77f8` (laneB).

## Findings (ranked)

### 1. HIGH — Φ anchor not settable; Phase-0 kill branch silently mis-anchors
`rl/ppo.py` hardcoded `phi_ckpt=(str(CKPT_DIR / start) if shaping == "value"
else None)` at the `collect()` call. The M43 plan's Phase-0 branch 0.c routes
an E0 KILL to a critic warm-start from `m39_retain_b.pt` via `--value-ckpt` —
in that branch Φ silently stayed on `--start`, so the A-phi arm would have run
with a Φ the pre-registration never described. The plan and the landing commit
both claim an explicit `--phi-ckpt` in `rl/ppo.py`; only `rl/collector.py` had
it.
**FIXED on this branch**: `--phi-ckpt` (default `None` → `--start`, the old
behavior) exposed in the CLI and threaded through `train()`;
`--phi-ckpt` without `--shaping value` is a CLI error. Tests:
`tests/test_ppo_cli.py`.

### 2. HIGH — `past=` opponent pool cross-contaminates gated arms
`rl/collector.py parse_pool` resolved `past=` with
`glob("ppo_*it*.pt")[-3:]` — every tag's promoted selves, sorted
alphabetically. Running A-phi after A-base fed A-base's selves into A-phi's
pool, breaking the single-variable comparison the two hashed specs
(`m43_laneA_base` vs `m43_laneA_phi`) exist to make. (`default_pool` uses
`ppo_it*.pt`, which is untagged-only — the bug was `parse_pool`'s wildcard.)
**FIXED on this branch**: `parse_pool(..., tag=...)` scopes the glob to
`ppo_<tag>_it*.pt` (untagged runs see only `ppo_it*.pt`); `rl/ppo.py` passes
its `--tag`. Tests: `tests/test_plan_ppo.py` (tag-scoping trio).

### 3. MED — `--shaping value`/`dev` was a silent no-op without `--race-shaping`
The collector builds a Φ net only when `race_shaping and shaping == "value"`
(same gating for `dev`), so `python -m rl.ppo --shaping value` alone ran a full
leg unshaped with no warning — the M17 "instrument the silent path" failure
shape, aimed at a pre-registered arm. The plan's A2 recipe passes both flags,
so a literal follow was safe; a typo'd rerun was not.
**FIXED on this branch**: CLI hard-fail (`p.error`) on a non-default
`--shaping` with a zero coefficient. Tests: `tests/test_ppo_cli.py`.

### 4. MED — gate config vs shipped fix-set can diverge (DEFERRED)
The gate battery measures the `model-c-pkgz:` spec token (= `conserve,
racemode2, racemode4, planzero`, resolved in `rl/matchrunner.py`), but the
shipped bundle reads its serve fixes from a hardcoded default string in the
tracked `submission/main.py` (currently `"conserve,planzero,ash,ashguard"`).
`tcg.shipping export` has no `--fixes` flag and does not regenerate `main.py`,
so gating one config and shipping another is a manual-edit-away accident.
`scripts/ship_verify.py` regex-reads the literal and checks fix-name validity,
but not equality with the gated token.
**Remediation (do on the box, before the next ship)**: add `--fixes` to
`tcg.shipping export` that templates the string into `submission/main.py`;
extend `scripts/ship_verify.py` to require the shipped fix set to equal the
gated spec token's package. Deferred here because it touches the ship artifact,
which this checkout cannot QC end-to-end. → BACKLOG.

### 5. MED — potential shaping does not telescope (DEFERRED)
`rl/collector.py` applies `F = coef·(Φ(s') − Φ(s))` with **no γ** and **no
terminal Φ subtraction**, so the shaped return keeps a residual
`coef·(Φ_last − Φ_first)` — the agent is paid for *ending* in a high-value
state. This is not the Ng/Harada/Russell policy-invariant form
(`F = γΦ(s') − Φ(s)`, with Φ(terminal)=0) that the M43 plan and BACKLOG #13(c)
invoke. Pre-existing (race/dev shaping share the site); with Φ = V(s) the
residual grows with how well the value head separates states, i.e. exactly
when M43's A-phi arm matters most.
**Remediation**: either apply the γ-discounted form and zero Φ at terminal
states, or pre-register the residual as intended. A reward-math change is
unverifiable in this checkout and would alter a pre-registered arm mid-flight —
M43 must decide before Lane A launches. → BACKLOG.

### 6. LOW — `--race-shaping` is a misnomer under value/dev shaping (DOC ONLY)
The coefficient flag is spelled `--race-shaping` whatever the potential. The
plan's recipes are internally consistent (`--shaping value --race-shaping
<coef>`), so this is legibility debt, not a defect. Propose a `--shaping-coef`
alias (keeping `--race-shaping` accepted) at M43 execution; not renamed here to
avoid churning pre-registered command lines.

### 7. LOW — doc drift (FIXED on this branch)
- `docs/MILESTONES.md` had no record past M26 (summary table) / M24 (notes),
  violating the CLAUDE.md hard rule; header still named `ppo_best_m20legB` as
  champion. → one-line-per-milestone catch-up M27–M42 + header fix.
- `ARCHITECTURE.md` §1 named the M20 champion and claimed `rl/ppo.py` never
  passes `collect(pool=...)` — false since M21. → corrected.
- `rl/ppo.py` module docstring claimed `compute_gae`/`ppo_update` were
  `NotImplementedError` stubs (M2-era). → rewritten.
- `CLAUDE.md` project shape omitted `rl/ppo.py`/`rl/collector.py`, said
  "milestone records M0–M14", and listed 4 of the 5 bundled twin files
  (`scaling.py`, added M42, was missing vs `scripts/ship_verify.py`
  `TWIN_FILES`). → corrected.
- Note for M43 execution: the plan's `--kl-coef "<M20's value — look up on the
  box>"` is **0.1**, and it is in this repo at `docs/M20.md`.

### 8. INFO — repository weight: 525 MB of ignored-but-tracked artifacts
96% of tracked bytes predate their own ignore rules: `data/` 342.6 MB (680
files), `checkpoints/` 171.4 MB (55 files, all pre-M31 — the live nets are not
in git at all), `runs/` 0.6 MB (130 files), `decks/gen/`, `dist/` (1 tarball —
plausibly `qc_battery`'s previous-ship mirror leg), plus `cg/` committed 4×
(~16 MB: `cg/`, `submission/cg/`, `submission_rules/cg/`,
`data/external/buddy/cg/`) despite README calling it user-provided.
**Decision (Piotr, 2026-08-11): keep everything tracked; document only.**
Rationale: fresh clones stay runnable (this review's own test runs worked only
because `cg/` and the derived `data/` parquets were present), and no history
rewrite is acceptable (commit SHAs are cited across the diaries). Revisit if
clone weight becomes a real cost.

## Retired in the accompanying cleanup

21 zero-reference milestone one-shots deleted from `scripts/` (m24×2, m28×2,
m29×1, m33×3, m38×1, m39×9, m40×2, `reweight_retreat.py`) after a repo-wide
reference grep — results live in the diaries, code recoverable from git
history (M25-cleanup precedent). Also: empty accidental `docs/CLAUDE.md`,
personal `.idea/`. **Not** touched, per the M37 audit's "0 imports ≠ dead"
lesson: `rl/value_train.py` (CLI of `scripts/m28_e_value.sh`),
`rl/behavior.py`, `rl/gate.py`/`rl/export.py` (test-pinned parity twins),
`M7_MODULES`, `tcg/deck_search.py` (M25 near-miss), `rl/mcts.py`, all
`scripts/*_probe.py` (count pinned = 22), sample agents, `docs/M*-plan.md`
(pre-registration records), and the rl/↔tcg/ duplication (tracked BACKLOG
item).

## Environment note

This checkout: engine `cg/` present, suite green offline (3 pre-existing
environment-dependent failures outside `ci_gate.py`'s `KNOWN_FAILURES`:
`test_fetch_offline_error_mentions_runbook` and 2 `test_leaderboard_decks`
param cases needing gitignored `data/kaggle/` files). Hermes CLI absent —
notification calls must stay non-fatal. Kaggle credentials absent — ingest
stages skip. The pipeline notebook (`notebooks/pipeline.ipynb`) encodes these
guards.

## Disposition summary

| # | Finding | Disposition |
|---|---------|-------------|
| 1 | Φ anchor not settable | FIXED (this branch) |
| 2 | `past=` cross-tag contamination | FIXED (this branch) |
| 3 | silent no-op shaping | FIXED (this branch) |
| 4 | gate-vs-shipped fix-set divergence | DEFERRED → BACKLOG (box) |
| 5 | non-telescoping shaping | DEFERRED → BACKLOG (M43 pre-registration decision) |
| 6 | `--race-shaping` misnomer | DOC ONLY (alias at M43 execution) |
| 7 | doc drift | FIXED (this branch) |
| 8 | ignored-but-tracked 525 MB / cg ×4 | DOC ONLY (user decision: keep tracked) |
