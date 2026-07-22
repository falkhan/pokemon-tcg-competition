# M26 plan — fix the ATTACH/Telepath fidelity gap (stop the self-mill deck-outs)

**Refined 2026-07-22** (from the post-mortem stub; diary [M26.md](M26.md)).

**Thesis (confirmed):** sub 54897966 peaked ~820 and trends down (725.9 @ n=40,
20W–20L); **11/20 losses are self-mill deck-outs**. Cause: energy under-attachment
(46.4% of turns vs teacher 57%), worst on Telepath Psychic Energy (id 19, the setup
engine), contested-attach share 16% vs teacher 45%. Per-class fidelity report (exact
val split): **ATTACH 0.408** (Telepath cell **0.343**, top confusion = ATTACK
instead) vs headline 0.668; second weak class **PLAY 0.486 at 23% share**
(supporter hoarding). Lever 1 of the stub is therefore DONE — the gap is real,
class-shaped, and aimed exactly where the stub guessed.

**Session decisions (Piotr):** levers A/B'd as separate arms + measured
composition; conditional PLAY×2 pre-approved; override lever gains the O2
"attach backstop" variant (guarantee an attach every turn one is legal).

## Phase 0 — refresh + re-confirm (DONE 07-22)

n=40 refresh, probes re-run, kill-gate (deck-out share ≥50%) passed — see diary.
Probe scripts `scripts/sub_behavior.py` + `scripts/attach_probe.py` committed as
the standard post-ship diagnostic.

## Phase 1 — `scripts/class_report.py` (formalize lever 1)

Reuses `BCDatasetV3`/`collate_v3`/`OptionScorerV3`; reconstructs the split exactly
as `train()` (`default_rng(0)`, 10% of games — data-dir order matters). Emits
headline, per-OptionType acc+share, within-ATTACH and within-PLAY per-card acc,
top confusions. **Acceptance test: reproduces 0.668 / ATTACH 0.408 / Telepath
0.343 / PLAY 0.486 on the shipped ckpt.** Runs after every M26 train.

## Phase 2 — mill-opponent baseline (zero new code)

`rl.matchrunner play --a model:checkpoints/m25_bc_alakazam_v3h.pt:clone54618168
--b model:checkpoints/m25_bc_grim_54861775.pt:data/kaggle/grimmsnarl_3121746f_deck.csv
-n 200 --workers 8 --seed {1,2}` (+ n=100 sanity vs `m25_bc_grim_54863653.pt`;
keep the more mill-faithful clone). Establishes the grim pin. If the shipped ckpt
doesn't bleed here, the clones aren't reproducing the stall punish (val
0.565/0.580) → opponent stays **advisory**, not a hard gate.

## Phase 3 — lever 2: ATTACH-class loss weighting

`--class-weight TYPE:k` (repeatable) in `plan_iter train`: after the
`--uniform-weights` block, `ds.weights[teacher_type == ATTACH] *= k`; consumed by
the existing weighted CE; `evaluate()` stays unweighted (M18a invariant).
**Dose law:** ATTACH touches 5.9% of rows (~16–24% effective loss share at k=3–5)
— an order of magnitude below the M21 kill regime (36% rows @10×) and nothing
like the M25 score-weighting kill (100% of rows). PLAY×2 touches 23% → conditional
arm only.

Arms (base recipe = the shipped one: `--data data/bc_m25_alakazam_v3h --init-v3h
checkpoints/m24_bc_54618168.pt --epochs 10 --lr 1e-4`):
**A** `m26_bc_attach3` (ATTACH:3) · **B** `m26_bc_attach5` (ATTACH:5) ·
**C** (conditional): winner-k + PLAY:2, only if A/B promotes with PLAY unimproved.

Gates per arm, cheapest first:
1. **Fidelity (free):** PROMOTE iff ATTACH ≥0.50 AND Telepath ≥0.45 AND headline
   ≥0.658 AND no other class −3pp+. Else KILL + diary the deltas.
2. **Screen:** n=200 seed 1 vs `rule:lucario` (clone deck); KILL <0.59.
3. **Confirm:** seed 2 → pooled n=400 ≥0.660 lucario; dragapult pooled n=400
   ≥0.3075 (EVAL-ONLY); kyogre n=200 ≥0.95; grim n=200×2 vs Phase-2 pin
   (advisory unless proven discriminative).

## Phase 4 — lever 3: flag-gated attach overrides (O1 + O2)

Both fire only at MAIN with `not obs.current.energyAttached`; option-list-driven:
- **O1 Telepath-priority:** legal ATTACH option with acted id 19 AND bench slot
  open → take it (model's best-scoring if several). Teacher mimicry of the
  contested-attach preference.
- **O2 attach backstop:** chosen action would end the turn (END, or ATTACK — the
  top observed confusion) while a legal energy ATTACH exists → take the model's
  highest-scoring ATTACH instead (attach doesn't consume the attack; the next
  MAIN re-offers it). **Guarantee: no turn ends with an unplayed legal attach.**
  Target choice = model's within-ATTACH scores; fallback if targets look bad in
  the behavior probe: `rl/generic_pilot.py::_attach_recipient_value`. Caveat:
  teacher attaches 57%, not 100% — O2 deliberately overshoots; measured arm.

Touch points, off by default (override law M8.1/M12/M13): matchrunner spec kinds
`modelt:` (O1) / `modela:` (O2) applied in **fn4 AND fn3** (spec kinds, not env
vars — jsonl run keys must not alias); same predicates in
`submission/main.py::agent()` behind per-fix flags (PKM_* env pattern, default
off); **scoped parity test** (synthetic obs through both pilots, identical
actions). A/B vs the no-flag baseline: non-inferiority ≥0.625 pooled n=400
lucario (else KILL) + improvement on the mill axis (grim WR / attach-rate).
O1+O2 compose; measured composition only. **Mandatory Piotr sign-off before any
override ships.**

## Phase 5 — pipeline hardening + ship

- `scripts/offline_behavior.py` (if Phase 4 runs): in-process games instrumented
  for attach-rate, Telepath contested share, deck-out losses — screen jsonl only
  stores result codes.
- Battery script gains the grim opponent; class_report post-train; parity test in
  the pre-ship checklist.
- Ship = Piotr's call, one ship, candidate matrix presented (fidelity + battery +
  grim + behavior per surviving arm). Deck stays `clone54618168` (md5 8e8cf124)
  unless Piotr changes it. Full ritual incl. model_monitor update; if an override
  ships, its flag is ON in the bundled main.py + parity green.
- Post-ship watch (~10–20 games): probes → attach-rate ≥~50%, Telepath contested
  ≥~30%, deck-out share of losses <50%, turns ↓ toward 16.6.

## Piotr decision points

Phase 0 kill-gate re-scope (passed) · Phase 4 override sign-off (mandatory) ·
Phase 5 ship + deck.

## Risks

Weighting distorting healthy classes (fidelity kill-gate first) · override-law
losing streak (off-by-default, non-inferiority, sign-off) · O2 overshoots teacher
attach rate (measured arm, never default) · fn4/fn3/main.py predicate divergence
(scoped parity test) · grim opponent may not reproduce the stall punish (advisory
pin) · class_report split fragility (acceptance test mandatory).

## Standing constraints (unchanged)

Workers ≤8 · dragapult never in a pool · one ship per milestone after the full
battery · deck md5 ritual · MDE refusal discipline · diary incrementally ·
Hermes telegram at phase transitions.
