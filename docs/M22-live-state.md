# M22 live A/B — current state

_Generated 2026-07-20T11:57:38+00:00 by `notebooks/m22_ab_monitor.ipynb`._

**Decidable: NO**

```
arm            n     W-L      liveWR  95% CI            score   last seen
54849475    34   13-21    0.382  [0.214, 0.550]    542.2  2026-07-20T11:08
54846434    35   15-20    0.429  [0.263, 0.594]    536.8  2026-07-20T10:47

observed gap : -4.6pp  (54849475 minus 54846434)
MDE at this n: 33.7pp  (80% power, alpha=0.05)

NO READ — the gap is INSIDE the noise floor. Resolving a true 4.6pp gap needs n=1838/arm (have 34 and 35).
Reporting a winner here would repeat the M21 error. Keep accruing.
Note: at the 10pp design target, n=393/arm is required.
```

| submission | arm | n | W–L | live WR | 95% CI | score |
|---|---|---|---|---|---|---|
| 54849475 | B3 — KL→0 self-play | 34 | 13–21 | 0.382 | [0.214, 0.55] | 542.2 |
| 54846434 | B2 — encoder-v4 + plan-PPO | 35 | 15–20 | 0.429 | [0.263, 0.594] | 536.8 |
| 54836093 | champion (context) | 45 | 22–23 | 0.489 | [0.343, 0.635] | 579.3 |

## For the next session

- Re-run `notebooks/m22_ab_monitor.ipynb` cell 2, then this cell, before quoting any number.
- If **Decidable: NO**, the arms are not separable yet — do not report a direction.
- B2 vs B3 differ by 2.0pp mirror / 3.5pp meta offline, which is below what this
  pipeline can resolve in reasonable time. This A/B may never resolve; if so that
  is the finding, and future pairs must be chosen to differ by ≥10pp.
- Context: `docs/M22.md` (diary), `docs/M22-plan.md` (plan).
