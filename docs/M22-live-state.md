# M22 live A/B — current state

_Generated 2026-07-21T22:48:38+00:00 by `notebooks/m22_ab_monitor.ipynb`._

**Decidable: NO**

```
arm            n     W-L      liveWR  95% CI            score   last seen
54885173    49   25-24    0.510  [0.370, 0.650]    723.0  2026-07-21T22:20
54864190    45   22-22    0.500  [0.354, 0.646]    564.5  2026-07-21T18:34
54849475    44   17-27    0.386  [0.239, 0.534]    541.9  2026-07-20T22:51
54846434    44   19-25    0.432  [0.284, 0.580]    531.5  2026-07-20T22:23

observed gap : +1.0pp  (54885173 minus 54864190)
MDE at this n: 28.9pp  (80% power, alpha=0.05)

NO READ — the gap is INSIDE the noise floor. Resolving a true 1.0pp gap needs n=37690/arm (have 49 and 45).
Reporting a winner here would repeat the M21 error. Keep accruing.
Note: at the 10pp design target, n=393/arm is required.
```

| submission | arm | n | W–L | live WR | 95% CI | score |
|---|---|---|---|---|---|---|
| 54885173 | M24 — replay-BC clone (Alakazam deck) | 49 | 25–24 | 0.51 | [0.37, 0.65] | 723.0 |
| 54864190 | M22c-RL — sample-agent teacher | 45 | 22–22 | 0.5 | [0.354, 0.646] | 564.5 |
| 54849475 | B3 — KL→0 self-play | 44 | 17–27 | 0.386 | [0.239, 0.534] | 541.9 |
| 54846434 | B2 — encoder-v4 + plan-PPO | 44 | 19–25 | 0.432 | [0.284, 0.58] | 531.5 |
| 54836093 | champion (context) | 45 | 22–23 | 0.489 | [0.343, 0.635] | 579.3 |

## For the next session

- Re-run `notebooks/m22_ab_monitor.ipynb` cell 2, then this cell, before quoting any number.
- If **Decidable: NO**, the arms are not separable yet — do not report a direction.
- B2 vs B3 differ by 2.0pp mirror / 3.5pp meta offline, which is below what this
  pipeline can resolve in reasonable time. This A/B may never resolve; if so that
  is the finding, and future pairs must be chosen to differ by ≥10pp.
- Context: `docs/M22.md` (diary), `docs/M22-plan.md` (plan).
