# Data Diagnostics — Autonomous_QUOTE_AGENTS.csv

This document records the exploratory analysis that shaped the model
design decisions in Agents 2 and 3. It exists so a reviewer can verify
the findings independently rather than take the conclusions on faith.

## Dataset shape
- 146,259 rows, 25 raw columns
- `Policy_Bind` (target): 22.2% Yes / 77.8% No — matches the brief's
  "1 in 5 quotes converts" framing closely
- `Policy_Bind_DT` null exactly when `Policy_Bind == 'No'` (clean, no
  leakage confusion once excluded as a feature)

## Data quality findings
| Finding | Evidence | Action taken |
|---|---|---|
| `Driving_Exp` = `Driver_Age` − 17 for every row | `(Driver_Age - Driving_Exp).nunique() == 1` | Dropped `Driving_Exp` as redundant |
| `Quote_Num` is not unique | 41,966+ repeated values; 75,004 rows (51%) belong to a repeated `Quote_Num` | Derived `is_requote_derived` / `true_requote_count` directly from repeats |
| Raw `Re_Quote` flag doesn't align with repeated `Quote_Num` | Cross-tab shows no consistent correspondence | Kept raw flag as a feature but don't rely on it alone |
| `Q_Valid_DT − Q_Creation_DT` constant at 59 days for every row | `.describe()` shows zero variance | Replaced with `quote_age_days` (days since creation vs. snapshot date) |
| Per-agent bind-rate variance matches pure sampling noise | Observed std 0.0183 vs. expected binomial std 0.0188 at n≈489/agent | `Agent_Num` treated as non-informative; not used as a router feature |

## Conversion signal analysis (the key finding)

**Question:** does `Policy_Bind` correlate with any input feature, alone
or in combination?

**Method 1 — univariate.** Bind rate grouped by `Prev_Accidents`,
`Prev_Citations`, `Sal_Range`, `Coverage`, `Driver_Age` bucket,
`Quoted_Premium` quintile, `Agent_Type`, `Region`, `Re_Quote`. Every
group's bind rate falls within ~1 percentage point of the 22.2% base
rate.

**Method 2 — multivariate.** 300-tree Random Forest, class-balanced, all
20 raw features (encoded), 80/20 stratified split. **ROC-AUC = 0.4997**
(chance = 0.50).

**Method 3 — engineered features.** Added `premium_to_salary`,
`premium_to_vehcost`, `risk_composite`, `true_requote_count` and re-ran
Random Forest and HistGradientBoosting. **RF AUC = 0.5039, HistGBM AUC =
0.5015** — no improvement.

**Method 4 — mutual information.** `mutual_info_classif` across all 25
features (raw + engineered). Maximum score: 0.015 (noise floor; nothing
exceeds ~0.02).

**Conclusion:** `Policy_Bind` is statistically independent of every
available feature in this dataset. This is consistent with (though not
provable to be) the label having been assigned via an independent random
draw at generation time, separate from the other columns. This is
reported as a genuine finding rather than worked around.

## Premium analysis
Mean `Quoted_Premium` is ~$744 across every `Risk_Tier` × `Coverage`
combination (std ~$76) — premium also does not vary meaningfully by
risk tier in this dataset. Agent 3 is therefore built around
**peer-group anomaly detection** (is this premium unusual vs. similar
quotes) rather than a supervised "premium predicts non-conversion"
model, since the latter has no basis in the data either.

## Implication for the pipeline
- **Agent 2** outputs calibrated probabilities that correctly cluster
  near the base rate — this is the honest, well-calibrated behavior
  given the finding above, not a modeling failure.
- **Agent 4**'s routing logic leans more heavily on `Risk_Tier` and
  `Premium_Flag` than `Bind_Score` for differentiation, since bind score
  alone carries little discriminative information in this dataset.
- **Business recommendation** (a genuine finding worth surfacing to
  stakeholders): if this reflects real production data, current intake
  fields do not explain conversion — worth investigating additional
  behavioral signals (e.g., site engagement, quote abandonment point,
  time-of-day, competitor comparison behavior) not present in this
  extract.
