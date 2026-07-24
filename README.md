# Autonomous Quote Agents — Multi-Agent Insurance Quote Pipeline

A 4-agent pipeline that autonomously processes auto insurance quotes:
risk profiling, conversion prediction, premium advising, and decision
routing — escalating to a human only when confidence genuinely demands it.

## Quick start
```bash
pip install -r requirements.txt
python train_and_save.py          # trains all 4 agents once, saves models/trained_pipeline.joblib
streamlit run dashboard/app.py    # dashboard loads the saved model - no retraining
```
(The dashboard also offers a "Train pipeline now" button on first launch if you skip the manual step above.)

Individual agent scripts still work standalone for testing/inspection:
```bash
python preprocessing.py
python agents/agent1_risk_profiler.py
python agents/agent2_conversion_predictor.py
python agents/agent3_premium_advisor.py
python agents/agent4_decision_router.py
python pipeline_graph.py            # full LangGraph-orchestrated batch run
```

## Architecture
```
CSV (146K rows)
   │
   ▼
preprocessing.py — cleaning, dedup, feature engineering
   │
   ▼
agents/agent1_risk_profiler.py       — composite score + K-Means → Risk_Tier
   ▼
agents/agent2_conversion_predictor.py — calibrated LR/SVM/RF → Bind_Score
   ▼
agents/agent3_premium_advisor.py     — peer z-score + LLM reasoning → Premium_Flag
   ▼
agents/agent4_decision_router.py     — rule fusion + RAG retrieval → Decision
   │
   ▼
pipeline_graph.py (LangGraph orchestration)
   │
   ▼
dashboard/app.py (Streamlit)
```

## Key design decisions (read before reviewing the code)
1. **The conversion signal was tested, not assumed.** `docs/data_diagnostics.md`
   documents an exhaustive check (univariate → multivariate RF → engineered
   features → mutual information) showing `Policy_Bind` is statistically
   independent of every available feature (best ROC-AUC ≈ 0.50). Agent 2 is
   built to handle this honestly — calibrated probabilities near the base
   rate, not an inflated accuracy number.
2. **Agent 1 has no ground-truth label** — `Risk_Tier` doesn't exist in the
   raw data, so it's built as an unsupervised composite score + clustering,
   not a supervised classifier trained on nothing.
3. **Agent 3 reframes "premium as blocker"** as peer-group anomaly
   detection, since `Quoted_Premium` also shows no bind correlation.
4. **RAG is structured, not text-based** — quotes are structured records,
   so Agent 4 retrieves similar historical quotes via feature-space nearest
   neighbors rather than text-embedding search.
5. **LangGraph, XGBoost, SHAP, LightGBM are used if installed**, with
   documented scikit-learn fallbacks (this sandbox has no internet access
   to install them — see comments in `pipeline_graph.py` and each agent
   file for exactly where the swap happens).
6. **Models are trained once and persisted** (`train_and_save.py` →
   `models/trained_pipeline.joblib` via joblib), not retrained on every
   dashboard launch. This also enables the "Predict New Quote" page,
   which scores a single new record in <100ms against the already-fitted
   agents rather than retraining. `preprocessing.py` is split into a fit
   step (learns snapshot date, outlier bounds from the training set) and
   a `transform_new_quote()` step that applies those saved stats to one
   new row, since recomputing an IQR from n=1 would be meaningless.

## Mapping to role requirements
| Requirement | Where it's demonstrated |
|---|---|
| Predictive modeling | Agents 1–3 |
| GenAI / LLM reasoning | Agent 3 premium reasoning, Agent 4 routing rationale + RAG summaries |
| RAG | Agent 4 `retrieve_similar` (structured nearest-neighbor retrieval) |
| ML technique breadth (kNN/Naive Bayes/SVM/Decision Forests) | Agent 2 model comparison (Logistic Regression, Linear SVM, Random Forest); Agent 1 K-Means |
| Data cleaning, dedup | `preprocessing.py` |
| Explaining findings to business teams clearly | `docs/data_diagnostics.md`, agent `.explain()` methods, dashboard diagnostics tab |
| Handling ambiguous requirements / managing risk | Full escalation-threshold design writeup in `agents/agent4_decision_router.py` docstring |

## Known limitations (documented, not hidden)
- Dashboard code is written against the standard Streamlit API but
  untested visually (no `streamlit` package in the build sandbox — no
  internet access to install it). Syntax-verified via `py_compile`.
- LangGraph/XGBoost/SHAP/LightGBM likewise untested directly in this
  sandbox for the same reason; fallback implementations were used for
  in-sandbox testing and are documented inline.
