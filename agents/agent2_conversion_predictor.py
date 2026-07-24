"""
Agent 2 - Conversion Predictor (FULLY AUTO)
=============================================
Scores each unbound quote with a calibrated bind probability (0-100%).

Honest diagnostic finding (see /docs/data_diagnostics.md for full evidence)
----------------------------------------------------------------------------
Exhaustive testing (marginal bind-rate checks across every feature,
a 500-tree class-weighted Random Forest on all raw features, engineered
interaction/ratio features, HistGradientBoosting, and mutual-information
scoring) all converge on the same result: Policy_Bind is statistically
independent of every available feature in this dataset (best ROC-AUC
achieved: ~0.50-0.51, i.e. no better than the 22.2% base rate; maximum
mutual information score across 25 features: 0.015, i.e. noise-floor).

Given that, this agent is built to do the honest thing a production
system should do when the inputs genuinely don't separate the classes:
  1. Try multiple model families (as the brief requires: a linear model,
     an SVM, and a tree ensemble), each explicitly justified.
  2. Handle the 22% class imbalance properly (class_weight='balanced'),
     even though imbalance handling can't manufacture signal that isn't
     there - it only prevents the model from being *worse* than the base
     rate by trivially predicting the majority class.
  3. Calibrate probabilities (Platt scaling) so the output is a
     trustworthy probability, not a false-precision score.
  4. Evaluate on PR-AUC, Brier score and a calibration curve - never
     accuracy, which is meaningless at this imbalance and would look
     artificially high just by predicting "No" every time.
  5. Report the diagnostic honestly as part of the explanation surfaced
     to the underwriter, rather than hiding it behind a confident-looking
     percentage.

This is presented as a legitimate finding, not a bug: it means, per this
dataset, conversion is not explained by risk profile, timing, coverage,
salary, or re-quote behavior alone - a genuinely useful insight for the
business (see write-up), even though it makes for a less flashy demo.
"""

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve,
)

FEATURE_COLS = [
    "Risk_Tier", "quote_age_days", "Coverage", "Agent_Type", "Region",
    "Sal_Range", "true_requote_count", "is_requote_derived",
]


class ConversionPredictorAgent:
    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.scaler = StandardScaler()
        self.cat_cols = ["Risk_Tier", "Coverage", "Agent_Type", "Region", "Sal_Range"]
        self.num_cols = ["quote_age_days", "true_requote_count", "is_requote_derived"]
        self.model = None
        self.diagnostics = {}

    def _prep(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_cat = df[self.cat_cols].astype(str)
        X_num = df[self.num_cols].astype(float).fillna(0)
        if fit:
            X_cat_enc = self.encoder.fit_transform(X_cat)
            X_num_scaled = self.scaler.fit_transform(X_num)
        else:
            X_cat_enc = self.encoder.transform(X_cat)
            X_num_scaled = self.scaler.transform(X_num)
        return np.hstack([X_cat_enc, X_num_scaled])

    def fit(self, df: pd.DataFrame, y: pd.Series):
        X = self._prep(df, fit=True)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Note: RBF-kernel SVC does not scale to 146K rows (O(n^2)-O(n^3));
        # LinearSVC is the practical choice for an SVM comparison at this
        # volume, wrapped in CalibratedClassifierCV since LinearSVC has no
        # native predict_proba.
        candidates = {
            "logistic": LogisticRegression(class_weight="balanced", max_iter=1000),
            "svm_linear": LinearSVC(class_weight="balanced", max_iter=2000, dual="auto"),
            "random_forest": RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
        }

        results = {}
        for name, base_model in candidates.items():
            calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
            calibrated.fit(Xtr, ytr)
            proba = calibrated.predict_proba(Xte)[:, 1]
            results[name] = {
                "roc_auc": roc_auc_score(yte, proba),
                "pr_auc": average_precision_score(yte, proba),
                "brier": brier_score_loss(yte, proba),
                "model": calibrated,
            }

        self.diagnostics["comparison"] = {
            k: {m: v[m] for m in ["roc_auc", "pr_auc", "brier"]}
            for k, v in results.items()
        }
        self.diagnostics["base_rate"] = float(y.mean())

        best_name = min(results, key=lambda k: results[k]["brier"])
        self.model = results[best_name]["model"]
        self.model_type = best_name
        self.diagnostics["selected_model"] = best_name
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prep(df, fit=False)
        proba = self.model.predict_proba(X)[:, 1]
        out = df.copy()
        out["Bind_Score"] = (proba * 100).round(1)
        return out

    def explain(self, row: pd.Series) -> str:
        base = self.diagnostics.get("base_rate", 0.222) * 100
        delta = row["Bind_Score"] - base
        direction = "above" if delta > 0 else ("below" if delta < 0 else "at")
        return (
            f"Bind score {row['Bind_Score']:.1f}% is {abs(delta):.1f} pts "
            f"{direction} the {base:.1f}% dataset base rate. Diagnostic "
            f"testing found no strong predictive feature for conversion in "
            f"this dataset (best model ROC-AUC ~{self.diagnostics.get('comparison', {}).get(self.model_type, {}).get('roc_auc', 0):.2f}), "
            f"so scores cluster near the base rate by design rather than "
            f"reflecting a confident individual prediction."
        )


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agent1_output.pkl"))
    y = (df["Policy_Bind"] == "Yes").astype(int)

    agent = ConversionPredictorAgent().fit(df, y)

    print("Model comparison (calibrated, test set):")
    for name, metrics in agent.diagnostics["comparison"].items():
        print(f"  {name:15s} ROC-AUC={metrics['roc_auc']:.3f}  "
              f"PR-AUC={metrics['pr_auc']:.3f}  Brier={metrics['brier']:.4f}")
    print(f"\nSelected model: {agent.diagnostics['selected_model']}")
    print(f"Base rate: {agent.diagnostics['base_rate']:.3f}")

    scored = agent.predict(df)
    print("\nBind_Score distribution:")
    print(scored["Bind_Score"].describe())

    print("\nSample explanation:")
    print(" -", agent.explain(scored.iloc[0]))

    scored.to_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agent2_output.pkl"))
    print("\nSaved agent2_output.pkl")
