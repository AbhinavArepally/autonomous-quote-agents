"""
Train once, save to disk.
==========================
Fits all 4 agents on the full cleaned dataset and saves everything needed
to score EITHER the full batch (dashboard) OR a single new quote (the
prediction form) without retraining: the 4 fitted agent objects plus the
preprocessing artifacts (snapshot date, outlier bounds).

Run this once (or whenever the underlying data changes):
    python train_and_save.py

Output: models/trained_pipeline.joblib
"""

import os
import time
import joblib
import pandas as pd

from preprocessing import load_raw, clean_and_engineer, PreprocessingArtifacts
from agents.agent1_risk_profiler import RiskProfilerAgent
from agents.agent2_conversion_predictor import ConversionPredictorAgent
from agents.agent3_premium_advisor import PremiumAdvisorAgent
from agents.agent4_decision_router import DecisionRouterAgent

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "Autonomous_QUOTE_AGENTS.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "trained_pipeline.joblib")


def train_all(csv_path: str = CSV_PATH, verbose: bool = True):
    t0 = time.time()
    if verbose:
        print("Loading & cleaning data...")
    raw = load_raw(csv_path)
    clean_df, report, artifacts = clean_and_engineer(raw)

    if verbose:
        print(f"  {clean_df.shape[0]:,} rows ready ({time.time()-t0:.1f}s)")
        print("Fitting Agent 1 - Risk Profiler...")
    risk_agent = RiskProfilerAgent().fit(clean_df)
    df1 = risk_agent.predict(clean_df)

    if verbose:
        print("Fitting Agent 2 - Conversion Predictor (this is the slow one)...")
    y = (df1["Policy_Bind"] == "Yes").astype(int)
    conversion_agent = ConversionPredictorAgent().fit(df1, y)
    df2 = conversion_agent.predict(df1)

    if verbose:
        print("Fitting Agent 3 - Premium Advisor...")
    premium_agent = PremiumAdvisorAgent().fit(df2)
    df3 = premium_agent.predict(df2)

    if verbose:
        print("Fitting Agent 4 - Decision Router (+ RAG index)...")
    router_agent = DecisionRouterAgent().fit_retrieval_index(df3)
    df3["Decision"] = df3.apply(router_agent.route, axis=1)
    df3["Route_Reason"] = df3.apply(router_agent.route_reason, axis=1)

    bundle = {
        "preprocessing_artifacts": artifacts,
        "risk_agent": risk_agent,
        "conversion_agent": conversion_agent,
        "premium_agent": premium_agent,
        "router_agent": router_agent,
        "data_quality_report": report.to_dict(),
        "conversion_diagnostics": conversion_agent.diagnostics,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    if verbose:
        print(f"\nSaved trained pipeline -> {MODEL_PATH} ({time.time()-t0:.1f}s total)")

    return bundle, df3


def load_trained_pipeline(model_path: str = MODEL_PATH) -> dict:
    """Load the saved bundle (used by both the dashboard and the single-quote
    prediction form so neither has to retrain)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python train_and_save.py` first."
        )
    return joblib.load(model_path)


def predict_single_quote(raw_fields: dict, bundle: dict = None) -> dict:
    """
    Score ONE new quote through all 4 loaded (already-trained) agents.
    Used by the dashboard's 'Predict New Quote' form.
    """
    from preprocessing import transform_new_quote

    if bundle is None:
        bundle = load_trained_pipeline()

    row_df = transform_new_quote(raw_fields, bundle["preprocessing_artifacts"])

    row_df = bundle["risk_agent"].predict(row_df)
    row_df = bundle["conversion_agent"].predict(row_df)
    row_df = bundle["premium_agent"].predict(row_df)

    router = bundle["router_agent"]
    row = row_df.iloc[0]
    decision = router.route(row)
    reason = router.route_reason(pd.Series({**row.to_dict(), "Decision": decision}))

    risk_explain = bundle["risk_agent"].explain(row)
    conv_explain = bundle["conversion_agent"].explain(row)
    prem_explain = bundle["premium_agent"].explain(row)

    similar_quotes = None
    if decision == "Escalate_to_Underwriter":
        similar_quotes = router.retrieve_similar_for_new(row)

    return {
        "Risk_Tier": row["Risk_Tier"],
        "risk_composite_score": float(row["risk_composite_score"]),
        "Bind_Score": float(row["Bind_Score"]),
        "Premium_Flag": row["Premium_Flag"],
        "Decision": decision,
        "Route_Reason": reason,
        "similar_quotes": similar_quotes,
        "explanations": {
            "risk": risk_explain,
            "conversion": conv_explain,
            "premium": prem_explain,
        },
    }


if __name__ == "__main__":
    bundle, scored_df = train_all()
    scored_df.to_pickle(os.path.join(PROJECT_ROOT, "data", "pipeline_final_output.pkl"))
    print("\nFinal decision distribution:")
    print(scored_df["Decision"].value_counts())

    print("\nSanity check - scoring a single new quote:")
    sample = {
        "Agent_Type": "EA", "Region": "A", "Policy_Type": "Standard",
        "HH_Vehicles": 2, "HH_Drivers": 2, "Driver_Age": 34,
        "Prev_Accidents": 0, "Prev_Citations": 1, "Gender": "F",
        "Marital_Status": "Married", "Education": "Bachelors",
        "Sal_Range": "> $ 60 K <= $ 90 K", "Coverage": "Balanced",
        "Veh_Usage": "Pleasure", "Annual_Miles_Range": "> 7.5 K & <= 15 K",
        "Vehicl_Cost_Range": "> $ 20 K <= $ 30 K", "Re_Quote": "No",
        "Quoted_Premium": 745.0,
    }
    result = predict_single_quote(sample, bundle)
    for k, v in result.items():
        if k != "explanations":
            print(f"  {k}: {v}")
    print("  explanations:")
    for k, v in result["explanations"].items():
        print(f"    {k}: {v}")
