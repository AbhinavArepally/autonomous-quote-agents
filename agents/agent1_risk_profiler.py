"""
Agent 1 - Risk Profiler (FULLY AUTO)
=====================================
Computes a Low / Medium / High risk tier for every quote.

Design justification
---------------------
There is no ground-truth "Risk_Tier" label in the source data (it's a
derived construct, not an observed outcome like Policy_Bind), so this is
NOT a supervised classification problem. Fitting kNN / Naive Bayes / an
SVM here would require inventing labels first and then "predicting" them
back - circular, and misleading if presented as supervised ML.

Instead: a domain-informed composite risk score (weighted combination of
accident history, citation history, age-based inexperience proxy, and
high-mileage exposure) is computed, then K-Means clustering is used to
learn natural break points in that composite score and assign Low /
Medium / High tiers from the resulting cluster ordering. This keeps a
real ML component (unsupervised clustering) in the loop while being
honest that no labeled target exists.

Weights are domain-informed (industry underwriting rules of thumb: prior
accidents and citations are the strongest loss predictors, followed by
inexperience and high annual mileage) rather than fit to an outcome,
since there is nothing reliable to fit them to.
"""

import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RISK_WEIGHTS = {
    "Prev_Accidents": 3.0,   # strongest loss predictor
    "Prev_Citations": 2.0,
    "young_driver": 1.5,     # Driver_Age < 25 proxy for inexperience
    "high_mileage": 1.0,     # Annual_Miles_Range top bucket
    "commercial_use": 1.0,   # Veh_Usage commercial/rideshare-type use
}


def compute_composite_score(df: pd.DataFrame) -> pd.Series:
    young = (df["Driver_Age"] < 25).astype(int)
    high_miles = (df["miles_mid_k"] > 45).astype(int)
    commercial = df["Veh_Usage"].astype(str).str.contains(
        "Commercial|Business", case=False, na=False
    ).astype(int)

    score = (
        RISK_WEIGHTS["Prev_Accidents"] * df["Prev_Accidents"].astype(int)
        + RISK_WEIGHTS["Prev_Citations"] * df["Prev_Citations"].astype(int)
        + RISK_WEIGHTS["young_driver"] * young
        + RISK_WEIGHTS["high_mileage"] * high_miles
        + RISK_WEIGHTS["commercial_use"] * commercial
    )
    return score


class RiskProfilerAgent:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_to_tier = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        score = compute_composite_score(df).values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(score)
        clusters = self.kmeans.fit_predict(scaled)

        # order clusters by their mean composite score -> Low/Med/High
        order = (
            pd.Series(score.flatten())
            .groupby(clusters)
            .mean()
            .sort_values()
            .index.tolist()
        )
        tier_names = ["Low", "Medium", "High"][: self.n_clusters]
        self.cluster_to_tier = {cl: tier for cl, tier in zip(order, tier_names)}
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "Call .fit() first"
        score = compute_composite_score(df)
        scaled = self.scaler.transform(score.values.reshape(-1, 1))
        clusters = self.kmeans.predict(scaled)
        tiers = pd.Series(clusters).map(self.cluster_to_tier).values

        out = df.copy()
        out["risk_composite_score"] = score.values
        out["Risk_Tier"] = tiers
        return out

    def explain(self, row: pd.Series) -> str:
        """Chain-of-thought style explanation for a single quote."""
        reasons = []
        if row["Prev_Accidents"] > 0:
            reasons.append(f"{int(row['Prev_Accidents'])} prior accident(s)")
        if row["Prev_Citations"] > 0:
            reasons.append(f"{int(row['Prev_Citations'])} prior citation(s)")
        if row["Driver_Age"] < 25:
            reasons.append(f"young/inexperienced driver (age {int(row['Driver_Age'])})")
        if row.get("miles_mid_k", 0) > 45:
            reasons.append("high annual mileage")
        if str(row.get("Veh_Usage", "")).lower().find("commercial") != -1 or \
           str(row.get("Veh_Usage", "")).lower().find("business") != -1:
            reasons.append(f"commercial/business vehicle use ({row.get('Veh_Usage')})")
        if not reasons:
            reasons.append("no material risk flags in accident, citation, age, or mileage history")
        return (
            f"Risk tier '{row['Risk_Tier']}' (composite score "
            f"{row['risk_composite_score']:.1f}) driven by: {', '.join(reasons)}."
        )


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clean_quotes.pkl"))
    agent = RiskProfilerAgent().fit(df)
    scored = agent.predict(df)

    print("Risk tier distribution:")
    print(scored["Risk_Tier"].value_counts())
    print("\nMean composite score by tier:")
    print(scored.groupby("Risk_Tier")["risk_composite_score"].mean().sort_values())

    print("\nSample explanations:")
    for i in [0, 5, 20]:
        print(" -", agent.explain(scored.iloc[i]))

    scored.to_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agent1_output.pkl"))
    print("\nSaved agent1_output.pkl")
