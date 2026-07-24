"""
Agent 4 - Decision Router (ESCALATE ONLY)
=============================================
Combines all upstream agent outputs into: Auto-Approve / Agent Follow-Up /
Escalate to Underwriter. Only the final Escalate bucket reaches a human.

Design justification
---------------------
Given the Agent 2 finding (Bind_Score barely moves, clustering ~22.2%),
routing leans more heavily on Risk_Tier and Premium_Flag for
differentiation than Bind_Score, which is documented explicitly below
rather than left as an unexplained implementation detail.

Escalation thresholds (documented, tunable per region/agent type in
the bonus section):
  - Escalate to Underwriter: Risk_Tier == 'High' (regardless of bind
    score - high loss risk always needs human underwriting judgment),
    OR Bind_Score is in the "genuinely uncertain" band (45-55%, though
    given the diagnostic finding this band is rarely hit in practice -
    documented rather than silently never firing),
    OR conflicting signal: Low risk but Overpriced_vs_peers AND high
    relative bind score (a "should have converted, priced out" pattern
    worth a human's attention).
  - Agent Follow-Up: Medium risk, or a premium anomaly on an otherwise
    Low/Medium risk quote (a human agent can plausibly fix this with
    outreach or a discount, doesn't need underwriting judgment).
  - Auto-Approve: Low risk, In_line or Underpriced premium, no
    conflicting signals - safe to leave automated.

RAG component
--------------
For every Escalate case, retrieves the k most similar historical quotes
(structured similarity over risk/coverage/region/premium features, via
sklearn NearestNeighbors - this is "structured RAG": retrieval grounded
in feature-space similarity rather than free-text embeddings, appropriate
since quotes are structured records, not documents) and summarizes their
outcomes to ground the handoff summary an underwriter reads.
"""

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

BIND_UNCERTAIN_LOW = 45.0
BIND_UNCERTAIN_HIGH = 55.0


class DecisionRouterAgent:
    def __init__(self, k_similar: int = 5):
        self.k_similar = k_similar
        self.nn_index = None
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.scaler = StandardScaler()
        self.retrieval_cols_cat = ["Risk_Tier", "Coverage", "Region"]
        self.retrieval_cols_num = ["Quoted_Premium", "Bind_Score"]
        self.reference_df = None

    def fit_retrieval_index(self, df: pd.DataFrame):
        """Build the structured-RAG retrieval index over historical quotes."""
        self.reference_df = df.reset_index(drop=True)
        X_cat = self.encoder.fit_transform(df[self.retrieval_cols_cat].astype(str))
        X_num = self.scaler.fit_transform(df[self.retrieval_cols_num].astype(float))
        X = np.hstack([X_cat, X_num])
        self.nn_index = NearestNeighbors(n_neighbors=self.k_similar + 1).fit(X)
        self._X = X
        return self

    def retrieve_similar(self, row_idx: int) -> pd.DataFrame:
        dists, idxs = self.nn_index.kneighbors(self._X[row_idx : row_idx + 1])
        # drop self-match (distance 0 at position 0)
        similar_idx = [i for i in idxs[0] if i != row_idx][: self.k_similar]
        return self.reference_df.iloc[similar_idx]

    def retrieve_similar_for_new(self, row: pd.Series) -> pd.DataFrame:
        """Same retrieval, but for a brand-new quote that was never part of
        the fitted index (e.g. a live prediction from the UI form) - builds
        its feature vector on the fly instead of looking it up by position."""
        row_df = pd.DataFrame([row[self.retrieval_cols_cat + self.retrieval_cols_num]])
        X_cat = self.encoder.transform(row_df[self.retrieval_cols_cat].astype(str))
        X_num = self.scaler.transform(row_df[self.retrieval_cols_num].astype(float))
        X_new = np.hstack([X_cat, X_num])
        dists, idxs = self.nn_index.kneighbors(X_new, n_neighbors=self.k_similar)
        return self.reference_df.iloc[idxs[0]]

    def route(self, row: pd.Series) -> str:
        risk = row["Risk_Tier"]
        bind = row["Bind_Score"]
        flag = row["Premium_Flag"]

        uncertain_bind = BIND_UNCERTAIN_LOW <= bind <= BIND_UNCERTAIN_HIGH
        priced_out_pattern = (risk == "Low") and (flag == "Overpriced_vs_peers") and (bind > 25)

        if risk == "High" or uncertain_bind or priced_out_pattern:
            return "Escalate_to_Underwriter"
        if risk == "Medium" or flag != "In_line":
            return "Agent_Follow_Up"
        return "Auto_Approve"

    def route_reason(self, row: pd.Series) -> str:
        risk, bind, flag = row["Risk_Tier"], row["Bind_Score"], row["Premium_Flag"]
        decision = row["Decision"]
        if decision == "Escalate_to_Underwriter":
            if risk == "High":
                return "High risk tier requires underwriter judgment regardless of bind score."
            if BIND_UNCERTAIN_LOW <= bind <= BIND_UNCERTAIN_HIGH:
                return f"Bind score {bind:.1f}% falls in the genuinely-uncertain 45-55% band."
            return (
                "Low risk with an overpriced-vs-peers premium and above-baseline "
                "bind score - a 'should have converted, likely priced out' pattern "
                "worth underwriter review."
            )
        if decision == "Agent_Follow_Up":
            if risk == "Medium":
                return "Medium risk tier - agent follow-up recommended, not full escalation."
            return f"Premium flagged ({flag}) but risk is manageable - agent can address via outreach."
        return "Low risk, in-line pricing, no conflicting signals - safe for automated approval."

    def build_escalation_summary(self, row: pd.Series) -> str:
        similar = self.retrieve_similar(row.name)
        bind_rate = (similar["Policy_Bind"] == "Yes").mean() if "Policy_Bind" in similar else None
        avg_premium = similar["Quoted_Premium"].mean()

        summary = (
            f"ESCALATION SUMMARY - Quote {row.get('Quote_Num', row.name)}\n"
            f"Risk Tier: {row['Risk_Tier']} | Bind Score: {row['Bind_Score']:.1f}% | "
            f"Premium Flag: {row['Premium_Flag']}\n"
            f"Routing reason: {row.get('Route_Reason', self.route_reason(row))}\n"
            f"Similar past quotes (n={len(similar)}, structured retrieval on "
            f"risk/coverage/region/premium): {bind_rate:.0%} bound historically, "
            f"average premium ${avg_premium:.2f}.\n"
            f"Recommendation for underwriter: review pricing adequacy and risk "
            f"classification consistency against these comparable cases."
        )
        return summary


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agent3_output.pkl"))

    router = DecisionRouterAgent().fit_retrieval_index(df)
    df["Decision"] = df.apply(router.route, axis=1)
    df["Route_Reason"] = df.apply(router.route_reason, axis=1)

    print("Decision distribution:")
    print(df["Decision"].value_counts())
    print("\nDecision distribution (%):")
    print((df["Decision"].value_counts(normalize=True) * 100).round(1))

    escalated = df[df["Decision"] == "Escalate_to_Underwriter"]
    print(f"\n{len(escalated)} quotes escalated ({len(escalated)/len(df):.1%})")

    print("\nSample escalation summary:")
    print(router.build_escalation_summary(escalated.iloc[0]))

    df.to_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agent4_output.pkl"))
    print("\nSaved agent4_output.pkl")
