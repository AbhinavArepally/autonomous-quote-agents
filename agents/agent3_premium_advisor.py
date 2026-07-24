"""
Agent 3 - Premium Advisor (HYBRID)
=====================================
For high-propensity unconverted quotes, reasons whether Quoted_Premium is
plausibly the conversion blocker and recommends an adjusted band.

Design justification
---------------------
Quoted_Premium showed no correlation with Policy_Bind in diagnostics
(same finding as Agent 2), so this cannot honestly be framed as "ML
predicts premium is the blocker" - there's no learnable relationship to
predict from. Also notable: mean premium is ~$744 across every Risk_Tier
x Coverage combination (std ~$76) - premium barely varies by risk in this
dataset either, which is itself worth flagging to the business.

Instead this agent is a genuine hybrid:
  - ML/statistical component: peer-group z-score. For each quote, compare
    Quoted_Premium against quotes with the same Risk_Tier + Coverage +
    Region (its true peer group), flag if it's a statistical outlier
    (|z| > 1.5).
  - LLM reasoning component: given the peer comparison, risk tier, salary
    range and vehicle cost range, the LLM reasons in natural language
    about whether the premium is a *plausible* blocker (e.g. an outlier
    premium for a Low-risk driver is a stronger blocker story than an
    outlier premium for a High-risk driver) and proposes an adjusted band
    or alternative coverage tier.

This keeps a genuine statistical anomaly-detection core (the honest thing
this data supports) while still satisfying the brief's "reasons whether
Quoted_Premium is the conversion blocker" requirement.
"""

import numpy as np
import os
import pandas as pd

Z_THRESHOLD = 1.5


class PremiumAdvisorAgent:
    def __init__(self, z_threshold: float = Z_THRESHOLD):
        self.z_threshold = z_threshold
        self.peer_stats = None

    def fit(self, df: pd.DataFrame):
        self.peer_stats = (
            df.groupby(["Risk_Tier", "Coverage", "Region"])["Quoted_Premium"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "peer_mean", "std": "peer_std", "count": "peer_n"})
        )
        # guard against zero/NaN std for tiny peer groups
        self.peer_stats["peer_std"] = self.peer_stats["peer_std"].replace(0, np.nan)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(
            self.peer_stats, on=["Risk_Tier", "Coverage", "Region"], how="left"
        )
        out["premium_z"] = (
            (out["Quoted_Premium"] - out["peer_mean"]) / out["peer_std"]
        ).fillna(0)
        out["Premium_Flag"] = np.where(
            out["premium_z"] > self.z_threshold, "Overpriced_vs_peers",
            np.where(out["premium_z"] < -self.z_threshold, "Underpriced_vs_peers", "In_line"),
        )
        return out

    def recommend_band(self, row: pd.Series) -> dict:
        """Statistical component: suggested adjusted band from peer stats."""
        if row["Premium_Flag"] == "Overpriced_vs_peers":
            target = row["peer_mean"] + 0.5 * row["peer_std"]
            return {"action": "lower_premium", "suggested_band": round(target, 2)}
        elif row["Premium_Flag"] == "Underpriced_vs_peers":
            return {"action": "review_pricing_adequacy", "suggested_band": round(row["peer_mean"], 2)}
        return {"action": "no_change", "suggested_band": round(row["Quoted_Premium"], 2)}

    def build_llm_prompt(self, row: pd.Series) -> str:
        """
        Prompt for the LLM reasoning layer. In production this is sent to
        the Claude API (see call_llm_reasoning below); kept as a separate
        method so it's independently testable/inspectable.
        """
        return (
            f"A quote has the following profile:\n"
            f"- Risk tier: {row['Risk_Tier']}\n"
            f"- Coverage: {row['Coverage']}\n"
            f"- Region: {row['Region']}\n"
            f"- Quoted premium: ₹{row['Quoted_Premium']:.2f}\n"
            f"- Peer group average premium (same risk tier, coverage, region): "
            f"₹{row['peer_mean']:.2f} (n={int(row['peer_n'])})\n"
            f"- Premium z-score vs peers: {row['premium_z']:.2f}\n"
            f"- Salary range: {row['Sal_Range']}, Vehicle cost range: {row['Vehicl_Cost_Range']}\n"
            f"- Bind score (from Conversion Predictor): {row.get('Bind_Score', 'n/a')}%\n\n"
            f"In 2-3 sentences: is the quoted premium a plausible reason this "
            f"quote hasn't converted? Recommend either an adjusted premium "
            f"band or an alternative coverage tier, and justify briefly."
        )

    def explain(self, row: pd.Series) -> str:
        """Template fallback explanation (used when no LLM call is made)."""
        rec = self.recommend_band(row)
        return (
            f"Premium ₹{row['Quoted_Premium']:.2f} vs. peer average "
            f"₹{row['peer_mean']:.2f} (z={row['premium_z']:.2f}) among "
            f"{int(row['peer_n'])} similar quotes ({row['Risk_Tier']} risk, "
            f"{row['Coverage']} coverage, {row['Region']}). "
            f"Flag: {row['Premium_Flag']}. Recommendation: {rec['action']} "
            f"toward ~₹{rec['suggested_band']:.2f}."
        )


def call_llm_reasoning(prompt: str) -> str:
    """
    Calls Claude for the natural-language reasoning step. Requires network
    + ANTHROPIC_API_KEY - not available in this offline sandbox, so this
    function is provided for completeness but falls back to a note when
    unreachable. In your environment with API access, this becomes a real
    api.anthropic.com/v1/messages call (model: claude-sonnet-4-6 or later).
    """
    try:
        import anthropic  # noqa
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as e:
        return f"[LLM reasoning unavailable in this environment: {e}]"


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agent2_output.pkl"))
    agent = PremiumAdvisorAgent().fit(df)
    scored = agent.predict(df)

    print("Premium_Flag distribution:")
    print(scored["Premium_Flag"].value_counts())

    print("\nSample explanation:")
    row = scored.iloc[0]
    print(" -", agent.explain(row))
    print("\nSample LLM prompt (for the flagged case below):")
    flagged = scored[scored["Premium_Flag"] != "In_line"].iloc[0]
    print(agent.build_llm_prompt(flagged))

    scored.to_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agent3_output.pkl"))
    print("\nSaved agent3_output.pkl")
