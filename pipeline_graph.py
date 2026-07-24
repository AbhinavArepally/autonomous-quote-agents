"""
Multi-Agent Orchestration Layer
==================================
Wires Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 sequentially, passing a
shared structured state between nodes, with a conditional edge at the end
for the 3-way routing decision.

Uses LangGraph if available (pip install langgraph). This sandbox has no
internet access to install it, so a minimal drop-in StateGraph
implementation is used as a fallback - same node/edge/state-passing
mental model, so swapping to real LangGraph in an environment with
internet access is a ~2 line change (see USE_LANGGRAPH below).
"""

import os
import pandas as pd
from typing import TypedDict, Optional

from agents.agent1_risk_profiler import RiskProfilerAgent
from agents.agent2_conversion_predictor import ConversionPredictorAgent
from agents.agent3_premium_advisor import PremiumAdvisorAgent
from agents.agent4_decision_router import DecisionRouterAgent

try:
    from langgraph.graph import StateGraph, END
    USE_LANGGRAPH = True
except ImportError:
    USE_LANGGRAPH = False


class QuoteState(TypedDict, total=False):
    df: pd.DataFrame            # working dataframe, enriched at each node
    stage: str
    diagnostics: dict


# ---------------------------------------------------------------------
# Fallback minimal graph runner (used when langgraph isn't installed)
# ---------------------------------------------------------------------
class _MinimalStateGraph:
    """Sequential graph runner mirroring LangGraph's node/edge API just
    enough for this pipeline. Not a general replacement for LangGraph."""

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, a, b):
        self.edges.append((a, b))

    def compile(self):
        order = [self.entry]
        current = self.entry
        edge_map = dict(self.edges)
        while current in edge_map and edge_map[current] != "__end__":
            current = edge_map[current]
            order.append(current)

        def run(state):
            for node_name in order:
                state = self.nodes[node_name](state)
            return state

        return run


class QuoteAgentPipeline:
    def __init__(self):
        self.risk_agent = RiskProfilerAgent()
        self.conversion_agent = ConversionPredictorAgent()
        self.premium_agent = PremiumAdvisorAgent()
        self.router_agent = DecisionRouterAgent()
        self.graph = None

    # ---- node functions, each consumes/produces QuoteState ----
    def _node_risk_profiler(self, state: QuoteState) -> QuoteState:
        df = state["df"]
        if not self.risk_agent.fitted:
            self.risk_agent.fit(df)
        state["df"] = self.risk_agent.predict(df)
        state["stage"] = "risk_profiled"
        return state

    def _node_conversion_predictor(self, state: QuoteState) -> QuoteState:
        df = state["df"]
        y = (df["Policy_Bind"] == "Yes").astype(int)
        if self.conversion_agent.model is None:
            self.conversion_agent.fit(df, y)
        state["df"] = self.conversion_agent.predict(df)
        state["diagnostics"] = state.get("diagnostics", {})
        state["diagnostics"]["conversion"] = self.conversion_agent.diagnostics
        state["stage"] = "conversion_scored"
        return state

    def _node_premium_advisor(self, state: QuoteState) -> QuoteState:
        df = state["df"]
        if self.premium_agent.peer_stats is None:
            self.premium_agent.fit(df)
        state["df"] = self.premium_agent.predict(df)
        state["stage"] = "premium_flagged"
        return state

    def _node_decision_router(self, state: QuoteState) -> QuoteState:
        df = state["df"]
        if self.router_agent.nn_index is None:
            self.router_agent.fit_retrieval_index(df)
        df = df.copy()
        df["Decision"] = df.apply(self.router_agent.route, axis=1)
        df["Route_Reason"] = df.apply(self.router_agent.route_reason, axis=1)
        state["df"] = df
        state["stage"] = "routed"
        return state

    def build(self):
        if USE_LANGGRAPH:
            g = StateGraph(QuoteState)
            g.add_node("risk_profiler", self._node_risk_profiler)
            g.add_node("conversion_predictor", self._node_conversion_predictor)
            g.add_node("premium_advisor", self._node_premium_advisor)
            g.add_node("decision_router", self._node_decision_router)
            g.set_entry_point("risk_profiler")
            g.add_edge("risk_profiler", "conversion_predictor")
            g.add_edge("conversion_predictor", "premium_advisor")
            g.add_edge("premium_advisor", "decision_router")
            g.add_edge("decision_router", END)
            self.graph = g.compile()
        else:
            g = _MinimalStateGraph()
            g.add_node("risk_profiler", self._node_risk_profiler)
            g.add_node("conversion_predictor", self._node_conversion_predictor)
            g.add_node("premium_advisor", self._node_premium_advisor)
            g.add_node("decision_router", self._node_decision_router)
            g.set_entry_point("risk_profiler")
            g.add_edge("risk_profiler", "conversion_predictor")
            g.add_edge("conversion_predictor", "premium_advisor")
            g.add_edge("premium_advisor", "decision_router")
            g.add_edge("decision_router", "__end__")
            self.graph = g.compile()
        return self

    def run(self, df: pd.DataFrame) -> QuoteState:
        assert self.graph is not None, "Call .build() first"
        initial_state: QuoteState = {"df": df, "stage": "start", "diagnostics": {}}
        return self.graph(initial_state) if not USE_LANGGRAPH else self.graph.invoke(initial_state)


if __name__ == "__main__":
    print(f"Using {'LangGraph' if USE_LANGGRAPH else 'fallback minimal graph (langgraph not installed in this sandbox)'}")

    raw = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "clean_quotes.pkl"))
    pipeline = QuoteAgentPipeline().build()
    result = pipeline.run(raw)

    out_df = result["df"]
    print("\nFinal pipeline output shape:", out_df.shape)
    print("\nDecision distribution:")
    print(out_df["Decision"].value_counts())
    print("\nStage reached:", result["stage"])

    out_df.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pipeline_final_output.pkl"))
    print("\nSaved pipeline_final_output.pkl")
