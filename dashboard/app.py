"""
Agent Operations Dashboard (Streamlit)
=========================================
Run with:  streamlit run dashboard/app.py

Loads a pre-trained pipeline (models/trained_pipeline.joblib - run
`python train_and_save.py` once first) rather than retraining on every
launch. If no trained model is found, offers to train it from this page.

Theming: colors come from TWO layers -
  1. .streamlit/config.toml - Streamlit's official theme engine, colors
     every native widget (buttons, sliders, active radio state, etc.)
  2. The CSS block below - adds gradient sidebar, custom KPI/explain
     cards, badges and animation on top of the themed base.
"""

import sys
import os

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_and_save import load_trained_pipeline, train_all, predict_single_quote

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pipeline_final_output.pkl")

st.set_page_config(page_title="Autonomous Quote Agents", layout="wide", page_icon="🚦")

# ---------------------------------------------------------------------
# Color system (vibrant, light - no black/dark backgrounds anywhere)
# ---------------------------------------------------------------------
VIOLET = "#7F5AF0"
VIOLET_DARK = "#5433FF"
TEAL = "#00C9A7"
SKY = "#00B4D8"
AMBER = "#FFB100"
CORAL = "#FF6B6B"
INK = "#23223A"
INK_SOFT = "#6B6B85"

DECISION_COLORS = {"Auto_Approve": TEAL, "Agent_Follow_Up": SKY, "Escalate_to_Underwriter": CORAL}
RISK_COLORS = {"Low": TEAL, "Medium": AMBER, "High": CORAL}
CHART_PALETTE = [VIOLET, TEAL, CORAL, AMBER, SKY]

# ---------------------------------------------------------------------
# CSS layer
# ---------------------------------------------------------------------
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding-top: 1.6rem; max-width: 1250px;}}
    html, body, [class*="css"] {{font-family: 'Inter', sans-serif;}}
    h1, h2, h3 {{font-family: 'Sora', sans-serif !important; font-weight: 800 !important;}}

    /* ---- Sidebar: vivid gradient, not dark ---- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(165deg, #8B6CF0 0%, #6C4CE0 45%, #4E32C9 100%);
        border-right: none;
    }}
    section[data-testid="stSidebar"] * {{color: white !important;}}
    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {{opacity: 0.75;}}

    /* radio group -> nav pills */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 6px;
        transition: background 0.15s ease;
        width: 100%;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: rgba(255,255,255,0.18);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background: rgba(255,255,255,0.95);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {{
        color: {VIOLET_DARK} !important;
        font-weight: 700 !important;
    }}

    /* ---- KPI / explain cards (custom HTML) ---- */
    .kpi-card {{
        background: white; border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 8px 22px rgba(84,51,255,0.08);
        border-top: 4px solid var(--accent, {VIOLET});
        transition: transform 0.15s ease;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-label {{font-size: 12.5px; color: {INK_SOFT}; font-weight: 600; margin-bottom: 6px;}}
    .kpi-value {{font-size: 26px; font-weight: 800; color: {INK};}}
    .kpi-delta {{font-size: 12px; font-weight: 700; margin-top: 3px; color: var(--accent, {VIOLET});}}

    .explain-card {{
        background: white; border-radius: 14px; padding: 16px 18px; margin-bottom: 10px;
        box-shadow: 0 6px 18px rgba(84,51,255,0.07);
        border-left: 4px solid {VIOLET};
    }}
    .explain-title {{font-weight: 700; font-size: 13.5px; margin-bottom: 5px; color: {INK};}}
    .explain-text {{font-size: 13px; color: {INK_SOFT}; line-height: 1.55;}}

    .badge {{display:inline-block; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700;}}

    .decision-banner {{
        text-align:center; padding: 24px; border-radius: 18px; margin: 6px 0 18px;
    }}
    .decision-banner .dlabel {{font-size:13px; opacity:0.8; font-weight:600;}}
    .decision-banner .dvalue {{font-size:28px; font-weight:800; margin-top:4px;}}

    .reason-box {{
        background: #FFF8EE; border-left: 4px solid {AMBER};
        padding: 14px 16px; border-radius: 10px; font-size: 13.5px; margin: 14px 0;
    }}

    div[data-testid="stMetric"] {{
        background: white; border-radius: 14px; padding: 14px 16px;
        box-shadow: 0 6px 18px rgba(84,51,255,0.07);
    }}
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value, delta=None, accent=VIOLET):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def badge(text, color):
    return f'<span class="badge" style="background:{color}22;color:{color};">{text}</span>'


def explain_card(title, text):
    st.markdown(f"""
    <div class="explain-card">
        <div class="explain-title">{title}</div>
        <div class="explain-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def style_fig(fig, height=300):
    fig.update_layout(
        height=height, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK),
    )
    return fig


@st.cache_resource(show_spinner="Loading trained pipeline...")
def get_bundle():
    try:
        return load_trained_pipeline()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def get_full_output():
    if os.path.exists(FULL_OUTPUT_PATH):
        return pd.read_pickle(FULL_OUTPUT_PATH)
    return None


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:12px;padding:4px 4px 22px;">
    <div style="font-size:26px;background:rgba(255,255,255,0.18);width:44px;height:44px;
         border-radius:13px;display:flex;align-items:center;justify-content:center;">🚦</div>
    <div>
        <div style="font-weight:800;font-size:16px;">Quote Agents</div>
        <div style="font-size:12px;opacity:0.75;">Autonomous Pipeline</div>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "View",
    ["📊 Overview", "🔍 Live Feed", "🚨 Escalation Queue", "🌎 Regional & Channel",
     "🧪 Model Diagnostics", "✍️ Predict New Quote"],
    label_visibility="collapsed",
)

st.sidebar.markdown("""
<div style="margin-top:20px;display:inline-flex;align-items:center;gap:8px;
     background:rgba(255,255,255,0.14);padding:8px 14px;border-radius:999px;
     font-size:12px;font-weight:600;">
    <span style="width:8px;height:8px;background:#00E5A0;border-radius:50%;display:inline-block;"></span>
    Pipeline live
</div>
""", unsafe_allow_html=True)

bundle = get_bundle()

if bundle is None:
    st.warning("No trained model found yet.")
    st.write("Train the pipeline once (takes ~30-60s), then it's saved to disk for every future launch.")
    if st.button("▶ Train pipeline now", type="primary"):
        with st.spinner("Training all 4 agents on the full dataset..."):
            bundle, scored_df = train_all()
            scored_df.to_pickle(FULL_OUTPUT_PATH)
        st.success("Trained and saved. Reloading...")
        st.cache_resource.clear()
        st.rerun()
    st.stop()

df = get_full_output()
if df is None:
    st.warning("Trained model found, but the full scored dataset cache is missing.")
    if st.button("▶ Score full dataset now", type="primary"):
        with st.spinner("Regenerating the full scored dataset..."):
            bundle, scored_df = train_all()
            scored_df.to_pickle(FULL_OUTPUT_PATH)
        st.cache_data.clear()
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------
if page == "📊 Overview":
    st.markdown('<h1 style="font-size:30px;margin-bottom:0;">Autonomous quote agents</h1>', unsafe_allow_html=True)
    st.caption("Risk Profiler → Conversion Predictor → Premium Advisor → Decision Router")
    st.write("")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total quotes", f"{len(df):,}", accent=VIOLET)
    with c2: kpi_card("Auto-approved", f"{(df['Decision']=='Auto_Approve').sum():,}",
                       f"{(df['Decision']=='Auto_Approve').mean():.1%}", accent=TEAL)
    with c3: kpi_card("Agent follow-up", f"{(df['Decision']=='Agent_Follow_Up').sum():,}",
                       f"{(df['Decision']=='Agent_Follow_Up').mean():.1%}", accent=SKY)
    with c4: kpi_card("Escalated", f"{(df['Decision']=='Escalate_to_Underwriter').sum():,}",
                       f"{(df['Decision']=='Escalate_to_Underwriter').mean():.1%}", accent=CORAL)
    with c5: kpi_card("Actual bind rate", f"{(df['Policy_Bind']=='Yes').mean():.1%}", accent=AMBER)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Risk tier distribution**")
            vc = df["Risk_Tier"].value_counts().reindex(["Low", "Medium", "High"])
            fig = go.Figure(go.Bar(
                x=vc.index, y=vc.values, marker_color=[RISK_COLORS[t] for t in vc.index],
                text=vc.values, textposition="outside", marker_line_width=0,
            ))
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("**Routing decisions**")
            vc = df["Decision"].value_counts()
            fig = go.Figure(go.Pie(
                labels=[l.replace("_", " ") for l in vc.index], values=vc.values,
                marker_colors=[DECISION_COLORS[k] for k in vc.index], hole=0.58,
                textfont=dict(color="white", size=12),
            ))
            st.plotly_chart(style_fig(fig), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown("**Bind score distribution**")
            fig = px.histogram(df, x="Bind_Score", nbins=20, color_discrete_sequence=[VIOLET])
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
            st.caption("Scores cluster near the 22.2% base rate — see Model Diagnostics for why.")

    with col4:
        with st.container(border=True):
            st.markdown("**Premium flag distribution**")
            vc = df["Premium_Flag"].value_counts()
            colors = [TEAL if "In" in k else (CORAL if "Over" in k else AMBER) for k in vc.index]
            fig = go.Figure(go.Bar(
                x=vc.values, y=[l.replace("_", " ") for l in vc.index], orientation="h",
                marker_color=colors, marker_line_width=0,
            ))
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)

# ---------------------------------------------------------------------
# Page: Live Feed
# ---------------------------------------------------------------------
elif page == "🔍 Live Feed":
    st.markdown('<h1 style="font-size:28px;">Live quote feed</h1>', unsafe_allow_html=True)
    n = st.slider("Rows to show", 10, 100, 25)
    view = df.sample(n)[["Quote_Num", "Risk_Tier", "Bind_Score", "Premium_Flag", "Decision", "Route_Reason"]].copy()

    def style_decision(val):
        color = DECISION_COLORS.get(val, "#888")
        return f"background-color:{color}22;color:{color};font-weight:700"

    def style_risk(val):
        color = RISK_COLORS.get(val, "#888")
        return f"background-color:{color}22;color:{color};font-weight:700"

    st.dataframe(
        view.style.map(style_decision, subset=["Decision"]).map(style_risk, subset=["Risk_Tier"]),
        use_container_width=True, height=560,
    )

# ---------------------------------------------------------------------
# Page: Escalation Queue
# ---------------------------------------------------------------------
elif page == "🚨 Escalation Queue":
    st.markdown('<h1 style="font-size:28px;">Escalation queue</h1>', unsafe_allow_html=True)
    esc = df[df["Decision"] == "Escalate_to_Underwriter"]

    c1, c2 = st.columns([1, 3])
    with c1:
        kpi_card("In queue", f"{len(esc):,}", f"{len(esc)/len(df):.1%} of volume", accent=CORAL)
    with c2:
        pick = st.selectbox("Select a quote to review", esc["Quote_Num"].head(200))

    row = esc[esc["Quote_Num"] == pick].iloc[0]

    st.markdown(f"""
    <div class="explain-card" style="border-left-color:{CORAL};">
        <div style="font-weight:800;font-size:16px;margin-bottom:8px;">Quote {row['Quote_Num']}</div>
        Risk tier: {badge(row['Risk_Tier'], RISK_COLORS[row['Risk_Tier']])}
        &nbsp; Bind score: <b>{row['Bind_Score']:.1f}%</b>
        &nbsp; Premium flag: {badge(row['Premium_Flag'].replace('_',' '), VIOLET)}
        <div class="reason-box"><b>Routing reason:</b> {row['Route_Reason']}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Retrieving similar historical quotes..."):
        similar = bundle["router_agent"].retrieve_similar(df.index.get_loc(row.name))
    st.markdown("**Similar historical quotes (RAG retrieval)**")
    st.dataframe(similar[["Quote_Num", "Risk_Tier", "Coverage", "Region", "Quoted_Premium", "Policy_Bind"]],
                 use_container_width=True)
    bind_rate = (similar["Policy_Bind"] == "Yes").mean()
    st.caption(f"{bind_rate:.0%} of these comparable quotes bound historically — "
               f"average premium ₹{similar['Quoted_Premium'].mean():.2f}.")

# ---------------------------------------------------------------------
# Page: Regional & Channel
# ---------------------------------------------------------------------
elif page == "🌎 Regional & Channel":
    st.markdown('<h1 style="font-size:28px;">Regional & channel intelligence</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**EA vs IA bind rate**")
            rates = df.groupby("Agent_Type")["Policy_Bind"].apply(lambda x: (x == "Yes").mean())
            fig = go.Figure(go.Bar(x=rates.index, y=rates.values, marker_color=[VIOLET, SKY],
                                    text=[f"{v:.1%}" for v in rates.values], textposition="outside",
                                    marker_line_width=0))
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig), use_container_width=True)
            st.caption("EA and IA bind rates are statistically indistinguishable (~22.2% each) — "
                       "a real finding, not a data gap.")
    with col2:
        with st.container(border=True):
            st.markdown("**Bind rate by region**")
            rates = df.groupby("Region")["Policy_Bind"].apply(lambda x: (x == "Yes").mean()).sort_index()
            fig = go.Figure(go.Bar(x=rates.index, y=rates.values, marker_color=CHART_PALETTE * 2,
                                    text=[f"{v:.1%}" for v in rates.values], textposition="outside",
                                    marker_line_width=0))
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig), use_container_width=True)

# ---------------------------------------------------------------------
# Page: Model Diagnostics
# ---------------------------------------------------------------------
elif page == "🧪 Model Diagnostics":
    st.markdown('<h1 style="font-size:28px;">Model diagnostics</h1>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="reason-box" style="border-left-color:{AMBER};">
        Exhaustive testing (marginal rates, multivariate Random Forest, engineered interaction
        features, gradient boosting, mutual information) found <b>Policy_Bind is statistically
        independent</b> of every available feature in this dataset (best ROC-AUC ≈ 0.50). Bind
        scores are calibrated estimates that correctly cluster near the base rate, not confident
        individual predictions.
    </div>
    """, unsafe_allow_html=True)

    comp = bundle["conversion_diagnostics"]["comparison"]
    comp_df = pd.DataFrame(comp).T.reset_index().rename(columns={"index": "model"})
    st.markdown("**Model comparison (calibrated, held-out test set)**")
    st.dataframe(comp_df, use_container_width=True)
    st.caption(f"Selected model: **{bundle['conversion_diagnostics']['selected_model']}** "
               f"(lowest Brier score) · base rate: {bundle['conversion_diagnostics']['base_rate']:.1%}")

    st.write("")
    st.markdown("**Data quality report**")
    dq = bundle["data_quality_report"]
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Rows processed", f"{dq['rows_out']:,}", accent=VIOLET)
    with c2: kpi_card("Re-quotes identified", f"{dq['requote_rows_identified']:,}", accent=SKY)
    with c3: kpi_card("Redundant columns dropped", len(dq["dropped_redundant_cols"]), accent=AMBER)
    st.write("")
    st.markdown('<ul class="notes-list">' + "".join(f"<li>{n}</li>" for n in dq["notes"]) + "</ul>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Page: Predict New Quote
# ---------------------------------------------------------------------
elif page == "✍️ Predict New Quote":
    st.markdown('<h1 style="font-size:28px;">Predict a new quote</h1>', unsafe_allow_html=True)
    st.caption("Runs a brand-new quote through all 4 trained agents live — no retraining.")

    with st.form("new_quote_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.markdown(f'<div style="color:{VIOLET_DARK};font-weight:800;font-size:13px;'
                            f'text-transform:uppercase;margin-bottom:8px;">Driver</div>', unsafe_allow_html=True)
                driver_age = st.slider("Driver age", 18, 65, 35)
                gender = st.selectbox("Gender", ["Female", "Male"])
                marital = st.selectbox("Marital status", ["Single", "Married", "Dirvorced", "Widow"])
                education = st.selectbox("Education", ["High School", "College", "Bachelors", "Masters", "Ph.D"])
                prev_acc = st.selectbox("Prior accidents", [0, 1])
                prev_cit = st.selectbox("Prior citations", [0, 1])
        with c2:
            with st.container(border=True):
                st.markdown(f'<div style="color:{VIOLET_DARK};font-weight:800;font-size:13px;'
                            f'text-transform:uppercase;margin-bottom:8px;">Policy</div>', unsafe_allow_html=True)
                policy_type = st.selectbox("Policy type", ["Car", "Truck", "Van"])
                coverage = st.selectbox("Coverage", ["Basic", "Balanced", "Enhanced"])
                veh_usage = st.selectbox("Vehicle usage", ["Pleasure", "Commute", "Business"])
                hh_vehicles = st.slider("Household vehicles", 1, 9, 2)
                hh_drivers = st.slider("Household drivers", 1, 9, 2)
                re_quote = st.selectbox("Re-quote", ["No", "Yes"])
        with c3:
            with st.container(border=True):
                st.markdown(f'<div style="color:{VIOLET_DARK};font-weight:800;font-size:13px;'
                            f'text-transform:uppercase;margin-bottom:8px;">Financials & channel</div>',
                            unsafe_allow_html=True)
                sal_range = st.selectbox("Salary range", [
                    "<= $ 25 K", "> $ 25 K <= $ 40 K", "> $ 40 K <= $ 60 K",
                    "> $ 60 K <= $ 90 K", "> $ 90 K "],
                    format_func=lambda x: x.replace("$", "₹").replace("K", "K"))
                vehcost_range = st.selectbox("Vehicle cost range", [
                    "<= $ 10 K", "> $ 10 K <= $ 20 K", "> $ 20 K <= $ 30 K",
                    "> $ 30 K <= $ 40 K", "> $ 40 K "],
                    format_func=lambda x: x.replace("$", "₹").replace("K", "L"))
                miles_range = st.selectbox("Annual mileage range", [
                    "<= 7.5 K", "> 7.5 K & <= 15 K", "> 15 K & <= 25 K", "> 25 K & <= 35 K",
                    "> 35 K & <= 45 K", "> 45 K & <= 55 K", "> 55 K"])
                quoted_premium = st.slider(
                    "Quoted Premium (₹)",
                    10000.0,   # Minimum
                    50000.0,   # Maximum
                    18000.0,   # Default
                    step=500.0
                )
                agent_type = st.selectbox("Agent type", ["EA", "IA"])
                region = st.selectbox("Region", ["A", "B", "C", "D", "E", "F", "G", "H"])

        st.write("")
        submitted = st.form_submit_button("Run through pipeline →", type="primary", use_container_width=True)

    if submitted:
        raw_fields = {
            "Agent_Type": agent_type, "Region": region, "Policy_Type": policy_type,
            "HH_Vehicles": hh_vehicles, "HH_Drivers": hh_drivers, "Driver_Age": driver_age,
            "Prev_Accidents": prev_acc, "Prev_Citations": prev_cit, "Gender": gender,
            "Marital_Status": marital, "Education": education, "Sal_Range": sal_range,
            "Coverage": coverage, "Veh_Usage": veh_usage, "Annual_Miles_Range": miles_range,
            "Vehicl_Cost_Range": vehcost_range, "Re_Quote": re_quote,
            "Quoted_Premium": quoted_premium,
        }
        with st.spinner("Running Risk Profiler → Conversion Predictor → Premium Advisor → Decision Router..."):
            result = predict_single_quote(raw_fields, bundle)

        st.write("")
        d_color = DECISION_COLORS[result["Decision"]]
        st.markdown(f"""
        <div class="decision-banner" style="background:{d_color}18;">
            <div class="dlabel">Final decision</div>
            <div class="dvalue" style="color:{d_color};">{result['Decision'].replace('_',' ')}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Risk tier", result["Risk_Tier"], accent=RISK_COLORS[result["Risk_Tier"]])
        with c2: kpi_card("Bind score", f"{result['Bind_Score']:.1f}%", accent=VIOLET)
        with c3: kpi_card("Premium flag", result["Premium_Flag"].replace("_", " "), accent=SKY)

        st.write("")
        st.markdown("**Why**")
        col1, col2 = st.columns(2)
        with col1:
            explain_card("🎯 Risk Profiler", result["explanations"]["risk"])
            explain_card("💲 Premium Advisor", result["explanations"]["premium"])
        with col2:
            explain_card("📈 Conversion Predictor", result["explanations"]["conversion"])
            explain_card("🧭 Decision Router", result["Route_Reason"])

        if result.get("similar_quotes") is not None:
            st.markdown("**Similar historical quotes (RAG retrieval)**")
            st.caption("Grounding for the underwriter handoff — retrieved live for this new quote.")
            sim = result["similar_quotes"]
            st.dataframe(
                sim[["Quote_Num", "Risk_Tier", "Coverage", "Region", "Quoted_Premium", "Policy_Bind"]],
                use_container_width=True,
            )
            bind_rate = (sim["Policy_Bind"] == "Yes").mean()
            st.caption(f"{bind_rate:.0%} of these comparable quotes bound historically — "
                       f"average premium ₹{sim['Quoted_Premium'].mean():.2f}.")