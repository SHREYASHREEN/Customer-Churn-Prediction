"""
ui.py - look and feel of the app: CSS, page headers, risk colours, the probability bar and reusable charts.
Only presentation code lives here; no machine-learning logic.
"""

import altair as alt
import pandas as pd
import streamlit as st

# one consistent palette
COLORS = {"churn": "#EF4444", "stay": "#10B981", "primary": "#2563EB", "amber": "#F59E0B",
          "ink": "#1F2937", "muted": "#64748B", "line": "#E2E8F0"}
RISK_COLORS = {"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"}
# darker versions with enough contrast for WHITE text on top
RISK_BANNER = {"Low Risk": "#047857", "Medium Risk": "#B45309", "High Risk": "#B91C1C"}
RISK_ICON = {"Low Risk": "🟢", "Medium Risk": "🟠", "High Risk": "🔴"}

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}
footer {visibility: hidden;}

/* banner at the top of every page */
.hero {background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%); border-radius: 14px;
       padding: 1.3rem 1.6rem; margin-bottom: 1.2rem;}
.hero-title {color: #FFFFFF; font-size: 1.65rem; font-weight: 700; line-height: 1.25;}
.hero-sub   {color: #DBEAFE; font-size: 1rem; margin-top: .3rem;}

/* prediction result banner */
.result-card {border-radius: 14px; padding: 1.2rem 1.5rem; margin: .4rem 0 1rem 0;}
.result-label {color: #F3F4F6; font-size: .85rem; letter-spacing: .06em; text-transform: uppercase;}
.result-main  {color: #FFFFFF; font-size: 1.9rem; font-weight: 800; line-height: 1.2;}
.result-sub   {color: #FFFFFF; font-size: 1.05rem; margin-top: .25rem;}

/* three-colour probability bar */
.pbar-wrap  {position: relative; margin: .6rem 0 1.6rem 0;}
.pbar-track {height: 22px; border-radius: 11px; position: relative;
             background: linear-gradient(to right, #10B981 0%, #10B981 30%, #F59E0B 30%, #F59E0B 50%, #EF4444 50%, #EF4444 100%);}
.pbar-marker {position: absolute; top: -6px; width: 6px; height: 34px; background: #111827;
              border: 2px solid #FFFFFF; border-radius: 3px; transform: translateX(-50%);}
.pbar-scale {position: relative; height: 18px; margin-top: 6px; color: #475569; font-size: .8rem;}
.pbar-scale span {position: absolute;}

/* risk-level cards */
.risk-card {border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid #E2E8F0; background: #FFFFFF; height: 100%;}
.risk-card h4 {margin: 0 0 .3rem 0; color: #111827;}
.risk-card p  {margin: .15rem 0; color: #374151;}

/* recommendation items */
.rec {border-left: 5px solid #2563EB; background: #FFFFFF; border-radius: 8px; padding: .7rem 1rem;
      margin-bottom: .7rem; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;}
.rec-title {font-weight: 700; color: #111827;}
.rec-meta  {color: #475569; font-size: .92rem;}
.pill {display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .75rem; font-weight: 700;
       color: #FFFFFF; margin-right: .5rem;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title, subtitle=""):
    sub = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="hero"><div class="hero-title">{title}</div>{sub}</div>', unsafe_allow_html=True)


def result_banner(label, risk, probability):
    color = RISK_BANNER[risk]
    st.markdown(
        f'<div class="result-card" style="background:{color}">'
        f'<div class="result-label">Prediction</div>'
        f'<div class="result-main">{RISK_ICON[risk]} {label}</div>'
        f'<div class="result-sub">Churn probability <b>{probability * 100:.1f}%</b> &nbsp;|&nbsp; {risk}</div></div>',
        unsafe_allow_html=True)


def probability_bar(probability):
    """A bar split into the three risk zones (0-30 green, 30-50 amber, 50-100 red) with a marker."""
    pos = min(max(probability, 0.0), 1.0) * 100
    shift = {0: "none", 100: "translateX(-100%)"}          # keep the end labels inside the bar's width
    ticks = "".join(f'<span style="left:{p}%;transform:{shift.get(p, "translateX(-50%)")}">{p}%</span>' for p in (0, 30, 50, 100))
    st.markdown(
        f'<div class="pbar-wrap"><div class="pbar-track"><div class="pbar-marker" style="left:{pos:.2f}%"></div></div>'
        f'<div class="pbar-scale">{ticks}</div></div>', unsafe_allow_html=True)


def risk_card(risk, rule, meaning, action):
    st.markdown(
        f'<div class="risk-card" style="border-top: 6px solid {RISK_COLORS[risk]}">'
        f'<h4>{RISK_ICON[risk]} {risk}</h4><p><b>{rule}</b></p><p>{meaning}</p><p><i>{action}</i></p></div>',
        unsafe_allow_html=True)


def recommendation_card(item):
    colors = {"High": "#B91C1C", "Medium": "#B45309", "Low": "#047857", "Optional": "#475569"}
    color = colors[item["priority"]]
    st.markdown(
        f'<div class="rec" style="border-left-color:{color}"><div class="rec-title">'
        f'<span class="pill" style="background:{color}">{item["priority"]}</span>{item["title"]}</div>'
        f'<div class="rec-meta"><b>Why:</b> {item["why"]}</div>'
        f'<div class="rec-meta"><b>Suggested action:</b> {item["action"]}</div></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- charts
def churn_rate_chart(df, column, title, order=None, horizontal=False, height=300):
    """Bar chart of the churn rate (%) per group, with the number of customers in the tooltip."""
    g = (df.groupby(column, observed=True)
           .agg(Customers=("Churned", "size"), Churned=("Churned", "sum")).reset_index())
    g["Churn rate (%)"] = (g["Churned"] / g["Customers"] * 100).round(1)
    tooltip = [column, "Customers", "Churned", "Churn rate (%)"]
    if horizontal:
        bars = alt.Chart(g).mark_bar(cornerRadiusEnd=4, color=COLORS["primary"]).encode(
            y=alt.Y(f"{column}:N", sort=order or "-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("Churn rate (%):Q", title="Churn rate (%)", scale=alt.Scale(domain=[0, max(60, g["Churn rate (%)"].max() + 8)])),
            tooltip=tooltip)
        labels = bars.mark_text(align="left", dx=5, color=COLORS["ink"]).encode(text=alt.Text("Churn rate (%):Q", format=".1f"))
    else:
        bars = alt.Chart(g).mark_bar(cornerRadiusEnd=4, color=COLORS["primary"]).encode(
            x=alt.X(f"{column}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
            y=alt.Y("Churn rate (%):Q", title="Churn rate (%)", scale=alt.Scale(domain=[0, max(60, g["Churn rate (%)"].max() + 8)])),
            tooltip=tooltip)
        labels = bars.mark_text(dy=-8, color=COLORS["ink"]).encode(text=alt.Text("Churn rate (%):Q", format=".1f"))
    return (bars + labels).properties(height=height, title=title)
