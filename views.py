"""
views.py - the pages of the Streamlit app.

Each render_*() function draws one page. All numbers come from the real dataset, the saved model and the
result files in outputs/ - nothing is typed in by hand.
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import churn_utils
import explain
import recommendations
import ui

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "Telco_customer_churn.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
SAMPLE_DIR = BASE_DIR / "sample_data"

MODEL_COLUMNS = {"Logistic Regression": "prob_logistic_regression", "Random Forest": "prob_random_forest",
                 "XGBoost": "prob_xgboost"}


# =============================================================================
# cached data loaders
# =============================================================================
@st.cache_data(show_spinner=False)
def load_dataset():
    """The real dataset (Excel file) with a few helper columns for charts."""
    df = pd.read_excel(DATA_FILE)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0.0)   # 11 blanks = tenure 0
    df["Churned"] = df["Churn Label"] == "Yes"
    df["Status"] = np.where(df["Churned"], "Churned", "Retained")
    df["Tenure Group"] = pd.cut(df["Tenure Months"], bins=[-1, 12, 24, 48, 60, 72],
                                labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "61-72 mo"]).astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_output(name):
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def get_evidence():
    return recommendations.compute_evidence(load_dataset())


@st.cache_resource(show_spinner=False)
def get_explainer(_art):
    return explain.build_explainer(_art)


def _missing_file_message(name):
    st.warning(f"The result file **outputs/{name}** was not found. Run `python train_model.py` (or the notebook) to create it.")


def _py(value):
    return value.item() if hasattr(value, "item") else value


def show_chart(chart):
    st.altair_chart(chart, width="stretch")


# =============================================================================
# 1. DASHBOARD
# =============================================================================
def render_dashboard():
    ui.page_header("📊 Customer Dashboard",
                   "Overview of the real customer dataset used to train the model "
                   "(Telco_customer_churn.xlsx, an IBM sample dataset of a fictional telecom company).")
    df = load_dataset()
    total = len(df)
    churned = int(df["Churned"].sum())
    stayed = total - churned

    cols = st.columns(5)
    cols[0].metric("Total customers", f"{total:,}", border=True)
    cols[1].metric("Churned", f"{churned:,}", border=True)
    cols[2].metric("Stayed", f"{stayed:,}", border=True)
    cols[3].metric("Churn rate", f"{churned / total * 100:.2f}%", border=True)
    cols[4].metric("Retention rate", f"{stayed / total * 100:.2f}%", border=True)

    st.write("")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            donut_data = pd.DataFrame({"Status": ["Retained", "Churned"], "Customers": [stayed, churned]})
            donut = alt.Chart(donut_data).mark_arc(innerRadius=70).encode(
                theta="Customers:Q",
                color=alt.Color("Status:N", scale=alt.Scale(domain=["Retained", "Churned"],
                                                            range=[ui.COLORS["stay"], ui.COLORS["churn"]]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Status", "Customers"]).properties(height=300, title="Churned vs retained customers")
            show_chart(donut)
    with right:
        with st.container(border=True):
            show_chart(ui.churn_rate_chart(df, "Contract", "Churn rate by contract type",
                                           order=["Month-to-month", "One year", "Two year"]))

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            show_chart(ui.churn_rate_chart(df, "Payment Method", "Churn rate by payment method", horizontal=True))
    with right:
        with st.container(border=True):
            show_chart(ui.churn_rate_chart(df, "Internet Service", "Churn rate by internet service",
                                           order=["DSL", "Fiber optic", "No"]))

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            show_chart(ui.churn_rate_chart(df, "Tenure Group", "Churn rate by tenure group",
                                           order=["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "61-72 mo"]))
    with right:
        with st.container(border=True):
            box = alt.Chart(df).mark_boxplot(size=60, extent="min-max").encode(
                x=alt.X("Status:N", title=None, sort=["Retained", "Churned"], axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Monthly Charges:Q", title="Monthly charges"),
                color=alt.Color("Status:N", legend=None, scale=alt.Scale(domain=["Retained", "Churned"],
                                                                         range=[ui.COLORS["stay"], ui.COLORS["churn"]])),
            ).properties(height=300, title="Monthly charges: retained vs churned")
            show_chart(box)
            avg = df.groupby("Status")["Monthly Charges"].mean()
            st.caption(f"Average monthly charge: retained {avg['Retained']:.2f}  |  churned {avg['Churned']:.2f}")

    st.caption("Churn rate = churned customers in the group / all customers in the group. Tenure groups: 0-12, 13-24, "
               "25-48, 49-60, 61-72 months. These are patterns in the data, not proof of causes.")


# =============================================================================
# 2. INDIVIDUAL PREDICTION
# =============================================================================
STATE_KEYS = {"Gender": "gender", "Senior Citizen": "senior", "Partner": "partner", "Dependents": "dependents",
              "Phone Service": "phone", "Internet Service": "internet", "Contract": "contract",
              "Paperless Billing": "paperless", "Payment Method": "payment"}
ADDON_KEYS = {"Online Security": "security", "Online Backup": "backup", "Device Protection": "protection",
              "Tech Support": "support", "Streaming TV": "tv", "Streaming Movies": "movies"}

DEFAULTS = {
    "gender": "Female", "senior": "No", "partner": "No", "dependents": "No", "phone": "Yes",
    "lines_with_phone": "No", "lines_no_phone": "No phone service", "internet": "DSL",
    "contract": "Month-to-month", "paperless": "Yes", "payment": "Electronic check",
    "tenure": 12, "monthly": 70.0, "auto_total": True, "total_manual": 840.0, "total_auto": 840.0,
}
for _key in ADDON_KEYS.values():
    DEFAULTS[f"{_key}_net"] = "No"
    DEFAULTS[f"{_key}_nonet"] = "No internet service"


@st.cache_data(show_spinner=False)
def get_examples(_art):
    """Three REAL customers from the held-out test set (low / medium / high predicted risk) for demonstrations."""
    predictions = load_output("test_predictions.csv")
    if predictions is None:
        return {}
    df = load_dataset()
    test_rows = df.loc[predictions["row_index"]]
    probabilities = churn_utils.predict_probabilities(test_rows[churn_utils.FEATURE_COLUMNS], _art)
    examples = {}
    for name, target in (("High-risk example", 0.85), ("Medium-risk example", 0.40), ("Low-risk example", 0.05)):
        position = int(np.argmin(np.abs(probabilities - target)))
        row = test_rows.iloc[position]
        examples[name] = {
            "customer": {c: _py(row[c]) for c in churn_utils.FEATURE_COLUMNS},
            "customer_id": str(row["CustomerID"]),
            "probability": float(probabilities[position]),
            "actual": "churned" if row["Churned"] else "stayed",
        }
    return examples


def apply_example(example):
    """Callback: copy a real customer into the form widgets (runs before the widgets are drawn)."""
    customer = example["customer"]
    for column, key in STATE_KEYS.items():
        st.session_state[key] = customer[column]
    st.session_state["lines_with_phone" if customer["Phone Service"] == "Yes" else "lines_no_phone"] = customer["Multiple Lines"]
    suffix = "net" if customer["Internet Service"] != "No" else "nonet"
    for column, key in ADDON_KEYS.items():
        st.session_state[f"{key}_{suffix}"] = customer[column]
    st.session_state["tenure"] = int(customer["Tenure Months"])
    st.session_state["monthly"] = float(customer["Monthly Charges"])
    st.session_state["auto_total"] = False
    st.session_state["total_manual"] = float(customer["Total Charges"])
    st.session_state["example_note"] = (f"Loaded real customer {example['customer_id']} from the held-out test set "
                                       f"(this customer actually {example['actual']}).")


def render_predict(art):
    ui.page_header("🔮 Predict Customer Churn",
                   "Enter one customer's details to estimate the probability that the customer will leave.")
    allowed = churn_utils.allowed_values(art)
    ranges = art["metadata"]["numeric_ranges"]
    for key, value in DEFAULTS.items():                      # first visit: fill the form with sensible defaults
        st.session_state.setdefault(key, value)

    # ---- example customers (real, from the test set)
    examples = get_examples(art)
    if examples:
        st.write("**Quick start:** load a real customer from the held-out test set, or fill in the form yourself.")
        buttons = st.columns(len(examples))
        for column, (name, example) in zip(buttons, examples.items()):
            column.button(f"📋 {name}", key=f"btn_{name}", on_click=apply_example, args=(example,), width="stretch")
        if st.session_state.get("example_note"):
            st.info(st.session_state["example_note"])

    # ---- the form
    with st.container(border=True):
        st.subheader("👤 Customer profile")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", allowed["Gender"], key="gender")
        senior = c2.selectbox("Senior Citizen", allowed["Senior Citizen"], key="senior")
        partner = c3.selectbox("Partner", allowed["Partner"], key="partner")
        dependents = c4.selectbox("Dependents", allowed["Dependents"], key="dependents")

    with st.container(border=True):
        st.subheader("📞 Services")
        c1, c2, c3 = st.columns(3)
        phone = c1.selectbox("Phone Service", allowed["Phone Service"], key="phone")
        if phone == "Yes":
            lines = c2.selectbox("Multiple Lines", ["No", "Yes"], key="lines_with_phone")
        else:
            lines = c2.selectbox("Multiple Lines", ["No phone service"], key="lines_no_phone", disabled=True,
                                 help="Locked: the customer has no phone service.")
        internet = c3.selectbox("Internet Service", allowed["Internet Service"], key="internet")

        has_internet = internet != "No"
        addon_options = ["No", "Yes"] if has_internet else ["No internet service"]
        suffix = "net" if has_internet else "nonet"
        if not has_internet:
            st.caption("🔒 The customer has no internet service, so the add-on services below are locked to "
                       "'No internet service'.")
        addon_values = {}
        cols = st.columns(3)
        for position, (column, key) in enumerate(ADDON_KEYS.items()):
            addon_values[column] = cols[position % 3].selectbox(column, addon_options, key=f"{key}_{suffix}",
                                                                disabled=not has_internet)

    with st.container(border=True):
        st.subheader("💳 Contract & billing")
        c1, c2, c3 = st.columns(3)
        contract = c1.selectbox("Contract", allowed["Contract"], key="contract")
        paperless = c2.selectbox("Paperless Billing", allowed["Paperless Billing"], key="paperless")
        payment = c3.selectbox("Payment Method", allowed["Payment Method"], key="payment")

        c1, c2, c3 = st.columns(3)
        tenure = c1.number_input("Tenure Months", min_value=int(ranges["Tenure Months"][0]),
                                 max_value=int(ranges["Tenure Months"][1]), step=1, key="tenure",
                                 help="How many months the customer has been with the company (0 = brand new).")
        monthly = c2.number_input("Monthly Charges", min_value=float(ranges["Monthly Charges"][0]),
                                  max_value=float(ranges["Monthly Charges"][1]), step=0.01, format="%.2f", key="monthly")
        auto_total = c3.toggle("Calculate Total Charges automatically", key="auto_total",
                               help="Total Charges = Tenure x Monthly Charges. Switch off to type your own value.")
        if auto_total:
            st.session_state["total_auto"] = round(min(tenure * monthly, float(ranges["Total Charges"][1])), 2)
            total = c3.number_input("Total Charges", key="total_auto", disabled=True, format="%.2f",
                                    min_value=0.0, max_value=float(ranges["Total Charges"][1]))
        else:
            total = c3.number_input("Total Charges", min_value=float(ranges["Total Charges"][0]),
                                    max_value=float(ranges["Total Charges"][1]), step=0.01, format="%.2f", key="total_manual",
                                    help="Must be 0 for a brand-new customer (tenure 0), otherwise roughly tenure x monthly charges.")

    customer = {"Gender": gender, "Senior Citizen": senior, "Partner": partner, "Dependents": dependents,
                "Tenure Months": int(tenure), "Phone Service": phone, "Multiple Lines": lines,
                "Internet Service": internet, **addon_values, "Contract": contract, "Paperless Billing": paperless,
                "Payment Method": payment, "Monthly Charges": float(monthly), "Total Charges": float(total)}
    customer = {column: customer[column] for column in churn_utils.FEATURE_COLUMNS}

    if st.button("🔮 Predict Churn", type="primary", width="stretch", key="predict_button"):
        problems = churn_utils.validate_customer(customer, ranges)
        if problems:
            st.error("Please correct the following before predicting:")
            for problem in problems:
                st.write(f"- {problem}")
            return
        try:
            result = churn_utils.predict_customer(customer, art)
            show_prediction(art, customer, result)
        except Exception as error:
            st.error("An error occurred while processing the prediction.")
            st.exception(error)


def show_prediction(art, customer, result):
    st.write("")
    ui.result_banner(result["label"], result["risk"], result["probability"])
    ui.probability_bar(result["probability"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted result", result["label"], border=True)
    c2.metric("Churn probability", f"{result['probability'] * 100:.2f}%", border=True)
    c3.metric("Risk category", result["risk"], border=True)
    st.caption(f"A customer is predicted to churn when the probability is above {churn_utils.DECISION_THRESHOLD:.0%}. "
               f"Risk levels: Low below {churn_utils.LOW_RISK_UPPER:.0%}, Medium {churn_utils.LOW_RISK_UPPER:.0%} to "
               f"{churn_utils.DECISION_THRESHOLD:.0%}, High above {churn_utils.DECISION_THRESHOLD:.0%}.")

    tab_why, tab_rec, tab_data = st.tabs(["🧠 Why this prediction?", "💡 Retention recommendations", "📋 Customer summary"])

    with tab_why:
        table, base, margin = explain.explain_customer(art, get_explainer(art), customer)
        top = table.head(8).copy()
        top["Label"] = top["Feature"] + " = " + top["Value"]
        chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("Label:N", sort=list(top["Label"]), title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Contribution:Q", title="Influence on the prediction (SHAP, log-odds)"),
            color=alt.Color("Effect:N", legend=alt.Legend(orient="bottom", title=None),
                            scale=alt.Scale(domain=["Raises churn risk", "Lowers churn risk"],
                                            range=[ui.COLORS["churn"], ui.COLORS["stay"]])),
            tooltip=["Feature", "Value", alt.Tooltip("Contribution:Q", format="+.3f"), "Effect"],
        ).properties(height=34 * len(top) + 40, title="Top factors behind this prediction (SHAP)")
        show_chart(chart)

        raises = table[table["Contribution"] > 0].head(3)
        lowers = table[table["Contribution"] < 0].head(3)
        left, right = st.columns(2)
        with left:
            st.markdown("**🔺 Factors raising the churn risk**")
            for row in raises.itertuples():
                st.write(f"- {row.Feature} = {row.Value}")
            if raises.empty:
                st.write("- none")
        with right:
            st.markdown("**🔻 Factors lowering the churn risk**")
            for row in lowers.itertuples():
                st.write(f"- {row.Feature} = {row.Value}")
            if lowers.empty:
                st.write("- none")
        st.caption("How to read: SHAP splits the model's output for THIS customer into contributions of each input "
                   "(the contributions of all options of one question, e.g. all three contract types, are added up). "
                   "Longer bar = bigger influence. This explains what the model relies on; it does not prove what causes churn.")

    with tab_rec:
        items = recommendations.build_recommendations(customer, result["probability"], result["risk"], get_evidence())
        for item in items:
            ui.recommendation_card(item)
        st.caption(recommendations.DISCLAIMER)

    with tab_data:
        summary = pd.DataFrame({"Field": list(customer.keys()), "Value": [str(v) for v in customer.values()]})
        st.dataframe(summary, hide_index=True, width="stretch")


# =============================================================================
# 3. BATCH PREDICTION
# =============================================================================
@st.cache_data(show_spinner=False, max_entries=5)
def cached_batch(csv_bytes, _art):
    return churn_utils.run_batch(csv_bytes, _art)


def render_batch(art):
    ui.page_header("📁 Batch Customer Prediction",
                   "Upload a CSV file with many customers and download the churn predictions for all of them.")

    with st.container(border=True):
        st.markdown(
            "**How it works**\n"
            "1. Download the template (or the sample file) and fill in your customers - **one row per customer**.\n"
            "2. Upload the CSV. The app checks the columns and every row before predicting.\n"
            "3. Valid rows get a churn probability, a predicted result and a risk category. Rows with problems are listed "
            "with the reason and are **not** predicted.\n"
            "4. Download the results as a CSV file.")
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download CSV template (3 example rows)", churn_utils.to_csv_bytes(churn_utils.template_dataframe()),
                           file_name="churn_batch_template.csv", mime="text/csv", width="stretch", key="dl_template")
        sample = SAMPLE_DIR / "sample_batch_customers.csv"
        if sample.exists():
            c2.download_button("⬇️ Download sample file (30 real customers)", sample.read_bytes(),
                               file_name="sample_batch_customers.csv", mime="text/csv", width="stretch", key="dl_sample")
        with st.expander("Required columns and allowed values"):
            allowed = churn_utils.allowed_values(art)
            ranges = art["metadata"]["numeric_ranges"]
            rows = [{"Column": "CustomerID (optional)", "Allowed values": "any text; copied to the results"}]
            for column in churn_utils.FEATURE_COLUMNS:
                if column in ranges:
                    low, high = ranges[column]
                    kind = "whole number" if column == "Tenure Months" else "number"
                    rows.append({"Column": column, "Allowed values": f"{kind} from {low} to {high}"})
                else:
                    rows.append({"Column": column, "Allowed values": ", ".join(allowed[column])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Values must match exactly (capital letters count). Total Charges may be blank only when Tenure Months is 0. "
                       "The same consistency rules as the single-customer form apply (e.g. no internet add-ons without internet).")

    uploaded = st.file_uploader("Upload a customer CSV file", type=["csv"], key="batch_file")
    if uploaded is None:
        st.info("No file uploaded yet. Download the template or the sample file above to try it out.")
        return

    outcome = cached_batch(uploaded.getvalue(), art)
    if outcome.fatal:
        st.error(f"❌ {outcome.fatal}")
        return
    for note in outcome.notes:
        st.info(note)

    results, rejected = outcome.results, outcome.rejected
    n_valid, n_bad = len(results), len(rejected)
    if n_bad == 0:
        st.success(f"✅ All {n_valid:,} rows were valid and have been predicted.")
    elif n_valid == 0:
        st.error(f"❌ None of the {n_bad:,} rows could be predicted. See the problems below.")
    else:
        st.warning(f"⚠️ {n_valid:,} of {outcome.n_rows:,} rows were predicted; {n_bad:,} row(s) were rejected (see below).")

    if n_valid:
        n_churn = int((results["Predicted_Churn"] == "Likely to Churn").sum())
        counts = results["Risk_Category"].value_counts()
        cols = st.columns(5)
        cols[0].metric("Customers predicted", f"{n_valid:,}", border=True)
        cols[1].metric("Likely to churn", f"{n_churn:,}", f"{n_churn / n_valid * 100:.1f}% of customers", delta_color="off", border=True)
        cols[2].metric("High risk", f"{int(counts.get('High Risk', 0)):,}", border=True)
        cols[3].metric("Medium risk", f"{int(counts.get('Medium Risk', 0)):,}", border=True)
        cols[4].metric("Low risk", f"{int(counts.get('Low Risk', 0)):,}", border=True)

        left, right = st.columns([1, 2])
        with left:
            with st.container(border=True):
                dist = pd.DataFrame({"Risk": ["Low Risk", "Medium Risk", "High Risk"],
                                     "Customers": [int(counts.get(k, 0)) for k in ("Low Risk", "Medium Risk", "High Risk")]})
                bars = alt.Chart(dist).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("Risk:N", sort=["Low Risk", "Medium Risk", "High Risk"], title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Customers:Q"),
                    color=alt.Color("Risk:N", legend=None, scale=alt.Scale(domain=list(ui.RISK_COLORS), range=list(ui.RISK_COLORS.values()))),
                    tooltip=["Risk", "Customers"])
                text = bars.mark_text(dy=-8).encode(text="Customers:Q", color=alt.value(ui.COLORS["ink"]))
                show_chart((bars + text).properties(height=260, title="Customers by risk level"))
        with right:
            with st.container(border=True):
                st.markdown("**Highest-risk customers in this file**")
                lead = [c for c in ("Source_Row", "CustomerID") if c in results.columns]
                top = results.sort_values("Churn_Probability", ascending=False).head(10)[
                    lead + ["Contract", "Tenure Months", "Monthly Charges", "Churn_Probability_Percent", "Risk_Category"]]
                st.dataframe(top, hide_index=True, width="stretch", column_config={
                    "Churn_Probability_Percent": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=100, format="%.1f%%")})

        st.subheader("Prediction results")
        choice = st.radio("Show", ["All", "High Risk", "Medium Risk", "Low Risk"], horizontal=True, key="batch_filter")
        shown = results if choice == "All" else results[results["Risk_Category"] == choice]
        shown = shown.sort_values("Churn_Probability", ascending=False)
        st.dataframe(shown, hide_index=True, width="stretch", column_config={
            "Churn_Probability_Percent": st.column_config.ProgressColumn("Churn probability (%)", min_value=0, max_value=100, format="%.1f")})
        st.download_button("⬇️ Download all results (CSV)", churn_utils.to_csv_bytes(results),
                           file_name="churn_predictions.csv", mime="text/csv", type="primary", width="stretch", key="dl_results")
        st.caption("Predicted_Churn = 'Likely to Churn' when the probability is above 50%. Risk levels: Low below 30%, "
                   "Medium 30-50%, High above 50%.")

    if n_bad:
        st.subheader("Rejected rows")
        st.caption("Source_Row is the line number in your CSV file (line 1 is the header). Fix these rows and upload again.")
        st.dataframe(rejected, hide_index=True, width="stretch")
        st.download_button("⬇️ Download rejected rows with reasons (CSV)", churn_utils.to_csv_bytes(rejected),
                           file_name="churn_rejected_rows.csv", mime="text/csv", width="stretch", key="dl_rejected")


# =============================================================================
# 4. RISK ANALYSIS AND RETENTION
# =============================================================================
def render_risk():
    ui.page_header("🧭 Risk Analysis & Retention",
                   "What the three risk levels mean and how a company can respond to each of them.")
    tab_levels, tab_playbook = st.tabs(["Risk levels", "Retention playbook"])

    with tab_levels:
        observed = load_output("risk_band_check.csv")
        low, high = churn_utils.LOW_RISK_UPPER, churn_utils.DECISION_THRESHOLD
        meanings = {
            "Low Risk": (f"Churn probability below {low:.0%}", "The model expects the customer to stay.",
                         "Keep service quality high; loyalty rewards or relevant add-on offers are optional."),
            "Medium Risk": (f"Churn probability {low:.0%} to {high:.0%} (inclusive)",
                            "Still predicted to stay, but the risk is clearly elevated.",
                            "Watch the account, run a satisfaction check, consider a targeted offer."),
            "High Risk": (f"Churn probability above {high:.0%}", "The model predicts that the customer will churn.",
                          "Prioritise a personalised retention contact."),
        }
        cols = st.columns(3)
        for column, risk in zip(cols, ("Low Risk", "Medium Risk", "High Risk")):
            with column:
                ui.risk_card(risk, *meanings[risk])
        st.write("")

        if observed is not None:
            st.markdown("**Does the risk level match reality?** Real churn rate of the held-out test customers in each level:")
            view = observed.copy()
            view["Actual_churn_rate"] = view["Actual_churn_rate"] * 100
            chart = alt.Chart(view).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Risk band:N", sort=["Low Risk", "Medium Risk", "High Risk"], title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Actual_churn_rate:Q", title="Customers who actually churned (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Risk band:N", legend=None, scale=alt.Scale(domain=list(ui.RISK_COLORS), range=list(ui.RISK_COLORS.values()))),
                tooltip=["Risk band", "Customers", "Actual_churners", alt.Tooltip("Actual_churn_rate:Q", format=".1f")])
            text = chart.mark_text(dy=-8).encode(text=alt.Text("Actual_churn_rate:Q", format=".1f"), color=alt.value(ui.COLORS["ink"]))
            show_chart((chart + text).properties(height=280))
            st.caption("The rate rises from Low to High, so the levels are meaningful. They are not perfect: some customers "
                       "in the Low level still churn and some in the High level stay.")
        else:
            _missing_file_message("risk_band_check.csv")

        st.info(f"**Decision threshold:** a customer is predicted to churn when the probability is above {high:.0%}. "
                "This standard threshold was **not changed**; all reported metrics use it.")

        threshold = load_output("threshold_analysis.csv")
        if threshold is not None:
            with st.expander("What would a different threshold do? (information only - the app keeps 50%)"):
                st.dataframe(threshold, hide_index=True, width="stretch", column_config={
                    c: st.column_config.NumberColumn(format="%.3f") for c in ("Precision", "Recall", "F1 Score")})
                st.caption("A lower threshold finds more churners (higher recall) but also flags more customers who would "
                           "have stayed (lower precision). Changing the threshold would need a business cost/benefit decision.")

    with tab_playbook:
        ev = get_evidence()
        st.markdown("The recommendation engine on the **Predict Customer** page uses these rules. The evidence column is "
                    "calculated from the real dataset.")
        playbook = pd.DataFrame([
            {"Trigger": "Month-to-month contract",
             "Evidence (churn rate in the dataset)": f"{ev['contract']['Month-to-month']:.0f}% vs {ev['contract']['Two year']:.0f}% on two-year contracts",
             "Suggested action": "Offer incentives for a 1- or 2-year contract"},
            {"Trigger": "Tenure of 12 months or less",
             "Evidence (churn rate in the dataset)": f"{ev['short_tenure']:.0f}% vs {ev['longer_tenure']:.0f}% for longer tenure",
             "Suggested action": "Onboarding support and engagement offers"},
            {"Trigger": f"Monthly charges in the top quarter (from {ev['monthly_p75']:.2f})",
             "Evidence (churn rate in the dataset)": f"{ev['high_monthly']:.0f}% vs {ev['other_monthly']:.0f}% for others",
             "Suggested action": "Review the plan; offer a better-matched plan or bundle"},
            {"Trigger": "Internet customer without Online Security",
             "Evidence (churn rate in the dataset)": f"{ev['no_security']:.0f}% vs {ev['security']:.0f}% with the service",
             "Suggested action": "Offer a security service trial"},
            {"Trigger": "Internet customer without Tech Support",
             "Evidence (churn rate in the dataset)": f"{ev['no_support']:.0f}% vs {ev['support']:.0f}% with the service",
             "Suggested action": "Offer a tech-support trial"},
            {"Trigger": "Pays by electronic check",
             "Evidence (churn rate in the dataset)": f"{ev['echeck']:.0f}% vs {ev['other_payment']:.0f}% for other methods",
             "Suggested action": "Incentive to switch to automatic payment"},
            {"Trigger": "Fibre-optic internet",
             "Evidence (churn rate in the dataset)": f"{ev['fiber']:.0f}% vs {ev['dsl']:.0f}% for DSL",
             "Suggested action": "Check service quality and price satisfaction"},
            {"Trigger": "High risk (probability above 50%)",
             "Evidence (churn rate in the dataset)": "see the risk-level check on the previous tab",
             "Suggested action": "Personalised retention contact"},
        ])
        st.dataframe(playbook, hide_index=True, width="stretch")
        st.warning(recommendations.DISCLAIMER)


# =============================================================================
# 5. MODEL PERFORMANCE
# =============================================================================
def render_performance(art):
    ui.page_header("📈 Model Performance",
                   "Results of the three compared models on the held-out test set (1,409 customers the models never saw "
                   "during training), plus cross-validation.")
    comparison = load_output("model_comparison.csv")
    predictions = load_output("test_predictions.csv")
    if comparison is None or predictions is None:
        _missing_file_message("model_comparison.csv / test_predictions.csv")
        return

    meta = art["metadata"]
    st.info(f"**Final model: {meta['final_model']}**, with {len(meta['feature_names'])} input features. "
            "The models perform very similarly and none is perfect; see the honest reading at the bottom of this page.")

    tab_cmp, tab_cm, tab_roc, tab_cv, tab_imp = st.tabs(
        ["Model comparison", "Confusion matrix", "ROC curve", "Cross-validation", "Feature importance"])

    with tab_cmp:
        st.dataframe(comparison, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.4f") for c in comparison.columns if c != "Model"})
        long = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="Model:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".4f")]).properties(height=330)
        show_chart(chart)

        ci = load_output("test_metric_confidence_intervals.csv")
        if ci is not None:
            with st.expander(f"How certain are the {meta['final_model']} numbers? (95% bootstrap confidence intervals)"):
                st.dataframe(ci, hide_index=True, width="stretch", column_config={
                    c: st.column_config.NumberColumn(format="%.4f") for c in ci.columns if c != "Metric"})
        with st.expander("What do these metrics mean?"):
            st.markdown(
                "- **Accuracy**: share of all customers classified correctly. Misleading here because most customers do not churn.\n"
                "- **Precision**: of the customers flagged as churners, how many really churn.\n"
                "- **Recall**: of the customers who really churn, how many the model finds.\n"
                "- **F1-score**: one number balancing precision and recall.\n"
                "- **ROC-AUC**: how well the model ranks churners above non-churners (0.5 = guessing, 1.0 = perfect).")

    with tab_cm:
        model_name = st.radio("Model", list(MODEL_COLUMNS), index=2, horizontal=True, key="cm_model")
        y_true = predictions["actual"].to_numpy()
        y_pred = (predictions[MODEL_COLUMNS[model_name]].to_numpy() > churn_utils.DECISION_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cells = pd.DataFrame({
            "Actual": ["Stayed", "Stayed", "Churned", "Churned"],
            "Predicted": ["Stay", "Churn", "Stay", "Churn"],
            "Customers": [tn, fp, fn, tp],
            "Meaning": ["Correct: stayed, predicted to stay", "False alarm: stayed, predicted to churn",
                        "Missed churner: churned, predicted to stay", "Correct: churned, predicted to churn"]})
        base = alt.Chart(cells).encode(
            x=alt.X("Predicted:N", sort=["Stay", "Churn"], title="Predicted", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Actual:N", sort=["Stayed", "Churned"], title="Actual"))
        heat = base.mark_rect(cornerRadius=6).encode(color=alt.Color("Customers:Q", scale=alt.Scale(scheme="blues"), legend=None),
                                                      tooltip=["Actual", "Predicted", "Customers", "Meaning"])
        label = base.mark_text(fontSize=24, fontWeight="bold").encode(
            text="Customers:Q", color=alt.condition(alt.datum.Customers > cells["Customers"].max() * 0.5,
                                                    alt.value("white"), alt.value("#111827")))
        left, right = st.columns([3, 2])
        with left:
            show_chart((heat + label).properties(height=330, title=f"{model_name} - confusion matrix (test set)"))
        with right:
            st.metric("Churners found (recall)", f"{tp / (tp + fn) * 100:.1f}%", f"{tp} of {tp + fn}", delta_color="off", border=True)
            st.metric("Churn alerts that are correct (precision)", f"{tp / (tp + fp) * 100:.1f}%", f"{tp} of {tp + fp}", delta_color="off", border=True)
            st.metric("Churners missed", f"{fn}", f"{fn / (tp + fn) * 100:.1f}% of churners", delta_color="off", border=True)

    with tab_roc:
        frames = []
        for model_name, column in MODEL_COLUMNS.items():
            fpr, tpr, _ = roc_curve(predictions["actual"], predictions[column])
            auc = roc_auc_score(predictions["actual"], predictions[column])
            frames.append(pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr,
                                        "Model": f"{model_name} (AUC {auc:.3f})"}))
        roc = pd.concat(frames)
        line = alt.Chart(roc).mark_line(strokeWidth=2.5).encode(
            x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)))
        diagonal = alt.Chart(pd.DataFrame({"False positive rate": [0, 1], "True positive rate": [0, 1]})).mark_line(
            strokeDash=[5, 5], color="#94A3B8").encode(x="False positive rate:Q", y="True positive rate:Q")
        show_chart((line + diagonal).properties(height=420, title="ROC curves (test set); dashed line = random guessing"))

    with tab_cv:
        cv = load_output("cross_validation_results.csv")
        if cv is None:
            _missing_file_message("cross_validation_results.csv")
        else:
            st.markdown("Each model was trained and tested 15 times (5 folds x 3 repeats) on different splits of the "
                        "real dataset; the table shows the **average ± standard deviation**.")
            view = pd.DataFrame({"Model": cv["Model"]})
            for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                               ("roc_auc", "ROC-AUC"), ("log_loss", "Log-loss (lower=better)"), ("brier", "Brier (lower=better)")]:
                view[label] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(cv[f"{key}_mean"], cv[f"{key}_std"])]
            st.dataframe(view, hide_index=True, width="stretch")
            paired = load_output("xgboost_vs_logistic_paired_cv.csv")
            if paired is not None:
                st.markdown("**XGBoost compared with Logistic Regression on the same 15 splits** (positive = XGBoost better):")
                st.dataframe(paired, hide_index=True, width="stretch", column_config={
                    "Mean difference (XGB - LR)": st.column_config.NumberColumn(format="%.4f"),
                    "Std of difference": st.column_config.NumberColumn(format="%.4f")})

    with tab_imp:
        importance = load_output("xgboost_feature_importance.csv")
        if importance is None:
            _missing_file_message("xgboost_feature_importance.csv")
        else:
            top = importance.head(15)
            chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4, color=ui.COLORS["primary"]).encode(
                y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                x=alt.X("Importance:Q", title="Importance (gain)"),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")]).properties(height=430, title="Top 15 XGBoost feature importances")
            show_chart(chart)
            st.caption("Feature importance shows which inputs the model relies on. It does not show what causes churn. "
                       "One-hot columns of the same question (e.g. Online Security_No / _Yes) share the credit.")

    st.divider()
    st.subheader("Honest reading of the results")
    selection = load_output("model_selection_evidence.csv")
    st.markdown(
        "- The test-set numbers come from **one** split of 1,409 customers; the confidence intervals above show how much they could move.\n"
        "- **Recall is modest**: the final model misses a large share of the customers who really churn.\n"
        "- XGBoost is kept as the final model mainly because it ranks customers best (highest ROC-AUC) and gives the best probability "
        "scores in cross-validation. The gap to Logistic Regression is small, and at the standard 50% threshold Logistic Regression "
        "is as good or better on several measures. Do **not** describe XGBoost as clearly superior or the model as perfect.")
    if selection is not None:
        with st.expander("Which model is best on each measure?"):
            st.dataframe(selection, hide_index=True, width="stretch")


# =============================================================================
# 6. EXPLAINABLE AI
# =============================================================================
@st.cache_data(show_spinner="Calculating SHAP values ...")
def cached_global_importance(_art):
    predictions = load_output("test_predictions.csv")
    customers = load_dataset().loc[predictions["row_index"], churn_utils.FEATURE_COLUMNS]
    return explain.global_importance(_art, get_explainer(_art), customers)


def render_explainability(art):
    ui.page_header("🧠 Explainable AI",
                   "Which customer details does the model rely on most? (SHAP explanations of the final XGBoost model)")
    if load_output("test_predictions.csv") is None:
        _missing_file_message("test_predictions.csv")
        return

    with st.container(border=True):
        st.markdown(
            "**What is SHAP?** SHAP splits the model's output for a customer into one contribution per input. "
            "Averaging the size of these contributions over the 1,409 held-out test customers gives the overall ranking below.\n\n"
            "To keep it readable, the contributions of all options of one question (for example the three contract types) are "
            "added together and shown under the question's name.")

    importance = cached_global_importance(art)
    top = importance.head(12)
    chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4, color=ui.COLORS["primary"]).encode(
        y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("Mean absolute SHAP:Q", title="Average influence on the prediction (SHAP, log-odds)"),
        tooltip=["Feature", alt.Tooltip("Mean absolute SHAP:Q", format=".3f")]).properties(height=400, title="Most influential customer details")
    with st.container(border=True):
        show_chart(chart)

    beeswarm = FIG_DIR / "shap_summary_beeswarm.png"
    if beeswarm.exists():
        with st.expander("Detailed SHAP summary plot (each dot = one test customer)"):
            st.image(str(beeswarm), width="stretch")
            st.caption("Each row is one model input. Each dot is a customer. Position left/right = pushes towards staying/churn; "
                       "colour = the input's value (red high, blue low; for one-hot inputs red = 'yes this option applies'). "
                       "Numeric inputs are shown in standardised units.")

    st.warning("**Correlation, model reliance and causation are different.** The model has learned patterns in historical data. "
               "A factor with a high SHAP value is something the model *uses*; it does not prove that changing that factor would "
               "change a customer's decision. To see the explanation for one customer, use **Predict Customer**.")


# =============================================================================
# 7. ABOUT
# =============================================================================
def render_about(art):
    ui.page_header("ℹ️ About this project", "Customer Churn Prediction and Retention Analysis System using Machine Learning")
    meta = art["metadata"]
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown(
            "**Goal.** Predict which telecom customers are likely to leave (churn), explain why, and suggest retention actions.\n\n"
            "**Dataset.** `Telco_customer_churn.xlsx`: 7,043 customers, 33 columns, 26.5% churned. "
            "An IBM sample dataset of a *fictional* company, so results show how the method behaves, not how real customers behave.\n\n"
            "**Target.** `Churn Label` (Yes = left, No = stayed).\n\n"
            "**Inputs.** 19 customer details (3 numeric, 16 categorical). Columns that would leak the answer "
            "(`Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`) are excluded.")
    with c2, st.container(border=True):
        st.markdown(
            "**Method.**\n"
            f"1. 80/20 stratified train/test split ({meta['n_train']:,} / {meta['n_test']:,} customers, random_state {meta['random_state']}).\n"
            "2. One-hot encoding + standard scaling, fitted on the training data only (46 model features).\n"
            "3. Logistic Regression, Random Forest and XGBoost compared; XGBoost is the final model.\n"
            "4. Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, cross-validation.\n"
            "5. SHAP for explanations; rule-based retention recommendations.\n\n"
            f"**Model file format.** {meta['model_format']}")
    with st.container(border=True):
        st.markdown("**Limitations**")
        st.markdown(
            "- Fictional sample data; no external validation.\n"
            "- Recall at the standard threshold is modest; some churners are missed and some alerts are false alarms.\n"
            "- Explanations and recommendations are based on patterns (association), not proven causes, and are not guarantees.\n"
            "- Login lasts only for the browser session; there is no password reset.")
        versions = meta["library_versions"]
        st.caption("Created with: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
