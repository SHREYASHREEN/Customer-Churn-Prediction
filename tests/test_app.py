"""
test_app.py - tests of the logged-in application: single prediction, validation, explanations,
recommendations and the numbers shown on every page.

Run from the project folder:   python tests/test_app.py

The REAL app.py runs with the REAL model files (no mock objects). Streamlit's AppTest fills in the widgets,
clicks buttons and lets us read what the app would display. It checks logic, messages and numbers; it does NOT
check how the page looks in a browser.
"""

import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.environ["CHURN_AUTH_DB"] = str(Path(tempfile.mkdtemp()) / "test_users.db")

import auth  # noqa: E402
import churn_utils  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

art = churn_utils.load_artifacts(str(PROJECT / "models"))
meta = art["metadata"]
results = []


def record(ok, name, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))


# ------------------------------------------------------------------ real customers + an INDEPENDENT probability calculation
df = pd.read_excel(PROJECT / "data" / "Telco_customer_churn.xlsx")
cols = meta["numerical_columns"] + meta["categorical_columns"]
X = df[cols].copy()
X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce").fillna(0.0)
y = (df["Churn Label"] == "Yes").astype(int)
_, X_test, _, y_test = train_test_split(X, y, test_size=meta["test_size"], random_state=meta["random_state"], stratify=y)

manual = np.hstack([(X_test[meta["numerical_columns"]].to_numpy(float) - art["scaler"].mean_) / art["scaler"].scale_] +
                   [(X_test[c].to_numpy()[:, None] == np.asarray(k)[None, :]).astype(float)
                    for c, k in zip(meta["categorical_columns"], art["encoder"].categories_)])
X_test = X_test.assign(prob=art["model"].predict_proba(manual)[:, 1], actual=y_test)   # does not use churn_utils


def nearest(target):
    return X_test.iloc[(X_test.prob - target).abs().argmin()]


cases = {
    "highest-probability customer": X_test.loc[X_test.prob.idxmax()],
    "lowest-probability customer": X_test.loc[X_test.prob.idxmin()],
    "near the 0.30 Low/Medium boundary": nearest(0.30),
    "just ABOVE 0.50 (churn, High Risk)": X_test[X_test.prob > 0.5].sort_values("prob").iloc[0],
    "at/just BELOW 0.50 (stay, Medium Risk)": X_test[X_test.prob <= 0.5].sort_values("prob").iloc[-1],
    "mid-range customer near 0.15": nearest(0.15),
}

# ------------------------------------------------------------------ helpers to drive the app
auth.register_user("app_tester", "App Tester", "Passw0rd123", "Passw0rd123")
SELECT_KEYS = {"Gender": "gender", "Senior Citizen": "senior", "Partner": "partner", "Dependents": "dependents",
               "Contract": "contract", "Paperless Billing": "paperless", "Payment Method": "payment"}
ADDONS = {"Online Security": "security", "Online Backup": "backup", "Device Protection": "protection",
          "Tech Support": "support", "Streaming TV": "tv", "Streaming Movies": "movies"}


def login(page=None):
    at = AppTest.from_file(str(PROJECT / "app.py"), default_timeout=120).run()
    at.text_input(key="login_username").set_value("app_tester")
    at.text_input(key="login_password").set_value("Passw0rd123")
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    if page:
        at.sidebar.radio(key="nav").set_value(page).run()
    return at


def open_predict():
    return login("🔮 Predict Customer")


def put(at, key, value):
    widget = at.selectbox(key=key)
    if widget.value != value:
        widget.set_value(value)


def fill(at, row, **override):
    """Enter a customer: set the controlling fields first and re-run so dependent fields adapt."""
    row = dict(row, **override)
    at.toggle(key="auto_total").set_value(False)
    put(at, "phone", row["Phone Service"])
    put(at, "internet", row["Internet Service"])
    at.run()
    if row["Phone Service"] == "Yes":
        put(at, "lines_with_phone", row["Multiple Lines"])
    for label, key in SELECT_KEYS.items():
        put(at, key, row[label])
    if row["Internet Service"] != "No":
        for label, key in ADDONS.items():
            put(at, f"{key}_net", row[label])
    at.number_input(key="tenure").set_value(int(row["Tenure Months"]))
    at.number_input(key="monthly").set_value(float(row["Monthly Charges"]))
    at.number_input(key="total_manual").set_value(float(row["Total Charges"]))
    return at


def predict(at):
    at.button(key="predict_button").click().run()
    return at


def page_html(at):
    return " ".join(m.value for m in at.markdown)


def metric_map(at):
    return {m.label: m.value for m in at.metric}


# =============================================================================== A. real customers
print("=== A. Real customers through the prediction form ===")
for name, row in cases.items():
    at = predict(fill(open_predict(), row))
    p = float(row.prob)
    risk = "High Risk" if p > 0.5 else "Medium Risk" if p >= 0.30 else "Low Risk"
    label = "Likely to Churn" if p > 0.5 else "Likely to Stay"
    m, html = metric_map(at), page_html(at)
    shown = float(m["Churn probability"].strip("%")) / 100 if "Churn probability" in m else -1
    marker = re.search(r'pbar-marker" style="left:([\d.]+)%', html)
    ok = (not at.exception and abs(shown - p) <= 0.00006 and m.get("Predicted result") == label and m.get("Risk category") == risk
          and f"{p * 100:.1f}%" in html and marker is not None and abs(float(marker.group(1)) - p * 100) <= 0.01)
    record(ok, f"{name}", f"true p={p:.4f} | page: {m.get('Churn probability')}, {m.get('Predicted result')}, {m.get('Risk category')} | "
                          f"actual: {'churned' if row.actual == 1 else 'stayed'}")

# label and risk can never disagree: sweep the boundaries with a fine grid of probabilities
sweep_ok = True
for p in np.linspace(0, 1, 2001):
    risk = churn_utils.risk_category(p)
    sweep_ok &= (risk == "High Risk") == (churn_utils.predicted_label(p) == "Likely to Churn")
    sweep_ok &= (risk == "Low Risk") == (p < 0.30)
record(sweep_ok and churn_utils.risk_category(0.30) == "Medium Risk" and churn_utils.risk_category(0.50) == "Medium Risk"
       and churn_utils.risk_category(0.5000001) == "High Risk" and churn_utils.risk_category(0.2999999) == "Low Risk",
       "Risk thresholds: Low < 30%, Medium 30-50% inclusive, High > 50%; 'Likely to Churn' only for High Risk")

# =============================================================================== B. the form adapts and protects
print("\n=== B. Form behaviour: locked fields, automatic total, limits ===")
at = open_predict()
at.selectbox(key="internet").set_value("No").run()
locked = [(at.selectbox(key=f"{k}_nonet").options, at.selectbox(key=f"{k}_nonet").disabled) for k in ADDONS.values()]
record(all(o == (["No internet service"], True) for o in locked), "Internet = No: all 6 add-on drop-downs lock to 'No internet service'")
at.selectbox(key="internet").set_value("DSL").run()
record(all(at.selectbox(key=f"{k}_net").options == ["No", "Yes"] and not at.selectbox(key=f"{k}_net").disabled for k in ADDONS.values()),
       "Internet = DSL: add-ons offer only 'No' / 'Yes'")
at.selectbox(key="phone").set_value("No").run()
lines = at.selectbox(key="lines_no_phone")
record(lines.options == ["No phone service"] and lines.disabled, "Phone Service = No: Multiple Lines locks to 'No phone service'")
at.selectbox(key="phone").set_value("Yes").run()
record(at.selectbox(key="lines_with_phone").options == ["No", "Yes"], "Phone Service = Yes: Multiple Lines offers 'No' / 'Yes'")

at.number_input(key="tenure").set_value(10).run()
at.number_input(key="monthly").set_value(50.0).run()
auto = at.number_input(key="total_auto")
record(auto.value == 500.0 and auto.disabled, "Automatic Total Charges = tenure x monthly (10 x 50 = 500) and is read-only")
ranges = meta["numeric_ranges"]
record(at.number_input(key="tenure").max == ranges["Tenure Months"][1] and at.number_input(key="monthly").min == ranges["Monthly Charges"][0]
       and at.number_input(key="monthly").max == ranges["Monthly Charges"][1], "Numeric fields are limited to the dataset range",
       f"tenure {ranges['Tenure Months']}, monthly {ranges['Monthly Charges']}")

# the default form (automatic total) must predict without errors
at = predict(open_predict())
record(not at.exception and "Churn probability" in metric_map(at), "The default form predicts without any change", str(metric_map(at)))

# =============================================================================== C. invalid input
print("\n=== C. Invalid input is refused with an explanation ===")
row = cases["mid-range customer near 0.15"]
for label, override, needle in [
    ("tenure 0 but Total Charges 840", {"Tenure Months": 0, "Total Charges": 840.0}, "Total Charges must be 0"),
    ("tenure 12, monthly 70, total 8000", {"Tenure Months": 12, "Monthly Charges": 70.0, "Total Charges": 8000.0}, "inconsistent"),
    ("tenure 12 but Total Charges 0", {"Tenure Months": 12, "Total Charges": 0.0}, "greater than 0"),
]:
    at = predict(fill(open_predict(), row, **override))
    record(needle in page_html(at) and "Churn probability" not in metric_map(at) and not at.exception,
           f"Refused: {label}", "explanation shown, no prediction")
at = predict(fill(open_predict(), row, **{"Tenure Months": 0, "Total Charges": 0.0}))
record("Churn probability" in metric_map(at) and not at.exception, "Accepted: brand-new customer (tenure 0, total 0)", metric_map(at).get("Churn probability", ""))

# =============================================================================== D. explanation + recommendations
print("\n=== D. Explanation and recommendations ===")
risky = dict(cases["highest-probability customer"])
at = predict(fill(open_predict(), pd.Series(risky)))
html = page_html(at)
record(len(at.get("arrow_vega_lite_chart")) + len(at.get("vega_lite_chart")) >= 1, "'Why this prediction' chart is drawn")
record("rec-title" in html and "Personalised retention contact" in html, "High-risk customer: first recommendation is the personalised retention contact")
sample = dict(Gender="Female", **{"Senior Citizen": "No", "Partner": "No", "Dependents": "No", "Tenure Months": 5, "Phone Service": "Yes",
              "Multiple Lines": "No", "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
              "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No", "Streaming Movies": "No",
              "Contract": "Month-to-month", "Paperless Billing": "Yes", "Payment Method": "Electronic check",
              "Monthly Charges": 95.0, "Total Charges": 475.0})
at = predict(fill(open_predict(), pd.Series(sample)))
html = page_html(at)
for needle, label in [("annual-contract incentive", "month-to-month contract"), ("Onboarding support", "short tenure"),
                      ("online-security service", "no online security"), ("Offer technical support", "no tech support"),
                      ("automatic payment", "electronic check"), ("Review whether the plan is a good fit", "high monthly charges"),
                      ("fibre service quality", "fibre optic")]:
    record(needle in html, f"Recommendation appears for: {label}")
loyal = dict(sample, **{"Tenure Months": 60, "Total Charges": 5700.0, "Monthly Charges": 95.0, "Contract": "Two year",
                        "Payment Method": "Bank transfer (automatic)", "Internet Service": "DSL", "Online Security": "Yes", "Tech Support": "Yes"})
at = predict(fill(open_predict(), pd.Series(loyal)))
html = page_html(at)
record("annual-contract incentive" not in html and "Onboarding support" not in html and "automatic payment" not in html,
       "Recommendations are NOT shown when the customer data does not trigger them (two-year, long tenure, autopay)")
record(any("not guaranteed to prevent churn" in c.value for c in at.caption), "The disclaimer 'not guaranteed to prevent churn' is shown (as a caption)")

# =============================================================================== E. example buttons (real customers)
print("\n=== E. Quick-start example buttons ===")
for button, expected in (("📋 High-risk example", "High Risk"), ("📋 Medium-risk example", "Medium Risk"), ("📋 Low-risk example", "Low Risk")):
    at = open_predict()
    at.button(key=f"btn_{button[2:]}").click().run()
    note = [i.value for i in at.info]
    at = predict(at)
    m = metric_map(at)
    record(not at.exception and any("real customer" in n and "held-out test set" in n for n in note) and m.get("Risk category") == expected,
           f"{button}: loads a real test-set customer and predicts {expected}", f"{m.get('Churn probability')} | {note[0][:60] if note else ''}")

# =============================================================================== F. numbers on the other pages
print("\n=== F. Numbers shown on the other pages match the data files ===")
at = login()
m = metric_map(at)
churned = int((df["Churn Label"] == "Yes").sum())
record(m.get("Total customers") == f"{len(df):,}" and m.get("Churned") == f"{churned:,}" and m.get("Stayed") == f"{len(df) - churned:,}"
       and m.get("Churn rate") == f"{churned / len(df) * 100:.2f}%" and m.get("Retention rate") == f"{(len(df) - churned) / len(df) * 100:.2f}%",
       "Dashboard KPIs equal the real dataset", str(m))
record(len(at.get("arrow_vega_lite_chart")) + len(at.get("vega_lite_chart")) >= 6, "Dashboard draws its 6 charts")

at = login("📈 Model Performance")
comparison = pd.read_csv(PROJECT / "outputs" / "model_comparison.csv")
shown = at.dataframe[0].value
record(not at.exception and np.allclose(shown.drop(columns="Model").to_numpy(float), comparison.drop(columns="Model").to_numpy(float)),
       "Model Performance table equals outputs/model_comparison.csv")
tp = pd.read_csv(PROJECT / "outputs" / "test_predictions.csv")
pred = (tp["prob_xgboost"] > 0.5).astype(int)
recall = ((pred == 1) & (tp["actual"] == 1)).sum() / (tp["actual"] == 1).sum()
record(metric_map(at).get("Churners found (recall)") == f"{recall * 100:.1f}%", "Confusion-matrix tab: recall equals the value computed from test_predictions.csv", metric_map(at).get("Churners found (recall)", ""))
at.radio(key="cm_model").set_value("Logistic Regression").run()
pred = (tp["prob_logistic_regression"] > 0.5).astype(int)
recall = ((pred == 1) & (tp["actual"] == 1)).sum() / (tp["actual"] == 1).sum()
record(not at.exception and metric_map(at).get("Churners found (recall)") == f"{recall * 100:.1f}%", "... and switches correctly to Logistic Regression")

at = login("🧭 Risk & Retention")
text = page_html(at)
record(not at.exception and "below 30%" in text and "above 50%" in text and "30% to 50% (inclusive)" in text, "Risk page states the 30% / 50% thresholds")
at = login("🧠 Explainable AI")
record(not at.exception and len(at.get("arrow_vega_lite_chart")) + len(at.get("vega_lite_chart")) >= 1 and any("Correlation" in w.value for w in at.warning),
       "Explainable AI page draws the SHAP chart and the correlation-vs-causation warning")
at = login("ℹ️ About")
record(not at.exception and "xgboost-ubj" in page_html(at), "About page shows the model format from the metadata")

print(f"\nSUMMARY: {sum(results)} passed, {len(results) - sum(results)} failed")
sys.exit(0 if all(results) else 1)
