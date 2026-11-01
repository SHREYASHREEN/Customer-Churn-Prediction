"""
test_batch.py - tests of batch (CSV) prediction.

Run from the project folder:   python tests/test_batch.py

Part A tests churn_utils.run_batch() (the code behind the Batch page) with valid, invalid and malformed files.
Part B uploads files through the real app (Streamlit AppTest) and checks what the page shows.
"""

import io
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.environ["CHURN_AUTH_DB"] = str(Path(tempfile.mkdtemp()) / "test_users.db")   # throw-away user database

import auth  # noqa: E402
import churn_utils  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

art = churn_utils.load_artifacts(str(PROJECT / "models"))
results = []


def record(ok, name, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))


def csv_of(df):
    return df.to_csv(index=False).encode("utf-8")


template = churn_utils.template_dataframe()
sample_bytes = (PROJECT / "sample_data" / "sample_batch_customers.csv").read_bytes()
errors_bytes = (PROJECT / "sample_data" / "sample_batch_with_errors_SYNTHETIC.csv").read_bytes()

# =============================================================================== A. LOGIC
print("=== A. run_batch() logic ===")

# 1. template
out = churn_utils.run_batch(churn_utils.to_csv_bytes(template), art)
record(out.fatal is None and len(out.results) == 3 and len(out.rejected) == 0, "Downloadable template is accepted (3 valid rows)")
need = {"Churn_Probability", "Churn_Probability_Percent", "Predicted_Churn", "Risk_Category", "Source_Row"}
record(need.issubset(out.results.columns), "Results contain probability, predicted class, risk category and source row")

# 2. batch result == single-customer result, row by row
same = True
for _, row in out.results.iterrows():
    single = churn_utils.predict_customer({c: row[c] for c in churn_utils.FEATURE_COLUMNS}, art)
    same &= abs(single["probability"] - row["Churn_Probability"]) <= 5e-5 and single["label"] == row["Predicted_Churn"] and single["risk"] == row["Risk_Category"]
record(same, "Batch prediction == single-customer prediction for every row (probability, class, risk)")

# 3. 30 real customers vs the dataset
out = churn_utils.run_batch(sample_bytes, art)
df = pd.read_excel(PROJECT / "data" / "Telco_customer_churn.xlsx").set_index("CustomerID")
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0.0)
direct = churn_utils.predict_probabilities(df.loc[out.results["CustomerID"], churn_utils.FEATURE_COLUMNS], art)
record(out.fatal is None and len(out.results) == 30 and len(out.rejected) == 0 and not out.notes,
       "Sample file of 30 real customers: all valid, no warnings")
record(np.allclose(out.results["Churn_Probability"].to_numpy(), direct, atol=5e-5),
       "Batch probabilities equal the probabilities computed directly from the dataset rows")
record(all((p > 0.5) == (lab == "Likely to Churn") and churn_utils.risk_category(p) == r
           for p, lab, r in zip(direct, out.results["Predicted_Churn"], out.results["Risk_Category"])),
       "Predicted class and risk category follow the 50% / 30% rules")

# 4. file with deliberate mistakes
out = churn_utils.run_batch(errors_bytes, art)
reasons = " | ".join(out.rejected["Problems"])
record(out.fatal is None and len(out.results) == 3 and len(out.rejected) == 5,
       "Synthetic error file: 3 rows predicted, 5 rejected")
for needle, label in [("Contract = 'Monthly' is not allowed", "misspelled category"),
                      ("Online Security must be 'No internet service'", "internet add-on without internet"),
                      ("Tenure Months is missing", "blank tenure"),
                      ("inconsistent", "billing inconsistent with tenure x monthly"),
                      ("Monthly Charges is missing or not a number (abc)", "text in a numeric column")]:
    record(needle in reasons, f"Rejected with a clear reason: {label}")
record(list(out.rejected["Source_Row"]) == [5, 6, 7, 8, 9], "Rejected rows report the correct CSV line numbers", str(list(out.rejected["Source_Row"])))
record(len(out.notes) == 3 and any("ignored" in n for n in out.notes) and any("matched" in n for n in out.notes) and any("set to 0" in n for n in out.notes),
       "Nothing is silent: renamed column, ignored column and blank-total rule are all reported")

# 5. whole-file problems
fatal_cases = {
    "empty file": b"",
    "header only (no customers)": csv_of(template.iloc[0:0]),
    "missing required columns": csv_of(template.drop(columns=["Contract", "Tenure Months"])),
    "semicolon-separated file": template.to_csv(index=False, sep=";").encode(),
    "binary garbage": bytes(range(256)) * 40,
    "duplicate column": csv_of(pd.concat([template, template[["Contract"]].rename(columns={"Contract": "contract"})], axis=1)),
    "more than 10,000 rows": csv_of(pd.concat([template] * 3334, ignore_index=True)),
    "larger than 5 MB": b"a" * (5 * 1024 * 1024 + 1),
}
for label, content in fatal_cases.items():
    o = churn_utils.run_batch(content, art)
    record(o.fatal is not None and o.results is None, f"Whole file refused: {label}", (o.fatal or "")[:90])
o = churn_utils.run_batch(csv_of(template.drop(columns=["Contract", "Tenure Months"])), art)
record("Contract" in o.fatal and "Tenure Months" in o.fatal, "Missing-column message names every missing column")

# 6. single bad cells
base = template.iloc[[0]].copy()


def one(**changes):
    row = base.copy()
    for k, v in changes.items():
        row[k] = v
    return churn_utils.run_batch(csv_of(row), art)


for label, change, needle in [
    ("tenure 3.5 (not whole)", {"Tenure Months": "3.5"}, "whole number"),
    ("tenure -1 (below range)", {"Tenure Months": "-1"}, "outside the range"),
    ("tenure 100 (above range)", {"Tenure Months": "100", "Total Charges": "8000"}, "outside the range"),
    ("monthly charge NaN", {"Monthly Charges": "nan"}, "not a number"),
    ("monthly charge inf", {"Monthly Charges": "inf"}, "not a number"),
    ("monthly charge blank", {"Monthly Charges": ""}, "missing"),
    ("total blank but tenure > 0", {"Total Charges": ""}, "only allowed when Tenure Months is 0"),
    ("lower-case category 'female'", {"Gender": "female"}, "not allowed"),
    ("phone No + multiple lines Yes", {"Phone Service": "No", "Multiple Lines": "Yes"}, "Multiple Lines must be"),
]:
    o = one(**change)
    record(len(o.rejected) == 1 and needle in o.rejected.iloc[0]["Problems"], f"Cell problem rejected: {label}")
o = one(Gender="  Female  ")
record(len(o.results) == 1, "Extra spaces around a value are tolerated (' Female ' accepted)")
o = one(**{"Tenure Months": "0", "Total Charges": ""})
record(len(o.results) == 1 and any("set to 0" in n for n in o.notes), "Blank Total Charges is accepted for tenure 0, with a note")

# 7. Excel-style file (BOM + Windows line endings)
excel_style = b"\xef\xbb\xbf" + sample_bytes.decode("utf-8-sig").replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
o = churn_utils.run_batch(excel_style, art)
record(o.fatal is None and len(o.results) == 30, "File with UTF-8 BOM and Windows (CRLF) line endings works")

# 8. download round trip
res = churn_utils.run_batch(sample_bytes, art).results
payload = churn_utils.to_csv_bytes(res)
back = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
record(payload.startswith(b"\xef\xbb\xbf") and list(back.columns) == list(res.columns) and len(back) == len(res)
       and np.allclose(back["Churn_Probability"], res["Churn_Probability"]) and (back["CustomerID"] == res["CustomerID"]).all(),
       "Results CSV download re-opens correctly (BOM for Excel, same columns, rows and probabilities)")
bad = churn_utils.run_batch(errors_bytes, art).rejected
back = pd.read_csv(io.BytesIO(churn_utils.to_csv_bytes(bad)), encoding="utf-8-sig")
record(len(back) == 5 and "Problems" in back.columns, "Rejected-rows CSV download re-opens correctly")
t = pd.read_csv(io.BytesIO(churn_utils.to_csv_bytes(template)), encoding="utf-8-sig")
record(list(t.columns) == ["CustomerID"] + churn_utils.FEATURE_COLUMNS, "Template CSV has exactly the required columns in order")

# =============================================================================== B. UI
print("\n=== B. Batch page in the real app (file upload through Streamlit AppTest) ===")
auth.register_user("batch_tester", "Batch Tester", "Passw0rd123", "Passw0rd123")


def open_batch_page():
    at = AppTest.from_file(str(PROJECT / "app.py"), default_timeout=120).run()
    at.text_input(key="login_username").set_value("batch_tester")
    at.text_input(key="login_password").set_value("Passw0rd123")
    at.button[0].click().run()
    at.sidebar.radio(key="nav").set_value("📁 Batch Prediction").run()
    return at


at = open_batch_page()
record(not at.exception and len(at.file_uploader) == 1 and len(at.get("download_button")) == 2,
       "Batch page opens with an uploader and template/sample download buttons")
at.file_uploader[0].upload("sample.csv", sample_bytes, "text/csv").run()
labels = {m.label: m.value for m in at.metric}
record(not at.exception and labels.get("Customers predicted") == "30" and any("All 30 rows" in s.value for s in at.success),
       "Upload of the 30-customer sample: summary shows 30 predicted customers", str(labels))
record(len(at.get("download_button")) == 3, "A 'download results' button appears after a successful upload")
expected_churn = int((churn_utils.run_batch(sample_bytes, art).results["Predicted_Churn"] == "Likely to Churn").sum())
record(labels.get("Likely to churn") == str(expected_churn), "'Likely to churn' count on the page equals the count from the data", f"{expected_churn}")

at = open_batch_page()
at.file_uploader[0].upload("errors.csv", errors_bytes, "text/csv").run()
labels = {m.label: m.value for m in at.metric}
record(not at.exception and any("3 of 8 rows were predicted" in w.value for w in at.warning) and labels.get("Customers predicted") == "3",
       "Upload with errors: warning says 3 of 8 rows predicted")
record(len(at.get("download_button")) == 4 and len(at.info) >= 3, "Rejected-rows download button and the 3 information notes are shown")

at = open_batch_page()
at.file_uploader[0].upload("wrong.csv", csv_of(template.drop(columns=["Contract"])), "text/csv").run()
record(not at.exception and any("Required column(s) missing: Contract" in e.value for e in at.error) and len(at.metric) == 0,
       "Upload with a missing column: clear error, no predictions shown")
at = open_batch_page()
at.file_uploader[0].upload("empty.csv", b"", "text/csv").run()
record(not at.exception and any("empty" in e.value for e in at.error), "Upload of an empty file: clear error")

print(f"\nSUMMARY: {sum(results)} passed, {len(results) - sum(results)} failed")
sys.exit(0 if all(results) else 1)
