"""
churn_utils.py
==============
Shared helper functions for the Customer Churn Prediction project.

The notebook, train_model.py, the Streamlit app and the tests all import this file,
so a customer is prepared for the model in exactly the same way everywhere.

Contents
--------
1. Thresholds and risk levels
2. Model files: saving a checksum manifest, loading + verifying the artifacts
3. validate_customer()   - rejects impossible single-customer input
4. build_model_input()   - scaling + one-hot encoding (identical to training)
5. predict_customer()    - one customer
6. run_batch()           - a whole CSV file, with strict validation
"""

import hashlib
import io
import json
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# 1. THRESHOLDS
# ---------------------------------------------------------------------------
# A customer is predicted to churn when the churn probability is STRICTLY ABOVE
# DECISION_THRESHOLD (the same tie rule scikit-learn and XGBoost use in predict()).
# Every metric in outputs/ was computed with this rule. It was NOT changed.
DECISION_THRESHOLD = 0.50

# Risk levels (presentation aid, chosen in advance, checked in outputs/risk_band_check.csv):
#   probability <  0.30            -> Low Risk
#   0.30 <= probability <= 0.50    -> Medium Risk   (predicted to stay, but elevated)
#   probability >  0.50            -> High Risk     (predicted to churn)
LOW_RISK_UPPER = 0.30

# The 19 customer attributes the model uses, in dataset order.
FEATURE_COLUMNS = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Tenure Months", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing",
    "Payment Method", "Monthly Charges", "Total Charges",
]
NUMERICAL_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges"]
INTERNET_ADDON_COLUMNS = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]

# Allowed ratio  Total Charges / (Tenure Months x Monthly Charges).
# In the dataset it lies between 0.69 and 1.57; 0.5 - 2.0 only rejects clearly inconsistent bills.
CHARGE_RATIO_MIN = 0.5
CHARGE_RATIO_MAX = 2.0

# ---------------------------------------------------------------------------
# 2. MODEL FILES
# ---------------------------------------------------------------------------
# The XGBoost model is saved in XGBoost's own portable format (.ubj, "Universal Binary JSON")
# instead of a Python pickle: it does not depend on Python/pickle internals and can be loaded
# by any XGBoost 2.x/3.x version.
MODEL_FILE = "final_xgboost_churn_model.ubj"
ENCODER_FILE = "final_churn_encoder.pkl"
SCALER_FILE = "final_churn_scaler.pkl"
NUMERICAL_FILE = "numerical_columns.json"
CATEGORICAL_FILE = "categorical_columns.json"
METADATA_FILE = "model_metadata.json"
MANIFEST_FILE = "manifest.json"
HASHED_FILES = [MODEL_FILE, ENCODER_FILE, SCALER_FILE]  # binary files: a changed byte = corrupted copy

REBUILD_HELP = ("To rebuild all model files from the dataset, open a terminal in the project folder and run:  "
                "python train_model.py")


class ArtifactError(RuntimeError):
    """Raised when the saved model files are missing, damaged or do not belong together."""


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(models_dir):
    """Record the SHA-256 checksum of every binary model file (called after training)."""
    manifest = {
        "note": "SHA-256 checksums of the binary model files. If a file is damaged (for example by a "
                "zip/git tool changing line endings) the app refuses to load it.",
        "sha256": {name: file_sha256(os.path.join(models_dir, name)) for name in HASHED_FILES},
    }
    with open(os.path.join(models_dir, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_manifest(models_dir):
    """Return a list of problems (empty list = all binary files are intact)."""
    problems = []
    path = os.path.join(models_dir, MANIFEST_FILE)
    if not os.path.exists(path):
        return [f"{MANIFEST_FILE} is missing"]
    with open(path, encoding="utf-8") as f:
        expected = json.load(f).get("sha256", {})
    for name in HASHED_FILES:
        file_path = os.path.join(models_dir, name)
        if not os.path.exists(file_path):
            problems.append(f"{name} is missing")
        elif file_sha256(file_path) != expected.get(name):
            problems.append(f"{name} is damaged (checksum does not match the manifest)")
    return problems


def load_artifacts(models_dir):
    """Load the model, encoder, scaler, column lists and metadata from `models_dir`.

    Runs integrity and consistency checks so a damaged or mismatched set of files fails with a
    clear message instead of silently producing wrong predictions.
    """
    models_dir = str(models_dir)
    problems = verify_manifest(models_dir)
    if problems:
        raise ArtifactError("Model files failed the integrity check: " + "; ".join(problems) + ". " + REBUILD_HELP)

    try:
        model = XGBClassifier()
        model.load_model(os.path.join(models_dir, MODEL_FILE))
        art = {
            "model": model,
            "encoder": joblib.load(os.path.join(models_dir, ENCODER_FILE)),
            "scaler": joblib.load(os.path.join(models_dir, SCALER_FILE)),
        }
        for key, name in (("numerical_columns", NUMERICAL_FILE), ("categorical_columns", CATEGORICAL_FILE),
                          ("metadata", METADATA_FILE)):
            with open(os.path.join(models_dir, name), encoding="utf-8") as f:
                art[key] = json.load(f)
    except Exception as error:  # damaged file, wrong library version ...
        raise ArtifactError(f"Could not load the model files ({type(error).__name__}: {error}). " + REBUILD_HELP) from error

    names = list(art["encoder"].get_feature_names_out(art["categorical_columns"]))
    expected_names = art["numerical_columns"] + names
    if art["model"].n_features_in_ != len(expected_names):
        raise ArtifactError(
            f"The model expects {art['model'].n_features_in_} features but the encoder/scaler produce "
            f"{len(expected_names)}. The files do not belong together. " + REBUILD_HELP)
    if list(art["metadata"]["feature_names"]) != expected_names:
        raise ArtifactError("Feature names in model_metadata.json do not match the saved encoder. " + REBUILD_HELP)
    return art


def allowed_values(art):
    """{column name: list of allowed text values} for the 16 categorical inputs."""
    return {col: [str(v) for v in cats]
            for col, cats in zip(art["categorical_columns"], art["encoder"].categories_)}


# ---------------------------------------------------------------------------
# 3. VALIDATION OF ONE CUSTOMER
# ---------------------------------------------------------------------------
def validate_customer(customer, numeric_ranges):
    """Check one customer (a dict) for impossible or out-of-range values.

    Returns a list of error messages; an empty list means the input is valid.
    `numeric_ranges` looks like {"Tenure Months": [0, 72], ...} (min/max found in the dataset).
    """
    errors = []

    # Rule 1: phone service and multiple lines must agree.
    if customer["Phone Service"] == "No" and customer["Multiple Lines"] != "No phone service":
        errors.append("Phone Service is 'No', so Multiple Lines must be 'No phone service'.")
    if customer["Phone Service"] == "Yes" and customer["Multiple Lines"] == "No phone service":
        errors.append("Phone Service is 'Yes', so Multiple Lines cannot be 'No phone service'.")

    # Rule 2: internet add-on services only exist if the customer has internet.
    has_internet = customer["Internet Service"] != "No"
    for col in INTERNET_ADDON_COLUMNS:
        if not has_internet and customer[col] != "No internet service":
            errors.append(f"Internet Service is 'No', so {col} must be 'No internet service'.")
        if has_internet and customer[col] == "No internet service":
            errors.append(f"The customer has internet, so {col} cannot be 'No internet service'.")

    # Rule 3: numeric values must lie inside the range found in the dataset.
    for col, (low, high) in numeric_ranges.items():
        value = customer[col]
        if value < low or value > high:
            errors.append(f"{col} = {value} is outside the range of the dataset ({low} to {high}).")

    # Rule 4: billing amounts must be consistent with each other.
    tenure, monthly, total = customer["Tenure Months"], customer["Monthly Charges"], customer["Total Charges"]
    if tenure == 0 and total != 0:
        errors.append("Tenure is 0 months, so Total Charges must be 0 (no billing yet).")
    if tenure > 0:
        if total <= 0:
            errors.append("Tenure is above 0 months, so Total Charges must be greater than 0.")
        elif monthly > 0:
            ratio = total / (tenure * monthly)
            if ratio < CHARGE_RATIO_MIN or ratio > CHARGE_RATIO_MAX:
                errors.append(
                    f"Total Charges ({total:.2f}) is inconsistent with Tenure x Monthly Charges "
                    f"({tenure * monthly:.2f}). Expected roughly between "
                    f"{CHARGE_RATIO_MIN * tenure * monthly:.0f} and {CHARGE_RATIO_MAX * tenure * monthly:.0f}.")
    return errors


# ---------------------------------------------------------------------------
# 4. PREPROCESSING (identical to training)
# ---------------------------------------------------------------------------
def build_model_input(df, art):
    """Raw customers (DataFrame) -> the numeric matrix the model expects.

    Training used:  [ scaled numeric columns | one-hot encoded categorical columns ]
    The same order and the same fitted scaler/encoder are used here.
    """
    numeric_part = art["scaler"].transform(df[art["numerical_columns"]])
    categorical_part = art["encoder"].transform(df[art["categorical_columns"]])
    return np.hstack([numeric_part, categorical_part])


def predict_probabilities(df, art):
    """Churn probability for every row of a DataFrame of raw customers."""
    return art["model"].predict_proba(build_model_input(df, art))[:, 1]


# ---------------------------------------------------------------------------
# 5. RISK LEVEL AND ONE-CUSTOMER PREDICTION
# ---------------------------------------------------------------------------
def risk_category(probability):
    """Map a churn probability to a risk level (see thresholds at the top)."""
    if probability > DECISION_THRESHOLD:
        return "High Risk"
    if probability >= LOW_RISK_UPPER:
        return "Medium Risk"
    return "Low Risk"


def risk_message(risk):
    """Plain-language sentence that always agrees with the risk level."""
    return {
        "High Risk": "Likely to Churn",
        "Medium Risk": "Likely to Stay, but with elevated churn risk",
        "Low Risk": "Likely to Stay",
    }[risk]


def predicted_label(probability):
    """'Likely to Churn' / 'Likely to Stay' using the decision threshold."""
    return "Likely to Churn" if probability > DECISION_THRESHOLD else "Likely to Stay"


def predict_customer(customer, art):
    """Predict for ONE customer (dict). Returns probability, predicted class and risk."""
    probability = float(predict_probabilities(pd.DataFrame([customer]), art)[0])
    risk = risk_category(probability)
    return {
        "probability": probability,
        "predicted_churn": probability > DECISION_THRESHOLD,
        "label": predicted_label(probability),
        "risk": risk,
        "message": risk_message(risk),
    }


# ---------------------------------------------------------------------------
# 6. BATCH PREDICTION FROM A CSV FILE
# ---------------------------------------------------------------------------
MAX_BATCH_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_BATCH_ROWS = 10_000
OPTIONAL_ID_COLUMN = "CustomerID"   # optional; carried through to the results if present


@dataclass
class BatchOutcome:
    fatal: str = None                                   # message if the whole file cannot be used
    notes: list = field(default_factory=list)           # non-fatal information for the user
    results: pd.DataFrame = None                        # valid rows + predictions
    rejected: pd.DataFrame = None                       # rows that were NOT predicted, with the reason
    n_rows: int = 0


def template_dataframe():
    """Three example rows that show the required format. They are SYNTHETIC (invented) and
    are only meant to illustrate the file layout; they are not part of any dataset."""
    rows = [
        ["EXAMPLE-001", "Female", "No", "No", "No", 3, "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes",
         "Month-to-month", "Yes", "Electronic check", 89.10, 267.30],
        ["EXAMPLE-002", "Male", "No", "Yes", "Yes", 34, "Yes", "Yes", "DSL", "Yes", "Yes", "No", "Yes", "No", "No",
         "One year", "No", "Mailed check", 58.20, 1978.80],
        ["EXAMPLE-003", "Female", "Yes", "Yes", "No", 60, "Yes", "No", "No", "No internet service", "No internet service",
         "No internet service", "No internet service", "No internet service", "No internet service",
         "Two year", "No", "Bank transfer (automatic)", 20.25, 1215.00],
    ]
    return pd.DataFrame(rows, columns=[OPTIONAL_ID_COLUMN] + FEATURE_COLUMNS)


def to_csv_bytes(df):
    """CSV as bytes with a UTF-8 BOM so that Excel on Windows opens it correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


def _to_number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def run_batch(csv_bytes, art):
    """Validate an uploaded CSV and predict every valid row.

    Nothing is accepted silently: missing columns stop the run, unused columns are reported,
    and every rejected row is listed with the reason.
    """
    out = BatchOutcome()
    if len(csv_bytes) == 0:
        out.fatal = "The file is empty."
        return out
    if len(csv_bytes) > MAX_BATCH_BYTES:
        out.fatal = f"The file is larger than {MAX_BATCH_BYTES // (1024 * 1024)} MB."
        return out

    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as error:
        out.fatal = f"The file could not be read as a CSV ({type(error).__name__}). Please upload a plain .csv file."
        return out

    # ---- columns ------------------------------------------------------------------
    canonical = {c.lower(): c for c in FEATURE_COLUMNS + [OPTIONAL_ID_COLUMN]}
    rename, seen = {}, {}
    for col in raw.columns:
        key = str(col).strip().lower()
        if key in canonical:
            if canonical[key] in seen:
                out.fatal = f"Column '{canonical[key]}' appears more than once ('{seen[canonical[key]]}' and '{col}')."
                return out
            seen[canonical[key]] = col
            rename[col] = canonical[key]
    unused = [str(c) for c in raw.columns if c not in rename]
    renamed = [f"'{old}' -> '{new}'" for old, new in rename.items() if old != new]
    raw = raw.rename(columns=rename)

    missing = [c for c in FEATURE_COLUMNS if c not in raw.columns]
    if missing:
        out.fatal = ("Required column(s) missing: " + ", ".join(missing) +
                     ". Download the template to see the exact column names.")
        return out
    if renamed:
        out.notes.append("Column names matched after ignoring capital letters/spaces: " + ", ".join(renamed) + ".")
    if unused:
        out.notes.append("These columns are not used by the model and were ignored: " + ", ".join(unused) + ".")

    if len(raw) == 0:
        out.fatal = "The file has a header but no customer rows."
        return out
    if len(raw) > MAX_BATCH_ROWS:
        out.fatal = f"The file has {len(raw):,} rows; the limit is {MAX_BATCH_ROWS:,}."
        return out
    out.n_rows = len(raw)

    # ---- row-by-row validation -------------------------------------------------------
    allowed = allowed_values(art)
    ranges = art["metadata"]["numeric_ranges"]
    has_id = OPTIONAL_ID_COLUMN in raw.columns
    good_rows, bad_rows = [], []
    blank_total_fixed = 0

    for position, record in enumerate(raw.to_dict("records")):
        line = position + 2                                   # +1 for the header, +1 because lines start at 1
        problems, customer = [], {}
        for col in FEATURE_COLUMNS:
            text = str(record[col]).strip()
            if col in NUMERICAL_COLUMNS:
                if text == "" and col == "Total Charges":      # decided below, once the tenure is known
                    customer[col] = None
                    continue
                value = _to_number(text)
                if value is None:
                    shown = text if text else "blank"
                    problems.append(f"{col} is missing or not a number ({shown})")
                elif col == "Tenure Months" and value != int(value):
                    problems.append(f"Tenure Months must be a whole number ({text})")
                else:
                    customer[col] = int(value) if col == "Tenure Months" else value
            else:
                if text not in allowed[col]:
                    shown = text if text else "(blank)"
                    problems.append(f"{col} = '{shown}' is not allowed (allowed: {', '.join(allowed[col])})")
                else:
                    customer[col] = text

        # blank Total Charges is only acceptable for brand-new customers (tenure 0) - same rule as training
        if "Total Charges" in customer and customer["Total Charges"] is None:
            if customer.get("Tenure Months") == 0:
                customer["Total Charges"] = 0.0
                blank_total_fixed += 1
            else:
                problems.append("Total Charges is blank (only allowed when Tenure Months is 0)")

        if not problems:
            problems.extend(validate_customer(customer, ranges))

        entry = {"Source_Row": line}
        if has_id:
            entry[OPTIONAL_ID_COLUMN] = str(record[OPTIONAL_ID_COLUMN]).strip()
        if problems:
            entry["Problems"] = "; ".join(problems)
            for col in FEATURE_COLUMNS:
                entry[col] = record[col]
            bad_rows.append(entry)
        else:
            entry.update(customer)
            good_rows.append(entry)

    if blank_total_fixed:
        out.notes.append(f"{blank_total_fixed} blank 'Total Charges' value(s) were set to 0 because the customer has "
                         "Tenure Months = 0 (the same rule used in training).")

    lead = ["Source_Row"] + ([OPTIONAL_ID_COLUMN] if has_id else [])
    out.rejected = pd.DataFrame(bad_rows, columns=lead + ["Problems"] + FEATURE_COLUMNS)
    result_columns = lead + FEATURE_COLUMNS + ["Churn_Probability", "Churn_Probability_Percent",
                                               "Predicted_Churn", "Risk_Category"]
    if not good_rows:
        out.results = pd.DataFrame(columns=result_columns)
        return out

    valid = pd.DataFrame(good_rows, columns=lead + FEATURE_COLUMNS)
    probabilities = predict_probabilities(valid, art)
    valid["Churn_Probability"] = np.round(probabilities, 4)
    valid["Churn_Probability_Percent"] = np.round(probabilities * 100, 1)
    valid["Predicted_Churn"] = [predicted_label(p) for p in probabilities]
    valid["Risk_Category"] = [risk_category(p) for p in probabilities]
    out.results = valid[result_columns]
    return out
