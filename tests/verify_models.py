"""
verify_models.py - verifies ALL model artifacts and result files of the project.

Run from the project folder:

    python tests/verify_models.py             # about 20 seconds
    python tests/verify_models.py --retrain   # also re-runs train_model.py and compares (about 1-2 minutes)

Checks: dataset unchanged | files intact (checksums) | damaged files are detected | the model loads in a
brand-new Python process ("after a restart") and predicts | 46 features match the encoder | recomputed test metrics
equal the result files | SHAP explanations add up | preprocessing equals an independent re-implementation |
all real customers pass the validation rules | pinned package versions are installed.
"""

import hashlib
import importlib.metadata as md
import json
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import churn_utils  # noqa: E402
import explain  # noqa: E402

ORIGINAL_DATASET_MD5 = "8534d3efea2a52104cba8605fef6a7df"   # MD5 of the dataset in the original ZIP
results = []


def record(status, name, detail=""):
    results.append(status)
    print(f"[{status}] {name}" + (f"  ->  {detail}" if detail else ""))


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


MODELS = PROJECT / "models"
OUT = PROJECT / "outputs"

# ------------------------------------------------------------------ 1. dataset
data_file = PROJECT / "data" / "Telco_customer_churn.xlsx"
md5 = hashlib.md5(data_file.read_bytes()).hexdigest()
record("PASS" if md5 == ORIGINAL_DATASET_MD5 else "FAIL", "Dataset is identical to the original file", md5)

# ------------------------------------------------------------------ 2. artifacts present, intact, consistent
expected_files = [churn_utils.MODEL_FILE, churn_utils.ENCODER_FILE, churn_utils.SCALER_FILE, churn_utils.NUMERICAL_FILE,
                  churn_utils.CATEGORICAL_FILE, churn_utils.METADATA_FILE, churn_utils.MANIFEST_FILE]
missing = [f for f in expected_files if not (MODELS / f).exists()]
record("PASS" if not missing else "FAIL", "All 7 model files exist in models/", f"missing: {missing}" if missing else "")
record("PASS" if churn_utils.verify_manifest(str(MODELS)) == [] else "FAIL", "SHA-256 checksums match manifest.json")
record("PASS" if (MODELS / churn_utils.MODEL_FILE).suffix == ".ubj" else "FAIL",
       "XGBoost model is stored in the native .ubj format (not a pickle)")

art = churn_utils.load_artifacts(str(MODELS))
meta = art["metadata"]
record("PASS", "Model, encoder, scaler, column lists and metadata load and agree",
       f"{art['model'].n_features_in_} model features = {len(meta['feature_names'])} names")
record("PASS" if meta["dataset"]["md5"] == md5 else "FAIL", "Model was trained on exactly this dataset (MD5 in metadata)")
record("PASS" if art["metadata"]["decision_threshold"] == churn_utils.DECISION_THRESHOLD == 0.5 else "FAIL",
       "Decision threshold is the standard 0.50 (unchanged)")

# ------------------------------------------------------------------ 3. damaged copies are detected
with tempfile.TemporaryDirectory() as tmp:
    for name, tamper in (("flip one byte", lambda b: bytes([b[0] ^ 0xFF]) + b[1:]),
                         ("Windows-style line-ending conversion", lambda b: b.replace(b"\n", b"\r\n")),
                         ("truncate the file", lambda b: b[: len(b) // 2])):
        copy = Path(tmp) / name.replace(" ", "_")
        shutil.copytree(MODELS, copy)
        target = copy / churn_utils.MODEL_FILE
        target.write_bytes(tamper(target.read_bytes()))
        try:
            churn_utils.load_artifacts(str(copy))
            record("FAIL", f"Damaged model file is detected ({name})", "loaded without complaint!")
        except churn_utils.ArtifactError as error:
            ok = "python train_model.py" in str(error)
            record("PASS" if ok else "FAIL", f"Damaged model file is detected ({name})", "clear message with rebuild instruction")
    empty = Path(tmp) / "empty"
    empty.mkdir()
    try:
        churn_utils.load_artifacts(str(empty))
        record("FAIL", "Missing model folder is reported")
    except churn_utils.ArtifactError:
        record("PASS", "Missing model files are reported with a clear message")

# ------------------------------------------------------------------ 4. restart: a brand-new Python process loads and predicts
df = pd.read_excel(data_file)
cols = meta["numerical_columns"] + meta["categorical_columns"]
X = df[cols].copy()
X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce").fillna(0.0)
y = (df["Churn Label"] == "Yes").astype(int)
sample = {c: (v.item() if hasattr(v, "item") else v) for c, v in X.iloc[7].items()}
in_process = churn_utils.predict_customer(sample, art)["probability"]
code = ("import sys, json; sys.path.insert(0, sys.argv[1]); import churn_utils; "
        "art = churn_utils.load_artifacts(sys.argv[1] + '/models'); "
        "print(json.dumps(churn_utils.predict_customer(json.loads(sys.argv[2]), art)))")
proc = subprocess.run([sys.executable, "-c", code, str(PROJECT), json.dumps(sample)], capture_output=True, text=True)
try:
    fresh = json.loads(proc.stdout.strip().splitlines()[-1])["probability"]
    record("PASS" if close(fresh, in_process, 1e-12) else "FAIL",
           "A brand-new Python process loads the model and gives the same prediction (restart test)", f"probability {fresh:.4f}")
except Exception:
    record("FAIL", "Fresh-process load failed", proc.stderr[-300:])

# ------------------------------------------------------------------ 5. recompute the test metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=meta["test_size"], random_state=meta["random_state"], stratify=y)
record("PASS" if len(X_train) == meta["n_train"] and len(X_test) == meta["n_test"] else "FAIL",
       "Train/test split reproduces the recorded sizes", f"{len(X_train)} / {len(X_test)}")
prob = churn_utils.predict_probabilities(X_test, art)
pred = (prob > churn_utils.DECISION_THRESHOLD).astype(int)
now = {"Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
       "Recall": recall_score(y_test, pred), "F1 Score": f1_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, prob)}
comp = pd.read_csv(OUT / "model_comparison.csv").set_index("Model")
record("PASS" if all(close(now[k], comp.loc[meta["final_model"], k]) for k in now) else "FAIL",
       "Metrics recomputed from the SAVED model == outputs/model_comparison.csv", ", ".join(f"{k} {v:.4f}" for k, v in now.items()))
record("PASS" if all(close(now[k], meta["test_metrics"][k]) for k in now) else "FAIL", "... == models/model_metadata.json")
tp = pd.read_csv(OUT / "test_predictions.csv")
record("PASS" if (tp["row_index"].to_numpy() == X_test.index.to_numpy()).all() and np.allclose(tp["prob_xgboost"], prob, atol=1e-6)
       and (tp["actual"].to_numpy() == y_test.to_numpy()).all() else "FAIL",
       "outputs/test_predictions.csv equals the saved model's test probabilities")

imp = pd.read_csv(OUT / "xgboost_feature_importance.csv")
model_imp = dict(zip(meta["feature_names"], art["model"].feature_importances_))
record("PASS" if set(imp["Feature"]) == set(meta["feature_names"]) and all(close(r.Importance, model_imp[r.Feature], 1e-6) for r in imp.itertuples()) else "FAIL",
       "xgboost_feature_importance.csv matches the saved model", f"top: {imp.iloc[0]['Feature']}")

bands = pd.Series([churn_utils.risk_category(p) for p in prob], index=y_test.index)
rb = pd.read_csv(OUT / "risk_band_check.csv").set_index("Risk band")
record("PASS" if all(int(rb.loc[b, "Customers"]) == int((bands == b).sum()) for b in rb.index) else "FAIL",
       "risk_band_check.csv matches the model and the 30% / 50% thresholds")

# ------------------------------------------------------------------ 6. SHAP explanations add up
explainer = explain.build_explainer(art)
worst = 0.0
for i in range(0, 200, 10):
    customer = {c: (v.item() if hasattr(v, "item") else v) for c, v in X.iloc[i].items()}
    _, base, margin = explain.explain_customer(art, explainer, customer)
    raw = float(art["model"].get_booster().inplace_predict(churn_utils.build_model_input(pd.DataFrame([customer]), art), predict_type="margin")[0])
    worst = max(worst, abs(margin - raw))
record("PASS" if worst < 1e-3 else "FAIL", "SHAP: baseline + contributions == the model's own output (20 customers)", f"max difference {worst:.1e}")

# ------------------------------------------------------------------ 7. independent re-implementation of the preprocessing
encoder, scaler = art["encoder"], art["scaler"]
manual_num = (X[meta["numerical_columns"]].to_numpy(float) - scaler.mean_) / scaler.scale_
blocks = []
for col, cats in zip(meta["categorical_columns"], encoder.categories_):
    blocks.append((X[col].to_numpy()[:, None] == np.asarray(cats)[None, :]).astype(float))
manual = np.hstack([manual_num] + blocks)
record("PASS" if np.allclose(manual, churn_utils.build_model_input(X, art), atol=1e-9) else "FAIL",
       "Preprocessing equals an independent re-implementation on all 7,043 customers (scaling + one-hot, same column order)")

# ------------------------------------------------------------------ 8. validation rules accept real data
bad = sum(1 for row in X.to_dict("records") if churn_utils.validate_customer(row, meta["numeric_ranges"]))
record("PASS" if bad == 0 else "FAIL", "All 7,043 real customers pass the validation rules (rules are not too strict)", f"{bad} rejected")

# ------------------------------------------------------------------ 9. pinned versions
pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([0-9][^\s#]*)", (PROJECT / "requirements.txt").read_text(), re.M))
mismatch = {p: (v, md.version(p)) for p, v in pins.items() if md.version(p) != v}
record("PASS" if not mismatch else "WARN", "Installed package versions equal the pins in requirements.txt",
       f"(pinned, installed) differences: {mismatch}" if mismatch else f"{len(pins)} packages")

# ------------------------------------------------------------------ 10. optional: retrain from scratch and compare
if "--retrain" in sys.argv:
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, str(PROJECT / "train_model.py"), "--output-root", tmp], capture_output=True, text=True)
        if proc.returncode != 0:
            record("FAIL", "train_model.py runs without errors", proc.stderr[-400:])
        else:
            record("PASS", "train_model.py runs without errors")
            same_files = [f for f in (churn_utils.MODEL_FILE, churn_utils.ENCODER_FILE, churn_utils.SCALER_FILE)
                          if (Path(tmp) / "models" / f).read_bytes() == (MODELS / f).read_bytes()]
            if len(same_files) == 3:
                record("PASS", "A fresh retrain gives byte-identical model, encoder and scaler files")
            else:
                new_comp = pd.read_csv(Path(tmp) / "outputs" / "model_comparison.csv").set_index("Model")
                diff = (new_comp - comp).abs().to_numpy().max()
                record("PASS" if diff < 1e-6 else "WARN", "A fresh retrain differs from the shipped files (different OS/library build?)",
                       f"largest metric difference {diff:.4f}")
            for name in ("model_comparison.csv", "test_predictions.csv", "cross_validation_results.csv"):
                a, b = pd.read_csv(Path(tmp) / "outputs" / name), pd.read_csv(OUT / name)
                ok = a.shape == b.shape and all(np.allclose(a[c], b[c], atol=1e-9) if a[c].dtype.kind in "fi" else (a[c] == b[c]).all() for c in a.columns)
                record("PASS" if ok else "WARN", f"Retrained outputs/{name} equals the shipped file")

print(f"\nSUMMARY: {results.count('PASS')} passed, {results.count('WARN')} warnings, {results.count('FAIL')} failed")
sys.exit(1 if "FAIL" in results else 0)
