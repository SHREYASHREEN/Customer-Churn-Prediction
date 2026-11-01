"""
train_model.py - rebuild every model file and result table from the dataset.

Run from the project folder:

    python train_model.py              # about 1 minute (includes cross-validation)
    python train_model.py --skip-cv    # faster, but cross_validation_results.csv etc. are NOT updated

This is the SAME pipeline as notebook/customer_churn_final.ipynb (same code, same random seed):
the notebook explains each step and draws the figures, this script is the quick one-command rebuild.
tests/verify_models.py checks that both give identical numbers.

What it writes
--------------
models/   final_xgboost_churn_model.ubj (native XGBoost format), final_churn_encoder.pkl, final_churn_scaler.pkl,
          numerical_columns.json, categorical_columns.json, model_metadata.json, manifest.json
outputs/  model_comparison.csv, classification_reports.txt, test_predictions.csv, cross_validation_results.csv,
          xgboost_vs_logistic_paired_cv.csv, model_selection_evidence.csv, threshold_analysis.csv,
          risk_band_check.csv, test_metric_confidence_intervals.csv, xgboost_feature_importance.csv,
          shap_mean_abs_importance.csv
"""

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import churn_utils  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = churn_utils.DECISION_THRESHOLD


def to_py(v):
    return int(v) if float(v).is_integer() else float(v)


def main():
    parser = argparse.ArgumentParser(description="Rebuild the churn model files and result tables.")
    parser.add_argument("--output-root", default=str(ROOT), help="folder that will receive models/ and outputs/ (default: project folder)")
    parser.add_argument("--skip-cv", action="store_true", help="skip the (slower) cross-validation step")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    models_dir, output_dir = out_root / "models", out_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = ROOT / "data" / "Telco_customer_churn.xlsx"

    # ------------------------------------------------------------------ 1. data
    print("1/8 Loading data ...")
    df = pd.read_excel(data_file)
    assert df.shape == (7043, 33), "Unexpected dataset shape - is this the original file?"
    model_df = df[churn_utils.FEATURE_COLUMNS + ["Churn Label"]].copy()
    model_df["Total Charges"] = pd.to_numeric(model_df["Total Charges"], errors="coerce")
    blank = model_df["Total Charges"].isna()
    assert (model_df.loc[blank, "Tenure Months"] == 0).all()      # all blanks are brand-new customers
    model_df["Total Charges"] = model_df["Total Charges"].fillna(0.0)

    # ------------------------------------------------------------------ 2. split
    X = model_df.drop(columns="Churn Label")
    y = (model_df["Churn Label"] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    NUMERICAL_COLUMNS = list(churn_utils.NUMERICAL_COLUMNS)
    CATEGORICAL_COLUMNS = [c for c in X.columns if c not in NUMERICAL_COLUMNS]
    assert len(CATEGORICAL_COLUMNS) == 16

    # ------------------------------------------------------------------ 3. preprocessing (fitted on TRAIN only)
    print("2/8 Preprocessing ...")
    encoder = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
    scaler = StandardScaler()
    X_train_cat = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS])
    X_test_cat = encoder.transform(X_test[CATEGORICAL_COLUMNS])
    X_train_num = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test_num = scaler.transform(X_test[NUMERICAL_COLUMNS])
    X_train_processed = np.hstack([X_train_num, X_train_cat])
    X_test_processed = np.hstack([X_test_num, X_test_cat])
    feature_names = NUMERICAL_COLUMNS + list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    assert X_train_processed.shape[1] == len(feature_names) == 46

    # ------------------------------------------------------------------ 4. models
    print("3/8 Training Logistic Regression, Random Forest and XGBoost ...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
    }
    for m in models.values():
        m.fit(X_train_processed, y_train)
    probs = {n: m.predict_proba(X_test_processed)[:, 1] for n, m in models.items()}
    preds = {n: (p > THRESHOLD).astype(int) for n, p in probs.items()}
    for n, m in models.items():
        assert (preds[n] == m.predict(X_test_processed)).all()

    # ------------------------------------------------------------------ 5. evaluation
    print("4/8 Evaluating on the test set ...")

    def evaluate(name, y_true, prob, pred):
        return {"Model": name, "Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred),
                "Recall": recall_score(y_true, pred), "F1 Score": f1_score(y_true, pred),
                "ROC-AUC": roc_auc_score(y_true, prob)}

    model_comparison = pd.DataFrame([evaluate(n, y_test, probs[n], preds[n]) for n in models])
    model_comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    report_text = ""
    for n in models:
        report_text += f"=== {n} ===\n" + classification_report(
            y_test, preds[n], target_names=["No Churn", "Churn"], digits=4) + "\n"
    (output_dir / "classification_reports.txt").write_text(report_text, encoding="utf-8")

    pd.DataFrame({"row_index": X_test.index, "actual": y_test.to_numpy(),
                  "prob_logistic_regression": probs["Logistic Regression"],
                  "prob_random_forest": probs["Random Forest"],
                  "prob_xgboost": probs["XGBoost"]}).to_csv(output_dir / "test_predictions.csv", index=False)

    # ------------------------------------------------------------------ 6. cross-validation
    if args.skip_cv:
        print("5/8 Cross-validation SKIPPED (--skip-cv): cross_validation_results.csv etc. were not updated.")
    else:
        print("5/8 Cross-validation (5 folds x 3 repeats, 3 models) ...")

        def make_pipeline(model):
            prep = ColumnTransformer([("num", StandardScaler(), NUMERICAL_COLUMNS),
                                      ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)])
            return Pipeline([("prep", prep), ("model", model)])

        cv_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
        }
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
        scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1",
                   "roc_auc": "roc_auc", "neg_log_loss": "neg_log_loss", "neg_brier": "neg_brier_score"}
        cv_raw = {n: cross_validate(make_pipeline(m), X, y, cv=cv, scoring=scoring) for n, m in cv_models.items()}

        rows = []
        for n, r in cv_raw.items():
            row = {"Model": n}
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                row[f"{key}_mean"] = r[f"test_{key}"].mean()
                row[f"{key}_std"] = r[f"test_{key}"].std()
            row["log_loss_mean"] = -r["test_neg_log_loss"].mean()
            row["log_loss_std"] = r["test_neg_log_loss"].std()
            row["brier_mean"] = -r["test_neg_brier"].mean()
            row["brier_std"] = r["test_neg_brier"].std()
            rows.append(row)
        cv_results = pd.DataFrame(rows)
        cv_results.to_csv(output_dir / "cross_validation_results.csv", index=False)

        paired_rows = []
        for key, label in [("roc_auc", "ROC-AUC gain"), ("f1", "F1 gain"), ("recall", "Recall gain"),
                           ("neg_log_loss", "Log-loss reduction (lower loss = better)")]:
            diff = cv_raw["XGBoost"][f"test_{key}"] - cv_raw["Logistic Regression"][f"test_{key}"]
            paired_rows.append({"Metric": label, "Mean difference (XGB - LR)": diff.mean(),
                                "Std of difference": diff.std(),
                                "Folds where XGBoost is better (of 15)": int((diff > 0).sum())})
        pd.DataFrame(paired_rows).to_csv(output_dir / "xgboost_vs_logistic_paired_cv.csv", index=False)

        higher_better = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        test_best = {m: model_comparison.loc[model_comparison[m].idxmax(), "Model"] for m in higher_better}
        cv_map = {"Accuracy": "accuracy_mean", "Precision": "precision_mean", "Recall": "recall_mean",
                  "F1 Score": "f1_mean", "ROC-AUC": "roc_auc_mean"}
        cv_best = {m: cv_results.loc[cv_results[c].idxmax(), "Model"] for m, c in cv_map.items()}
        cv_best["Log-loss (lower better)"] = cv_results.loc[cv_results["log_loss_mean"].idxmin(), "Model"]
        cv_best["Brier score (lower better)"] = cv_results.loc[cv_results["brier_mean"].idxmin(), "Model"]
        test_best["Log-loss (lower better)"] = "-"
        test_best["Brier score (lower better)"] = "-"
        selection = pd.DataFrame({"Best on single test split": pd.Series(test_best),
                                  "Best in 15-run cross-validation": pd.Series(cv_best)})
        selection.index.name = "Measure"
        selection.to_csv(output_dir / "model_selection_evidence.csv")

    # ------------------------------------------------------------------ 7. final model analysis
    print("6/8 Threshold table, risk bands, confidence intervals ...")
    FINAL_MODEL_NAME = "XGBoost"
    final_model = models[FINAL_MODEL_NAME]
    final_probs = probs[FINAL_MODEL_NAME]

    rows = []
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        p = (final_probs > t).astype(int)
        rows.append({"Threshold": t, "Customers flagged": int(p.sum()), "Precision": precision_score(y_test, p),
                     "Recall": recall_score(y_test, p), "F1 Score": f1_score(y_test, p)})
    pd.DataFrame(rows).to_csv(output_dir / "threshold_analysis.csv", index=False)

    bands = pd.Series([churn_utils.risk_category(p) for p in final_probs], index=y_test.index, name="Risk band")
    risk_check = (pd.DataFrame({"Risk band": bands, "churned": y_test})
                  .groupby("Risk band")["churned"].agg(Customers="count", Actual_churners="sum", Actual_churn_rate="mean")
                  .reindex(["Low Risk", "Medium Risk", "High Risk"]))
    risk_check.to_csv(output_dir / "risk_band_check.csv")

    rng = np.random.default_rng(RANDOM_STATE)
    y_arr = y_test.to_numpy()
    boot = {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": [], "ROC-AUC": []}
    for _ in range(1000):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        yb, pb = y_arr[idx], final_probs[idx]
        cb = (pb > THRESHOLD).astype(int)
        boot["Accuracy"].append(accuracy_score(yb, cb))
        boot["Precision"].append(precision_score(yb, cb))
        boot["Recall"].append(recall_score(yb, cb))
        boot["F1 Score"].append(f1_score(yb, cb))
        boot["ROC-AUC"].append(roc_auc_score(yb, pb))
    pd.DataFrame({"Metric": list(boot),
                  "Test-set value": [float(model_comparison.loc[model_comparison["Model"] == FINAL_MODEL_NAME, m].iloc[0]) for m in boot],
                  "95% CI lower": [np.percentile(v, 2.5) for v in boot.values()],
                  "95% CI upper": [np.percentile(v, 97.5) for v in boot.values()]}
                 ).to_csv(output_dir / "test_metric_confidence_intervals.csv", index=False)

    print("7/8 Feature importance and SHAP ...")
    (pd.DataFrame({"Feature": feature_names, "Importance": final_model.feature_importances_})
       .sort_values("Importance", ascending=False).reset_index(drop=True)
       .to_csv(output_dir / "xgboost_feature_importance.csv", index=False))
    shap_values = shap.TreeExplainer(final_model).shap_values(X_test_processed)
    assert shap_values.shape == X_test_processed.shape
    (pd.DataFrame({"Feature": feature_names, "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0)})
       .sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True)
       .to_csv(output_dir / "shap_mean_abs_importance.csv", index=False))

    # ------------------------------------------------------------------ 8. save model files
    print("8/8 Saving model files ...")
    final_model.save_model(str(models_dir / churn_utils.MODEL_FILE))            # native XGBoost format
    joblib.dump(encoder, models_dir / churn_utils.ENCODER_FILE)
    joblib.dump(scaler, models_dir / churn_utils.SCALER_FILE)
    (models_dir / churn_utils.NUMERICAL_FILE).write_text(json.dumps(NUMERICAL_COLUMNS, indent=2), encoding="utf-8")
    (models_dir / churn_utils.CATEGORICAL_FILE).write_text(json.dumps(CATEGORICAL_COLUMNS, indent=2), encoding="utf-8")

    metadata = {
        "pipeline": "Experiment A (canonical): StandardScaler + one-hot (drop=None), 46 features, blank Total Charges set to 0",
        "final_model": FINAL_MODEL_NAME,
        "model_format": "xgboost-ubj (native XGBoost format)",
        "model_file": churn_utils.MODEL_FILE,
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "decision_threshold": churn_utils.DECISION_THRESHOLD,
        "low_risk_upper": churn_utils.LOW_RISK_UPPER,
        "numerical_columns": NUMERICAL_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "feature_names": feature_names,
        "numeric_ranges": {c: [to_py(X[c].min()), to_py(X[c].max())] for c in NUMERICAL_COLUMNS},
        "dataset": {"file": "data/Telco_customer_churn.xlsx", "rows": int(len(df)),
                    "md5": hashlib.md5(data_file.read_bytes()).hexdigest()},
        "test_metrics": {k: float(v) for k, v in
                         model_comparison[model_comparison["Model"] == FINAL_MODEL_NAME].iloc[0].drop("Model").items()},
        "library_versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                             "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__},
    }
    (models_dir / churn_utils.METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    churn_utils.write_manifest(str(models_dir))

    # ---- self-check: reload from disk exactly as the app does and compare
    art = churn_utils.load_artifacts(str(models_dir))
    X_check = churn_utils.build_model_input(X_test, art)
    assert np.allclose(X_check, X_test_processed), "App preprocessing differs from training preprocessing!"
    assert np.allclose(art["model"].predict_proba(X_check)[:, 1], final_probs), "Reloaded model differs!"
    print("\nOK - saved files reload correctly and reproduce the test-set probabilities.")
    print(model_comparison.round(4).to_string(index=False))
    print(f"\nFiles written to: {models_dir}  and  {output_dir}")


if __name__ == "__main__":
    main()
