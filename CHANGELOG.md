# CHANGELOG

## Version for the mini-project exhibition (this package)

### Audit of the uploaded `Customer_Churn_Project_zip.zip` (what was found, by running the files)
| Finding | Evidence |
|---|---|
| **No authentication anywhere in the ZIP.** `app.py` was the original 345-line app, byte-identical to the very first submission. | Searched all files for login/register/hash/user-store code: none. MD5 comparison with the archived original. |
| The ZIP was a **hybrid**: corrected `models/`, `churn_utils.py` and tests, but the *original* `app.py`, notebook (3 `NameError` cells), `requirements.txt` (unpinned, no `openpyxl`) and a stale `outputs/model_comparison.csv` (XGBoost 0.8105 while the saved model/metadata said 0.8133). | MD5 comparison; the shipped `verify_project.py` reported 2 failures + 3 warnings on the ZIP as uploaded. |
| Root-level and `notebook/` pickles are the **30-feature** model (Experiment B). | They load and report 30 features. They were **not** used. |
| "XGBoostError: input stream corrupted" **could not be reproduced**: every pickle in the ZIP loaded with XGBoost 3.4.1 on Linux. | Load test of 5 model files. Probable causes on a user machine are a damaged copy (e.g. line-ending conversion) or a version mismatch; both are now handled (see below). |

### Changes
| Area | Change | Files |
|---|---|---|
| **Model serialization** | XGBoost model now saved in XGBoost's **native `.ubj` format** (no pickle). SHA-256 checksums in `manifest.json`; the loader refuses damaged/mismatched files with a clear message and rebuild instruction. Encoder/scaler stay as sklearn objects (versions pinned). Column lists are now JSON. | `churn_utils.py`, `models/*` |
| **Reproducible rebuild** | New `train_model.py` rebuilds all model files and result tables in about a minute. The notebook and the script produce **byte-identical** model files and identical CSVs (tested). | `train_model.py`, notebook |
| **Notebook** | Saves the native model, JSON columns, manifest, metadata with dataset checksum and `test_predictions.csv`; runs top to bottom (26/26 code cells, 0 errors). | `notebook/customer_churn_final.ipynb` |
| **Authentication (new)** | Register, login, logout; PBKDF2-SHA256 salted hashes; lockout; all pages protected. | `auth.py`, `app.py` |
| **Application (rebuilt)** | 7 pages: dashboard, predict, batch, risk & retention, model performance, explainable AI, about. Sidebar navigation, themed, native metric cards, Altair charts. | `app.py`, `views.py`, `ui.py`, `.streamlit/config.toml` |
| **Prediction page** | Form in 3 sections, 3 real-customer example buttons, dependent fields locked, automatic total charges, validation, probability bar, SHAP explanation grouped by original feature, retention recommendations with dataset evidence. | `views.py`, `explain.py`, `recommendations.py` |
| **Batch prediction (new)** | Strict CSV validation (columns, categories, numbers, consistency), per-row rejection reasons, summary, downloads, template and sample files. | `churn_utils.run_batch`, `sample_data/` |
| **Risk levels** | Low < 30 %, Medium 30-50 % inclusive, High > 50 %; threshold 0.50 unchanged; labels always agree with the level. | `churn_utils.py` |
| **Dependencies** | Pinned versions (added `altair`); installation tested in a fresh virtual environment. | `requirements.txt` |
| **Tests (new/extended)** | `run_all_tests.py` + `verify_models.py`, `test_auth.py`, `test_batch.py`, `test_app.py`. | `tests/` |
| **Housekeeping** | `.gitattributes` keeps model files binary; `.gitignore` excludes the user database. Everything from the earlier uploads is in `archive/` (unchanged, not used). Stale/duplicate files removed from the active project. | `archive/` |
| **Docs** | New README and `docs/VIVA_GUIDE.md` (numbers generated from the result files); chart previews. | `README.md`, `docs/` |

### Results
Unchanged from the previous corrected package (same pipeline, same seed, same numbers): XGBoost test accuracy 0.8133, precision 0.6773, recall 0.5668, F1 0.6172, ROC-AUC 0.8584. The stale CSV in the uploaded ZIP (0.8105) was regenerated.

### Problems found and fixed while building/testing (honest list)
* A **ROC chart** axis showed "False positive rate, x" (a reference line leaked its field names) - fixed after rendering the real chart.
* The **SHAP chart** axis title was cut off, and bar-value labels on the risk chart were pale on white - fixed.
* Probability-bar end labels (0 % / 100 %) hung half outside the bar - fixed.
* The batch error file originally mixed several problems per row - kept deliberately (shows several reasons at once).
* Several of my *own tests* were wrong at first (a lockout test confounded by earlier failed logins, a disclaimer searched in the wrong element type, two placeholder checks removed). They were corrected so every test checks exactly what it says.
* An earlier draft of the valid-range limits used the training split only and would have rejected a real customer (monthly charge 18.25); limits now come from the whole dataset.

## Earlier versions
See `archive/original_submission/` (your first upload) and `archive/uploaded_zip_2_as_received/` (the hybrid ZIP as received).
