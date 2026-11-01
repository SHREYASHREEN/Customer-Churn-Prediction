# Customer Churn Prediction and Retention Analysis System using Machine Learning
# Customer Churn Prediction and Retention Analysis System

🚀 **Live Demo:** https://customer-churn-prediction-biqkerg4mysctrkt6gajs8.streamlit.app/

A Streamlit application that predicts which telecom customers are likely to leave (**churn**), explains the prediction,
suggests retention actions and can score a whole CSV file of customers. It is built on a real customer dataset
(`data/Telco_customer_churn.xlsx`, 7,043 customers) and an XGBoost model trained with a reproducible pipeline.

> **Read this first - honest status**
> * The app, the model files and all four test suites were run and passed **on Linux with Python 3.12**. They were **not** run on Windows, and the pages were never opened in a real web browser (only through Streamlit's headless test framework), so **no screenshots are included**. See section 9 for exactly what was and was not tested.
> * Your uploaded ZIP contained **no authentication code** (its `app.py` was the original 345-line app). The login system here is **new**. If you have your own newer version on your computer, nothing from it is lost: your original files are in `archive/`.

---

## 1. Features

| Page | What it does |
|---|---|
| 🔐 Login / Register | Allows users to create an account and log in. |
| 📊 Dashboard | Displays customer statistics and churn analysis charts. |
| 🔮 Predict Customer | Predicts whether an individual customer is likely to churn and displays the churn probability. |
| 📁 Batch Prediction | Uploads a CSV file and predicts churn for multiple customers. |
| 🧭 Risk & Retention | Categorizes customers into Low, Medium and High risk and provides retention recommendations. |
| 📈 Model Performance | Displays model evaluation metrics, confusion matrix, ROC curve and feature importance. |

**Risk levels:** Low < 30% · Medium 30%–50% · High > 50%.
**Risk levels (fixed, consistent everywhere):** Low < 30 % · Medium 30 %–50 % (inclusive) · High > 50 %. A customer is predicted to churn when the probability is above 50 %; this standard decision threshold was **not changed**.

---

## 2. Quick start on Windows 10/11 (VS Code + PowerShell)

1. Install **Python 3.12** from python.org (tick *Add python.exe to PATH*). Check: `py -3.12 --version`.
2. Unzip the project, open VS Code, **File > Open Folder** > `Customer_Churn_Project`.
3. Open a terminal: **Terminal > New Terminal** (it must say `...\Customer_Churn_Project>`). Then run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

* If PowerShell says *running scripts is disabled*, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that terminal, then activate again. (In Command Prompt use `.venv\Scripts\activate.bat`.)
* The browser opens at http://localhost:8501. Stop the app with `Ctrl+C`.
* **Always start the app from the project folder** (where `app.py` is), so the theme in `.streamlit\config.toml` and the relative paths work.

### If the model does not load ("input stream corrupted" or an integrity message)
The app checks the SHA-256 checksum of every model file and shows a clear message. Zip/Git tools that change line endings can damage binary files. The fix takes about a minute and always works because it builds the files on **your** machine:

```powershell
python train_model.py
streamlit run app.py
```

(`python train_model.py --skip-cv` is faster but leaves the cross-validation tables unchanged.) Rebuilding on a different computer can change the metrics in the 3rd-4th decimal place; the app reads its numbers from the rebuilt files, so everything stays consistent.

---

## 3. How to use the app

**Register and log in**
1. Open the **Register** tab, enter your name, a username (3-30 letters/digits/`_ . -`) and a password (8+ characters with a letter and a digit), press *Create account*.
2. Open the **Login** tab, sign in. Use **Log out** at the bottom of the sidebar. (A page refresh also logs you out.)

**Individual prediction** (sidebar > *Predict Customer*)
1. Press an example button (real customers from the test set) **or** fill in the three form sections yourself. Fields adapt automatically: with *Internet Service = No* the add-ons lock to "No internet service"; with *Phone Service = No*, *Multiple Lines* locks. *Total Charges* is calculated automatically (switch the toggle off to type it).
2. Press **Predict Churn**. Read the result banner, the probability bar, then the tabs: *Why this prediction?*, *Retention recommendations*, *Customer summary*.
3. Impossible input (e.g. tenure 0 with total charges 840) is refused with an explanation.

**Batch prediction** (sidebar > *Batch Prediction*)
1. Download **the template** (3 invented example rows) or **the sample file** (30 real customers from the test set).
2. Upload your CSV. The 19 columns must be present (an optional `CustomerID` column is carried through); values must match the allowed values listed on the page.
3. Read the summary cards and charts, filter by risk level, then **Download all results (CSV)**. Rejected rows (with the reason and CSV line number) can be downloaded too.
4. To demonstrate validation, upload `sample_data/sample_batch_with_errors_SYNTHETIC.csv` (8 *invented* customers: 3 are accepted, 5 rejected with reasons).

---

## 4. Project structure

```
Customer_Churn_Project/
├── app.py                   router: login gate, sidebar navigation, loads the model
├── auth.py                  registration / login / logout (SQLite + PBKDF2 hashing)
├── views.py                 all pages (dashboard, predict, batch, risk, performance, XAI, about)
├── ui.py                    look and feel: CSS, banners, probability bar, reusable charts
├── churn_utils.py           shared ML code: model loading + checksums, validation, preprocessing, risk, batch
├── explain.py               SHAP explanations (grouped by original feature)
├── recommendations.py       rule-based retention recommendations (with dataset evidence)
├── train_model.py           one-command rebuild of all model files and result tables
├── requirements.txt         pinned, tested package versions
├── .streamlit/config.toml   theme and upload limit
├── data/Telco_customer_churn.xlsx      ORIGINAL dataset (unchanged)
├── notebook/customer_churn_final.ipynb the explained training notebook (figures, SHAP, everything)
├── models/                  final_xgboost_churn_model.ubj (native XGBoost format), encoder, scaler, column lists,
│                            model_metadata.json, manifest.json (checksums)
├── outputs/                 result tables (CSV) and outputs/figures/ (11 paper figures, 300 dpi)
├── sample_data/             sample_batch_customers.csv (30 real customers), sample_batch_with_errors_SYNTHETIC.csv
├── tests/                   run_all_tests.py, verify_models.py, test_auth.py, test_batch.py, test_app.py
├── docs/                    VIVA_GUIDE.md, chart_previews/
├── auth_data/               created automatically on first registration (users.db) - never share it
└── archive/                 everything from your earlier uploads, unchanged (NOT used by the app)
```

---

## 5. Dataset, features and training (what the model really does)

| Item | Value |
|---|---|
| Dataset | `data/Telco_customer_churn.xlsx` - IBM *Telco customer churn* sample data (a **fictional** telecom company). 7,043 customers, 33 columns. The file is never modified (MD5 checked by the tests). |
| Target | `Churn Label` (Yes = churned, No = stayed): 1,869 churned (26.54%), 5,174 stayed |
| Input features (19) | 16 categorical (gender, senior citizen, partner, dependents, phone service, multiple lines, internet service, 6 internet add-ons, contract, paperless billing, payment method) and 3 numeric (tenure months, monthly charges, total charges) |
| Excluded | IDs/constants/location columns, and the **leakage** columns `Churn Value` (the target itself), `Churn Score`, `CLTV`, `Churn Reason` |
| Missing values | 11 blank `Total Charges`, **all** customers with tenure 0 (never billed) -> set to 0 (a rule that needs no statistics from the data, so no leakage) |
| Encoding | One-hot (`OneHotEncoder`, unknown categories ignored) -> 43 columns |
| Scaling | `StandardScaler` on the 3 numeric columns |
| Model input | 46 features = 3 scaled numeric + 43 one-hot columns (this exact order is used everywhere) |
| Train/test split | 80 / 20, stratified by churn, `random_state=42`: 5,634 training / 1,409 test customers |
| Leakage control | Encoder and scaler are fitted on the **training set only**; cross-validation re-fits them inside every fold |
| Models compared | Logistic Regression, Random Forest (200 trees), XGBoost (200 trees, depth 4, learning rate 0.05, subsample 0.8, colsample 0.8). No hyper-parameter tuning. |
| Final model | XGBoost, saved in XGBoost's **native `.ubj` format** (more robust than a Python pickle) + SHA-256 manifest |
| Decision rule | Churn when probability > 0.50 (unchanged) |

`notebook/customer_churn_final.ipynb` explains every step and draws the figures; `train_model.py` runs the same pipeline from the command line. `tests/verify_models.py --retrain` checks that both give **identical** model files and result tables.

---

## 6. Results (held-out test set: 1,409 customers the models never saw)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8020 | 0.6435 | 0.5695 | 0.6043 | 0.8489 |
| Random Forest | 0.7949 | 0.6412 | 0.5160 | 0.5719 | 0.8330 |
| XGBoost | 0.8133 | 0.6773 | 0.5668 | 0.6172 | 0.8584 |

Final model **XGBoost**: of the 374 customers who really churned the model finds 212 (recall 56.7%) and misses 162; of its 313 churn alerts 212 are correct (precision 67.7%). Confusion matrix: TN = 934, FP = 101, FN = 162, TP = 212.

95 % bootstrap confidence intervals for XGBoost (how much the test numbers could move):

| Metric | Test-set value | 95% CI lower | 95% CI upper |
|---|---|---|---|
| Accuracy | 0.8133 | 0.7935 | 0.8339 |
| Precision | 0.6773 | 0.6229 | 0.7252 |
| Recall | 0.5668 | 0.5154 | 0.6175 |
| F1 Score | 0.6172 | 0.5727 | 0.6581 |
| ROC-AUC | 0.8584 | 0.8371 | 0.8773 |

**Cross-validation** (5 folds x 3 repeats, mean ± standard deviation):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log-loss (lower=better) | Brier (lower=better) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8105 ± 0.0064 | 0.6642 ± 0.0179 | 0.5800 ± 0.0155 | 0.6190 ± 0.0116 | 0.8573 ± 0.0066 | 0.4032 ± 0.0080 | 0.1310 ± 0.0027 |
| Random Forest | 0.7956 ± 0.0078 | 0.6458 ± 0.0173 | 0.5092 ± 0.0333 | 0.5688 ± 0.0237 | 0.8362 ± 0.0067 | 0.4623 ± 0.0228 | 0.1408 ± 0.0032 |
| XGBoost | 0.8068 ± 0.0066 | 0.6629 ± 0.0179 | 0.5541 ± 0.0162 | 0.6034 ± 0.0126 | 0.8617 ± 0.0063 | 0.3962 ± 0.0081 | 0.1295 ± 0.0029 |

**Risk levels checked against reality** (test customers in each level that really churned):

| Risk band | Customers | Actual_churners | Actual_churn_rate |
|---|---|---|---|
| Low Risk | 882 | 87 | 9.9 % |
| Medium Risk | 214 | 75 | 35.0 % |
| High Risk | 313 | 212 | 67.7 % |

### Honest reading
* XGBoost has the best ROC-AUC and the best probability scores (log-loss, Brier) in cross-validation, which is why it is the final model for an app that shows probabilities. The gain over Logistic Regression is small (ROC-AUC +0.0044).
* At the standard 0.50 threshold, Logistic Regression is as good or better on accuracy, precision, recall and F1 in cross-validation (recall 0.580 vs 0.554; XGBoost has the higher recall in only 0 of 15 folds). **Do not say XGBoost is clearly superior.**
* Recall is modest (56.7%): the model misses a large share of churners. A lower threshold would catch more (see `outputs/threshold_analysis.csv`) but that is a business decision, so the threshold was not changed.
* The data is a fictional sample; the model shows how the method behaves, not how real customers behave.
* SHAP and feature importance show what the **model uses**, not what **causes** churn.

---

## 7. Model files and what "restart" means
`models/` holds everything the app needs: `final_xgboost_churn_model.ubj`, `final_churn_encoder.pkl`, `final_churn_scaler.pkl`, `numerical_columns.json`, `categorical_columns.json`, `model_metadata.json`, `manifest.json`. On start the app (1) checks the checksums, (2) loads the model with XGBoost's own loader, (3) checks that the model's 46 features match the encoder/scaler. Any problem produces a clear message instead of wrong predictions. `tests/verify_models.py` also proves that a **brand-new Python process** can load the model and give the same prediction.

---

## 8. Security notes (authentication)
Passwords are never stored: each user has a random 16-byte salt and a PBKDF2-HMAC-SHA256 hash (600,000 iterations); hashes are compared in constant time; SQL is parameterised (injection attempts are tested); the same message is shown for an unknown user and a wrong password; 5 wrong passwords lock the account for 5 minutes. **Limitations:** the login lasts only for the browser session (a refresh logs you out), there is no password reset or e-mail verification, and it is not meant for production use. Do not commit or share `auth_data/users.db`.

---

## 9. Testing: what was done and what was not

Run everything with `python tests\run_all_tests.py` (about 1-2 minutes; add `--retrain` to also rebuild the model from scratch and compare, about 40 seconds more).

| Suite | What it checks | Result when this package was built |
|---|---|---|
| `verify_models.py` | dataset unchanged; checksums; **damaged-file detection** (flipped byte, Windows line endings, truncation); fresh-process load ("restart"); 46 features match; recomputed metrics = CSVs; SHAP adds up; independent re-implementation of preprocessing; all 7,043 real customers pass validation; retrain gives byte-identical model files | all passed (22; 27 with `--retrain`) |
| `test_auth.py` | hashing, registration rules, duplicate users, SQL-injection attempts, lockout, login/logout, page protection in the real app | all passed (40) |
| `test_batch.py` | valid/invalid/malformed CSVs, batch = single prediction, download round-trip, upload through the real app | all passed (46) |
| `test_app.py` | 6 real customers incl. boundary cases at 30 % and 50 %, locked fields, invalid input, recommendations, explanation chart, example buttons, numbers on every page = data files | all passed (40) |

**Not tested (be honest in the viva):** Windows 10/11 (all tests ran on Linux, Python 3.12); a real web browser (Streamlit's headless `AppTest` runs the real script but cannot see the page); mobile/responsive layout; more than one simultaneous user. The charts and custom HTML components were rendered to images from the app's own specifications for a visual check (`docs/chart_previews/`); these are **not screenshots of the running app**. To take real screenshots: run the app, press `Win + Shift + S`, and save them for your report.

---

## 10. Limitations
Fictional sample data and no external validation; one train/test split for the headline numbers; modest recall and no treatment of class imbalance; no hyper-parameter tuning; the 0.50 threshold and 30 %/50 % risk levels are conventions, not cost-optimised; recommendations are rule-based ideas built from association, not proven causes or guarantees; the login is a demo-grade system; pickles (encoder, scaler) must only be loaded from trusted sources and need the pinned library versions.

## 11. What changed from your uploaded ZIP
See `CHANGELOG.md`. In short: the ZIP's `app.py`, notebook, `requirements.txt` and `model_comparison.csv` were your original versions (mixed with the earlier corrected `models/`), so the shipped project failed its own verification; everything was rebuilt on the verified 46-feature pipeline. The archived 30-feature model was **not** used.
