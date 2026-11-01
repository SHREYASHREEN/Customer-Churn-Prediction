# Viva guide - Customer Churn Prediction and Retention Analysis System

Numbers below come from the result files of this package (`outputs/`). If you rebuild the model on another computer they may change slightly in the 3rd-4th decimal - quote the numbers you see in *your* app.

## A. The 60-second summary
"My project predicts which telecom customers are likely to leave. I trained three models - Logistic Regression, Random Forest and XGBoost - on a real 7,043-customer dataset, kept XGBoost as the final model, and built a Streamlit application around it with login, a dashboard, single and batch prediction, risk levels, SHAP explanations and retention recommendations. The final model reaches about 0.86 ROC-AUC and 81% accuracy on customers it had never seen, but it only finds about 57% of the customers who really churn, so I describe it honestly as a useful but imperfect model."

## B. Feature by feature (what / how / where / one-line answer)

| Feature | How it works | File | One-line answer |
|---|---|---|---|
| **Authentication** | Register stores a random salt + PBKDF2-SHA256 hash in SQLite; login recomputes and compares; 5 wrong tries lock for 5 minutes; every page is blocked until `session_state` holds a user | `auth.py`, gate in `app.py` | "Passwords are never stored, only salted hashes, so even I cannot read them." |
| **Dashboard** | Loads the Excel file, counts churn, groups by contract/payment/internet/tenure and draws Altair charts | `views.render_dashboard` | "Every number is calculated from the real dataset, nothing is typed in." |
| **Individual prediction** | The form builds a dictionary of 19 values, validates it, scales/encodes it exactly as in training, and calls the model's `predict_proba` | `views.render_predict`, `churn_utils.predict_customer` | "The same preprocessing function is used by the notebook and the app, so training and prediction cannot disagree." |
| **Input validation** | Rules: no add-ons without internet, no multiple lines without phone, numbers inside the dataset range, total charges consistent with tenure x monthly. The UI locks impossible choices and validates again on submit | `churn_utils.validate_customer` | "The model never saw impossible customers, so I don't let users create them." |
| **Risk categories** | probability < 30 % low; 30-50 % medium; > 50 % high. High = predicted churn | `churn_utils.risk_category` | "Thresholds are a presentation choice fixed in advance; I checked on the test set that real churn rises from 9.9 % to 67.7 % across the levels." |
| **Retention recommendations** | If-then rules on the customer's data (e.g. month-to-month, short tenure, no online security); each shows the churn rate of that group in the dataset | `recommendations.py` | "They are suggestions backed by dataset patterns, not guarantees." |
| **Batch prediction** | Reads the CSV as text, checks columns, validates every row, predicts the valid ones in one vectorised call, lists rejected rows with reasons, offers CSV downloads | `churn_utils.run_batch`, `views.render_batch` | "Nothing is accepted silently: bad columns stop the run and bad rows are reported." |
| **Model performance** | Reads the saved evaluation tables and the stored test predictions; draws confusion matrix and ROC curve from them | `views.render_performance` | "The page shows real saved results, including the weak spots." |
| **Explainable AI (SHAP)** | SHAP splits each prediction into contributions per input; one-hot columns are added back together per original question | `explain.py` | "It shows what the model relies on; it does not prove what causes churn." |
| **Model file (.ubj) + checksums** | XGBoost's own format; SHA-256 manifest detects damaged copies; `train_model.py` rebuilds everything in about a minute | `churn_utils.load_artifacts`, `train_model.py` | "A pickle can break between library versions; the native format and a checksum make loading dependable." |
| **Tests** | 4 suites, about 150 automatic checks incl. damaged files, SQL injection, malformed CSVs | `tests/` | "I tested what I claim, and I list what I did not test." |

## C. Core ML concepts in simple words
* **Churn** - a customer stops using the service. **Why predict it** - keeping a customer is usually cheaper than winning a new one.
* **Feature** - one input column (e.g. contract type). **Target** - the column we predict (`Churn Label`).
* **Training vs testing data** - the model learns on 80 % and is judged on the other 20 % it never saw. **Stratified** - both parts keep the same ~26.5 % churn share.
* **Data leakage** - the model accidentally sees the answer. I removed `Churn Value`, `Churn Score`, `CLTV` and `Churn Reason`, and fit the encoder/scaler on training data only.
* **One-hot encoding** - turns "Contract = One year" into 0/1 columns. **Scaling** - puts numbers on a similar scale.
* **Logistic Regression** - weighs each feature and turns the sum into a probability. **Random Forest** - many decision trees voting. **XGBoost** - trees built one after another, each fixing the previous mistakes.
* **Accuracy** - share of correct answers (misleading: 73 % of customers don't churn). **Precision** - of those flagged, how many really churn. **Recall** - of those who churn, how many we find. **F1** - balance of both. **ROC-AUC** - how well churners are ranked above stayers (0.5 = guessing, 1 = perfect).
* **Threshold** - the probability above which we say "churn" (0.50, unchanged).
* **Cross-validation** - repeating the experiment on 15 different splits to see if a result is stable.

## D. Likely viva questions and answers
1. **Which dataset?** IBM's Telco customer churn sample: 7,043 customers, 33 columns, about 26.5 % churned. It is a fictional company, so results are illustrative.
2. **Which columns do you use and which do you exclude?** 19 customer details. I exclude IDs/location and four leakage columns (`Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`).
3. **How do you handle missing values?** Only `Total Charges` has 11 blanks, all for customers with tenure 0 who were never billed, so I set them to 0.
4. **How do you encode and scale?** One-hot for the 16 categorical columns and standard scaling for the 3 numeric ones, fitted on training data only; the model sees 46 features.
5. **How did you split the data?** 80/20, stratified, fixed seed 42, so results are reproducible.
6. **Why three algorithms?** To compare a simple baseline (Logistic Regression), a bagging ensemble (Random Forest) and a boosting ensemble (XGBoost) fairly on the same split.
7. **Why XGBoost?** It has the best ROC-AUC (0.858) and the best probability scores in cross-validation. But the gap to Logistic Regression is small, and at the 0.50 threshold Logistic Regression is as good or better on some measures - I say so openly.
8. **Is your model perfect?** No. Accuracy is 81.3%, but recall is only 56.7%: it misses about 43% of churners.
9. **Why not just use accuracy?** Because always predicting "stays" already gives about 73 % accuracy while finding no churner at all.
10. **What is the confusion matrix of your final model?** TN 934, FP 101, FN 162, TP 212 on 1,409 test customers.
11. **What is ROC-AUC?** The chance that a random churner gets a higher probability than a random non-churner; mine is 0.858.
12. **How reliable are the test numbers?** One split of 1,409 customers; the 95 % interval for accuracy is 0.793-0.834, so small differences between models are within noise.
13. **How do you avoid overfitting/leakage?** Separate test set, preprocessing fitted on training data only, cross-validation with preprocessing inside each fold, leakage columns removed.
14. **What are feature importance and SHAP?** Importance = how much the model used each input overall. SHAP = how much each input pushed *this* prediction up or down. The top inputs here are Contract_Month-to-month, Online Security_No, Internet Service_Fiber optic.
15. **Does SHAP prove why customers leave?** No - it explains the model, which learned correlations. It is not causation.
16. **What is the risk category?** Low < 30 %, Medium 30-50 %, High > 50 %. High means "predicted to churn". I checked that real churn rates rise across the levels.
17. **Did you change the decision threshold?** No, it is the standard 0.50. A lower threshold would raise recall but cause more false alarms - a business decision.
18. **How does the app predict one customer?** The form values become a dictionary, are validated, scaled and one-hot encoded with the saved scaler/encoder, and the saved XGBoost model returns the churn probability.
19. **How does batch prediction work and what if the file is wrong?** Same preprocessing for every row. Missing columns stop the run; bad rows are rejected with the reason and line number; the rest are predicted and can be downloaded.
20. **How are passwords protected?** Salted PBKDF2-SHA256 hashes (600,000 iterations), constant-time comparison, parameterised SQL, lockout after 5 failures.
21. **What was the "input stream corrupted" error?** XGBoost refusing a damaged or incompatible model file. I save the model in XGBoost's native format with a checksum, so damage is detected with a clear message, and `python train_model.py` rebuilds everything.
22. **How do you know the app's prediction equals the notebook's?** `verify_models.py` recomputes the test metrics from the saved files, compares preprocessing with an independent implementation, and the retrained model files are byte-identical.
23. **What are your limitations?** Fictional data, one split, modest recall, no tuning, conventional thresholds, rule-based recommendations, demo-grade login, not tested on Windows or in a real browser.
24. **What is your contribution?** A complete, reproducible, tested churn pipeline and application with honest evaluation - not a new algorithm.
25. **How could you improve it?** Tune hyper-parameters with nested cross-validation, handle class imbalance or choose the threshold by cost/benefit, check probability calibration, validate on another dataset, add proper user management.

## E. Do NOT claim
* That XGBoost is "clearly the best" or the model is "highly accurate" without mentioning recall.
* That the factors in SHAP/feature importance *cause* churn, or that the recommendations *will* prevent it.
* That it works on real customers, or that you tested on Windows/in a browser if you have not done so yourself.

## F. A 5-minute exhibition demo
1. **Register and log in** (say: hashed passwords). 2. **Dashboard** - "26.5 % churn; month-to-month and short-tenure customers churn most". 3. **Predict Customer** - press *High-risk example*, press *Predict*, show the probability bar, then the *Why this prediction?* tab and the recommendations. 4. Change *Internet Service* to *No* to show the locked add-ons. 5. **Batch Prediction** - upload `sample_data/sample_batch_customers.csv`, filter *High Risk*, download the CSV; then upload the `..._SYNTHETIC.csv` file to show rejected rows. 6. **Model Performance** - show the table, confusion matrix and the honest-reading box. 7. **Explainable AI** - show the SHAP ranking and the causation warning.
