ARCHIVE OF THE ORIGINAL SUBMISSION (kept for the record - NOT used by the current project)
===========================================================================================
Everything here is exactly as it was in customer_churn ZIP.zip. The current project does not read any of these files.

customer_churn_final_ORIGINAL.ipynb          the original notebook (cells were executed out of order; "Run All" failed with NameError)
customer_churn_prediction_EXPERIMENT_B.ipynb the earlier experiment (30 features, drop='first', no scaling for XGBoost) - this produced the model the OLD app used
app.py / README.md / requirements.txt        original versions (the old app loaded Experiment B files while the README/CSV described Experiment A)
models_experiment_B_old_app/                 the 4 pickles the old app.py loaded
models_experiment_A_original/                the Experiment A model/encoder/scaler originally saved by the notebook
outputs/                                     original result CSVs (the old model_comparison.csv was Experiment A)

Removed from the project (not archived because they were empty or unrelated): Untitled.ipynb, Untitled1.ipynb,
.ipynb_checkpoints/, and a stray .sf/ folder (Salesforce tooling metadata, unrelated to this project).
The dataset itself was NOT moved or changed: data/Telco_customer_churn.xlsx.
