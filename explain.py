"""
explain.py - explanations of the churn model with SHAP.

SHAP gives every model input a contribution to ONE prediction (in "log-odds" units: positive pushes
towards churn, negative towards staying). The model sees 46 columns (3 numeric + 43 one-hot columns), which
is hard to read, so the contributions of all one-hot columns that belong to the same original question
(e.g. the three Contract columns) are ADDED UP and shown under the original feature name.

Important: this shows what the MODEL relies on. It does not prove what CAUSES a customer to leave.
"""

import numpy as np
import pandas as pd
import shap

import churn_utils


def build_explainer(art):
    """SHAP TreeExplainer for the loaded XGBoost model (fast for tree models)."""
    return shap.TreeExplainer(art["model"])


def feature_groups(art):
    """{original feature name: [column positions in the model matrix]}.

    Matrix layout = [3 numeric columns | one-hot block of categorical column 1 | block 2 | ...]
    """
    groups = {col: [i] for i, col in enumerate(art["numerical_columns"])}
    start = len(art["numerical_columns"])
    for col, categories in zip(art["categorical_columns"], art["encoder"].categories_):
        groups[col] = list(range(start, start + len(categories)))
        start += len(categories)
    return groups


def explain_customer(art, explainer, customer):
    """Explain one customer.

    Returns (table, base_value, margin):
      table      DataFrame with Feature, Value, Contribution (sorted by size), Effect
      base_value the model's average output (log-odds) before looking at the customer
      margin     base_value + sum of contributions = the model's raw output (log-odds) for this customer
    """
    x = churn_utils.build_model_input(pd.DataFrame([customer]), art)
    shap_values = np.asarray(explainer.shap_values(x))[0]
    base_value = float(np.ravel(explainer.expected_value)[0])

    rows = []
    for feature, positions in feature_groups(art).items():
        value = customer[feature]
        if feature in art["numerical_columns"]:
            shown = f"{value:g}"
        else:
            shown = str(value)
        contribution = float(shap_values[positions].sum())
        rows.append({"Feature": feature, "Value": shown, "Contribution": contribution,
                     "Effect": "Raises churn risk" if contribution > 0 else "Lowers churn risk"})
    table = pd.DataFrame(rows)
    table = table.reindex(table["Contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    margin = base_value + float(shap_values.sum())
    return table, base_value, margin


def global_importance(art, explainer, customers):
    """Average absolute SHAP contribution per ORIGINAL feature over many customers (a DataFrame of raw rows)."""
    x = churn_utils.build_model_input(customers, art)
    shap_values = np.asarray(explainer.shap_values(x))
    rows = [{"Feature": feature, "Mean absolute SHAP": float(np.abs(shap_values[:, positions].sum(axis=1)).mean())}
            for feature, positions in feature_groups(art).items()]
    return pd.DataFrame(rows).sort_values("Mean absolute SHAP", ascending=False).reset_index(drop=True)
