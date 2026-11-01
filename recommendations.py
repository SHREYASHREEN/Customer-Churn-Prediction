"""
recommendations.py - simple, rule-based retention suggestions.

Every suggestion is triggered by the customer's OWN data (and the predicted risk) and shows the evidence from
the real dataset (how often similar customers churned). These are ideas to discuss with the customer, NOT
guaranteed ways to prevent churn.
"""

import pandas as pd

DISCLAIMER = ("These suggestions are rule-based ideas built from patterns in the dataset. They are not guaranteed "
              "to prevent churn, and the dataset shows association, not cause.")

SHORT_TENURE_MONTHS = 12


def _rate(mask_churn):
    return float(mask_churn.mean() * 100) if len(mask_churn) else float("nan")


def compute_evidence(df):
    """Churn rates (%) of customer groups in the real dataset. `df` needs the raw columns and 'Churn Label'."""
    churned = df["Churn Label"] == "Yes"
    internet = df["Internet Service"] != "No"
    monthly_p75 = float(df["Monthly Charges"].quantile(0.75))
    return {
        "contract": {k: _rate(churned[df["Contract"] == k]) for k in df["Contract"].unique()},
        "short_tenure": _rate(churned[df["Tenure Months"] <= SHORT_TENURE_MONTHS]),
        "longer_tenure": _rate(churned[df["Tenure Months"] > SHORT_TENURE_MONTHS]),
        "monthly_p75": monthly_p75,
        "high_monthly": _rate(churned[df["Monthly Charges"] >= monthly_p75]),
        "other_monthly": _rate(churned[df["Monthly Charges"] < monthly_p75]),
        "no_security": _rate(churned[internet & (df["Online Security"] == "No")]),
        "security": _rate(churned[internet & (df["Online Security"] == "Yes")]),
        "no_support": _rate(churned[internet & (df["Tech Support"] == "No")]),
        "support": _rate(churned[internet & (df["Tech Support"] == "Yes")]),
        "echeck": _rate(churned[df["Payment Method"] == "Electronic check"]),
        "other_payment": _rate(churned[df["Payment Method"] != "Electronic check"]),
        "fiber": _rate(churned[df["Internet Service"] == "Fiber optic"]),
        "dsl": _rate(churned[df["Internet Service"] == "DSL"]),
    }


def build_recommendations(customer, probability, risk, ev):
    """List of suggestions (dicts with priority, title, why, action), most important first."""
    pct = probability * 100
    items = []

    # 1) always: an action based on the predicted risk level
    if risk == "High Risk":
        items.append({"priority": "High", "title": "Personalised retention contact",
                      "why": f"The predicted churn probability is {pct:.1f}% (High Risk).",
                      "action": "Have a retention specialist contact the customer soon with a personalised offer. "
                                "Ask about satisfaction, billing and service problems before offering a discount."})
    elif risk == "Medium Risk":
        items.append({"priority": "Medium", "title": "Proactive check-in",
                      "why": f"The predicted churn probability is {pct:.1f}% (Medium Risk): the customer is expected to "
                             "stay, but the risk is elevated.",
                      "action": "Monitor the account, send a short satisfaction survey, and consider one targeted offer "
                                "from the points below."})
    else:
        items.append({"priority": "Low", "title": "Keep the customer satisfied",
                      "why": f"The predicted churn probability is {pct:.1f}% (Low Risk).",
                      "action": "No urgent action is needed. Maintain service quality; loyalty rewards or "
                                "relevant add-on offers can be considered."})

    # 2) rules based on the customer's own details (each shows dataset evidence)
    risky = risk != "Low Risk"
    base = lambda p: p if risky else "Optional"   # for a Low-Risk customer these are optional ideas

    if customer["Contract"] == "Month-to-month":
        items.append({"priority": base("High"), "title": "Offer an annual-contract incentive",
                      "why": f"The customer is on a month-to-month contract. In this dataset {ev['contract']['Month-to-month']:.0f}% "
                             f"of month-to-month customers churned, versus {ev['contract']['Two year']:.0f}% on two-year contracts.",
                      "action": "Offer a discount, bonus service or price lock for moving to a 1-year or 2-year contract."})

    if customer["Tenure Months"] <= SHORT_TENURE_MONTHS:
        items.append({"priority": base("High"), "title": "Onboarding support and engagement offers",
                      "why": f"The customer has been with the company for {customer['Tenure Months']} month(s). In this dataset "
                             f"{ev['short_tenure']:.0f}% of customers with {SHORT_TENURE_MONTHS} months or less churned, versus "
                             f"{ev['longer_tenure']:.0f}% of longer-tenured customers.",
                      "action": "Run a welcome / check-in call, explain the services in use, and offer a small engagement reward "
                                "during the first year."})

    if customer["Monthly Charges"] >= ev["monthly_p75"]:
        items.append({"priority": base("Medium"), "title": "Review whether the plan is a good fit",
                      "why": f"Monthly charges ({customer['Monthly Charges']:.2f}) are in the top quarter of customers "
                             f"(from {ev['monthly_p75']:.2f}). In this dataset {ev['high_monthly']:.0f}% of these customers churned, "
                             f"versus {ev['other_monthly']:.0f}% of the others.",
                      "action": "Check whether all paid services are used and offer a better-matched plan or bundle discount."})

    has_internet = customer["Internet Service"] != "No"
    if has_internet and customer["Online Security"] == "No":
        items.append({"priority": base("Medium"), "title": "Offer an online-security service",
                      "why": f"The customer has internet but no Online Security. In this dataset {ev['no_security']:.0f}% of such "
                             f"customers churned, versus {ev['security']:.0f}% of customers who have it.",
                      "action": "Offer a free trial or discounted bundle of the security service."})
    if has_internet and customer["Tech Support"] == "No":
        items.append({"priority": base("Medium"), "title": "Offer technical support",
                      "why": f"The customer has internet but no Tech Support. In this dataset {ev['no_support']:.0f}% of such "
                             f"customers churned, versus {ev['support']:.0f}% of customers who have it.",
                      "action": "Offer a trial of the tech-support service or a proactive help session."})

    if customer["Payment Method"] == "Electronic check":
        items.append({"priority": base("Medium"), "title": "Encourage automatic payment",
                      "why": f"The customer pays by electronic check. In this dataset {ev['echeck']:.0f}% of these customers churned, "
                             f"versus {ev['other_payment']:.0f}% of customers using other payment methods.",
                      "action": "Offer a small incentive to switch to automatic bank transfer or credit-card payment."})

    if customer["Internet Service"] == "Fiber optic":
        items.append({"priority": base("Medium"), "title": "Check fibre service quality and price satisfaction",
                      "why": f"In this dataset {ev['fiber']:.0f}% of fibre-optic customers churned, versus {ev['dsl']:.0f}% of DSL customers.",
                      "action": "Ask about speed, reliability and price; resolve any open service tickets."})

    # the risk-based action always stays first; the customer-specific ideas are ordered by priority
    order = {"High": 0, "Medium": 1, "Low": 2, "Optional": 3}
    first, rest = items[0], sorted(items[1:], key=lambda item: order[item["priority"]])   # stable sort
    return [first] + rest
