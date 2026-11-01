import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# LOAD MODEL FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "final_xgboost_churn_model.pkl")
)

encoder = joblib.load(
    os.path.join(BASE_DIR, "final_churn_encoder.pkl")
)

numerical_columns = joblib.load(
    os.path.join(BASE_DIR, "numerical_columns.pkl")
)

categorical_columns = joblib.load(
    os.path.join(BASE_DIR, "categorical_columns.pkl")
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether the customer "
    "is likely to churn."
)


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.header("Enter Customer Details")

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "No phone service", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=0.01
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    online_security = st.selectbox(
        "Online Security",
        ["No", "No internet service", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "No internet service", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "No internet service", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "No internet service", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "No internet service", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "No internet service", "Yes"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
        step=0.01
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

if st.button("🔮 Predict Churn", use_container_width=True):

    # --------------------------------------------------------
    # CREATE RAW INPUT DATA
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Senior Citizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure Months": [tenure],
        "Phone Service": [phone_service],
        "Multiple Lines": [multiple_lines],
        "Internet Service": [internet_service],
        "Online Security": [online_security],
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Paperless Billing": [paperless_billing],
        "Payment Method": [payment_method],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges]
    })


    # --------------------------------------------------------
    # KEEP EXACT COLUMN ORDER
    # --------------------------------------------------------

    input_data = input_data[
        categorical_columns + numerical_columns
    ]


    # --------------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------------

    try:

        encoded_data = encoder.transform(
            input_data[categorical_columns]
        )

        if hasattr(encoded_data, "toarray"):
            encoded_data = encoded_data.toarray()


        # ----------------------------------------------------
        # CREATE ENCODED DATAFRAME
        # ----------------------------------------------------

        encoded_feature_names = encoder.get_feature_names_out(
            categorical_columns
        )

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_feature_names
        )


        # ----------------------------------------------------
        # NUMERICAL DATA
        # ----------------------------------------------------

        numerical_df = input_data[
            numerical_columns
        ].copy()


        # ----------------------------------------------------
        # COMBINE FEATURES
        # ----------------------------------------------------

        model_input = pd.concat(
            [
                numerical_df.reset_index(drop=True),
                encoded_df.reset_index(drop=True)
            ],
            axis=1
        )


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(model_input)[0]

        probability = model.predict_proba(
            model_input
        )[0][1]


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.subheader("Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to CHURN"
            )

        else:

            st.success(
                "✅ Customer is likely to STAY"
            )


        # ----------------------------------------------------
        # CHURN PROBABILITY
        # ----------------------------------------------------

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability >= 0.70:

            risk = "🔴 High Risk"

        elif probability >= 0.40:

            risk = "🟠 Medium Risk"

        else:

            risk = "🟢 Low Risk"


        st.write(
            f"### Churn Risk: {risk}"
        )


    except Exception as e:

        st.error(
            "An error occurred while processing the prediction."
        )

        st.exception(e)