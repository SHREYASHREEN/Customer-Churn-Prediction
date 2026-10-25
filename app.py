import os
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# ==========================================================
# 1. APPLICATION CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

st.set_page_config(
    page_title="ChurnIQ | Customer Retention",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# 2. DATABASE AND USER AUTHENTICATION
# ==========================================================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        200_000,
    ).hex()


def register_user(username, password):
    username = username.strip()

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO users
                (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    salt,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

        return True, "Account created successfully. You can now log in."

    except sqlite3.IntegrityError:
        return False, "That username already exists. Please choose another."


def verify_user(username, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT username, password_hash, salt
            FROM users
            WHERE username = ?
            """,
            (username.strip(),),
        ).fetchone()

    if not row:
        return False

    candidate_hash = hash_password(password, row[2])

    return hmac.compare_digest(candidate_hash, row[1])


init_db()


# ==========================================================
# 3. APPLICATION STYLING
# ==========================================================

st.markdown(
    """
    <style>
        /* Main application background and text */
        .stApp {
            background-color: #f5f7fb;
            color: #172b4d;
        }

        [data-testid="stHeader"] {
            background-color: #f5f7fb;
        }

        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }

        /* Main heading banner */
        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 18px;
            background: linear-gradient(
                115deg,
                #102a43 0%,
                #176b87 65%,
                #21a0a0 100%
            );
            color: #ffffff;
            margin-bottom: 1rem;
        }

        .hero h1 {
            color: #ffffff !important;
            margin: 0;
            font-size: 2.2rem;
        }

        .hero p {
            color: #e5f4f8 !important;
            margin: 0.45rem 0 0 0;
            font-size: 1rem;
        }

        /* General headings and text */
        h1, h2, h3, h4, p, label {
            color: #172b4d;
        }

        [data-testid="stCaptionContainer"] {
            color: #526579;
        }

        /* Input boxes */
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            background-color: #ffffff !important;
            color: #172b4d !important;
            -webkit-text-fill-color: #172b4d !important;
            border-color: #cbd5e1 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #172b4d !important;
            border-color: #cbd5e1 !important;
        }

        div[data-baseweb="select"] input {
            color: #172b4d !important;
        }

        /* Buttons */
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 9px;
            font-weight: 600;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 1rem 1.2rem;
            border-radius: 14px;
            border: 1px solid #e4eaf2;
            box-shadow: 0 2px 8px rgba(16, 42, 67, 0.04);
        }

        div[data-testid="stMetric"] label {
            color: #526579 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #102a43 !important;
        }

        /* Login and registration card */
        .auth-card {
            background-color: #ffffff;
            border: 1px solid #e3eaf2;
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            color: #172b4d;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: #172b4d !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #eaf0f7;
        }

        section[data-testid="stSidebar"] * {
            color: #172b4d;
        }

        /* Hide Streamlit footer */
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 4. LOAD TRAINED MACHINE LEARNING FILES
# ==========================================================

@st.cache_resource
def load_artifacts():
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

    return model, encoder, numerical_columns, categorical_columns


try:
    (
        model,
        encoder,
        numerical_columns,
        categorical_columns,
    ) = load_artifacts()

except Exception as exc:
    st.error(
        "The machine learning model files could not be loaded. "
        "Please check that all four .pkl files exist in the repository."
    )
    st.exception(exc)
    st.stop()


# ==========================================================
# 5. SESSION STATE
# ==========================================================

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ==========================================================
# 6. LOGIN AND REGISTRATION PAGE
# ==========================================================

def show_auth():

    left, center, right = st.columns([1, 1.1, 1])

    with center:

        st.markdown(
            """
            <div class="hero">
                <h1>📈 ChurnIQ</h1>
                <p>
                    Customer retention intelligence powered by
                    machine learning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        login_tab, register_tab = st.tabs(
            ["🔐 Login", "✨ Create account"]
        )

        # ---------------- LOGIN ----------------

        with login_tab:

            st.subheader("Welcome back")

            st.caption(
                "Sign in to access the customer churn prediction dashboard."
            )

            with st.form("login_form"):

                username = st.text_input(
                    "Username",
                    key="login_username",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                )

                submitted = st.form_submit_button(
                    "Login securely",
                    use_container_width=True,
                    type="primary",
                )

            if submitted:

                if verify_user(username, password):

                    st.session_state.authenticated_user = username.strip()
                    st.session_state.last_prediction = None

                    st.rerun()

                else:

                    st.error("Incorrect username or password.")

        # ---------------- REGISTRATION ----------------

        with register_tab:

            st.subheader("Create your account")

            st.caption(
                "Register an account to access the demonstration."
            )

            with st.form("register_form"):

                new_username = st.text_input(
                    "Choose a username",
                    key="register_username",
                )

                new_password = st.text_input(
                    "Create password (8+ characters)",
                    type="password",
                    key="register_password",
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    key="register_confirm",
                )

                submitted_register = st.form_submit_button(
                    "Create account",
                    use_container_width=True,
                    type="primary",
                )

            if submitted_register:

                if new_password != confirm_password:

                    st.error("The passwords do not match.")

                else:

                    success, message = register_user(
                        new_username,
                        new_password,
                    )

                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        st.divider()

        st.caption(
            "ChurnIQ | Academic machine learning demonstration"
        )

        st.caption(
            "Passwords are stored as salted PBKDF2 hashes. "
            "This demonstration login is not production-grade "
            "identity management."
        )


# ==========================================================
# 7. CUSTOMER CHURN PREDICTION FUNCTION
# ==========================================================

def predict_customer(input_data):

    # Arrange input columns in the order expected by the encoder.
    input_data = input_data[
        list(categorical_columns) + list(numerical_columns)
    ]

    # Encode categorical features using the saved encoder.
    encoded_data = encoder.transform(
        input_data[list(categorical_columns)]
    )

    if hasattr(encoded_data, "toarray"):
        encoded_data = encoded_data.toarray()

    encoded_feature_names = encoder.get_feature_names_out(
        list(categorical_columns)
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_feature_names,
    )

    # Keep numerical features.
    numerical_df = input_data[
        list(numerical_columns)
    ].copy()

    # Combine numerical and encoded categorical features.
    model_input = pd.concat(
        [
            numerical_df.reset_index(drop=True),
            encoded_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Obtain the prediction and probability.
    prediction = int(model.predict(model_input)[0])

    probability = float(
        model.predict_proba(model_input)[0][1]
    )

    # Classify the estimated churn risk.
    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return prediction, probability, risk


# ==========================================================
# 8. CUSTOMER PREDICTION DASHBOARD
# ==========================================================

def show_dashboard():

    username = st.session_state.authenticated_user

    # ---------------- SIDEBAR ----------------

    with st.sidebar:

        st.markdown("## 📈 ChurnIQ")

        st.caption("CUSTOMER RETENTION PLATFORM")

        st.divider()

        st.write("👤 **Signed in as**")

        st.write(username)

        st.success("Session active")

        st.divider()

        st.markdown("### Navigation")

        st.info(
            "🔮 Churn Prediction\n\n"
            "Enter a customer's details to estimate churn risk."
        )

        st.divider()

        if st.button(
            "↪ Logout",
            use_container_width=True,
        ):

            st.session_state.authenticated_user = None
            st.session_state.last_prediction = None

            st.rerun()

        st.caption("ChurnIQ · MCA ML Project")

    # ---------------- MAIN HEADER ----------------

    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Intelligence</h1>
            <p>
                Understand customer risk and support proactive
                retention decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- SUMMARY CARDS ----------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prediction engine",
        "XGBoost",
        help="Trained machine learning classification model",
    )

    col2.metric(
        "Risk bands",
        "3 levels",
        help="Low, Medium and High risk",
    )

    col3.metric(
        "Workspace",
        "Private session",
        help="Available after login",
    )

    st.write("")

    # ---------------- CUSTOMER FORM ----------------

    st.subheader("🔮 Predict customer churn")

    st.write(
        "Complete the customer profile below, then select "
        "**Predict churn**."
    )

    with st.form("customer_prediction_form"):

        st.markdown("#### 1. Customer profile")

        left, right = st.columns(2)

        with left:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
            )

            partner = st.selectbox(
                "Partner",
                ["No", "Yes"],
            )

            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"],
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12,
                step=1,
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["No", "Yes"],
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "No phone service", "Yes"],
            )

        with right:

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["No", "Yes"],
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=0.01,
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=1000.0,
                step=0.01,
            )

        st.markdown("#### 2. Internet and service options")

        col_a, col_b, col_c = st.columns(3)

        with col_a:

            online_security = st.selectbox(
                "Online Security",
                ["No", "No internet service", "Yes"],
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "No internet service", "Yes"],
            )

        with col_b:

            device_protection = st.selectbox(
                "Device Protection",
                ["No", "No internet service", "Yes"],
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["No", "No internet service", "Yes"],
            )

        with col_c:

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "No internet service", "Yes"],
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "No internet service", "Yes"],
            )

        submitted = st.form_submit_button(
            "🔮 Predict churn",
            type="primary",
            use_container_width=True,
        )

    # ---------------- RUN PREDICTION ----------------

    if submitted:

        input_data = pd.DataFrame(
            {
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
                "Total Charges": [total_charges],
            }
        )

        try:

            prediction, probability, risk = predict_customer(
                input_data
            )

            st.session_state.last_prediction = {
                "prediction": prediction,
                "probability": probability,
                "risk": risk,
                "customer": input_data.iloc[0].to_dict(),
            }

        except Exception as exc:

            st.error(
                "Could not process this customer profile. "
                "Please check the model files and input features."
            )

            st.exception(exc)

    # ---------------- DISPLAY RESULTS ----------------

    result = st.session_state.get("last_prediction")

    if result:

        st.divider()

        st.subheader("📊 Prediction result")

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Churn probability",
            f"{result['probability'] * 100:.1f}%",
        )

        result2.metric(
            "Risk category",
            result["risk"],
        )

        result3.metric(
            "Model outcome",
            (
                "Likely to churn"
                if result["prediction"] == 1
                else "Likely to stay"
            ),
        )

        if result["prediction"] == 1:

            st.error(
                "⚠️ This customer is predicted to churn. "
                "Consider reviewing their contract, service "
                "experience, and retention options."
            )

        else:

            st.success(
                "✅ This customer is predicted to stay."
            )

        st.progress(
            min(max(result["probability"], 0.0), 1.0),
            text="Estimated churn probability",
        )

        st.caption(
            "Risk bands: High ≥ 70% · Medium ≥ 40% and < 70% · "
            "Low < 40%. This is a model estimate, not a guarantee."
        )

        with st.expander("View submitted customer profile"):

            st.dataframe(
                pd.DataFrame([result["customer"]]),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    st.caption(
        "ChurnIQ · Machine Learning Customer Churn Prediction · "
        "For academic demonstration."
    )


# ==========================================================
# 9. APPLICATION ENTRY POINT
# ==========================================================

if st.session_state.authenticated_user:

    show_dashboard()

else:

    show_auth()
