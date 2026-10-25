import os
import sqlite3
import hashlib
import hmac
import secrets
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

st.set_page_config(
    page_title="ChurnIQ | Customer Retention",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Authentication storage ----------
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
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 200_000
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
                "INSERT INTO users(username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, datetime.now().isoformat(timespec="seconds")),
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username already exists. Please choose another."

def verify_user(username, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT username, password_hash, salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if not row:
        return False
    candidate = hash_password(password, row[2])
    return hmac.compare_digest(candidate, row[1])

init_db()

# ---------- Styling ----------
```python
st.markdown("""
<style>
    /* Main application background and default text */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #F5F7FB !important;
        color: #243447 !important;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }

    /* Make headings readable */
    h1, h2, h3, h4, h5, h6,
    p, label, li, span,
    [data-testid="stMarkdownContainer"] p {
        color: #243447;
    }

    /* Header */
    [data-testid="stHeader"] {
        background: #F5F7FB;
    }

    /* ChurnIQ branding card */
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(
            115deg,
            #102A43 0%,
            #176B87 65%,
            #219A9A 100%
        );
        color: #FFFFFF !important;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 2.2rem;
    }

    .hero p {
        color: #F0FAFF !important;
        margin: .45rem 0 0 0;
        font-size: 1rem;
    }

    /* Login and registration card */
    .auth-card {
        background-color: #FFFFFF;
        color: #243447;
        border: 1px solid #DCE5EF;
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
    }

    /* Input labels */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    label,
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        color: #243447 !important;
    }

    /* Text fields */
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #243447 !important;
        -webkit-text-fill-color: #243447 !important;
        border-color: #C9D5E3 !important;
    }

    /* Placeholder text */
    input::placeholder,
    textarea::placeholder {
        color: #77869A !important;
        -webkit-text-fill-color: #77869A !important;
        opacity: 1 !important;
    }

    /* Login and registration tabs */
    button[data-baseweb="tab"] {
        color: #34465A !important;
    }

    button[data-baseweb="tab"] p {
        color: #34465A !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FF4B4B !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FF4B4B !important;
    }

    /* Primary buttons */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 9px;
        font-weight: 600;
    }

    /* Metrics cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        color: #243447;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #E4EAF2;
        box-shadow: 0 2px 8px rgba(16, 42, 67, 0.04);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #243447;
    }

    /* Supporting text */
    .small-label {
        color: #526579;
        font-size: .9rem;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)
```

@st.cache_resource
def load_artifacts():
    return (
        joblib.load(os.path.join(BASE_DIR, "final_xgboost_churn_model.pkl")),
        joblib.load(os.path.join(BASE_DIR, "final_churn_encoder.pkl")),
        joblib.load(os.path.join(BASE_DIR, "numerical_columns.pkl")),
        joblib.load(os.path.join(BASE_DIR, "categorical_columns.pkl")),
    )

try:
    model, encoder, numerical_columns, categorical_columns = load_artifacts()
except Exception as exc:
    st.error("The model files could not be loaded. Keep the model .pkl files in the project root.")
    st.exception(exc)
    st.stop()

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "Login"

def show_auth():
    left, center, right = st.columns([1, 1.1, 1])
    with center:
        st.markdown("""
        <div class="hero">
          <h1>📈 ChurnIQ</h1>
          <p>Customer retention intelligence powered by machine learning.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        login_tab, register_tab = st.tabs(["🔐 Login", "✨ Create account"])
        with login_tab:
            st.subheader("Welcome back")
            st.caption("Sign in to access the customer churn prediction dashboard.")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login securely", use_container_width=True, type="primary")
            if submitted:
                if verify_user(username, password):
                    st.session_state.authenticated_user = username.strip()
                    st.session_state.last_prediction = None
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
        with register_tab:
            st.subheader("Create your account")
            st.caption("Create a local account to use the demo.")
            with st.form("register_form", clear_on_submit=True):
                new_username = st.text_input("Choose a username", key="register_username")
                new_password = st.text_input("Create password (8+ characters)", type="password", key="register_password")
                confirm_password = st.text_input("Confirm password", type="password", key="register_confirm")
                submitted_register = st.form_submit_button("Create account", use_container_width=True, type="primary")
            if submitted_register:
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    ok, message = register_user(new_username, new_password)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Demo authentication · Passwords are stored as salted PBKDF2 hashes.")
        st.caption("Use a demo account for mentor presentations; this local login is not production-grade identity management.")

def predict_customer(input_data):
    input_data = input_data[categorical_columns + numerical_columns]
    encoded_data = encoder.transform(input_data[categorical_columns])
    if hasattr(encoded_data, "toarray"):
        encoded_data = encoded_data.toarray()
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)
    numerical_df = input_data[numerical_columns].copy()
    model_input = pd.concat(
        [numerical_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1
    )
    prediction = int(model.predict(model_input)[0])
    probability = float(model.predict_proba(model_input)[0][1])
    risk = "High" if probability >= .70 else "Medium" if probability >= .40 else "Low"
    return prediction, probability, risk

def show_dashboard():
    username = st.session_state.authenticated_user
    with st.sidebar:
        st.markdown("## 📈 ChurnIQ")
        st.caption("CUSTOMER RETENTION PLATFORM")
        st.divider()
        st.write("👤 **Signed in as**")
        st.write(username)
        st.success("Session active")
        st.divider()
        st.markdown("### Navigation")
        st.info("🔮 Churn Prediction\n\nEnter a customer's details to estimate churn risk.")
        st.divider()
        if st.button("↪ Logout", use_container_width=True):
            st.session_state.authenticated_user = None
            st.session_state.last_prediction = None
            st.rerun()
        st.caption("ChurnIQ · MCA ML Project")

    st.markdown("""
    <div class="hero">
      <h1>Customer Churn Intelligence</h1>
      <p>Understand customer risk and support proactive retention decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction engine", "XGBoost", help="Trained machine-learning classification model")
    c2.metric("Risk bands", "3 levels", help="Low, Medium and High risk")
    c3.metric("Workspace", "Private session", help="Available after login")
    st.write("")
    st.subheader("🔮 Predict customer churn")
    st.write("Complete the customer profile below, then select **Predict churn**.")
    with st.form("customer_prediction_form"):
        st.markdown("#### 1. Customer profile")
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
        with right:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
            ])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.01)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=0.01)
        st.markdown("#### 2. Internet and service options")
        a, b, c = st.columns(3)
        with a:
            online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
        with b:
            device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
        with c:
            streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
        submitted = st.form_submit_button("🔮 Predict churn", type="primary", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame({
            "Gender": [gender], "Senior Citizen": [senior_citizen],
            "Partner": [partner], "Dependents": [dependents],
            "Tenure Months": [tenure], "Phone Service": [phone_service],
            "Multiple Lines": [multiple_lines], "Internet Service": [internet_service],
            "Online Security": [online_security], "Online Backup": [online_backup],
            "Device Protection": [device_protection], "Tech Support": [tech_support],
            "Streaming TV": [streaming_tv], "Streaming Movies": [streaming_movies],
            "Contract": [contract], "Paperless Billing": [paperless_billing],
            "Payment Method": [payment_method], "Monthly Charges": [monthly_charges],
            "Total Charges": [total_charges],
        })
        try:
            prediction, probability, risk = predict_customer(input_data)
            st.session_state.last_prediction = {
                "prediction": prediction, "probability": probability, "risk": risk,
                "customer": input_data.iloc[0].to_dict()
            }
        except Exception as exc:
            st.error("Could not process this customer profile. Please check the model files and input features.")
            st.exception(exc)

    result = st.session_state.get("last_prediction")
    if result:
        st.divider()
        st.subheader("📊 Prediction result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn probability", f"{result['probability'] * 100:.1f}%")
        r2.metric("Risk category", result["risk"])
        r3.metric("Model outcome", "Likely to churn" if result["prediction"] == 1 else "Likely to stay")
        if result["prediction"] == 1:
            st.error("⚠️ This customer is predicted to churn. Consider reviewing their contract, service experience, and retention options.")
        else:
            st.success("✅ This customer is predicted to stay.")
        st.progress(min(max(result["probability"], 0.0), 1.0), text="Estimated churn probability")
        st.caption("Risk bands: High ≥ 70% · Medium ≥ 40% and < 70% · Low < 40%. This is a model estimate, not a guarantee.")
        with st.expander("View submitted customer profile"):
            st.dataframe(pd.DataFrame([result["customer"]]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("ChurnIQ · Machine Learning Customer Churn Prediction · For academic demonstration.")

if st.session_state.authenticated_user:
    show_dashboard()
else:
    show_auth()
