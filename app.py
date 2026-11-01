"""
Customer Churn Prediction and Retention Analysis System
=======================================================
Run from the project folder with:    streamlit run app.py

Flow: login/register (auth.py)  ->  sidebar navigation  ->  pages (views.py)
The model files are loaded from the models/ folder by churn_utils.py (the same code the notebook uses).
"""

import streamlit as st

import auth
import churn_utils
import ui
import views
from pathlib import Path

st.set_page_config(page_title="Churn Prediction & Retention System", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

# ---------------------------------------------------------------- 1. nobody gets past this line without logging in
if not auth.is_logged_in():
    auth.render_auth_page()
    st.stop()

# ---------------------------------------------------------------- 2. load the model files (once, then cached)
MODELS_DIR = Path(__file__).resolve().parent / "models"


@st.cache_resource(show_spinner="Loading model files ...")
def get_artifacts():
    return churn_utils.load_artifacts(MODELS_DIR)


try:
    art = get_artifacts()
except churn_utils.ArtifactError as error:
    st.error("The model files could not be loaded, so predictions are not available.")
    st.code(str(error))
    if st.sidebar.button("Log out"):
        auth.logout()
    st.stop()

# ---------------------------------------------------------------- 3. sidebar: user, navigation, model info
PAGES = {
    "📊 Dashboard": lambda: views.render_dashboard(),
    "🔮 Predict Customer": lambda: views.render_predict(art),
    "📁 Batch Prediction": lambda: views.render_batch(art),
    "🧭 Risk & Retention": lambda: views.render_risk(),
    "📈 Model Performance": lambda: views.render_performance(art),
}

user = auth.current_user()
with st.sidebar:
    st.markdown("### 📊 Churn Analytics")
    st.caption("Customer Churn Prediction & Retention Analysis using Machine Learning")
    st.markdown(f"👤 **{user['full_name']}**  \n`{user['username']}`")
    st.divider()
    page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
    st.divider()
    meta = art["metadata"]
    st.caption(f"Model: **{meta['final_model']}**  \nTest ROC-AUC: **{meta['test_metrics']['ROC-AUC']:.3f}**  \n"
               f"Features: **{len(meta['feature_names'])}**")
    if st.button("🚪 Log out", key="logout_button", width="stretch"):
        auth.logout()

# ---------------------------------------------------------------- 4. show the chosen page
PAGES[page]()
