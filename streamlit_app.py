import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from datetime import datetime
import time

# ------------------ Streamlit setup ------------------
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
.main-header h1 {
    color: white; font-size: 3rem; font-weight: 700;
    text-align: center; margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.main-header p {
    color: rgba(255,255,255,0.9); text-align: center;
    font-size: 1.2rem; margin-top: 0.5rem;
}
.metric-card {
    background: white; padding: 1.5rem; border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    border-left: 5px solid #667eea; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Load models & encoders ------------------
@st.cache_resource
def load_all():
    with st.spinner("🔄 Loading model & encoders..."):
        model = joblib.load('lgb_model.pkl')
        le_cat = joblib.load('le_cat.pkl')
        le_gender = joblib.load('le_gender.pkl')
        le_job = joblib.load('le_job.pkl')
        le_merchant = joblib.load('le_merchant.pkl')
        with open('feature_names.json') as f:
            feature_names = json.load(f)
        try:
            with open('threshold.txt') as f:
                threshold = float(f.read())
        except:
            threshold = 0.5
        return model, le_cat, le_gender, le_job, le_merchant, feature_names, threshold

model, le_cat, le_gender, le_job, le_merchant, feature_names, threshold = load_all()

# ------------------ Sidebar ------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h2 style="color: white;">🛡️ FraudGuard AI</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Dashboard", "🔍 Manual Prediction"]
)

# ------------------ Dashboard Page ------------------
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ FraudGuard AI</h1>
        <p>Advanced Machine Learning-Powered Credit Card Fraud Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#667eea;">🎯 Accuracy</h3><h2>99.0%</h2>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#667eea;">⚡ Speed</h3><h2>0.2s</h2>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#667eea;">🔍 Features</h3><h2>13</h2>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#667eea;">📊 Data</h3><h2>1.3M</h2>
        </div>""", unsafe_allow_html=True)

# ------------------ Manual Prediction Page ------------------
elif page == "🔍 Manual Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Manual Transaction Analysis</h1>
        <p>Enter transaction details to get fraud prediction</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("fraud_form"):
        col1, col2 = st.columns(2)

        with col1:
            amt = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=10000.0, value=100.0)
            category = st.text_input("Transaction Category", value="misc_net")
            merchant = st.text_input("Merchant Name", value="fraud_Koepp, Krajcik and Kozey")
            unix_time = st.number_input("Unix Timestamp", min_value=0, value=int(time.time()))

        with col2:
            gender = st.selectbox("Gender", ["M", "F"])
            job = st.text_input("Job Title", value="Psychologist, counselling")
            city_pop = st.number_input("City Population", min_value=1, max_value=10000000, value=1000000)
            merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=40.7589)
            merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-73.9851)

        submitted = st.form_submit_button("🔍 Analyze Transaction")

    if submitted:
        try:
            # Build input with dummy cc_num & zip
            input_data = {
                'cc_num': 1234567812345678,
                'merchant': merchant,
                'category': category,
                'amt': amt,
                'gender': gender,
                'zip': 10001,
                'lat': 40.7128,  # dummy lat
                'long': -74.0060, # dummy long
                'city_pop': city_pop,
                'job': job,
                'unix_time': unix_time,
                'merch_lat': merch_lat,
                'merch_long': merch_long
            }

            df_input = pd.DataFrame([input_data])

            # Encode categorical columns
            df_input['category'] = le_cat.transform(df_input['category'])
            df_input['gender'] = le_gender.transform(df_input['gender'])
            df_input['job'] = le_job.transform(df_input['job'])
            df_input['merchant'] = le_merchant.transform(df_input['merchant'])

            # Match feature order
            df_input = df_input[feature_names]

            with st.spinner("🔄 Analyzing..."):
                prob = model.predict_proba(df_input)[0][1]
                pred = "FRAUD" if prob >= threshold else "LEGITIMATE"

            st.success(f"✅ Prediction: {pred} | Fraud Probability: {prob:.2%}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
