import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from datetime import datetime
import time

# ✅ Page config
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Custom CSS (full styling you shared)
st.markdown("""
<style>
/* ... (keep your full CSS here) ... */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
/* rest of your CSS exactly as before */
</style>
""", unsafe_allow_html=True)

# ✅ Load models & encoders
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

# ✅ Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h2 style="color: white;">🛡️ FraudGuard AI</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Dashboard", "🔍 Manual Prediction", "📊 Batch Analysis", "⚙️ Settings"]
)

# ---------------- Dashboard ----------------
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🛡️ FraudGuard AI</h1>
        <p>Advanced Machine Learning-Powered Credit Card Fraud Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card"><h3 style="color:#667eea;">🎯 Accuracy</h3><h2>99.0%</h2></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card"><h3 style="color:#667eea;">⚡ Speed</h3><h2>0.2s</h2></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card"><h3 style="color:#667eea;">🔍 Features</h3><h2>13</h2></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card"><h3 style="color:#667eea;">📊 Data</h3><h2>1.3M</h2></div>""", unsafe_allow_html=True)

# ---------------- Manual Prediction ----------------
elif page == "🔍 Manual Prediction":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🔍 Manual Transaction Analysis</h1>
        <p>Enter transaction details to get instant fraud prediction</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("fraud_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            amt = st.number_input("Transaction Amount ($)", value=100.0)
            category = st.text_input("Transaction Category", value="misc_net")
            merchant = st.text_input("Merchant Name", value="fraud_Koepp, Krajcik and Kozey")
            unix_time = st.number_input("Unix Timestamp", value=int(time.time()))

        with col2:
            gender = st.selectbox("Gender", ["M", "F"])
            job = st.text_input("Job Title", value="Psychologist, counselling")
            city_pop = st.number_input("City Population", value=1000000)
            merch_lat = st.number_input("Merchant Latitude", value=40.7589)
            merch_long = st.number_input("Merchant Longitude", value=-73.9851)

        submitted = st.form_submit_button("🔍 Analyze Transaction")

    if submitted:
        try:
            # ✅ Add dummy cc_num & zip
            input_data = {
                'cc_num': 1234567812345678,
                'merchant': merchant,
                'category': category,
                'amt': amt,
                'gender': gender,
                'zip': 10001,
                'lat': 40.7128,   # dummy customer lat
                'long': -74.0060, # dummy customer long
                'city_pop': city_pop,
                'job': job,
                'unix_time': unix_time,
                'merch_lat': merch_lat,
                'merch_long': merch_long
            }

            df_input = pd.DataFrame([input_data])
            df_input['category'] = le_cat.transform(df_input['category'])
            df_input['gender'] = le_gender.transform(df_input['gender'])
            df_input['job'] = le_job.transform(df_input['job'])
            df_input['merchant'] = le_merchant.transform(df_input['merchant'])

            # ✅ Use exact feature order
            df_input = df_input[feature_names]

            with st.spinner("🔄 Analyzing transaction..."):
                prob = model.predict_proba(df_input)[0][1]
                prediction = "FRAUD" if prob >= threshold else "LEGITIMATE"

            st.success(f"✅ Prediction: {prediction} | Fraud Probability: {prob:.2%}")

        except Exception as e:
            st.error(f"❌ Error processing transaction: {str(e)}")

# ---------------- Batch Analysis ----------------
elif page == "📊 Batch Analysis":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>📊 Batch Transaction Analysis</h1>
        <p>Upload CSV file for bulk fraud detection</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown("""<h4>✅ File loaded successfully!</h4>""", unsafe_allow_html=True)
            st.dataframe(df.head())

            # Encode categorical columns
            df['category'] = le_cat.transform(df['category'])
            df['gender'] = le_gender.transform(df['gender'])
            df['job'] = le_job.transform(df['job'])
            df['merchant'] = le_merchant.transform(df['merchant'])

            # Ensure feature order
            df = df[feature_names]

            with st.spinner("🔄 Running predictions..."):
                probs = model.predict_proba(df)[:,1]
                preds = (probs >= threshold).astype(int)

            # Add results to dataframe
            df['fraud_probability'] = probs
            df['is_fraud_predicted'] = preds

            # Show summary
            st.markdown("""<h4>📊 Results Summary</h4>""", unsafe_allow_html=True)
            st.metric("Total transactions", len(df))
            st.metric("Fraud detected", preds.sum())
            st.metric("Avg fraud probability", f"{np.mean(probs):.2%}")

            # Show risk distribution
            hist_values = np.histogram(probs, bins=30)
            chart_data = pd.DataFrame({
                'Risk Score': hist_values[1][:-1],
                'Number of Transactions': hist_values[0]
            })
            st.bar_chart(chart_data.set_index('Risk Score'))

            # Download button
            csv = df.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Results CSV",
                data=csv,
                file_name=f'fraud_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ---------------- Settings ----------------
elif page == "⚙️ Settings":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>⚙️ Settings</h1>
        <p>Configure fraud detection model settings</p>
    </div>
    """, unsafe_allow_html=True)

    new_threshold = st.slider(
        "Fraud Threshold (higher = stricter)",
        0.0, 1.0, value=threshold, step=0.01
    )
    st.info(f"Current threshold: {new_threshold:.2f}")

    # Feature info table
    feature_descriptions = {
        'cc_num': 'Credit card number',
        'merchant': 'Merchant name',
        'category': 'Transaction category',
        'amt': 'Transaction amount',
        'gender': 'Customer gender',
        'zip': 'ZIP code',
        'lat': 'Customer latitude',
        'long': 'Customer longitude',
        'city_pop': 'City population',
        'job': 'Customer job title',
        'unix_time': 'Transaction timestamp',
        'merch_lat': 'Merchant latitude',
        'merch_long': 'Merchant longitude'
    }

    st.markdown("""<h4>📋 Model Features</h4>""", unsafe_allow_html=True)
    feature_df = pd.DataFrame([
        {'Feature': f, 'Description': feature_descriptions.get(f, 'N/A')}
        for f in feature_names
    ])
    st.table(feature_df)

    st.markdown("""
    <div class="info-indicator fade-in">
        📍 Note: cc_num and zip are added as dummy fields during prediction to match training data.
    </div>
    """, unsafe_allow_html=True)

