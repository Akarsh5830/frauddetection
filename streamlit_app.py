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

# ✅ BIG custom CSS (keep your detailed styles)
st.markdown("""
<style>
/* Main container styling */
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
.input-card, .result-card {
    background: #fff; padding: 2rem; border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin: 2rem 0;
}
.success-indicator {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: 600;
}
.warning-indicator {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: 600;
}
.info-indicator {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: 600;
}
/* Plus your other CSS: button, slider, progress bar, etc... */
</style>
""", unsafe_allow_html=True)

# ✅ Load model & encoders
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
    <h2 style="color: white; margin-bottom: 2rem;">🛡️ FraudGuard AI</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Dashboard", "🔍 Manual Prediction", "📊 Batch Analysis", "⚙️ Settings"]
)

# ------------------ Dashboard ------------------
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🛡️ FraudGuard AI</h1>
        <p>Advanced Machine Learning-Powered Credit Card Fraud Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    # Big metric cards section
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#667eea;">🎯 Model Accuracy</h3>
            <h2 style="color:#2c3e50;">99.0%</h2>
            <p style="color:#7f8c8d; font-size:0.9rem;">LightGBM Performance</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#667eea;">⚡ Processing Speed</h3>
            <h2 style="color:#2c3e50;">0.2s</h2>
            <p style="color:#7f8c8d; font-size:0.9rem;">Per Transaction</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#667eea;">🔍 Features</h3>
            <h2 style="color:#2c3e50;">13</h2>
            <p style="color:#7f8c8d; font-size:0.9rem;">Available Features</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#667eea;">📊 Dataset</h3>
            <h2 style="color:#2c3e50;">1.3M</h2>
            <p style="color:#7f8c8d; font-size:0.9rem;">Training Records</p>
        </div>""", unsafe_allow_html=True)

    # Quick start card
    st.markdown("""
    <div class="input-card fade-in">
        <h2 style="color:#2c3e50; margin-bottom:1rem;">🚀 Quick Start</h2>
        <p style="color:#34495e; font-size:1.1rem; margin-bottom:2rem;">
            Use the sidebar to analyze a single transaction or upload a CSV for batch analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Features and performance side-by-side
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("""
        <div class="result-card">
            <h3 style="color:#667eea; margin-bottom:1rem;">🎯 Model Features</h3>
            <ul style="color:#34495e; line-height:1.8;">
                <li>🏪 Merchant Information</li>
                <li>📂 Transaction Category</li>
                <li>💰 Transaction Amount</li>
                <li>👤 Customer Gender</li>
                <li>📍 Geographic Location</li>
                <li>🏢 Customer Job</li>
                <li>⏰ Transaction Time</li>
                <li>🌍 City Population</li>
                <li>🪪 Card Number (dummy)</li>
                <li>🏠 Zip Code (dummy)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="result-card">
            <h3 style="color:#667eea; margin-bottom:1rem;">📈 Model Performance</h3>
            <div style="margin-bottom:1rem;">
                <p style="margin:0.5rem 0; color:#34495e;"><strong>Precision (Fraud):</strong> 0.27</p>
                <div style="background:#ecf0f1; border-radius:10px; height:8px;">
                    <div style="background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); width:27%; height:100%; border-radius:10px;"></div>
                </div>
            </div>
            <div style="margin-bottom:1rem;">
                <p style="margin:0.5rem 0; color:#34495e;"><strong>Recall (Fraud):</strong> 0.72</p>
                <div style="background:#ecf0f1; border-radius:10px; height:8px;">
                    <div style="background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); width:72%; height:100%; border-radius:10px;"></div>
                </div>
            </div>
            <div style="margin-bottom:1rem;">
                <p style="margin:0.5rem 0; color:#34495e;"><strong>F1-Score (Fraud):</strong> 0.39</p>
                <div style="background:#ecf0f1; border-radius:10px; height:8px;">
                    <div style="background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); width:39%; height:100%; border-radius:10px;"></div>
                </div>
            </div>
            <div style="margin-bottom:1rem;">
                <p style="margin:0.5rem 0; color:#34495e;"><strong>Overall Accuracy:</strong> 0.99</p>
                <div style="background:#ecf0f1; border-radius:10px; height:8px;">
                    <div style="background:linear-gradient(90deg,#667eea 0%,#764ba2 100%); width:99%; height:100%; border-radius:10px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------ Manual Prediction Page ------------------
elif page == "🔍 Manual Prediction":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🔍 Manual Transaction Analysis</h1>
        <p>Enter transaction details to get instant fraud prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # Transaction input form
    with st.form("fraud_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input(
                "💰 Transaction Amount ($)",
                min_value=0.01, max_value=10000.0, value=100.0, step=0.01,
                help="Enter the transaction amount"
            )
            category = st.text_input(
                "📂 Transaction Category",
                value="misc_net",
                help="Example: grocery_pos, travel, shopping_net..."
            )
            merchant = st.text_input(
                "🏪 Merchant Name",
                value="fraud_Koepp, Krajcik and Kozey",
                help="Merchant name from dataset"
            )
            unix_time = st.number_input(
                "⏰ Unix Timestamp",
                min_value=0, value=int(time.time()),
                help="Current time in Unix format"
            )
        with col2:
            gender = st.selectbox(
                "👤 Customer Gender", ["M", "F"],
                help="Gender of the customer"
            )
            job = st.text_input(
                "🏢 Customer Job",
                value="Psychologist, counselling",
                help="Customer job title"
            )
            city_pop = st.number_input(
                "🌍 City Population",
                min_value=1, max_value=10000000, value=1000000,
                help="Population of the city"
            )
            merch_lat = st.number_input(
                "📍 Merchant Latitude", min_value=-90.0, max_value=90.0, value=40.7589, step=0.0001
            )
            merch_long = st.number_input(
                "📍 Merchant Longitude", min_value=-180.0, max_value=180.0, value=-73.9851, step=0.0001
            )

        submitted = st.form_submit_button("🔍 Analyze Transaction")

    # ------------------ Prediction Logic ------------------
    if submitted:
        try:
            # ✅ Add dummy fields to fix shape mismatch
            input_data = {
                'cc_num': 1234567812345678,
                'merchant': merchant,
                'category': category,
                'amt': amt,
                'gender': gender,
                'zip': 10001,
                'lat': 40.7128,     # Dummy customer lat
                'long': -74.0060,   # Dummy customer long
                'city_pop': city_pop,
                'job': job,
                'unix_time': unix_time,
                'merch_lat': merch_lat,
                'merch_long': merch_long
            }

            # Convert to DataFrame
            df_input = pd.DataFrame([input_data])

            # Encode categorical fields
            df_input['category'] = le_cat.transform(df_input['category'])
            df_input['gender'] = le_gender.transform(df_input['gender'])
            df_input['job'] = le_job.transform(df_input['job'])
            df_input['merchant'] = le_merchant.transform(df_input['merchant'])

            # ✅ Use correct feature order
            df_input = df_input[feature_names]

            # Predict
            with st.spinner("🔄 Analyzing transaction..."):
                prob = model.predict_proba(df_input)[0][1]
                prediction = "FRAUD" if prob >= threshold else "LEGITIMATE"
                confidence = max(prob, 1 - prob)

            # ------------------ Display Results ------------------
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Fraud Probability", f"{prob:.2%}")
                if prob >= 0.8:
                    st.markdown('<div class="warning-indicator">🚨 HIGH RISK</div>', unsafe_allow_html=True)
                elif prob >= 0.5:
                    st.markdown('<div class="info-indicator">⚠️ MEDIUM RISK</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-indicator">✅ LOW RISK</div>', unsafe_allow_html=True)
            with col_b:
                st.metric("Prediction", prediction)
                st.metric("Confidence", f"{confidence:.2%}")

            # Show summary
            st.markdown("""
            <div class="result-card fade-in">
                <h3 style="color:#667eea; margin-bottom:1rem;">📋 Transaction Summary</h3>
            </div>
            """, unsafe_allow_html=True)

            summary_data = {
                'Field': ['Amount', 'Category', 'Merchant', 'Transaction Time'],
                'Value': [
                    f"${amt:.2f}",
                    category.replace('_',' ').title(),
                    merchant,
                    datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            st.table(pd.DataFrame(summary_data))

        except Exception as e:
            st.error(f"❌ Error processing transaction: {str(e)}")

# ------------------ Batch Analysis Page ------------------
elif page == "📊 Batch Analysis":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>📊 Batch Transaction Analysis</h1>
        <p>Upload a CSV file for bulk fraud detection</p>
    </div>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload your transactions CSV file",
        type=["csv"],
        help="Make sure your CSV has correct columns matching the dataset"
    )

    if uploaded_file:
        try:
            # Read uploaded CSV
            df = pd.read_csv(uploaded_file)

            # Show preview
            st.markdown("""
            <div class="result-card fade-in">
                <h3 style="color:#667eea; margin-bottom:1rem;">📋 Data Preview</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)

            # Encode categorical fields
            df['category'] = le_cat.transform(df['category'])
            df['gender'] = le_gender.transform(df['gender'])
            df['job'] = le_job.transform(df['job'])
            df['merchant'] = le_merchant.transform(df['merchant'])

            # Ensure correct feature order
            df = df[feature_names]

            with st.spinner("🔄 Running predictions..."):
                probs = model.predict_proba(df)[:,1]
                preds = (probs >= threshold).astype(int)

            # Add results to DataFrame
            df['fraud_probability'] = probs
            df['is_fraud_predicted'] = preds

            # ------------------ Summary Metrics ------------------
            total = len(df)
            fraud_count = preds.sum()
            fraud_percent = (fraud_count / total) * 100
            avg_prob = np.mean(probs)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", f"{total:,}")
            with col2:
                st.metric("Fraud Detected", f"{fraud_count:,}")
            with col3:
                st.metric("Fraud Rate", f"{fraud_percent:.2f}%")
            with col4:
                st.metric("Avg Risk Score", f"{avg_prob:.2%}")

            # ------------------ Risk Score Chart ------------------
            st.markdown("""
            <div class="result-card fade-in">
                <h3 style="color:#667eea; margin-bottom:1rem;">📊 Risk Score Distribution</h3>
            </div>
            """, unsafe_allow_html=True)

            hist_values = np.histogram(probs, bins=30)
            chart_df = pd.DataFrame({
                'Risk Score': hist_values[1][:-1],
                'Count': hist_values[0]
            })
            st.bar_chart(chart_df.set_index('Risk Score'))

            # ------------------ Download Button ------------------
            st.markdown("""
            <div class="result-card fade-in">
                <h3 style="color:#667eea; margin-bottom:1rem;">📥 Download Results</h3>
            </div>
            """, unsafe_allow_html=True)

            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Complete Analysis (CSV)",
                data=csv,
                file_name=f'fraud_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("✅ Make sure your CSV has correct columns and data.")


# ------------------ Settings Page ------------------
elif page == "⚙️ Settings":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>⚙️ Model Settings</h1>
        <p>Configure fraud detection parameters</p>
    </div>
    """, unsafe_allow_html=True)

    # Slider to adjust threshold
    new_threshold = st.slider(
        "🔧 Fraud Detection Threshold (higher = stricter)",
        min_value=0.0, max_value=1.0, value=threshold, step=0.01,
        help="Change the cutoff: higher means fewer false positives, lower means catching more fraud"
    )
    st.info(f"Current threshold: {new_threshold:.2f}")

    # ------------------ Feature Table ------------------
    st.markdown("""
    <div class="result-card fade-in">
        <h3 style="color:#667eea; margin-bottom:1rem;">📋 Model Features</h3>
    </div>
    """, unsafe_allow_html=True)

    # Feature descriptions
    feature_descriptions = {
        'cc_num': 'Credit card number (dummy)',
        'merchant': 'Merchant name',
        'category': 'Transaction category',
        'amt': 'Transaction amount',
        'gender': 'Customer gender',
        'zip': 'ZIP code (dummy)',
        'lat': 'Customer latitude (dummy)',
        'long': 'Customer longitude (dummy)',
        'city_pop': 'City population',
        'job': 'Customer job title',
        'unix_time': 'Transaction timestamp',
        'merch_lat': 'Merchant latitude',
        'merch_long': 'Merchant longitude'
    }

    feature_df = pd.DataFrame([
        {'Feature': feat, 'Description': feature_descriptions.get(feat, 'N/A')}
        for feat in feature_names
    ])

    st.table(feature_df)

    # ------------------ Note ------------------
    st.markdown("""
    <div class="info-indicator fade-in">
        📍 Note: Some fields like cc_num and zip are dummy values to match training shape.
    </div>
    """, unsafe_allow_html=True)


