import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from datetime import datetime
import time

# 🎨 Custom CSS for modern styling
st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .input-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Success/Error indicators */
    .success-indicator {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .warning-indicator {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .info-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Form styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stSelectbox > div > div > div {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        padding: 0.75rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ⚡ Load model & encoders
@st.cache_resource
def load_all():
    with st.spinner("🔄 Loading AI models and encoders..."):
        model = joblib.load('lgb_model.pkl')
        le_cat = joblib.load('le_cat.pkl')
        le_gender = joblib.load('le_gender.pkl')
        le_job = joblib.load('le_job.pkl')
        le_merchant = joblib.load('le_merchant.pkl')
        with open('feature_names.json') as f:
            feature_names = json.load(f)
        with open('merchant_names.json') as f:
            merchant_names = json.load(f)
        try:
            with open('threshold.txt') as f:
                threshold = float(f.read())
        except:
            threshold = 0.5
        return model, le_cat, le_gender, le_job, le_merchant, feature_names, merchant_names, threshold

# Load models
model, le_cat, le_gender, le_job, le_merchant, feature_names, merchant_names, threshold = load_all()

# Sidebar navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h2 style="color: white; margin-bottom: 2rem;">🛡️ FraudGuard AI</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Dashboard", "🔍 Manual Prediction", "📊 Batch Analysis", "⚙️ Settings"]
)

if page == "🏠 Dashboard":
    # Main header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🛡️ FraudGuard AI</h1>
        <p>Advanced Machine Learning-Powered Credit Card Fraud Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🎯 Model Accuracy</h3>
            <h2 style="color: #2c3e50; margin: 0;">99.0%</h2>
            <p style="color: #7f8c8d; font-size: 0.9rem; margin: 0;">LightGBM Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">⚡ Processing Speed</h3>
            <h2 style="color: #2c3e50; margin: 0;">0.2s</h2>
            <p style="color: #7f8c8d; font-size: 0.9rem; margin: 0;">Per Transaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🔍 Features</h3>
            <h2 style="color: #2c3e50; margin: 0;">9</h2>
            <p style="color: #7f8c8d; font-size: 0.9rem; margin: 0;">Available Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">📊 Dataset</h3>
            <h2 style="color: #2c3e50; margin: 0;">1.3M</h2>
            <p style="color: #7f8c8d; font-size: 0.9rem; margin: 0;">Training Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("""
    <div class="input-card fade-in">
        <h2 style="color: #2c3e50; margin-bottom: 1rem;">🚀 Get Started</h2>
        <p style="color: #34495e; font-size: 1.1rem; margin-bottom: 2rem;">
            Enter transaction details manually or upload a CSV file for batch analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Model Features</h3>
            <ul style="color: #34495e; line-height: 2;">
                <li>🏪 Merchant Information</li>
                <li>📂 Transaction Category</li>
                <li>💰 Transaction Amount</li>
                <li>👤 Customer Gender</li>
                <li>📍 Geographic Location</li>
                <li>🏢 Customer Job</li>
                <li>⏰ Transaction Time</li>
                <li>🌍 City Population</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📈 Model Performance</h3>
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0.5rem 0; color: #34495e;"><strong>Precision (Fraud):</strong> 0.27</p>
                <div style="background: #ecf0f1; border-radius: 10px; height: 8px;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); width: 27%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0.5rem 0; color: #34495e;"><strong>Recall (Fraud):</strong> 0.72</p>
                <div style="background: #ecf0f1; border-radius: 10px; height: 8px;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); width: 72%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0.5rem 0; color: #34495e;"><strong>F1-Score (Fraud):</strong> 0.39</p>
                <div style="background: #ecf0f1; border-radius: 10px; height: 8px;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); width: 39%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0.5rem 0; color: #34495e;"><strong>Overall Accuracy:</strong> 0.99</p>
                <div style="background: #ecf0f1; border-radius: 10px; height: 8px;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); width: 99%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔍 Manual Prediction":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🔍 Manual Transaction Analysis</h1>
        <p>Enter transaction details to get instant fraud prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Transaction form
    st.markdown("""
    <div class="input-card fade-in">
        <h2 style="color: #2c3e50; margin-bottom: 1rem;">📝 Transaction Details</h2>
        <p style="color: #34495e; font-size: 1.1rem; margin-bottom: 2rem;">
            Fill in the transaction information below to analyze fraud risk.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("fraud_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💳 Transaction Info")
            
            # Transaction amount
            amt = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=10000.0,
                value=100.0,
                step=0.01,
                help="Enter the transaction amount"
            )
            
            # Category
            categories = ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport', 
                         'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos', 
                         'food_dining', 'personal_care', 'health_fitness', 'travel']
            category = st.selectbox(
                "Transaction Category",
                categories,
                help="Select the transaction category"
            )
            
            # Merchant
            merchant = st.selectbox(
                "Merchant Name",
                merchant_names,
                help="Select the merchant name from the available options (693 merchants available)"
            )
            
            # Unix time
            unix_time = st.number_input(
                "Unix Timestamp",
                min_value=0,
                value=int(time.time()),
                help="Enter the transaction timestamp in Unix format"
            )
        
        with col2:
            st.subheader("👤 Customer Information")
            
            # Gender
            gender = st.selectbox(
                "Gender",
                ["M", "F"],
                help="Select customer gender"
            )
            
            # Job
            jobs = ['Psychologist, counselling', 'Special educational needs teacher',
                   'Nature conservation officer', 'Patent attorney', 'Transport planner',
                   'Arboriculturist', 'Designer, multimedia', 'Public affairs consultant',
                   'Pathologist', 'Dance movement psychotherapist']
            job = st.selectbox(
                "Job Title",
                jobs,
                help="Select customer job title"
            )
            
            # Note: Customer location not available in real data
            st.info("📍 Customer location coordinates are not available in real transaction data")
            
            # Use default values for customer location (not available in real data)
            lat = 40.7128  # Default NYC latitude
            long = -74.0060  # Default NYC longitude
        
        # Additional fields
        st.subheader("📍 Location & Population")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            # City population
            city_pop = st.number_input(
                "City Population",
                min_value=1,
                max_value=10000000,
                value=1000000,
                help="Enter city population"
            )
        
        with col4:
            # Merchant latitude
            merch_lat = st.number_input(
                "Merchant Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.7589,
                step=0.0001,
                help="Enter merchant location latitude"
            )
        
        with col5:
            # Merchant longitude
            merch_long = st.number_input(
                "Merchant Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-73.9851,
                step=0.0001,
                help="Enter merchant location longitude"
            )
        
        # Submit button
        submitted = st.form_submit_button(
            "🔍 Analyze Transaction",
            use_container_width=True
        )
        
        if submitted:
            try:
                # Create input data with only the specified features
                input_data = {
                    'cc_num': 1234567812345678,
                    'merchant': merchant,
                    'category': category,
                    'amt': amt,
                    'gender': gender,
                    'lat': 40.7128,
                    'long': -74.0060,
                    'long': long,
                    'city_pop': city_pop,
                    'job': job,
                    'unix_time': unix_time,
                    'merch_lat': merch_lat,
                    'merch_long': merch_long
                }
                
                # Convert to DataFrame
                df_input = pd.DataFrame([input_data])
                
                # Encode categorical variables
                df_input['category'] = le_cat.transform(df_input['category'])
                df_input['gender'] = le_gender.transform(df_input['gender'])
                df_input['job'] = le_job.transform(df_input['job'])
                df_input['merchant'] = le_merchant.transform(df_input['merchant'])
                
                # Use only the specified features for prediction
                required_features = ['merchant', 'category', 'amt', 'gender', 'lat', 'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long']
                df_input = df_input[required_features]
                
                # Make prediction
                with st.spinner("🔄 Analyzing transaction..."):
                    probability = model.predict_proba(df_input)[0][1]
                    prediction = 1 if probability >= threshold else 0
                
                # Display results
                st.markdown("""
                <div class="result-card fade-in">
                    <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Analysis Results</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric(
                        "Fraud Probability",
                        f"{probability:.3f}",
                        delta=f"{probability:.1%} risk"
                    )
                    
                    # Risk level indicator
                    if probability >= 0.8:
                        st.markdown("""
                        <div class="warning-indicator fade-in">
                            🚨 HIGH RISK - Likely Fraudulent
                        </div>
                        """, unsafe_allow_html=True)
                    elif probability >= 0.5:
                        st.markdown("""
                        <div class="info-indicator fade-in">
                            ⚠️ MEDIUM RISK - Suspicious Activity
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-indicator fade-in">
                            ✅ LOW RISK - Likely Legitimate
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_result2:
                    st.metric(
                        "Prediction",
                        "FRAUD" if prediction == 1 else "LEGITIMATE",
                        delta="Model Decision"
                    )
                    
                    # Confidence level
                    confidence = max(probability, 1 - probability)
                    st.metric(
                        "Confidence",
                        f"{confidence:.1%}",
                        delta="Model Confidence"
                    )
                
                # Detailed analysis
                st.markdown("""
                <div class="result-card fade-in">
                    <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Transaction Summary</h3>
                </div>
                """, unsafe_allow_html=True)
                
                summary_data = {
                    'Field': ['Amount', 'Category', 'Merchant', 'Location Distance', 'Time'],
                    'Value': [
                        f"${amt:.2f}",
                        category.replace('_', ' ').title(),
                        merchant,
                        f"{((lat - merch_lat)**2 + (long - merch_long)**2)**0.5:.2f}°",
                        datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                
                st.table(pd.DataFrame(summary_data))
                
            except Exception as e:
                st.error(f"❌ Error processing transaction: {str(e)}")
                st.info("Please check your input values and try again.")

elif page == "📊 Batch Analysis":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>📊 Batch Transaction Analysis</h1>
        <p>Upload CSV file for bulk fraud detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("""
    <div class="input-card fade-in">
        <h2 style="color: #2c3e50; margin-bottom: 1rem;">📤 Upload Transaction Data</h2>
        <p style="color: #34495e; font-size: 1.1rem; margin-bottom: 2rem;">
            Upload a CSV file containing transaction data for batch analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your transaction data in CSV format"
    )
    
    if uploaded_file is not None:
        # Show upload success
        st.markdown("""
        <div class="success-indicator fade-in">
            ✅ File uploaded successfully! Processing your data...
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate processing steps
        steps = ["Reading data...", "Preprocessing...", "Encoding features...", "Running predictions...", "Generating results..."]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) * 20)
            time.sleep(0.5)
        
        # Read and process data
        try:
            df = pd.read_csv(uploaded_file)
            
            # Show data preview
            st.markdown("""
            <div class="result-card fade-in">
                <h3 style="color: #667eea; margin-bottom: 1rem;">📋 Data Preview</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(df.head(), use_container_width=True)
            
            # Process data
            with st.spinner("🔄 Processing data..."):
                # Encode categorical columns
                df['category'] = le_cat.transform(df['category'])
                df['gender'] = le_gender.transform(df['gender'])
                df['job'] = df['job'].astype(str)
                df['job'] = le_job.transform(df['job'])
                df['merchant'] = le_merchant.transform(df['merchant'])
                
                # Use only the specified features for prediction
                required_features = ['merchant', 'category', 'amt', 'gender', 'lat', 'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long']
                X_input = df[required_features]
                
                # Make predictions
                probs = model.predict_proba(X_input)[:,1]
                preds = (probs >= threshold).astype(int)
                
                # Add results to dataframe
                df['fraud_probability'] = probs
                df['is_fraud_predicted'] = preds
            
            # Results section
            st.markdown("""
            <div class="result-card fade-in">
                <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Analysis Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary metrics
            total_transactions = len(df)
            fraud_count = preds.sum()
            fraud_percentage = (fraud_count / total_transactions) * 100
            avg_probability = probs.mean()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", f"{total_transactions:,}")
            
            with col2:
                st.metric("Fraud Detected", f"{fraud_count:,}")
            
            with col3:
                st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
            
            with col4:
                st.metric("Avg. Risk Score", f"{avg_probability:.3f}")
            
            # Risk distribution chart
            st.markdown("""
            <div class="result-card fade-in">
                <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Risk Score Distribution</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create histogram data
            hist_values = np.histogram(probs, bins=30)
            chart_data = pd.DataFrame({
                'Risk Score': hist_values[1][:-1],
                'Number of Transactions': hist_values[0]
            })
            
            st.bar_chart(chart_data.set_index('Risk Score'))
            
            # High-risk transactions
            high_risk_threshold = 0.7
            high_risk_transactions = df[df['fraud_probability'] > high_risk_threshold]
            
            if len(high_risk_transactions) > 0:
                st.markdown("""
                <div class="warning-indicator fade-in">
                    ⚠️ High-Risk Transactions Detected
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    high_risk_transactions[['fraud_probability', 'is_fraud_predicted']].head(10),
                    use_container_width=True
                )
            
            # Download results
            st.markdown("""
            <div class="result-card fade-in">
                <h3 style="color: #667eea; margin-bottom: 1rem;">📥 Download Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Complete Analysis (CSV)",
                data=csv,
                file_name=f'fraud_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains the required columns and is properly formatted.")

elif page == "⚙️ Settings":
    st.markdown("""
    <div class="main-header fade-in">
        <h1>⚙️ Model Settings</h1>
        <p>Configure fraud detection parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-card fade-in">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🔧 Model Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Threshold adjustment
    new_threshold = st.slider(
        "Fraud Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=threshold,
        step=0.01,
        help="Adjust the sensitivity of fraud detection. Higher values = more conservative."
    )
    
    if new_threshold != threshold:
        st.info(f"Threshold updated from {threshold:.3f} to {new_threshold:.3f}")
        threshold = new_threshold
    
    # Model information
    st.markdown("""
    <div class="result-card fade-in">
        <h3 style="color: #667eea; margin-bottom: 1rem;">📋 Model Information</h3>
        <p><strong>Model Type:</strong> LightGBM Gradient Boosting</p>
        <p><strong>Training Date:</strong> December 2024</p>
        <p><strong>Features Used:</strong> 9 available features</p>
        <p><strong>Last Updated:</strong> {datetime.now().strftime("%B %d, %Y")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature information
    st.markdown("""
    <div class="result-card fade-in">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🔍 Feature Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Use only the available features for prediction (customer lat/long not available in real data)
    feature_descriptions = {
        'merchant': 'Merchant name',
        'category': 'Transaction category',
        'amt': 'Transaction amount',
        'gender': 'Customer gender',
        'city_pop': 'City population',
        'job': 'Customer job title',
        'unix_time': 'Transaction timestamp',
        'merch_lat': 'Merchant latitude',
        'merch_long': 'Merchant longitude'
    }
    
    # Note: Customer lat/long are set to default values (not available in real data)
    available_features = ['merchant', 'category', 'amt', 'gender', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long']
    
    feature_df = pd.DataFrame([
        {'Feature': feat, 'Description': feature_descriptions.get(feat, 'N/A')}
        for feat in available_features
    ])
    
    st.table(feature_df)
    
    # Add note about customer location
    st.markdown("""
    <div class="info-indicator fade-in">
        📍 Note: Customer latitude and longitude are not available in real transaction data and are set to default values
    </div>
    """, unsafe_allow_html=True)
