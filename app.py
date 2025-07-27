import streamlit as st
import pandas as pd
import joblib
import json

# ⚡ Load model & encoders
@st.cache_resource
def load_all():
    model = joblib.load('lgb_model.pkl')
    le_cat = joblib.load('le_cat.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_job = joblib.load('le_job.pkl')
    le_merchant = joblib.load('le_merchant.pkl')
    with open('feature_names.json') as f:
        feature_names = json.load(f)
    with open('threshold.txt') as f:
        threshold = float(f.read())
    return model, le_cat, le_gender, le_job, le_merchant, feature_names, threshold

model, le_cat, le_gender, le_job, le_merchant, feature_names, threshold = load_all()

# 🎨 App title
st.title("🔍 Fraud Detection App")
st.write("Upload transactions CSV to predict fraud / not fraud")

# 📤 Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # 📄 Read data
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded data preview:", df.head())

    # ✏️ Encode categorical columns
    df['category'] = le_cat.transform(df['category'])
    df['gender'] = le_gender.transform(df['gender'])
    df['job'] = df['job'].astype(str)
    df['job'] = le_job.transform(df['job'])
    df['merchant'] = le_merchant.transform(df['merchant'])

    # 🧹 Drop unused text columns, keep numerical only
    cols_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'dob', 'trans_num']
    X_input = df.drop(cols_to_drop, axis=1)

    # ✅ Ensure same column order as training
    X_input = X_input[feature_names]

    # 🔮 Predict probabilities & apply threshold
    probs = model.predict_proba(X_input)[:,1]
    preds = (probs >= threshold).astype(int)

    # 🪄 Show result
    df['fraud_probability'] = probs
    df['is_fraud_predicted'] = preds
    st.write("✅ Prediction result:", df[['fraud_probability', 'is_fraud_predicted']].head())

    # 📥 Download predictions
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )
