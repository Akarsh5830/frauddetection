import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load model & encoders
@st.cache_resource
def load_all():
    model = joblib.load('lgb_model.pkl')
    le_cat = joblib.load('le_cat.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_job = joblib.load('le_job.pkl')
    le_merchant = joblib.load('le_merchant.pkl')
    with open('feature_names.json') as f:
        feature_names = json.load(f)
    return model, le_cat, le_gender, le_job, le_merchant, feature_names

model, le_cat, le_gender, le_job, le_merchant, feature_names = load_all()

# 🎨 App title & description
st.set_page_config(page_title="Fraud Detection App", page_icon="💳")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
Model built  using LightGBM on balanced data (SMOTE).
</div><br>""", unsafe_allow_html=True)

# ℹ️ Show label mappings
with st.expander("🔍 View label mappings"):
    st.write("**Category mapping:**")
    st.json(dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_).tolist())))
    st.write("**Gender mapping:**")
    st.json(dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_).tolist())))
    st.write("**Job mapping:**")
    st.json(dict(zip(le_job.classes_, le_job.transform(le_job.classes_).tolist())))
    st.write("**Merchant mapping:**")
    st.json(dict(zip(le_merchant.classes_, le_merchant.transform(le_merchant.classes_).tolist())))

# 📝 Input section
st.subheader("🛠️ Enter transaction details")

category_input = st.selectbox("📦 Category", le_cat.classes_)
gender_input = st.selectbox("👤 Gender", le_gender.classes_)
job_input = st.selectbox("💼 Job", le_job.classes_)
merchant_input = st.selectbox("🏪 Merchant", le_merchant.classes_)

amt = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0)
city_pop = st.number_input("🏙️ City Population", min_value=0)
unix_time = st.number_input("🕒 Transaction Unix Time", min_value=0)
merch_lat = st.number_input("📍 Merchant Latitude", format="%.6f")
merch_long = st.number_input("📍 Merchant Longitude", format="%.6f")

# ✅ Encode inputs
category_enc = le_cat.transform([category_input])[0]
gender_enc = le_gender.transform([gender_input])[0]
job_enc = le_job.transform([job_input])[0]
merchant_enc = le_merchant.transform([merchant_input])[0]

if st.button("🚀 Predict Fraud", type="primary"):
    data = pd.DataFrame([[
        amt, city_pop, unix_time, merch_lat, merch_long,
        category_enc, gender_enc, job_enc, merchant_enc
    ]], columns=[
        'amt', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
        'category', 'gender', 'job', 'merchant'
    ])
    # Match training column order
    data = data.reindex(columns=feature_names)

    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely Fraud! (Confidence: {proba*100:.1f}%)")
    else:
        st.success(f"✅ Looks Legit! (Fraud probability: {proba*100:.1f}%)")

    # 📊 Feature importance
    st.subheader("📊 Feature Importance")
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color="#4A90E2")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("🔍 Model Feature Importance")
    st.pyplot(fig)

# 📌 Footer
st.markdown("""
<hr style='margin-top:20px;'>
<div style='text-align:center;'>
    <i>Built with ❤️ by Akarsh Yadav</i>
</div>
""", unsafe_allow_html=True)
