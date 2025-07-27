import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# ✅ Load model & encoders (cached)
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

st.title("💳 Credit Card Fraud Detection App")
st.markdown("""
Model built  using LightGBM on balanced data (SMOTE).
""")

# ✅ Show code→label mappings to user in expander
with st.expander("ℹ️ See label mappings"):
    st.write("**Category mapping:**")
    st.json(dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_).tolist())))
    st.write("**Gender mapping:**")
    st.json(dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_).tolist())))
    st.write("**Job mapping:**")
    st.json(dict(zip(le_job.classes_, le_job.transform(le_job.classes_).tolist())))
    st.write("**Merchant mapping:**")
    st.json(dict(zip(le_merchant.classes_, le_merchant.transform(le_merchant.classes_).tolist())))

st.subheader("📝 Input transaction details:")

# ✅ Instead of numbers, show dropdowns with real text, then encode
category_input = st.selectbox("Category", le_cat.classes_)
gender_input = st.selectbox("Gender", le_gender.classes_)
job_input = st.selectbox("Job", le_job.classes_)
merchant_input = st.selectbox("Merchant", le_merchant.classes_)

amt = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
city_pop = st.number_input("City Population", min_value=0)
unix_time = st.number_input("Transaction Unix Time", min_value=0)
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")

# ✅ Encode text inputs
category_enc = le_cat.transform([category_input])[0]
gender_enc = le_gender.transform([gender_input])[0]
job_enc = le_job.transform([job_input])[0]
merchant_enc = le_merchant.transform([merchant_input])[0]

if st.button("🔍 Predict Fraud"):
    # Build input vector in same order as feature_names
    data = pd.DataFrame([[
        amt, city_pop, unix_time, merch_lat, merch_long,
        category_enc, gender_enc, job_enc, merchant_enc
    ]], columns=[
        'amt', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
        'category', 'gender', 'job', 'merchant'
    ])

    # Adjust column order to match training (feature_names)
    data = data.reindex(columns=feature_names)

    pred = model.predict(data)[0]

    if pred == 1:
        st.error("⚠️ Transaction is predicted to be **FRAUD**!")
    else:
        st.success("✅ Transaction looks **legit / not fraud**.")



st.caption("Built by Akarsh Yadav 🚀")
