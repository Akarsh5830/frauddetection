import streamlit as st
import joblib
import json
import pandas as pd

# ✅ Load model & encoders once
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

# ✅ Show modern header & info box
with open('templates/header.html', 'r', encoding='utf-8') as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# ✅ Put ALL inputs inside the card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📝 Enter transaction details:")

    col1, col2 = st.columns(2)

    with col1:
        category_input = st.selectbox("Category", le_cat.classes_)
        job_input = st.selectbox("Job", le_job.classes_)
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        unix_time = st.number_input("Transaction Unix Time", min_value=0)

    with col2:
        gender_input = st.selectbox("Gender", le_gender.classes_)
        merchant_input = st.selectbox("Merchant", le_merchant.classes_)
        city_pop = st.number_input("City Population", min_value=0)
        merch_lat = st.number_input("Merchant Latitude", format="%.6f")
        merch_long = st.number_input("Merchant Longitude", format="%.6f")

    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Predict button
if st.button("🔍 Predict Fraud"):
    # Encode categorical
    category_enc = le_cat.transform([category_input])[0]
    gender_enc = le_gender.transform([gender_input])[0]
    job_enc = le_job.transform([job_input])[0]
    merchant_enc = le_merchant.transform([merchant_input])[0]

    # Build dataframe
    data = pd.DataFrame([[
        amt, city_pop, unix_time, merch_lat, merch_long,
        category_enc, gender_enc, job_enc, merchant_enc
    ]], columns=[
        'amt', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
        'category', 'gender', 'job', 'merchant'
    ])

    # Reorder columns
    data = data.reindex(columns=feature_names)

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"⚠️ Transaction predicted as **FRAUD**! (probability: {prob*100:.1f}%)")
    else:
        st.success(f"✅ Transaction predicted as **NOT fraud** (probability: {prob*100:.1f}%)")

# ✅ Footer
st.markdown("""
<footer style="text-align:center; font-size:14px; color:#95a5a6; margin:40px 0;">
Built by Akarsh Yadav 🚀
</footer>
""", unsafe_allow_html=True)
