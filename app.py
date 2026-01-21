import streamlit as st
import pandas as pd
import joblib


# Load the saved model and column list
model = joblib.load('q_commerce_model.pkl')
model_columns = joblib.load('model_columns.pkl')


st.set_page_config(page_title="Q-Commerce Predictor", page_icon="🛍️")


st.title("🛍️ 10-Minute Delivery Spending Predictor")
st.write("This app predicts monthly spending based on the Random Forest model from the Capstone project.")


# --- Sidebar Inputs ---
st.sidebar.header("User Parameters")


age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Monthly Income (USD)", 1000, 10000, 4000)
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
freq = st.sidebar.slider("App Opening Frequency", 1, 100, 20)
orders = st.sidebar.slider("Orders Last Month", 0, 40, 5)
basket = st.sidebar.number_input("Avg Basket Size ($)", 5, 100, 25)
is_pro = st.sidebar.selectbox("Pro Member?", ["No", "Yes"])
dist = st.sidebar.slider("Distance to Dark Store (km)", 0.5, 5.0, 1.5)


# --- Prediction Logic ---
if st.button("Predict Spending"):
    # Convert 'Yes/No' to 1/0
    pro_val = 1 if is_pro == "Yes" else 0
    
    # Create input DataFrame matching the EXACT order of columns in X
    input_df = pd.DataFrame([[
        age, city_tier, income, freq, orders, basket, pro_val, dist
    ]], columns=model_columns)
    
    prediction = model.predict(input_df)[0]
    
    st.success(f"### Predicted Monthly Spend: ${prediction:.2f}")
    
    # Failure Analysis Note (for Distinction points)
    st.info("**Model Note:** This prediction is based on statistical averages. As identified in the Failure Analysis, high-value impulse outliers or logistics delays beyond 4km may cause actual spending to vary.")