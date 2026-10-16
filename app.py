import streamlit as st
import pandas as pd
import joblib

# Load the saved Random Forest model
model = joblib.load('random_forest_model.pkl')

st.title("🍔 Food Delivery Order Cancellation Predictor")
st.write("Enter the order and delivery details below to predict if an order is likely to be cancelled.")

# Create input widgets for the user
first_mile_distance = st.number_input("First Mile Distance (km)", min_value=0.0, max_value=50.0, value=2.0)
last_mile_distance = st.number_input("Last Mile Distance (km)", min_value=0.0, max_value=50.0, value=5.0)
alloted_orders = st.number_input("Allotted Orders", min_value=0, max_value=100, value=5)
delivered_orders = st.number_input("Delivered Orders", min_value=0, max_value=100, value=4)
session_time = st.number_input("Session Time (hours)", min_value=0.0, max_value=24.0, value=2.5)
allot_delay_min = st.number_input("Allotment Delay (minutes)", min_value=0.0, max_value=120.0, value=3.0)

# Button to trigger prediction
if st.button("Predict Cancellation Risk"):
    # Organize input data into a DataFrame
    input_data = pd.DataFrame([[
        first_mile_distance, 
        last_mile_distance, 
        alloted_orders, 
        delivered_orders, 
        session_time, 
        allot_delay_min
    ]], columns=[
        'first_mile_distance', 'last_mile_distance', 
        'alloted_orders', 'delivered_orders', 'session_time',
        'allot_delay_min'
    ])
    
    # Make prediction and get probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of cancellation
    
    # Display results
    st.subheader("Prediction Results:")
    if prediction == 1:
        st.error(f"⚠️ **High Risk:** This order is predicted to be **CANCELLED** (Cancellation Probability: {probability * 100:.2f}%)")
    else:
        st.success(f"✅ **Low Risk:** This order is predicted to be **SUCCESSFUL** (Cancellation Probability: {probability * 100:.2f}%)")