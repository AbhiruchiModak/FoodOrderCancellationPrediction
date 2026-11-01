import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Food Delivery Cancellation Predictor",
    page_icon="🛵",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    return model

model = load_model()

st.title("🛵 Food Delivery Order Cancellation Predictor")
st.write("Enter all the order and partner parameters below to evaluate cancellation risk.")

with st.form("prediction_form"):
    st.subheader("1. Delivery Partner & Distance Metrics")
    alloted_orders = st.number_input("Allotted Orders", min_value=0.0, value=45.0, step=1.0)
    delivered_orders = st.number_input("Delivered Orders", min_value=0.0, value=40.0, step=1.0)
    first_mile_distance = st.number_input("First Mile Distance (km)", min_value=0.0, value=1.5, step=0.1)
    last_mile_distance = st.number_input("Last Mile Distance (km)", min_value=0.0, value=3.2, step=0.1)
    session_time = st.number_input("Session Time (hours)", min_value=0.0, value=3.5, step=0.1)

    st.subheader("2. Operational Delays & Time Metrics")
    allot_delay_min = st.number_input("Allotment Delay (minutes)", min_value=0.0, value=2.0, step=0.1)
    accept_delay_min = st.number_input("Acceptance Delay (minutes)", min_value=0.0, value=1.0, step=0.1)
    pickup_delay_min = st.number_input("Pickup Delay (minutes)", min_value=0.0, value=5.0, step=0.1)
    transit_time_min = st.number_input("Estimated Transit Time (minutes)", min_value=1.0, value=20.0, step=1.0)
    
    order_hour = st.slider("Order Hour (0-23)", min_value=0, max_value=23, value=19)
    order_dayofweek = st.selectbox("Day of the Week", options=[0, 1, 2, 3, 4, 5, 6], 
                                   format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
    is_weekend = 1 if order_dayofweek >= 5 else 0

    submitted = st.form_submit_button("Predict Cancellation Risk")

if submitted:
    # CREATE INPUT DATAFRAME WITH ALL REQUIRED TRAINING FEATURES
    input_data = pd.DataFrame({
        'alloted_orders': [alloted_orders],
        'delivered_orders': [delivered_orders],
        'first_mile_distance': [first_mile_distance],
        'last_mile_distance': [last_mile_distance],
        'session_time': [session_time],
        'allot_delay_min': [allot_delay_min],
        'accept_delay_min': [accept_delay_min],
        'pickup_delay_min': [pickup_delay_min],
        'transit_time_min': [transit_time_min],
        'order_hour': [order_hour],
        'order_dayofweek': [order_dayofweek],
        'is_weekend': [is_weekend]
    })

    try:
        # Get probabilities and handle under-sampling bias threshold
        probabilities = model.predict_proba(input_data)
        cancel_probability = probabilities[0][1] # Probability of cancellation (Class 1)

        CUSTOM_THRESHOLD = 0.60 
        is_cancelled = cancel_probability >= CUSTOM_THRESHOLD

        # Display Results
        st.divider()
        st.subheader("Prediction Result")
        
        if is_cancelled:
            st.error(f"⚠️ **High Risk: Order likely to be CANCELLED** (Probability: {cancel_probability:.2%})")
        else:
            st.success(f"✅ **Low Risk: Order likely to be DELIVERED** (Cancellation Probability: {cancel_probability:.2%})")
            
        with st.expander("View Diagnostic Details"):
            st.write(f"Raw Cancellation Probability: {cancel_probability:.4f}")
            st.write(f"Decision Threshold: {CUSTOM_THRESHOLD}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")