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

# Load the trained model (and scaler/preprocessor if saved together)
@st.cache_resource
def load_model():
    # If you saved a dictionary containing model and features, load that instead:
    # data = joblib.load('model_bundle.pkl')
    # return data['model'], data['features']
    
    model = joblib.load('random_forest_model.pkl')
    return model

model = load_model()

st.title("🛵 Food Delivery Order Cancellation Predictor")
st.write("Enter the order parameters below to evaluate cancellation risk.")

with st.form("prediction_form"):
    st.subheader("Order & Delivery Metrics")
    
    # Input fields matching the exact features seen at fit time
    alloted_orders = st.number_input("Allotted Orders", min_value=0.0, value=45.0, step=1.0)
    delivered_orders = st.number_input("Delivered Orders", min_value=0.0, value=40.0, step=1.0)
    first_mile_distance = st.number_input("First Mile Distance (km)", min_value=0.0, value=1.5, step=0.1)
    last_mile_distance = st.number_input("Last Mile Distance (km)", min_value=0.0, value=3.2, step=0.1)
    session_time = st.number_input("Session Time (hours)", min_value=0.0, value=3.5, step=0.1)
    
    submitted = st.form_submit_button("Predict Cancellation Risk")

if submitted:
    # 1. CREATE INPUT DATAFRAME WITH EXACT TRAINING COLUMNS
    input_data = pd.DataFrame({
        'alloted_orders': [alloted_orders],
        'delivered_orders': [delivered_orders],
        'first_mile_distance': [first_mile_distance],
        'last_mile_distance': [last_mile_distance],
        'session_time': [session_time]
    })

    try:
        # 2. GET PROBABILITIES
        probabilities = model.predict_proba(input_data)
        cancel_probability = probabilities[0][1] # Probability of class 1 (Cancelled)

        # 3. ADJUST THRESHOLD FOR UNDER-SAMPLING BIAS
        CUSTOM_THRESHOLD = 0.60 
        is_cancelled = cancel_probability >= CUSTOM_THRESHOLD

        # Display Results
        st.divider()
        st.subheader("Prediction Result")
        
        if is_cancelled:
            st.error(f"⚠️ **High Risk: Order likely to be CANCELLED** (Probability: {cancel_probability:.2%})")
        else:
            st.success(f"✅ **Low Risk: Order likely to be DELIVERED** (Cancellation Probability: {cancel_probability:.2%})")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")