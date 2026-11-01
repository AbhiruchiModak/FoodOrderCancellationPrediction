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
    st.subheader("Order & Operational Metrics")
    
    # Input fields corresponding to your feature engineering pipeline
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
    # 1. CREATE INPUT DATAFRAME WITH EXACT TRAINING FEATURE NAMES AND ORDER
    # Replace these keys with the exact list of features used during model training!
    input_data = pd.DataFrame({
        'allot_delay_min': [allot_delay_min],
        'accept_delay_min': [accept_delay_min],
        'pickup_delay_min': [pickup_delay_min],
        'transit_time_min': [transit_time_min],
        'order_hour': [order_hour],
        'order_dayofweek': [order_dayofweek],
        'is_weekend': [is_weekend]
    })

    try:
        # 2. GET PROBABILITIES INSTEAD OF BLIND PREDICTIONS
        # model.predict_proba() returns [prob_class_0, prob_class_1]
        probabilities = model.predict_proba(input_data)
        cancel_probability = probabilities[0][1] # Probability of cancellation (Class 1)

        # 3. ADJUST THRESHOLD FOR UNDER-SAMPLING BIAS
        # Random Under-Sampling pushes probabilities up. 
        # Adjusting the decision threshold (e.g., to 0.65 or 0.70) corrects false positives.
        CUSTOM_THRESHOLD = 0.60 
        
        is_cancelled = cancel_probability >= CUSTOM_THRESHOLD

        # Display Results
        st.divider()
        st.subheader("Prediction Result")
        
        if is_cancelled:
            st.error(f"⚠️ **High Risk: Order likely to be CANCELLED** (Probability: {cancel_probability:.2%})")
        else:
            st.success(f"✅ **Low Risk: Order likely to be DELIVERED** (Cancellation Probability: {cancel_probability:.2%})")
            
        with st.expander("View Model Diagnostics"):
            st.write(f"Raw Cancellation Probability: {cancel_probability:.4f}")
            st.write(f"Decision Threshold: {CUSTOM_THRESHOLD}")
            st.json(input_data.to_dict(orient='records')[0])

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Tip: Ensure the feature names and column order in `input_data` match the exact columns used when training your Random Forest model.")