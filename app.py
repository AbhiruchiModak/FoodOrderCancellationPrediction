import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Food Order Cancellation Predictor",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food Order Cancellation Predictor")
st.write("Enter the order logistics and rider metrics below to get the cancellation risk prediction.")

MODEL_FILE = "cancellation_model.pkl"

@st.cache_resource
def load_bundle():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        return None

bundle = load_bundle()

if bundle is None:
    st.error(f"⚠️ Model file '{MODEL_FILE}' not found! Please place your exported .pkl file in the same folder as app.py.")
else:
    model = bundle['model']
    features = bundle['features']        # Exact FINAL_FEATURES from your notebook
    threshold = bundle['threshold']
    calibrator = bundle['calibrator']
    base_rate = bundle['base_rate']
    model_name = bundle['model_name']
    
    st.info(f"Loaded **{model_name}** model using {len(features)} optimized features.")

    # 2. Input Fields Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Distance & Timing")
        first_mile = st.number_input("First Mile Distance (km)", min_value=0.0, max_value=25.0, value=1.8, step=0.1)
        last_mile = st.number_input("Last Mile Distance (km)", min_value=0.0, max_value=30.0, value=3.5, step=0.1)
        allot_delay = st.number_input("Allotment Delay (mins)", min_value=0.0, max_value=60.0, value=1.2, step=0.1)
        order_hour = st.slider("Order Hour (0-23)", 0, 23, 14)
        day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    with col2:
        st.subheader("🚴 Rider Metrics")
        alloted_orders = st.number_input("Allotted Orders (30d)", min_value=0, max_value=500, value=45)
        delivered_orders = st.number_input("Delivered Orders (30d)", min_value=0, max_value=500, value=44)
        undelivered_orders = st.number_input("Undelivered Orders (30d)", min_value=0, max_value=50, value=1)
        session_time = st.number_input("Active Session Time (hrs)", min_value=0.0, max_value=24.0, value=4.5, step=0.25)
        lifetime_orders = st.number_input("Lifetime Order Count", min_value=0, max_value=5000, value=350)
        reassigned = st.selectbox("Is Order Reassigned?", options=[0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")

    # 3. Feature Engineering (matching notebook logic)
    total_distance = first_mile + last_mile
    alot_div = alloted_orders if alloted_orders > 0 else np.nan
    cancel_rate_30d = undelivered_orders / alot_div
    delivery_rate_30d = delivered_orders / alot_div
    is_weekend = float(day_of_week in (5, 6))

    raw_data = {
        'first_mile_distance': first_mile,
        'last_mile_distance': last_mile,
        'total_distance': total_distance,
        'alloted_orders': float(alloted_orders),
        'delivered_orders': float(delivered_orders),
        'undelivered_orders': float(undelivered_orders),
        'lifetime_order_count': float(lifetime_orders),
        'session_time': float(session_time),
        'allot_delay_min': float(allot_delay),
        'cancel_rate_30d': cancel_rate_30d,
        'delivery_rate_30d': delivery_rate_30d,
        'order_hour': int(order_hour),
        'order_dayofweek': int(day_of_week),
        'is_weekend': is_weekend,
        'reassigned_order': float(reassigned)
    }

    # Construct DataFrame and select features
    input_df = pd.DataFrame([{f: raw_data.get(f, np.nan) for f in features}])[features].astype(float)

    def logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    st.markdown("---")

    # 4. Prediction Execution
    if st.button("Predict Cancellation Risk", type="primary", use_container_width=True):
        # FIX: Added .to_numpy() to pass a clean NumPy array to the pipeline, avoiding imputer/dataframe clashes
        p_raw = float(model.predict_proba(input_df.to_numpy())[:, 1][0])
        
        # Calibrated risk
        calibrated_risk = float(calibrator.predict_proba(logit([p_raw]))[0, 1])
        
        st.subheader("Prediction Results")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Estimated Cancellation Risk", f"{calibrated_risk:.1%}")
        with col_m2:
            st.metric("Typical Dataset Base Rate", f"{base_rate:.1%}")
            
        if p_raw >= threshold:
            st.error(f"⚠️ **HIGH RISK ({model_name})**: Flagged for early operational intervention (Threshold: {threshold:.3f}).")
        else:
            st.success(f"✅ **NORMAL**: Order stability looks healthy (Threshold: {threshold:.3f}).")