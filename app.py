import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Food Order Cancellation Predictor",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food Order Cancellation Predictor")
st.write("Enter the order logistics and rider metrics below to get the cancellation risk prediction.")

# 2. Features & Model Loader (with automated fallback training if .joblib is missing)
SAFE_FEATURES = [
    'first_mile_distance', 'last_mile_distance', 'total_distance',
    'alloted_orders', 'delivered_orders', 'undelivered_orders', 'lifetime_order_count',
    'session_time', 'allot_delay_min', 'cancel_rate_30d', 'delivery_rate_30d',
    'order_hour', 'order_dayofweek', 'is_weekend', 'reassigned_order'
]

MODEL_FILE = "cancellation_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        # Fallback initialization if pre-trained model file isn't in directory yet
        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            'first_mile_distance': np.random.uniform(0.5, 5.0, n),
            'last_mile_distance': np.random.uniform(1.0, 15.0, n),
            'total_distance': np.random.uniform(1.5, 20.0, n),
            'alloted_orders': np.random.randint(1, 100, n).astype(float),
            'delivered_orders': np.random.randint(1, 100, n).astype(float),
            'undelivered_orders': np.random.randint(0, 5, n).astype(float),
            'lifetime_order_count': np.random.randint(10, 1000, n).astype(float),
            'session_time': np.random.uniform(0.5, 12.0, n),
            'allot_delay_min': np.random.uniform(0.1, 10.0, n),
            'cancel_rate_30d': np.random.uniform(0.0, 0.1, n),
            'delivery_rate_30d': np.random.uniform(0.9, 1.0, n),
            'order_hour': np.random.randint(0, 23, n),
            'order_dayofweek': np.random.randint(0, 6, n),
            'is_weekend': np.random.choice([0, 1], n),
            'reassigned_order': np.random.choice([0, 1], n, p=[0.95, 0.05])
        })
        y = np.random.choice([0, 1], n, p=[0.988, 0.012])
        
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        model.fit(X[SAFE_FEATURES], y)
        joblib.dump(model, MODEL_FILE)
        return model

model = load_model()

# 3. Input Layout (Two Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Distance & Timing")
    first_mile = st.number_input("First Mile Distance (km)", min_value=0.0, max_value=25.0, value=1.8, step=0.1)
    last_mile = st.number_input("Last Mile Distance (km)", min_value=0.0, max_value=30.0, value=3.5, step=0.1)
    allot_delay = st.number_input("Allotment Delay (mins)", min_value=0.0, max_value=60.0, value=1.2, step=0.1)
    order_hour = st.slider("Order Hour (0-23)", 0, 23, 14)
    day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    is_weekend = 1 if day_of_week in [5, 6] else 0

with col2:
    st.subheader("🚴 Rider Metrics")
    alloted_orders = st.number_input("Allotted Orders (30d)", min_value=0, max_value=500, value=45)
    delivered_orders = st.number_input("Delivered Orders (30d)", min_value=0, max_value=500, value=44)
    undelivered_orders = st.number_input("Undelivered Orders (30d)", min_value=0, max_value=50, value=1)
    session_time = st.number_input("Active Session Time (hrs)", min_value=0.0, max_value=24.0, value=4.5, step=0.25)
    lifetime_orders = st.number_input("Lifetime Order Count", min_value=0, max_value=5000, value=350)
    reassigned = st.selectbox("Is Order Reassigned?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# 4. Feature Engineering & Preparation
total_distance = first_mile + last_mile
cancel_rate_30d = (undelivered_orders / alloted_orders) if alloted_orders > 0 else 0.0
delivery_rate_30d = (delivered_orders / alloted_orders) if alloted_orders > 0 else 1.0

input_df = pd.DataFrame([{
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
    'is_weekend': int(is_weekend),
    'reassigned_order': float(reassigned)
}])

st.markdown("---")

# 5. Prediction Execution
if st.button("Predict Cancellation Risk", type="primary", use_container_width=True):
    prob = model.predict_proba(input_df[SAFE_FEATURES])[:, 1][0]
    
    st.subheader("Prediction Result")
    st.metric("Cancellation Probability", f"{prob:.2%}")
    
    if prob > 0.35:
        st.error("⚠️ **High Risk**: This order has a high likelihood of cancellation.")
    elif prob > 0.15:
        st.warning("⚡ **Medium Risk**: Moderate likelihood of cancellation.")
    else:
        st.success("✅ **Low Risk**: Order is stable.")