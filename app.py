import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title='Order Cancellation Risk Predictor',
    page_icon='🚚',
    layout='centered',
)

# --- LOAD MODEL BUNDLE ---

def load_bundle():
  bundle = joblib.load('cancellation_model.pkl')

  # Hotfix patch for old scikit-learn imputer compatibility
  model = bundle['model']
  for name, step in getattr(model, 'steps', []):
    if isinstance(step, SimpleImputer) and not hasattr(step, '_fill_dtype'):
      step._fill_dtype = getattr(step, 'statistics_', [0.0]).dtype

  return bundle


try:
  bundle = load_bundle()
  final_model = bundle['model']
  FINAL_FEATURES = bundle['features']
  THRESHOLD = bundle['threshold']
  calibrator = bundle['calibrator']
  BASE_RATE = bundle['base_rate']
  medians = bundle.get('medians', {})
except Exception as e:
  st.error(
      f"Error loading model bundle: {e}. Make sure 'cancellation_model.pkl' is"
      ' in the same directory.'
  )
  st.stop()

# --- HELPER DEFINITIONS ---
DEPENDS = {
    'total_distance': ['first_mile_distance', 'last_mile_distance'],
    'cancel_rate_30d': ['alloted_orders', 'undelivered_orders'],
    'delivery_rate_30d': ['alloted_orders', 'delivered_orders'],
    'is_weekend': ['order_dayofweek'],
}

QUESTIONS = {
    'first_mile_distance': 'Distance rider -> restaurant (km)',
    'last_mile_distance': 'Distance restaurant -> customer (km)',
    'alloted_orders': "Rider's orders allotted in the last 30 days",
    'delivered_orders': "Rider's orders delivered in the last 30 days",
    'undelivered_orders': (
        "Rider's orders cancelled/undelivered in the last 30 days"
    ),
    'lifetime_order_count': 'Rider lifetime delivered orders',
    'session_time': 'Minutes the rider has been online today',
    'allot_delay_min': 'Minutes between order placed and rider allotted',
    'order_hour': 'Hour the order was placed (0-23)',
    'order_dayofweek': 'Day of week (0=Mon ... 6=Sun)',
    'reassigned_order': 'Was the order reassigned to this rider? (1=yes, 0=no)',
}

needed_raw = []
for f in FINAL_FEATURES:
  for r in DEPENDS.get(f, [f]):
    if r not in needed_raw:
      needed_raw.append(r)


def make_row(raw):
  """raw: dict of answers -> one-row DataFrame with model's features."""
  g = lambda k: float(raw[k]) if raw.get(k) not in (None, '') else np.nan
  r = {k: g(k) for k in QUESTIONS}
  r['total_distance'] = r['first_mile_distance'] + r['last_mile_distance']
  alot = (
      r['alloted_orders']
      if r['alloted_orders'] and r['alloted_orders'] > 0
      else np.nan
  )
  r['cancel_rate_30d'] = r['undelivered_orders'] / alot
  r['delivery_rate_30d'] = r['delivered_orders'] / alot
  r['is_weekend'] = (
      float(r['order_dayofweek'] in (5, 6))
      if not np.isnan(r['order_dayofweek'])
      else np.nan
  )
  return pd.DataFrame([r])[FINAL_FEATURES]


# --- UI LAYOUT ---
st.title('🚚 Order Cancellation Risk Predictor')
st.markdown(
    'Enter or adjust the order and rider parameters below to evaluate'
    ' cancellation risk in real-time.'
)

st.sidebar.header('Order & Rider Parameters')

user_raw = {}
for k in needed_raw:
  default_val = float(medians.get(k, 0.0))
  user_raw[k] = st.sidebar.number_input(
      label=QUESTIONS.get(k, k), value=default_val, format='%.2f'
  )

# --- PREDICTION LOGIC ---
if st.sidebar.button('Evaluate Risk', type='primary'):
  try:
    row = make_row(user_raw)
    
    # Ensure no NaNs are passed to the model
    if row.isna().any().any():
      st.error("Error: Some required input features are missing or NaN. Please check all sidebar fields.")
      st.stop()

    p_raw = float(final_model.predict_proba(row)[:, 1][0])
    
    # Pass raw probability correctly in 2D array format to calibrator
    risk = float(calibrator.predict_proba([[p_raw]])[0, 1])

    is_high_risk = risk >= THRESHOLD

    st.subheader('Prediction Results')

    col1, col2, col3 = st.columns(3)
    col1.metric('Estimated Risk', f'{risk:.1%}')
    col2.metric('Typical Baseline Rate', f'{BASE_RATE:.1%}')
    col3.metric('Risk vs Typical', f'{risk / BASE_RATE:.1f}x')

    st.markdown('---')

    if is_high_risk:
      st.error(
          '🚨 **Decision: HIGH RISK - Flag for intervention** (Probability'
          f' {risk:.2%} >= Threshold {THRESHOLD:.2f})'
      )
    else:
      st.success(
          f'✅ **Decision: Normal** (Probability {risk:.2%} < Threshold'
          f' {THRESHOLD:.2f})'
      )

  except Exception as ex:
    st.error(
        f'An error occurred during prediction: {ex}. Please check input values.'
    )