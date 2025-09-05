import json
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Demand Forecasting (LightGBM)", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("lgbm_model.joblib")

model = load_model()

def get_feature_schema(m):
    # Try best-known ways to recover names & count from LightGBM models
    names = None
    n = None

    # 1) scikit-learn API sometimes exposes n_features_in_
    n = getattr(m, "n_features_in_", None)

    # 2) Feature names via wrapper or booster
    #    (depends on how it was trained/saved)
    for attr in ("feature_name_",):
        if hasattr(m, attr):
            names = getattr(m, attr)
            break
    if names is None and hasattr(m, "booster_"):
        try:
            names = m.booster_.feature_name()
        except Exception:
            names = None

    # Fallbacks
    if names is not None:
        names = list(names)
        n = len(names)
    elif n is not None:
        names = [f"f{i}" for i in range(n)]
    else:
        # Absolute fallback if nothing is available
        n = 14
        names = [f"f{i}" for i in range(n)]

    return names, n

feature_names, n_features = get_feature_schema(model)

st.title("📈 Demand Forecasting (LightGBM)")
st.caption("Enter values for all features expected by the trained model (order matters).")

with st.expander("Paste a JSON row instead (optional)"):
    eg = {name: 0 for name in feature_names}
    txt = st.text_area(
        "JSON object with feature:value pairs",
        value=json.dumps(eg, indent=2),
        height=180
    )
    use_json = st.checkbox("Use JSON above")
    json_row = None
    if use_json:
        try:
            json_row = json.loads(txt)
            st.success("Parsed JSON.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            use_json = False

inputs = []
if not use_json:
    st.subheader("Manual inputs")
    cols = st.columns(2)
    for i, name in enumerate(feature_names):
        with cols[i % 2]:
            # You can tweak ranges/types per feature once you know them
            val = st.number_input(name, value=0.0, format="%.6f")
            inputs.append(val)
else:
    # Build inputs in the exact feature order
    try:
        inputs = [json_row[name] for name in feature_names]
    except KeyError as e:
        st.error(f"Missing key in JSON: {e}")

if st.button("Predict"):
    try:
        X = np.array([inputs], dtype=float)
        if X.shape[1] != n_features:
            st.error(f"Provided {X.shape[1]} features; model expects {n_features}.")
        else:
            yhat = model.predict(X)
            st.success(f"Prediction: {float(yhat[0]):.6f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
