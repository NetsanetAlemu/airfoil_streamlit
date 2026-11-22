import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Page config ----------
st.set_page_config(
    page_title="Airfoil Noise Predictor",
    page_icon="✈️",
    layout="wide",
)

# ---------- Load artifacts ----------
BASE = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    lr  = joblib.load(BASE / "linear_model.pkl")
    dt  = joblib.load(BASE / "decision_tree_model.pkl")
    knn = joblib.load(BASE / "knn_model.pkl")
    scaler = joblib.load(BASE / "scaler.pkl")
    with open(BASE / "meta.json") as f:
        meta = json.load(f)
    return lr, dt, knn, scaler, meta

lr, dt, knn, scaler, meta = load_artifacts()
FEATURES = meta["features"]
RANGES   = meta["feature_ranges"]
METRICS  = meta["metrics"]

# ---------- Header ----------
st.title("✈️ Airfoil Self-Noise Predictor")
st.markdown(
    "Predict the **scaled sound pressure level (dB)** of an airfoil from "
    "aerodynamic and acoustic parameters, using three machine-learning models "
    "trained on the NASA Airfoil Self-Noise dataset (1,503 wind-tunnel measurements)."
)

# ---------- Explainer infographic ----------
img_path = BASE / "assets" / "explainer.jpeg"
if img_path.exists():
    st.image(str(img_path), use_container_width=True,
             caption="The five factors that affect airfoil noise — applicable to airplanes, helicopters, and wind turbines.")
else:
    st.info("ℹ️ Add your explainer image at `assets/explainer.jpeg` to display it here.")

st.divider()

# ---------- Feature explanations ----------
with st.expander("📚 What do the 5 input features mean? (click to expand)", expanded=False):
    st.markdown("""
    | Feature | Plain-English meaning | Typical effect on noise |
    | :--- | :--- | :--- |
    | **Frequency (Hz)** | Pitch of the sound | Higher frequency → generally lower overall level in this dataset |
    | **Angle of attack (°)** | How tilted the wing is to the wind | Higher → more turbulence → more noise |
    | **Chord length (m)** | Width of the airfoil (front to back) | Larger → changes flow → affects noise |
    | **Free-stream velocity (m/s)** | Speed of oncoming air | Faster → louder (non-linear effect) |
    | **Suction-side thickness (m)** | Thickness of the slow air layer on top of the wing | Thicker → more turbulence → more noise |
    """)

# ---------- Sidebar: model choice ----------
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Choose a model for prediction",
    ["KNN (best, R²=0.893)", "Decision Tree (R²=0.885)", "Linear Regression (R²=0.558)"],
)
model_map = {
    "KNN (best, R²=0.893)":          ("knn",  knn),
    "Decision Tree (R²=0.885)":      ("dt",   dt),
    "Linear Regression (R²=0.558)":  ("linear", lr),
}
model_key, model_obj = model_map[model_choice]
m = METRICS[model_key]
st.sidebar.metric("Test R²", f"{m['R2']:.3f}")
st.sidebar.metric("Test RMSE", f"{m['RMSE']:.2f} dB")

# ---------- Input sliders ----------
st.subheader("🎛️ Input Features")
cols = st.columns(2)
inputs = {}

# Slider definitions with sensible step sizes
slider_config = {
    "frequency":               dict(step=50.0,  format="%.0f", help="Sound frequency in Hz"),
    "angle_of_attack":         dict(step=0.1,   format="%.1f", help="Wing tilt relative to airflow (degrees)"),
    "chord_length":            dict(step=0.005, format="%.3f", help="Airfoil width front-to-back (m)"),
    "free_stream_velocity":    dict(step=0.5,   format="%.1f", help="Oncoming air speed (m/s)"),
    "suction_side_thickness":  dict(step=0.0005,format="%.4f", help="Boundary-layer displacement thickness (m)"),
}

for i, feat in enumerate(FEATURES):
    lo, hi = RANGES[feat]
    default = (lo + hi) / 2
    with cols[i % 2]:
        inputs[feat] = st.slider(
            label=feat.replace("_", " ").title(),
            min_value=float(lo),
            max_value=float(hi),
            value=float(default),
            step=slider_config[feat]["step"],
            format=slider_config[feat]["format"],
            help=slider_config[feat]["help"],
        )

# ---------- Predict ----------
if st.button("🔮 Predict Sound Pressure Level", type="primary", use_container_width=True):
    input_df = pd.DataFrame([inputs])[FEATURES]
    input_scaled = scaler.transform(input_df)
    pred = model_obj.predict(input_scaled)[0]

    # Categorize the noise level
    if pred < 115:   cat, color = "Low",    "🟢"
    elif pred < 125: cat, color = "Moderate","🟡"
    elif pred < 135: cat, color = "High",   "🟠"
    else:            cat, color = "Very High","🔴"

    st.success(f"### Predicted SPL: **{pred:.2f} dB** {color} ({cat})")
    st.caption(
        "Reference: 100 dB ≈ chainsaw, 120 dB ≈ jet engine at close range, "
        "140 dB ≈ pain threshold."
    )

    # Quick comparison across all three models
    st.subheader("🔍 All Three Models on This Input")
    comp_cols = st.columns(3)
    for col, (name, mdl) in zip(comp_cols, [("Linear", lr), ("Decision Tree", dt), ("KNN", knn)]):
        p = mdl.predict(input_scaled)[0]
        col.metric(name, f"{p:.2f} dB")
    st.caption(
        "The spread across models reveals whether the input lies in a region where "
        "the relationship is strongly non-linear."
    )

# ---------- Feature importance (static image from training) ----------
st.divider()
st.subheader("📊 Feature Importance (from training)")
with st.expander("Show feature importance charts"):
    dt_importances = dt.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(dt_importances)
    ax.barh(np.array(FEATURES)[order], dt_importances[order], color="#1f77b4")
    ax.set_xlabel("Decision Tree importance (variance reduction)")
    ax.set_title("Top features driving airfoil noise")
    st.pyplot(fig)

st.divider()
st.caption(
    "Built with Streamlit • Dataset: NASA Airfoil Self-Noise (UCI ID 291) • "
    "Models: Linear Regression, Decision Tree, KNN."
)