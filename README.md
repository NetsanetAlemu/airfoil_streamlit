# ✈️ Airfoil Self-Noise Predictor

An end-to-end machine learning project that predicts the **scaled sound pressure level (dB)** of an airfoil from five aerodynamic and acoustic features. Includes a fully interactive **Streamlit** web app for real-time prediction.

Built as part of an MSc AI assignment exploring the complete ML workflow: EDA → preprocessing → univariate, bivariate, and multivariate modeling → feature importance → deployment.

---

## 🎯 Problem Statement

Aircraft and wind-turbine designers need to predict airfoil noise from geometry and airflow parameters to design quieter blades — without expensive wind-tunnel tests. This project builds regression models that predict **sound pressure level (dB)** from five measured features.

**Dataset:** [NASA Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self-noise) (UCI ML Repository, ID 291)
- **1,503** wind-tunnel measurements
- **5** numeric features
- **0** missing values
- **Target:** `scaled sound pressure level` (dB), range ≈ 103–141 dB

### Features

| Feature | Plain-English meaning |
| :--- | :--- |
| **Frequency (Hz)** | Pitch of the sound |
| **Angle of attack (°)** | How much the wing tilts against the wind |
| **Chord length (m)** | Width of the wing from front to back |
| **Free-stream velocity (m/s)** | Speed of the oncoming air |
| **Suction-side thickness (m)** | Thickness of the slow air layer on top of the wing |

---

## 📊 Results

| Stage | Model | R² | RMSE (dB) | MAE (dB) |
| :--- | :--- | :--- | :--- | :--- |
| Univariate | Linear Regression (frequency) | 0.095 | 6.734 | 5.411 |
| Bivariate 4A | LR (frequency + velocity) | 0.127 | 6.612 | 5.238 |
| Bivariate 4B | LR (frequency + suction thickness) | 0.339 | 5.756 | 4.658 |
| Multivariate | Linear Regression (all 5) | 0.558 | 4.704 | 3.672 |
| Multivariate | **Decision Tree** (tuned, depth=12) | **0.885** | **2.398** | **1.796** |
| Multivariate | **KNN** (tuned, k=3, distance-weighted) | **0.893** | **2.314** | **1.711** |

**Best model:** KNN Regressor (R² = 0.893, RMSE = 2.31 dB) — meets the success criteria (R² > 0.80, RMSE < 3 dB).

### Key Findings

1. **A single feature is insufficient** — `frequency` alone explains only 9.5% of variance.
2. **Linear models plateau at R² ≈ 0.56** — the true relationship is non-linear and heteroscedastic.
3. **Non-linear models are essential** — Decision Tree and KNN both exceed the success criteria.
4. **Top-2 features are `frequency` and `suction_side_thickness`**, confirmed by three independent importance methods (Linear coefficients, Decision Tree variance reduction, permutation importance).
5. **Multicollinearity matters** — `suction_side_thickness` was underestimated by Linear Regression due to its strong correlation with `angle_of_attack` (r ≈ 0.75). Tree-based and permutation methods revealed its true importance.

---

## 🖥️ Streamlit App

The app provides:
- An explainer infographic showing the real-world context (airplanes, helicopters, wind turbines)
- Plain-English definitions of all 5 input features
- Interactive sliders for each feature
- Real-time prediction with all three models side by side
- Model metrics and feature importance charts

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/airfoil_streamlit.git
cd airfoil_streamlit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
