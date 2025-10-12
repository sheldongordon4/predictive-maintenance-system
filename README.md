# Engine Failure Prediction – RUL Estimation with NASA CMAPSS

This project uses machine learning to estimate the Remaining Useful Life (RUL) of turbofan engines based on NASA’s CMAPSS dataset. The system is designed to enable predictive maintenance, helping engineers and technicians prevent failure, reduce downtime, and optimize maintenance planning.

## Project Highlights
- **Data-Driven Maintenance**
Uses real-world engine data to forecast failures before they happen.
- **ML Model Trained for RUL Prediction**
Compares Support Vector Regressor, Random Forest, and LightGBM — with LightGBM selected for its superior performance.
- **Evaluation Metrics**
  - **MAE**: 18.01
  - **RMSE**: 24.31
- **Deployed with Streamlit**
A real-time dashboard for monitoring engine health and predicting risk.

---

## Dataset Description
NASA CMAPSS dataset from Kaggle includes:
- **Train Set**: Full run-to-failure history for 100 engines
- **Test Set**: Partial lifecycle up to a cut-off point
- **RUL File**: Ground truth for Remaining Useful Life

Each entry contains:
- Engine ID
- Cycle number
- 3 operational settings
- 21 sensor readings

📎 [Dataset on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## Feature Engineering

- Constructed RUL as `RUL = EOL - cycle_time`
- Dropped low-variance sensors
- Selected sensors showing degradation trends
- Applied `StandardScaler` for normalization
- Ensured feature consistency across train/test sets

---

## Model Development

Tested three regression models:
| Model                  | RMSE |
|------------------------|------|
| Support Vector Regressor | 33.31 |
| Random Forest Regressor  | 31.74 |
| **LightGBM**              | **31.32** (selected) |

✅ LightGBM was selected for its speed, accuracy, and built-in feature importance insights.

---

## Maintenance Use Case

The system classifies engine health based on RUL:

| Predicted RUL  | Maintenance Action       |
|----------------|--------------------------|
| ≤ 15 cycles    | Immediate Repair         |
| 16–47 cycles   | Schedule Inspection      |
| > 47 cycles    | Continue Normal Operation|

This enables:
- Proactive failure prevention  
- Cost-optimized maintenance planning  
- Improved equipment safety

---

## Streamlit Dashboard

Launch the app with:

```bash
streamlit run monitor_engine.py
```

Features:

- Real-time RUL prediction
- Engine-wise health visualization
- Sensor degradation trend plots
- Maintenance risk classification
- CSV batch upload for bulk predictions

---

## License

MIT License. See LICENSE for details.

---

## Acknowledgments

NASA Prognostics Center
CMAPSS Dataset (2008)
scikit-learn, LightGBM, Streamlit

---
