# Engine Failure Prediction – RUL Estimation with NASA CMAPSS

This project uses machine learning to estimate the Remaining Useful Life (RUL) of turbofan engines based on NASA’s CMAPSS dataset. The system is designed to enable predictive maintenance, helping engineers and technicians prevent failure, reduce downtime, and optimize maintenance planning.

## Project Highlights
- **Data-Driven Maintenance**
Uses real-world engine data to forecast failures before they happen.
- **ML Model Trained for RUL Prediction**
Compares Support Vector Regressor, Random Forest, and LightGBM — with LightGBM selected for its superior performance.
- **Evaluation Metrics** (on the held-out CMAPSS test set)
  - **MAE**: 17.93
  - **RMSE**: 24.68
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

---

## Feature Engineering

- Constructed RUL as `RUL = EOL - cycle_time`
- Dropped low-variance/near-constant sensors and operational settings (see `pipeline.FEATURES_TO_DROP`)
- Selected sensors showing degradation trends
- Imputed missing values (mean) and applied `StandardScaler` for normalization — both fitted once and reused identically at inference (see `pipeline.py`)
- Ensured feature consistency across train/test sets via the shared `pipeline.py` module, used by both the notebook and `monitor_engine.py`
- **Outlier removal was evaluated and deliberately not applied**: IQR-filtering the degradation sensors disproportionately deleted near-failure rows (the sensors drift *because* the engine is failing, so IQR flags exactly those cycles as "outliers"), which measurably hurt Repair-class recall. See the "Handle Outliers" section of the notebook for the analysis.

---

## Model Development

Tested three regression models (3-fold cross-validated RMSE):
| Model                  | RMSE |
|------------------------|------|
| Support Vector Regressor | 38.72 |
| Random Forest Regressor  | 36.48 |
| **LightGBM**              | **36.24** (selected) |

✅ LightGBM was selected for its efficiency, explainability, and strong predicitve performance. Final hyperparameters are chosen automatically by `GridSearchCV` in the notebook, not hardcoded, so they stay correct if the training data changes.

---

## Environment Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sheldongordon4/predictive-maintenance-system.git
cd predictive-maintenance-system
```

### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate.bat   # Windows
```

### 3️⃣ Install dependencies
If you have `requirements.txt`:
```bash
pip install -r requirements.txt
```

Otherwise:
```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn streamlit joblib
```

---

## Training the Model (Notebook)

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open and run `engine_model_dev.ipynb` top to bottom. It trains and
   evaluates the RUL model, and its final cells save **two** artifacts to the
   project root:
   - `lightgbm_model.pkl` — the trained LightGBM regressor
   - `preprocessor.pkl` — the fitted imputer + `StandardScaler` used to scale
     features before training

   Both come from `pipeline.py` (`pipeline.MODEL_FILENAME` /
   `pipeline.PREPROCESSOR_FILENAME`), the module shared by the notebook and
   `monitor_engine.py`. **They must always be regenerated together** — the
   model's learned split thresholds are only valid on data scaled by the
   preprocessor it was trained alongside.

These files are saved locally and are automatically ignored by Git.

---

## Pretrained Model Loading

If you already have both trained artifacts and simply want to run the app
without retraining:

1. Place `lightgbm_model.pkl` and `preprocessor.pkl` in the project root
   (next to `monitor_engine.py`).
2. Start the app:
   ```bash
   streamlit run monitor_engine.py
   ```

`monitor_engine.py` loads both files via `pipeline.load_artifacts()`. If
either is missing, the app shows an error naming the missing file instead of
starting.

---

## Maintenance Decision Logic

The system classifies engine health based on RUL:

| Predicted RUL  | Maintenance Action       |
|----------------|--------------------------|
| ≤ 15 cycles    | Immediate Repair         |
| 16–47 cycles   | Schedule Inspection      |
| > 47 cycles    | Continue Normal Operation|

The 15-cycle "Repair" cutoff is fixed in `monitor_engine.py`. The 47-cycle
"Inspect" boundary shown above is just the value used during evaluation —
in the running app it's an adjustable sidebar slider (default 30, range
10–70), so operators can tune it to their own risk tolerance without
changing code.

This enables:
- Proactive failure prevention  
- Cost-optimized maintenance planning  
- Improved equipment safety

---

## File Management & Ignore Rules

`.gitignore` ensures unnecessary or large files aren’t tracked:
```
# Python / Jupyter
__pycache__/
.ipynb_checkpoints/

# System
.DS_Store

# Large or sensitive
*.mp4
*.pdf
*.docx
*.pkl
.env

# Data (with an explicit exception for the small CMAPSS sample CSVs)
*.csv
!training_data.csv
!test_data.csv
```

---

## Optional: Using Git LFS for Large Files
If you must version large binaries (e.g., `.pkl`, `.mp4`):
```bash
git lfs install
git lfs track "*.mp4" "*.pkl"
git add .gitattributes
git commit -m "chore: track large files with Git LFS"
git push
```

---

## Run the Streamlit App
Once the model exists:
```bash
streamlit run monitor_engine.py
```

Features:
- Real-time Remaining Useful Life (RUL) prediction  
- Sensor trend visualization  
- Maintenance risk classification  
- Batch CSV predictions  

---

## License
MIT License — see `LICENSE` file for details.

---

## Acknowledgments
- NASA Prognostics Data Repository  
- CMAPSS Dataset (2008)  
- scikit-learn, LightGBM, Streamlit  

---
