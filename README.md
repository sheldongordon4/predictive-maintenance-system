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

✅ LightGBM was selected for its efficiency, explainability, and strong predicitve performance.

---

## ⚙️ Environment Setup

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
2. Open and run:
   - `engine_model_dev.ipynb` → trains and evaluates the RUL model.  
3. Save the trained model:
   ```python
   import joblib
   joblib.dump(model, "lightgbm_model.pkl")
   ```

This file will be saved locally and is automatically ignored by Git.

---

## Pretrained Model Loading

If you already have a trained model and simply want to run the app without retraining:

1. Place your model file here:
   ```
   predictive-maintenance-system/models/lightgbm_model.pkl
   ```
   (Create the `models/` folder if it doesn’t exist.)

2. In `monitor_engine.py`, ensure the model is loaded like this:
   ```python
   import joblib
   model = joblib.load("models/lightgbm_model.pkl")
   ```

3. Then start the app:
   ```bash
   streamlit run monitor_engine.py
   ```

> 💡 If you want to use a different path, update `monitor_engine.py` accordingly under the model-loading section.

---

## Maintenance Decision Logic

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
