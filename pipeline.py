"""Shared feature engineering / inference pipeline for the CMAPSS RUL model.

This module is the single source of truth for everything that has to stay
identical between training and serving:

  - the raw CMAPSS sensor column names
  - which columns are dropped before modeling
  - the exact feature set/order the model is fit and predicted on
  - loading the fitted preprocessor (imputer + scaler) and trained model
  - running the full inference pipeline end to end

Both ``engine_model_dev.ipynb`` (training) and ``monitor_engine.py``
(serving) import from here instead of redefining this logic, so the two
can no longer silently drift apart the way they had: the notebook fit a
SimpleImputer + StandardScaler before training the LightGBM model, but
only the model was ever saved -- the Streamlit app was calling
``model.predict()`` on raw, unscaled sensor values.
"""

import os

import joblib
import pandas as pd

# Raw CMAPSS column names, in file order (matches train_data.txt /
# test_data.txt / training_data.csv / test_data.csv).
RAW_COLUMNS = [
    "engine_id", "cycle_time", "op_set_1", "op_set_2", "op_set_3",
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
    "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
    "PCNfR_dmd", "W31", "W32",
]

# Columns dropped before modeling: identifiers, operational settings, and
# sensors that are constant/near-constant in this dataset. This must stay
# identical to what was used to produce preprocessor.pkl / lightgbm_model.pkl
# -- changing it means retraining and re-exporting both artifacts.
FEATURES_TO_DROP = [
    "engine_id", "op_set_1", "op_set_2", "op_set_3",
    "T2", "P2", "P15", "epr", "farB", "Nf_dmd", "PCNfR_dmd",
]

# The exact feature columns, in the exact order, the model was trained on.
MODEL_FEATURES = [c for c in RAW_COLUMNS if c not in FEATURES_TO_DROP]

MODEL_FILENAME = "lightgbm_model.pkl"
PREPROCESSOR_FILENAME = "preprocessor.pkl"


def drop_features(df, columns_to_drop=FEATURES_TO_DROP):
    """Drop selected columns from a dataframe (columns not present are ignored).

    Returns a new dataframe without the dropped columns.
    """
    return df.drop(columns=[c for c in columns_to_drop if c in df.columns])


def _artifact_dir():
    return os.path.dirname(os.path.abspath(__file__))


def load_artifacts(artifact_dir=None):
    """Load the fitted preprocessor (imputer + scaler) and trained model.

    Both files are produced together at the end of ``engine_model_dev.ipynb``
    and must come from the same training run -- the model's learned split
    thresholds are only meaningful on data scaled by this exact preprocessor.

    Returns (preprocessor, model).
    """
    artifact_dir = artifact_dir or _artifact_dir()
    preprocessor_path = os.path.join(artifact_dir, PREPROCESSOR_FILENAME)
    model_path = os.path.join(artifact_dir, MODEL_FILENAME)

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Missing '{preprocessor_path}'. Re-run engine_model_dev.ipynb "
            f"through the model-training cell -- it saves {PREPROCESSOR_FILENAME} "
            f"alongside {MODEL_FILENAME}, and the two must come from the same "
            "training run."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing '{model_path}'. Re-run engine_model_dev.ipynb to train "
            "and export it."
        )

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model


def validate_schema(df):
    """Raise a clear error if required raw sensor columns are missing."""
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded data is missing required column(s): {missing}. "
            f"Expected raw CMAPSS columns: {RAW_COLUMNS}"
        )


def predict_rul(df, preprocessor, model):
    """Run the full inference pipeline on raw sensor data.

    Validates the schema, selects the pinned model feature set (ignoring any
    extra columns such as 'engine_id', 'RUL' or 'EOL' that may be present),
    applies the SAME imputer + scaler fitted during training, then predicts.

    Returns a pandas Series of predicted RUL, aligned to df's index.
    """
    validate_schema(df)

    X = df[MODEL_FEATURES]
    X_transformed = preprocessor.transform(X)
    X_transformed = pd.DataFrame(X_transformed, columns=MODEL_FEATURES, index=df.index)

    predictions = model.predict(X_transformed)
    return pd.Series(predictions, index=df.index, name="predicted_rul")
