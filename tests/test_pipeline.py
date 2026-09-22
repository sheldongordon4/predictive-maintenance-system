"""Tests for pipeline.py -- the feature engineering / inference pipeline
shared between engine_model_dev.ipynb (training) and monitor_engine.py
(serving).

These use small synthetic data and a throwaway preprocessor/model fitted
in-memory; they never touch the real lightgbm_model.pkl / preprocessor.pkl,
which are gitignored and not required to run this suite.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pipeline as pl


def make_raw_df(n_rows=10, engine_id=1):
    """A small dataframe with every raw CMAPSS column, deterministic so
    tests are reproducible."""
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n_rows) for col in pl.RAW_COLUMNS}
    data['engine_id'] = engine_id
    data['cycle_time'] = np.arange(1, n_rows + 1)
    return pd.DataFrame(data)


class TestColumnConstants:
    """Guard the invariants the rest of the pipeline (and monitor_engine.py)
    assumes hold between RAW_COLUMNS / FEATURES_TO_DROP / MODEL_FEATURES."""

    def test_model_features_excludes_dropped_columns(self):
        assert not set(pl.MODEL_FEATURES) & set(pl.FEATURES_TO_DROP)

    def test_model_features_is_subset_of_raw_columns(self):
        assert set(pl.MODEL_FEATURES) <= set(pl.RAW_COLUMNS)

    def test_model_features_has_no_duplicates(self):
        assert len(pl.MODEL_FEATURES) == len(set(pl.MODEL_FEATURES))

    def test_features_to_drop_are_all_real_columns(self):
        # Catches a typo in FEATURES_TO_DROP that would silently no-op.
        assert set(pl.FEATURES_TO_DROP) <= set(pl.RAW_COLUMNS)

    def test_cycle_time_is_a_model_feature_but_eol_is_not(self):
        # cycle_time is a genuine predictive feature; EOL only ever exists
        # as a training-time intermediate and must never reach the model.
        assert 'cycle_time' in pl.MODEL_FEATURES
        assert 'EOL' not in pl.MODEL_FEATURES


class TestDropFeatures:
    def test_drops_listed_columns(self):
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        out = pl.drop_features(df, ['a', 'c'])
        assert list(out.columns) == ['b']

    def test_ignores_columns_not_present(self):
        df = pd.DataFrame({'a': [1], 'b': [2]})
        out = pl.drop_features(df, ['a', 'does_not_exist'])
        assert list(out.columns) == ['b']

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({'a': [1], 'b': [2]})
        pl.drop_features(df, ['a'])
        assert list(df.columns) == ['a', 'b']

    def test_default_drop_list_leaves_exactly_model_features(self):
        df = make_raw_df()
        out = pl.drop_features(df)
        assert set(out.columns) == set(pl.MODEL_FEATURES)


class TestValidateSchema:
    def test_passes_with_all_required_columns(self):
        pl.validate_schema(make_raw_df())  # should not raise

    def test_passes_with_extra_columns_present(self):
        df = make_raw_df()
        df['some_extra_column'] = 1
        pl.validate_schema(df)  # should not raise

    def test_raises_with_missing_column(self):
        df = make_raw_df().drop(columns=['T30'])
        with pytest.raises(ValueError, match='T30'):
            pl.validate_schema(df)

    def test_raises_lists_all_missing_columns(self):
        df = make_raw_df().drop(columns=['T30', 'Nc'])
        with pytest.raises(ValueError, match='Nc'):
            pl.validate_schema(df)


@pytest.fixture
def fitted_preprocessor_and_model():
    """A small but real preprocessor + model, fitted on synthetic data with
    the exact MODEL_FEATURES schema. Stands in for the notebook's actual
    `pipe` / `lgb_model` without needing the committed .pkl artifacts."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(50, len(pl.MODEL_FEATURES))),
        columns=pl.MODEL_FEATURES,
    )
    y = rng.normal(loc=100, size=50)

    preprocessor = make_pipeline(
        SimpleImputer(strategy='mean').set_output(transform='pandas'),
        StandardScaler(),
    )
    # Re-wrap as a DataFrame before fitting, same as engine_model_dev.ipynb
    # does -- StandardScaler's own output isn't set to pandas, so
    # fit_transform() returns a bare ndarray otherwise.
    X_transformed = pd.DataFrame(preprocessor.fit_transform(X), columns=pl.MODEL_FEATURES)

    model = LinearRegression()
    model.fit(X_transformed, y)

    return preprocessor, model


class TestPredictRul:
    def test_returns_series_named_predicted_rul(self, fitted_preprocessor_and_model):
        preprocessor, model = fitted_preprocessor_and_model
        result = pl.predict_rul(make_raw_df(), preprocessor, model)
        assert isinstance(result, pd.Series)
        assert result.name == 'predicted_rul'

    def test_result_aligned_to_input_index(self, fitted_preprocessor_and_model):
        preprocessor, model = fitted_preprocessor_and_model
        df = make_raw_df().set_index(pd.RangeIndex(100, 110))
        result = pl.predict_rul(df, preprocessor, model)
        assert list(result.index) == list(df.index)

    def test_ignores_extra_columns(self, fitted_preprocessor_and_model):
        # A stray ground-truth 'RUL'/'EOL' column, or anything else not in
        # MODEL_FEATURES, must not change the prediction.
        preprocessor, model = fitted_preprocessor_and_model
        df = make_raw_df()
        df_with_extra = df.copy()
        df_with_extra['RUL'] = 999
        df_with_extra['EOL'] = 999
        pd.testing.assert_series_equal(
            pl.predict_rul(df, preprocessor, model),
            pl.predict_rul(df_with_extra, preprocessor, model),
        )

    def test_raises_on_missing_required_column(self, fitted_preprocessor_and_model):
        preprocessor, model = fitted_preprocessor_and_model
        df = make_raw_df().drop(columns=['T30'])
        with pytest.raises(ValueError):
            pl.predict_rul(df, preprocessor, model)


class TestLoadArtifacts:
    def test_raises_informative_error_when_both_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=pl.PREPROCESSOR_FILENAME):
            pl.load_artifacts(artifact_dir=str(tmp_path))

    def test_raises_informative_error_when_model_missing(self, tmp_path):
        joblib.dump({'fake': 'preprocessor'}, tmp_path / pl.PREPROCESSOR_FILENAME)
        with pytest.raises(FileNotFoundError, match=pl.MODEL_FILENAME):
            pl.load_artifacts(artifact_dir=str(tmp_path))

    def test_loads_both_artifacts_successfully(self, tmp_path, fitted_preprocessor_and_model):
        preprocessor, model = fitted_preprocessor_and_model
        joblib.dump(preprocessor, tmp_path / pl.PREPROCESSOR_FILENAME)
        joblib.dump(model, tmp_path / pl.MODEL_FILENAME)

        loaded_preprocessor, loaded_model = pl.load_artifacts(artifact_dir=str(tmp_path))

        df = make_raw_df()
        pd.testing.assert_series_equal(
            pl.predict_rul(df, preprocessor, model),
            pl.predict_rul(df, loaded_preprocessor, loaded_model),
        )
