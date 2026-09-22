"""Smoke test for monitor_engine.py using Streamlit's own AppTest framework.

This exists because a plain HTTP check (e.g. `curl` against a running
`streamlit run` process) does NOT catch script-execution errors: the initial
HTTP response is just the static page shell, served before the Python script
ever runs. That gap let a real bug reach production -- get_artifacts() was
called (and could emit a cache-miss spinner, itself a Streamlit command)
before st.set_page_config(), which Streamlit requires to be the first
Streamlit command in the script. `curl`-based checks in CI reported success
throughout; only actually executing the script (as AppTest does, and as a
real browser session does via websocket) surfaces this class of error.

Requires lightgbm_model.pkl / preprocessor.pkl to be present (gitignored,
not committed) -- skipped otherwise, e.g. in CI, which intentionally runs
without them (see test_app_shows_friendly_error_without_model_artifacts).
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

import pipeline

ARTIFACTS_PRESENT = os.path.exists(pipeline.MODEL_FILENAME) and os.path.exists(
    pipeline.PREPROCESSOR_FILENAME
)


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="model artifacts not present (gitignored)")
def test_app_runs_without_exception():
    at = AppTest.from_file("monitor_engine.py")
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Engine Health Monitoring Dashboard"


@pytest.mark.skipif(
    ARTIFACTS_PRESENT,
    reason="model artifacts present in repo root; pipeline.load_artifacts() always "
    "resolves relative to pipeline.py's own location (not cwd), so their absence "
    "can't be simulated in-process here -- this only actually runs in CI, which "
    "intentionally has no committed .pkl files.",
)
def test_app_shows_friendly_error_without_model_artifacts():
    at = AppTest.from_file("monitor_engine.py")
    at.run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(
        pipeline.MODEL_FILENAME in e.value or pipeline.PREPROCESSOR_FILENAME in e.value
        for e in at.error
    )
