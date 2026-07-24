
from tests.conftest import MockDataFactory
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

# Threshold slider range: min=0.0, max=DEFAULT_THRESHOLDS.medium (0.75), default=0.59
_DEFAULT_THRESHOLD = 0.59
_MODIFIED_THRESHOLD = 0.70  # within valid slider range [0.0, 0.75]
_MODIFIED_FAISS_TOP_K = 10  # within valid range [1, 20]



@patch("src.core.webhook.send_plagiarism_alert")
@patch("src.core.embedding_model.embed_chunks", side_effect=MockDataFactory.embed_chunks)
def test_app_settings_reset_to_defaults(mock_embed, mock_webhook):
    """Verify the Reset to Factory Defaults button restores all settings."""
    at = AppTest.from_file("app/streamlit_app.py")

    # Simulate an authenticated admin session
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["role"] = "admin"

    # Initial render
    at.run(timeout=30)
    assert not at.exception

    # Modify settings to non-default values
    at.slider(key="threshold_slider").set_value(_MODIFIED_THRESHOLD)
    at.checkbox(key="chunk_matrix_checkbox").check()
    at.slider(key="faiss_top_k_slider").set_value(_MODIFIED_FAISS_TOP_K)

    # Re-render to propagate the changes
    at.run(timeout=30)
    assert not at.exception

    # Confirm the values were accepted
    assert at.session_state["threshold_slider"] == _MODIFIED_THRESHOLD
    assert at.session_state["chunk_matrix_checkbox"] is True
    assert at.session_state["faiss_top_k_slider"] == _MODIFIED_FAISS_TOP_K

    # Locate the Reset to Factory Defaults button (guard against None keys)
    reset_btn = next(
        (btn for btn in at.button if btn.key and "reset_defaults_button" in btn.key),
        None,
    )
    assert (
        reset_btn is not None
    ), "Reset to Factory Defaults button was not found in the sidebar"

    # Click the reset button and re-render
    reset_btn.click().run(timeout=30)
    assert not at.exception

    # Confirm all settings have reverted to factory defaults
    assert at.session_state["threshold_slider"] == _DEFAULT_THRESHOLD
    assert at.session_state["chunk_matrix_checkbox"] is False
    assert at.session_state["faiss_top_k_slider"] == 5
