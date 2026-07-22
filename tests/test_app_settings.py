import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from streamlit.testing.v1 import AppTest

def mock_embed_chunks(chunks, batch_size=64):
    if not chunks:
        return np.array([])
    val = 1.0 / (384 ** 0.5)
    return np.full((len(chunks), 384), val, dtype="float32")

@patch("src.core.webhook.send_plagiarism_alert")
@patch("src.core.embedding_model.embed_chunks", side_effect=mock_embed_chunks)
def test_app_settings_reset_to_defaults(mock_embed, mock_webhook):
    # Instantiate AppTest
    at = AppTest.from_file("app/streamlit_app.py")
    
    # Simulate authentication in session state
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["role"] = "admin"
    at.session_state["page"] = "dashboard"
    
    # Run the app first to render the widgets
    at.run()
    assert not at.exception

    # Modify settings values
    at.slider(key="threshold_slider").set_value(0.85)
    at.checkbox(key="chunk_matrix_checkbox").check()
    at.slider(key="faiss_top_k_slider").set_value(15)
    
    # Run the app to propagate modified values
    at.run()
    assert not at.exception
    
    # Verify settings values are updated
    assert at.session_state["threshold_slider"] == 0.85
    assert at.session_state["chunk_matrix_checkbox"] is True
    assert at.session_state["faiss_top_k_slider"] == 15
    
    # Locate the Reset button and click it
    reset_btn = None
    for btn in at.button:
        if "reset_defaults_button" in btn.key:
            reset_btn = btn
            break
            
    assert reset_btn is not None
    reset_btn.click().run()
    
    # Ensure no exceptions occurred
    assert not at.exception
    
    # Verify settings values have reverted to their defaults in session state
    # Plagiarism threshold defaults to DEFAULT_THRESHOLDS.plagiarism (0.59)
    assert at.session_state["threshold_slider"] == 0.59
    assert at.session_state["chunk_matrix_checkbox"] is False
    assert at.session_state["faiss_top_k_slider"] == 5
