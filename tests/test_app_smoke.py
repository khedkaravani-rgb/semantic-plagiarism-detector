import io
import pytest
import numpy as np
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from reportlab.pdfgen import canvas

def generate_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    # Write text in lines to make it clean
    words = text.split()
    lines = []
    for i in range(0, len(words), 8):
        lines.append(" ".join(words[i:i+8]))
    
    y = 750
    for line in lines:
        c.drawString(50, y, line)
        y -= 20
        
    c.showPage()
    c.save()
    return buf.getvalue()

def mock_embed_chunks(chunks, batch_size=64):
    if not chunks:
        return np.array([])
    # Return L2-normalised vectors of shape (len(chunks), 384)
    # 1.0 / sqrt(384) ensures L2 norm is 1.0.
    val = 1.0 / (384 ** 0.5)
    return np.full((len(chunks), 384), val, dtype="float32")

@pytest.fixture(autouse=True)
def clean_smoke_test_env():
    import os
    from src.db.corpus_db import clear_all_data
    clear_all_data()
    index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "corpus.index"))
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
        except Exception:
            pass
    yield
    clear_all_data()
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
        except Exception:
            pass


@patch("src.core.webhook.send_plagiarism_alert")
@patch("src.core.embedding_model.embed_chunks", side_effect=mock_embed_chunks)
def test_app_smoke(mock_embed, mock_webhook):
    # Instantiate AppTest
    at = AppTest.from_file("app/streamlit_app.py")
    
    # Simulate authentication in session state
    at.session_state["authenticated"] = True
    at.session_state["username"] = "admin"
    at.session_state["role"] = "admin"
    at.session_state["page"] = "dashboard"
    
    # Initial run to display uploader
    at.run()
    
    # Assert uploader is found
    assert len(at.file_uploader) > 0
    
    # Generate 2 PDFs with text > 20 words to trigger plagiarism warnings
    import uuid
    sample_text = (
        "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural "
        "intelligence displayed by humans and other animals. This field of computer science is "
        "highly focused on study, research and development of agents that perceive their environment "
        "and take actions that maximize their chance of successfully achieving their goals."
    )
    unique_suffix = f" Run identifier: {uuid.uuid4()}"
    pdf1 = generate_pdf(sample_text + unique_suffix + " doc1")
    pdf2 = generate_pdf(sample_text + unique_suffix + " doc2")
    
    # Upload via the native AppTest FileUploader.upload method
    at.file_uploader[0].upload("doc1.pdf", pdf1, "application/pdf")
    at.file_uploader[0].upload("doc2.pdf", pdf2, "application/pdf")
    
    # Execute full pipeline
    at.run(timeout=30)
    
    # Ensure no exceptions occurred during pipeline execution
    assert not at.exception
    
    # Check if metrics are rendered correctly (should be 5 summary metrics)
    assert len(at.metric) >= 5
    
    # Verify warnings are present and severity badge is correct (🔴 High since similarity = 100%)
    badge_found = False
    for md in at.markdown:
        if "High" in md.value:
            badge_found = True
            break
    assert badge_found, "High plagiarism warning badge was not rendered"
    
    # Find the "Run FAISS Search" button and click it to ensure FAISS search works
    faiss_btn = None
    for btn in at.button:
        if "Run FAISS" in btn.label:
            faiss_btn = btn
            break
            
    assert faiss_btn is not None
    faiss_btn.click().run()
    assert not at.exception
    
    # Verify webhook alert was triggered
    mock_webhook.assert_called_once()
