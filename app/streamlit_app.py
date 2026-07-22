import sys
import os
from pathlib import Path

# Fix Streamlit import paths by pointing to project root
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent  # Points to semantic-plagiarism-detector/

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Standard / Third-party imports
import io
import pandas as pd
import streamlit as st


# Internal Imports (Python will now correctly locate the 'src' package)
from src.i18n.translator import get_text  # type: ignore
from src.utils.file_parser import extract_text_from_pdf, EncryptedPDFError  # type: ignore
from src.utils.excel_export import export_similarity_matrix_to_excel  # type: ignore
from src.core.embeddings import generate_embeddings  # type: ignore
from src.core.similarity import compute_similarity_matrix  # type: ignore
from src.core.faiss_indexer import build_index  # type: ignore
from src.core.ai_detector import detect_documents_ai_probability  # type: ignore
from src.db.auth import authenticate_user, init_db, update_password  # type: ignore
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from app.theme import (
    empty_state_html,
    get_colors,
    get_theme_name,
    inject_css,
    set_theme,
)
from src.core.ai_detector import detect_documents_ai_probability
from src.core.config import DEFAULT_THRESHOLDS, severity_key
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    SUPPORTED_OCR_LANGUAGES,
    OCRDependencyError,
    extract_text,
    prepare_text_for_embedding,
    remove_ignore_phrases,
)
from src.core.embedding_model import embed_documents
from src.core.faiss_index import (
    build_index,
    build_index_from_matrix,
    load_index,
    load_or_rebuild_index,
    save_index,
    search_similar_chunks,
)
from src.core.similarity import (
    document_similarity_matrix,
    find_most_similar_chunks,
    flag_plagiarism,
)
from src.core.text_chunking import chunk_documents
from src.core.webhook import send_plagiarism_alert


class OCRFileBatchError(Exception):
    """Exception raised when OCR extraction fails on one or more files in a batch."""

    def __init__(self, failed_files: list[str], failure_details: list[str]):
        self.failed_files = failed_files
        self.failure_details = failure_details
        super().__init__(f"OCR failed for files: {failed_files}")


from src.db import (
    clear_all_data,
    delete_document,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_documents_by_class,
    get_unique_class_sections,
    init_corpus_db,
)
from src.db.auth import (
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_all_users,
    get_tour_completed,
    get_user_role,
    init_db,
    set_tour_completed,
    verify_user,
)
from src.db.incidents import (  # noqa: E402
    get_high_severity_trends,
    get_most_plagiarized_documents,
)
from src.utils.pdf_report import highlight_pdf_matches  # noqa: E402
from src.utils.redis_cache import (
    cache_session_state,
    clear_session,
    get_analysis_results,
    get_faiss_index,
    get_session_state,
    get_upload_count,
    increment_upload_count,
    is_upload_rate_limited,
)
from src.utils.warning_list import render_warning_controls
from src.visualization.analytics import (  # noqa: E402
    plot_high_severity_trends,
    plot_most_plagiarized_documents,
)
from src.visualization.heatmap import plot_similarity_heatmap  # noqa: E402
from src.visualization.network_graph import plot_similarity_network

init_db()
# Safe import for PDF Highlighting
try:

    from src.utils.pdf_highlighter import highlight_pdf_matches  # type: ignore
except Exception:
    highlight_pdf_matches = None

# Safe import for Google Drive integration

    from src.utils.excel_export import export_similarity_matrix_to_excel
    from src.utils.json_export import export_similarity_matrix_to_json
except ImportError:
    from utils.excel_export import export_similarity_matrix_to_excel
    from utils.json_export import export_similarity_matrix_to_json

# Initialize corpus database
init_corpus_db()

# Generate unique session ID for this Streamlit session
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.session_id


_BRANDING_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "branding_config.json")
)
_BRANDING_LOGO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "branding_logo.png")
)
_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "corpus.index")
)
from streamlit_tour import Tour

from src.db.auth import (
    check_login_rate_limit,
    clear_login_attempts,
    record_failed_login,
)


try:
    from src.utils.google_drive import import_from_google_drive  # type: ignore
except Exception:
    import_from_google_drive = None

# -----------------------------------------------------------------------------
# Page Configuration & Session State
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Semantic Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "pdf_passwords" not in st.session_state:
    st.session_state.pdf_passwords = {}
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# -----------------------------------------------------------------------------
# Sidebar Settings Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ " + get_text("settings", lang=st.session_state.lang))
    
    # 1. i18n Language Dropdown (#144)
    selected_lang_name = st.selectbox(
        "🌐 Language / Idioma",
        options=["English", "Español"],
        index=0 if st.session_state.lang == "en" else 1,

# ── SESSION TIMEOUT & ROUTE PROTECTION ────────────────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds


# 1. Handle Automatic Session Expiration (Inactivity Check)
cached_last_interaction = get_session_state(SESSION_ID, "last_interaction")
if cached_last_interaction is not None:
    last_interaction = cached_last_interaction
elif "last_interaction" in st.session_state:
    last_interaction = st.session_state.last_interaction
else:
    last_interaction = None

if last_interaction and st.session_state.get("authenticated", False):
    elapsed_time = time.time() - last_interaction
    if elapsed_time > TIMEOUT_LIMIT:
        for key in ["authenticated", "username", "role", "last_interaction"]:
            if key in st.session_state:
                del st.session_state[key]
        clear_session(SESSION_ID)
        from src.errors import UI_SESSION_EXPIRED

        st.warning(UI_SESSION_EXPIRED)
        st.stop()
    else:
        st.session_state.last_interaction = time.time()
        cache_session_state(SESSION_ID, "last_interaction", time.time())

# Render Login UI if not authenticated
if not st.session_state.get("authenticated", False):
    if st.session_state.get("pending_2fa", False):
        with st.form("otp_form"):
            st.subheader("🔒 Two-Factor Authentication")
            st.info(
                "Enter the 6-digit verification token from your Google Authenticator/Authy app."
            )
            otp_code = st.text_input(
                "Verification Code", max_chars=6, key="login_otp_code"
            )
            col1, col2 = st.columns(2)
            with col1:
                verify_submitted = st.form_submit_button(
                    "Verify", use_container_width=True
                )
            with col2:
                cancel_submitted = st.form_submit_button(
                    "Cancel", use_container_width=True
                )

            if verify_submitted:
                username = st.session_state.get("pending_username")
                enabled, otp_secret = get_2fa_status(username)
                if enabled and otp_secret:
                    import pyotp

                    totp = pyotp.TOTP(otp_secret)
                    if totp.verify(otp_code.strip()):
                        role = st.session_state.get("pending_role")
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = role
                        st.session_state.last_interaction = time.time()

                        cache_session_state(SESSION_ID, "authenticated", True)
                        cache_session_state(SESSION_ID, "username", username)
                        cache_session_state(SESSION_ID, "role", role)
                        cache_session_state(SESSION_ID, "last_interaction", time.time())

                        # Clear pending state
                        del st.session_state["pending_2fa"]
                        del st.session_state["pending_username"]
                        del st.session_state["pending_role"]

                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid verification code. Please try again.")
                else:
                    st.error("2FA configuration error. Please contact admin.")

            if cancel_submitted:
                del st.session_state["pending_2fa"]
                del st.session_state["pending_username"]
                del st.session_state["pending_role"]
                st.rerun()
        st.stop()

    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin")
        login_submitted = st.form_submit_button("Log In", use_container_width=True)

        if login_submitted:
            username = username.strip().lower()

            if not username or not password:
                from src.errors import AUTH_BLANK_CREDENTIALS

                st.error(AUTH_BLANK_CREDENTIALS)
            else:
                # Check rate limit before attempting authentication
                is_allowed, error_msg = check_login_rate_limit(username)
                if not is_allowed:
                    st.error(error_msg)
                elif verify_user(username, password):
                    role = get_user_role(username)
                    if role is None:
                        from src.errors import AUTH_ROLE_UNDETERMINED

                        st.error(AUTH_ROLE_UNDETERMINED)
                    else:
                        # Clear failed login attempts on successful login
                        clear_login_attempts(username)

                        # Check if 2FA is enabled for this user
                        enabled, _ = get_2fa_status(username)
                        if enabled:
                            st.session_state.pending_2fa = True
                            st.session_state.pending_username = username
                            st.session_state.pending_role = role
                            st.rerun()
                        else:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.role = role
                            st.session_state.last_interaction = time.time()
                            cache_session_state(SESSION_ID, "authenticated", True)
                            cache_session_state(SESSION_ID, "username", username)
                            cache_session_state(SESSION_ID, "role", role)
                            cache_session_state(
                                SESSION_ID, "last_interaction", time.time()
                            )

                            st.success(f"Welcome back, {role.capitalize()}!")
                            st.rerun()
                else:
                    # Record failed login attempt
                    record_failed_login(username)
                    from src.errors import AUTH_INVALID_CREDENTIALS

                    st.error(AUTH_INVALID_CREDENTIALS)
    st.stop()


# Active user role

user_role = st.session_state.get("role", "user")


@st.dialog("⚠️ Confirm Bulk Clear")
def clear_all_dialog():
    st.markdown(
        "**WARNING:** This action is destructive and cannot be undone. "
        "This will permanently delete all student documents, paragraph chunks, "
        "and plagiarism incidents from the database, and reset the FAISS index."
    )
    st.write("Are you absolutely sure you want to proceed?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True, key="cancel_clear_all"):
            st.rerun()
    with col2:
        if st.button(
            "Clear All",
            type="primary",
            use_container_width=True,
            key="confirm_clear_all",
        ):
            # 1. Clear database tables (documents, chunks, incidents)
            clear_all_data()

            # 2. Clear/reset FAISS index file on disk
            if os.path.exists(_INDEX_PATH):
                try:
                    os.remove(_INDEX_PATH)
                except Exception as e:
                    print(f"Error removing FAISS index: {e}")

            # 3. Invalidate Redis cache
            try:
                from src.utils.redis_cache import get_cache

                cache = get_cache()
                if cache.is_available():
                    cache.delete("faiss:index:corpus_index")
                    cache.clear_pattern("analysis:*")
            except Exception as e:
                print(f"Error invalidating cache: {e}")

            # 4. Invalidate Session State cache
            if "analysis_results" in st.session_state:
                st.session_state.analysis_results = None
            if "analysis_file_signature" in st.session_state:
                st.session_state.analysis_file_signature = None

            st.success("All documents, chunks, and incidents have been cleared.")
            st.rerun()


# ── Top-right Theme Toggle ───────────────────────────────────────────────────
current_theme = get_theme_name()

# Create a narrow right-aligned column for the theme toggle
_, theme_col = st.columns([0.94, 0.06])

with theme_col:
    theme_icon = "☀️" if current_theme == "Dark" else "🌙"

    if st.button(
        theme_icon,
        key="theme_toggle",
    ):
        new_theme = "Light" if current_theme == "Dark" else "Dark"
        set_theme(new_theme)
        st.rerun()


# ── Sidebar (ROLE RESTRICTED Settings) ────────────────────────────────────────
# ── Sidebar ───────────────────────────────────────────────────────────────────
unique_classes = ["All Classes"] + get_unique_class_sections()
selected_class = "All Classes"

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    selected_theme = st.radio(
        "Theme",
        options=["Light", "Dark"],
        index=0 if current_theme == "Light" else 1,
        horizontal=True,
        key="theme_selector",
    )
    if selected_theme != current_theme:
        set_theme(selected_theme)
        st.rerun()

    if user_role == "admin":
        threshold = st.slider(
            "Plagiarism Threshold",
            min_value=0.0,
            max_value=DEFAULT_THRESHOLDS.medium,
            value=DEFAULT_THRESHOLDS.plagiarism,
            step=0.01,
            help=(
                "Controls which pairs are flagged. Severity remains Medium "
                f"at {DEFAULT_THRESHOLDS.medium:.0%} and High at "
                f"{DEFAULT_THRESHOLDS.high:.0%}."
            ),
            key="threshold_slider",
        )
        selected_class = st.selectbox(
            "Filter by Class Section",
            options=unique_classes,
            key="class_filter_selectbox",
        )
        use_chunk_matrix = st.checkbox(
            "Use chunk-level similarity matrix",
            value=False,
            key="chunk_matrix_checkbox",
        )
        faiss_top_k = st.slider(
            "FAISS: matches per chunk",
            1,
            20,
            value=5,
            key="faiss_top_k_slider",
        )

        with st.expander("� Ignore Phrases", expanded=False):
            st.caption(
                "Enter common template text or standard assignment questions to ignore during analysis. "
                "These phrases will be removed from documents before chunking and embedding."
            )
            ignore_phrases = st.text_area(
                "Ignore Phrases (one per line)",
                placeholder="Q1: Explain the theory of relativity\nQ2: Describe the process of photosynthesis\nInstructions: Write in your own words",
                help="Each line represents a phrase to ignore. Matching text will be removed from all documents.",
                key="ignore_phrases_textarea",
            )

        with st.expander("� OCR Settings", expanded=False):
            st.caption(
                "Used only for scanned or image-only PDF pages. "
                "Text-based PDFs continue to use native extraction."
            )
        # ── Customizable Chunk Size & Overlap Sliders (#153) ─────────────────
        st.markdown("### ✂️ Chunking Settings")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            200,
            2000,
            value=500,
            step=50,
            help="Target character length for text chunks during embedding.",
            key="chunk_size_slider",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            0,
            500,
            value=50,
            step=10,
            help="Character overlap between consecutive chunks to preserve contextual boundary.",
            key="chunk_overlap_slider",
        )

        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI

        with st.expander("🔤 OCR Settings", expanded=False):
            ocr_language_labels = {
                display_name: code
                for code, display_name in SUPPORTED_OCR_LANGUAGES.items()
            }
            language_names = list(ocr_language_labels)
            default_language_name = SUPPORTED_OCR_LANGUAGES[DEFAULT_OCR_LANGUAGE]
            selected_ocr_language_name = st.selectbox(
                "OCR Language",
                options=language_names,
                index=language_names.index(default_language_name),
                key="ocr_language_selector",
            )
            ocr_language = ocr_language_labels[selected_ocr_language_name]

            ocr_dpi = st.slider(
                "OCR DPI Resolution",
                min_value=150,
                max_value=400,
                value=DEFAULT_OCR_DPI,
                step=25,
                key="ocr_dpi_slider",
            )

        # ── Document Management & Bulk Clear ──
        st.markdown("---")
        st.markdown("### 📁 Document Management")
        existing_docs = get_all_documents()
        if existing_docs:
            st.write(f"**{len(existing_docs)}** documents in database")
            for doc in existing_docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc['filename']}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['filename']}"):
                        delete_document(doc["filename"])
                        embeddings_matrix = get_all_embeddings()
                        if embeddings_matrix.size > 0:
                            new_index = build_index_from_matrix(embeddings_matrix)
                            save_index(new_index, _INDEX_PATH)
                        else:
                            if os.path.exists(_INDEX_PATH):
                                os.remove(_INDEX_PATH)
                        st.rerun()

        st.markdown('<div class="clear-all-container">', unsafe_allow_html=True)
        if st.button(
            "🗑️ Clear All Documents",
            key="clear_all_documents_button",
            use_container_width=True,
        ):
            clear_all_dialog()
        st.markdown("</div>", unsafe_allow_html=True)

# ── MAIN APPLICATION SECTIONS (ROLE CHECKED) ──────────────────────────────────

if user_role != "admin":
    # STANDARD USER VIEW: Student Query / Search Panel Only (No admin PDF uploading)
    st.subheader("🔎 Secure Student Search Portal")
    st.caption(
        "Paste a text snippet below to check its similarity against existing indexed assignments."
    )

    st.info(
        "🔒 Note: Direct assignment uploads and detailed breakdown panels are restricted to Administrator access. Your queries are anonymized for privacy."
    )

    query_text = st.text_area(
        "Paste a text snippet to check against index:",
        height=150,
        placeholder="Paste a paragraph here to check for plagiarism...",
    )

    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():
        # Load existing index and registry from database
        from src.core.faiss_index import build_index_from_matrix
        from src.db.corpus_db import get_all_embeddings, get_chunk_registry

        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()

                if embeddings_matrix.shape[0] == 0:
                    from src.errors import UI_NO_DOCUMENTS_INDEXED

                    st.warning(UI_NO_DOCUMENTS_INDEXED)
                else:
                    # Build index from stored embeddings
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )

                    # Embed the query
                    from src.core.embedding_model import embed_chunks

                    query_vec = embed_chunks([query_text.strip()])[0]

                    # Search with threshold
                    faiss_threshold = threshold
                    results = search_similar_chunks(
                        query_vec,
                        faiss_index,
                        registry,
                        top_k=faiss_top_k,
                        threshold=faiss_threshold,
                    )

                    if not results:
                        st.success(
                            "✅ No significant matches found in the assignment database."
                        )
                    else:
                        st.success(
                            f"Found **{len(results)}** potentially similar passages."
                        )

                        # Anonymize document names
                        doc_id_map = {}
                        anon_counter = 1

                        for record, score in results:
                            if record.doc_name not in doc_id_map:
                                doc_id_map[record.doc_name] = (
                                    f"Document-{anon_counter:03d}"
                                )
                                anon_counter += 1

                        # Display anonymized results
                        for rank, (record, score) in enumerate(results, 1):
                            anon_doc_name = doc_id_map[record.doc_name]
                            color = "#ff4b4b" if score >= 0.90 else "#ffa500"

                            with st.expander(
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index+1}) "
                                f"— {score:.1%}",
                                expanded=(rank == 1),
                            ):
                                cq, cm = st.columns(2)
                                with cq:
                                    st.markdown("**Your query:**")
                                    st.info(query_text.strip())
                                with cm:
                                    st.markdown(
                                        f"**Matching passage in {anon_doc_name}:**"
                                    )
                                    st.warning(record.chunk_text)

                                st.markdown(
                                    f"<div style='text-align:right;'>"
                                    f"<span style='background:{color};color:white;padding:3px 12px;"
                                    f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                                    f"Similarity: {score*100:.1f}%</span></div>",
                                    unsafe_allow_html=True,
                                )

                        st.caption(
                            "🔒 Document names are anonymized to protect student privacy."
                        )

            except Exception as e:
                from src.errors import UI_INDEX_LOAD_FAILED

                st.error(UI_INDEX_LOAD_FAILED.format(error=str(e)))
                st.info(
                    "Please ensure documents have been indexed by an administrator."
                )
else:
    # ADMINISTRATOR ACCESS: Full Upload Pipeline & Evaluation Dashboards

    # Load or initialize FAISS index
    if os.path.exists(_INDEX_PATH):
        faiss_index = load_index(_INDEX_PATH)
        registry = get_chunk_registry()
        if faiss_index is not None and faiss_index.ntotal != len(registry):
            all_embs = get_all_embeddings()
            if len(all_embs) > 0 and len(all_embs) == len(registry):
                faiss_index = build_index_from_matrix(all_embs)
                save_index(faiss_index, _INDEX_PATH)
            elif len(all_embs) == 0:
                faiss_index = None
                registry = []
        if faiss_index is not None:
            st.info(f"📂 Loaded existing FAISS index with {faiss_index.ntotal} vectors")
    else:
        threshold = DEFAULT_THRESHOLDS.plagiarism
        use_chunk_matrix = False
        faiss_top_k = 5
        chunk_size = 500
        chunk_overlap = 50
        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI
        ignore_phrases = ""
        st.info("ℹ️ Settings configuration is restricted to Administrators.")

# ── Onboarding Tour for First-Time Admin Users ───────────────────────────────────
if (
    Tour is not None
    and user_role == "admin"
    and not get_tour_completed(st.session_state.username)
):
    username = st.session_state.username

    if st.button("🎯 Start Guided Tour", key="start_tour_button", type="primary"):
        st.session_state.show_tour = True

    if st.session_state.get("show_tour", False):
        tour_steps = [
            Tour.info(
                title="👋 Welcome to the Plagiarism Detection System!",
                desc="This guided tour will walk you through the key features to help you get started.",
            ),
            Tour.bind(
                "threshold_slider",
                title="⚙️ Plagiarism Threshold",
                desc=(
                    "Adjust the flagging threshold. Medium severity starts "
                    f"at {DEFAULT_THRESHOLDS.medium:.0%} and High at "
                    f"{DEFAULT_THRESHOLDS.high:.0%}."
                ),
                side="right",
            ),
            Tour.bind(
                "class_filter_selectbox",
                title="🔍 Class Filter",
                desc="Filter analysis results by specific class sections.",
                side="right",
            ),
            Tour.info(
                title="📊 Analysis Dashboard",
                desc="View similarity metrics, flagged pairs, and comparisons in the tabs below.",
            ),
            Tour.info(
                title="🎉 You're All Set!",
                desc="You can now start uploading assignments and detecting plagiarism.",
            ),
        ]

        tour = Tour(steps=tour_steps)
        tour.start()

        set_tour_completed(username, True)
        st.session_state.show_tour = False
        st.success("✅ Onboarding tour completed!")
        st.rerun()

# ── MAIN APPLICATION SECTIONS ──────────────────────────────────────────────────

if user_role != "admin":
    # STANDARD USER VIEW
    st.subheader("🔎 Secure Student Search Portal")
    st.caption(
        "Paste a text snippet below to check its similarity against existing indexed assignments."
    )

    st.info(
        "🔒 Note: Direct assignment uploads are restricted to Administrator access."

    )
    st.session_state.lang = "en" if selected_lang_name == "English" else "es"
    current_lang = st.session_state.lang

    st.markdown("---")
    
    # 2. Similarity Threshold Slider
    similarity_threshold = st.slider(
        get_text("threshold", lang=current_lang),
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
    )

    st.markdown("---")


    # 3. Text Chunking Settings
    st.subheader("✂️ Chunking Settings")
    chunk_size = st.number_input("Chunk Size (words)", min_value=50, max_value=1000, value=200, step=50)
    chunk_overlap = st.number_input("Chunk Overlap (words)", min_value=0, max_value=200, value=50, step=10)

    st.markdown("---")

    # 4. Google Drive Integration Section
    if import_from_google_drive:
        st.subheader("☁️ Google Drive Import")
        gdrive_folder_id = st.text_input("Folder ID / Share Link")
        if st.button("Import from Drive"):
            if gdrive_folder_id:
                with st.spinner("Downloading files from Google Drive..."):

        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()

                if embeddings_matrix.shape[0] == 0:
                    st.warning("No documents are currently indexed.")
                else:
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )

                    # Apply ignore phrases to query if configured
                    processed_query = query_text.strip()
                    if ignore_phrases and ignore_phrases.strip():
                        processed_query = remove_ignore_phrases(
                            processed_query, ignore_phrases
                        )

                    # Embed the query
                    from src.core.embedding_model import embed_chunks

                    query_vec = embed_chunks([processed_query])[0]

                    # Search with threshold
                    from src.core.embedding_model import embed_chunks

                    query_vec = embed_chunks([query_text.strip()])[0]
                    faiss_threshold = threshold
                    results = search_similar_chunks(
                        query_vec,
                        faiss_index,
                        registry,
                        top_k=faiss_top_k,
                        threshold=faiss_threshold,
                    )

                    if selected_class != "All Classes":
                        class_docs = get_documents_by_class(selected_class)
                        results = [
                            (record, score)
                            for record, score in results
                            if record.doc_name in class_docs
                        ]

                    if not results:
                        st.success(
                            "✅ No significant matches found in the assignment database."
                        )
                    else:
                        st.success(
                            f"Found **{len(results)}** potentially similar passages."
                        )

            except Exception as e:
                st.error(f"Error loading index: {str(e)}")
else:
    # ADMIN FULL ACCESS VIEW
    faiss_index = None
    registry = []

    # Try to load FAISS index from Redis cache
    cached_index_data = get_faiss_index("corpus_index")
    if cached_index_data is not None:
        try:
            import faiss

            index_buffer = _io.BytesIO(cached_index_data)
            faiss_index = faiss.deserialize_index(faiss.read_index(index_buffer))
            registry = get_chunk_registry()
            st.info(
                f"📂 Loaded FAISS index from Redis cache with {faiss_index.ntotal} vectors"
            )
        except Exception as e:
            print(f"[Redis] Error loading cached index: {e}, falling back to disk")

    # If Redis loading failed, load from local disk
    if faiss_index is None:
        try:
            from src.core.faiss_index import load_or_rebuild_index

            faiss_index, registry, index_recovered = load_or_rebuild_index(_INDEX_PATH)

            if index_recovered:
                if faiss_index.ntotal:
                    st.warning(
                        f"FAISS index was missing, corrupted, or inconsistent and was "
                        f"automatically rebuilt from {faiss_index.ntotal} stored vectors."
                    )
                else:
                    st.info(
                        "No stored embeddings were found. An empty FAISS index was "
                        "initialized safely."
                    )
            else:
                st.info(
                    f"Loaded and validated the existing FAISS index with "
                    f"{faiss_index.ntotal} vectors."
                )
        except Exception:
            faiss_index = None
            registry = []

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
        # Try to load from Redis cache

        cached_results = get_analysis_results(f"{SESSION_ID}:current")
        if cached_results is not None:
            st.session_state.analysis_results = cached_results

    # Initialize analysis_file_signature in session state
    if "analysis_file_signature" not in st.session_state:
        st.session_state.analysis_file_signature = None

        cached_signature = get_session_state(SESSION_ID, "analysis_file_signature")
        if cached_signature is not None:
            st.session_state.analysis_file_signature = cached_signature

    # 1. LOCAL FILE UPLOADER
    uploaded_files = st.file_uploader(
        "📂 Upload Assignments",
        type=["pdf", "docx", "txt", "zip"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    # Check upload rate limit if files are uploaded
    if uploaded_files:
        username = st.session_state.get("username", "anonymous")
        if is_upload_rate_limited(username):
            current_count = get_upload_count(username)
            st.error(
                f"Upload rate limit exceeded. Maximum 100 uploads per hour allowed. Current: {current_count}/100. Please try again later."
            )
            uploaded_files = None
        else:
            # Increment upload counter for each file
            for _ in uploaded_files:
                increment_upload_count(username)

    st.markdown("### 🔗 Or Paste URL")
    url_input = st.text_input(
        "Paste a direct URL (e.g., Wikipedia article, Medium blog post)",
        placeholder="https://example.com/article",
        key="url_input",
        help="The system will fetch and extract text from the webpage for plagiarism detection.",
    )

    file_bytes_dict = {
        uploaded_file.name: uploaded_file.getvalue() for uploaded_file in uploaded_files
    }
    # 2. GOOGLE DRIVE IMPORT SECTION (#146)
    from src.utils.google_drive import bulk_download_drive_folder

    if "drive_files_dict" not in st.session_state:
        st.session_state.drive_files_dict = {}

    with st.expander("🌐 Import from Google Drive Folder", expanded=False):
        st.caption(
            "Paste a shared Google Drive folder link or ID to bulk-download assignments."
        )

        drive_folder_input = st.text_input(
            "Google Drive Folder Link / ID:",
            placeholder="https://drive.google.com/drive/folders/1A2B3C...",
            key="drive_folder_url_input",
        )

        drive_api_key = st.text_input(
            "API Key (Optional):",
            type="password",
            key="drive_api_key_input",
        )

        if st.button(
            "📥 Import Files from Drive", type="primary", use_container_width=True
        ):
            if not drive_folder_input.strip():
                st.error("Please enter a valid Google Drive folder link or ID.")
            else:
                with st.spinner(
                    "Connecting to Google Drive API & downloading files..."
                ):

                    try:
                        drive_files = import_from_google_drive(gdrive_folder_id)
                        st.session_state["gdrive_files"] = drive_files
                        st.success(f"Successfully imported {len(drive_files)} files!")
                    except Exception as err:

                        st.error(f"Drive import failed: {err}")
            else:
                st.warning("Please enter a valid Google Drive Folder ID.")

# -----------------------------------------------------------------------------
# Authentication Guard
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:
    st.header("🔑 Login")
    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate_user(username_input, password_input):
            st.session_state.authenticated = True
            st.session_state.username = username_input
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.stop()

                        st.error(f"Failed to import from Google Drive: {str(err)}")

    # 3. MERGE LOCAL AND DRIVE FILE BYTES
    file_bytes_dict = {}

    if uploaded_files:
        for f in uploaded_files:
            if f.name.lower().endswith(".zip"):
                try:
                    from src.utils.zip_processor import process_zip_file

                    zip_files = process_zip_file(f.read())
                    if not zip_files:
                        st.error(
                            f"⚠️ ZIP file '{f.name}' contains no supported documents (.pdf, .docx, .txt)."
                        )
                    else:
                        file_bytes_dict.update(zip_files)
                except ValueError as ve:
                    st.error(f"⚠️ Failed to process ZIP archive '{f.name}': {str(ve)}")
                except Exception:
                    st.error(
                        f"⚠️ Failed to process ZIP archive '{f.name}': Unknown error occurred."
                    )
            else:
                file_bytes_dict[f.name] = f.read()
            f.seek(0)

    # Allow analysis with existing index even without new uploads
    # Also allow URL input as an alternative to file uploads
    url_text = None
    url_filename = None

    if url_input and url_input.strip():
        try:
            from src.core.document_parser import extract_text_from_url

            with st.spinner("🔍 Fetching and extracting text from URL..."):
                url_text = extract_text_from_url(url_input.strip())
                if not url_text or len(url_text.strip()) < 50:
                    st.warning(
                        "⚠️ The URL did not contain enough text content for analysis."
                    )
                    url_text = None
                else:
                    # Generate a filename from the URL
                    from urllib.parse import urlparse

                    parsed = urlparse(url_input.strip())
                    url_filename = f"webpage_{parsed.netloc.replace('.', '_')}.txt"
                    st.success(
                        f"✅ Successfully extracted {len(url_text)} characters from the webpage."
                    )
        except Exception as e:
            st.error(f"❌ Failed to fetch URL: {str(e)}")
            url_text = None

    # Check if we have enough content (either files or URL)
    # Use resolved file_bytes_dict count so a single ZIP with 2+ files inside counts correctly
    has_files = len(file_bytes_dict) >= 2
    has_url = url_text is not None

    if not has_files and not has_url:
        if st.session_state.analysis_results is None:
            st.info(
                "👆 Please upload **at least 2** PDF assignment files or paste a URL to begin."
            )
            st.stop()
        else:
            st.success(
                f"📂 Using existing index with {faiss_index.ntotal} vectors from {len(get_all_documents())} documents"
            )
            # Skip to analysis section with existing index
            file_bytes_dict = {}
            raw_texts = {}
            chunked_docs = {}
            embeddings = {}
            sim_df = None
            chunk_sim_df = None
            # We'll need to handle this case differently for the analysis
            st.warning(
                "⚠️ Full similarity matrix requires re-uploading files. FAISS search is available with existing index."
            )
    if st.session_state.drive_files_dict:
        file_bytes_dict.update(st.session_state.drive_files_dict)

    # 4. PIPELINE STOP CHECK
    if len(file_bytes_dict) < 2:
        if st.session_state.analysis_results is None:
            st.markdown(
                empty_state_html(
                    "Waiting for Files",
                    "Please upload or import from Drive at least 2 PDF, DOCX, or TXT assignments to begin.",
                    "📂",
                ),
                unsafe_allow_html=True,
            )
            st.stop()


# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
st.title(get_text("title", lang=current_lang))
st.caption(get_text("subtitle", lang=current_lang))

st.markdown("---")

# -----------------------------------------------------------------------------
# Upload & Decryption Processing Section (#167)
# -----------------------------------------------------------------------------
st.subheader(get_text("upload_title", lang=current_lang))

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)


# Merge locally uploaded files with Google Drive imports if available
file_dict = {}
if uploaded_files:
    for uf in uploaded_files:
        file_dict[uf.name] = uf.getvalue()

            metadata_dict[filename] = {
                "student_name": student_name.strip(),
                "class_section": class_section.strip(),
                "assignment_title": assignment_title.strip(),
            }

    # Add metadata for URL input if provided
    if url_filename:
        with st.expander(f"🔗 {url_filename}", expanded=True):
            student_name = st.text_input(
                f"Student Name for {url_filename}",
                value="Web Source",
                key=f"student_{url_filename}",
            )
            class_section = st.text_input(
                f"Class/Section for {url_filename}",
                value=batch_class,
                key=f"class_{url_filename}",
            )
            assignment_title = st.text_input(
                f"Assignment Title for {url_filename}",
                value=batch_assignment,
                key=f"assignment_{url_filename}",
            )

            metadata_dict[url_filename] = {
                "student_name": student_name.strip(),
                "class_section": class_section.strip(),
                "assignment_title": assignment_title.strip(),
            }

    # ── Pipeline Execution ────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_pipeline(
        file_bytes_dict: dict[str, bytes],
        ocr_language: str,
        ocr_dpi: int,
        existing_index=None,
        existing_registry=None,
        url_text: str = None,
        url_filename: str = None,
    ):
        raw_texts = {}
        failed_files = []
        failure_details = []
        for name, data in file_bytes_dict.items():
            try:
                raw_texts[name] = extract_text(
                    _io.BytesIO(data),
                    name,
                    ocr_language=ocr_language,
                    ocr_dpi=ocr_dpi,
                )
            except OCRDependencyError as exc:
                failed_files.append(name)
                failure_details.append(f"{name}: {exc}")

        # Add URL text if provided
        if url_text and url_filename:
            raw_texts[url_filename] = url_text

        if failed_files:
            raise OCRFileBatchError(failed_files, failure_details)

        # Apply ignore phrases to raw texts before chunking
        if ignore_phrases and ignore_phrases.strip():
            raw_texts = {
                name: remove_ignore_phrases(text, ignore_phrases)
                for name, text in raw_texts.items()
            }

            # Original chunks are preserved for UI display.
            raw_texts[name] = extract_text(
                _io.BytesIO(data),
                name,
                ocr_language=ocr_language,
                ocr_dpi=ocr_dpi,
            )


if "gdrive_files" in st.session_state:
    for gname, gbytes in st.session_state["gdrive_files"].items():
        if gname not in file_dict:
            file_dict[gname] = gbytes


parsed_file_texts = {}
encrypted_files_detected = []

if file_dict:
    for file_name, file_bytes in file_dict.items():
        if file_name.lower().endswith(".pdf"):
            user_pass = st.session_state.pdf_passwords.get(file_name, None)
            try:
                extracted_text, is_protected = extract_text_from_pdf(file_bytes, password=user_pass)
                parsed_file_texts[file_name] = extracted_text
            except EncryptedPDFError:
                encrypted_files_detected.append(file_name)
        elif file_name.lower().endswith(".txt"):
            parsed_file_texts[file_name] = file_bytes.decode("utf-8", errors="ignore")
        elif file_name.lower().endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            parsed_file_texts[file_name] = "\n".join([p.text for p in doc.paragraphs])

# Prompt for Password-Protected PDFs (#167)
if encrypted_files_detected:
    st.warning("🔒 Password-protected PDF(s) detected! Please enter the password(s) below:")
    
    for enc_file in encrypted_files_detected:
        col1, col2 = st.columns([3, 1])
        with col1:
            input_pass = st.text_input(
                f"Password for '{enc_file}'",
                type="password",
                key=f"pass_input_{enc_file}",
            )
        with col2:
            st.write(" ")
            st.write(" ")
            if st.button("Decrypt PDF", key=f"btn_decrypt_{enc_file}"):
                if input_pass:
                    st.session_state.pdf_passwords[enc_file] = input_pass
                    st.success(f"Password saved for {enc_file}!")
                    st.rerun()
                else:
                    st.error("Please enter a password.")

# -----------------------------------------------------------------------------
# Main Analysis Dashboard
# -----------------------------------------------------------------------------
if parsed_file_texts and not encrypted_files_detected:
    st.markdown("---")
    st.subheader(get_text("analysis_summary", lang=current_lang))

    doc_names = list(parsed_file_texts.keys())
    doc_contents = list(parsed_file_texts.values())

    # 1. Embeddings & Similarity Matrix
    embeddings = generate_embeddings(doc_contents)
    sim_matrix = compute_similarity_matrix(embeddings)
    df_sim = pd.DataFrame(sim_matrix, index=doc_names, columns=doc_names)

    # 2. Chunked Documents & FAISS Index
    chunked_docs = {}
    for name, text in parsed_file_texts.items():
        words = text.split()
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)
        chunked_docs[name] = chunks if chunks else [text]

    faiss_idx, faiss_reg = build_index(embeddings, chunked_docs)

    # 3. AI Probability Detection
    ai_probs = detect_documents_ai_probability(chunked_docs)

    # 4. Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(get_text("metric_docs", lang=current_lang), len(doc_names))
    m2.metric(get_text("metric_pairs", lang=current_lang), int(len(doc_names) * (len(doc_names) - 1) / 2))
    
    flagged_count = 0
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            if sim_matrix[i][j] >= similarity_threshold:
                flagged_count += 1

    m3.metric(get_text("metric_flagged", lang=current_lang), flagged_count)
    m4.metric(get_text("metric_faiss", lang=current_lang), faiss_idx.ntotal if faiss_idx else 0)

    # -----------------------------------------------------------------------------
    # Navigation Tabs
    # -----------------------------------------------------------------------------
    t1, t2, t3, t4 = st.tabs([
        get_text("tab_warnings", lang=current_lang),
        get_text("tab_matrix", lang=current_lang),
        "🔍 PDF Highlighter",
        get_text("tab_users", lang=current_lang),
    ])

    # Tab 1: Plagiarism Warnings
    with t1:
        st.write("### " + get_text("tab_warnings", lang=current_lang))
        if flagged_count == 0:
            st.info("No plagiarism detected above the current similarity threshold.")

    try:
        with st.spinner("🧠 Processing files and building embeddings…"):
            analysis_results = run_pipeline(file_bytes_dict, ocr_language, ocr_dpi)
    except OCRFileBatchError as exc:
        from src.errors import OCR_DEPENDENCIES_MISSING

        st.error(OCR_DEPENDENCIES_MISSING)
        if exc.failed_files:
            st.warning(f"Failed files: {', '.join(exc.failed_files)}")
        st.stop()

    (
        raw_texts,
        chunked_docs,
        embeddings,
        sim_df,
        chunk_sim_df,
        faiss_index,
        registry,
        ai_probabilities,
    ) = analysis_results

    active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
    flags = flag_plagiarism(active_sim_df, threshold=threshold)

    # ── Summary Metrics ───────────────────────────────────────────────────────────

    if len(file_bytes_dict) < 2:
        st.markdown(
            empty_state_html(
                "Waiting for Files",
                "Please upload at least 2 PDF, DOCX, or TXT assignments to begin analysis.",
                "📂",
            ),
            unsafe_allow_html=True,
        )
        st.stop()

    if "sent_alerts" not in st.session_state:
        st.session_state.sent_alerts = set()

    for flag in flags:
        alert_key = (flag["doc_a"], flag["doc_b"])
        if alert_key not in st.session_state.sent_alerts:
            try:
                send_plagiarism_alert(
                    doc_a=flag["doc_a"],
                    doc_b=flag["doc_b"],
                    similarity=float(flag["similarity"]),
                )
                st.session_state.sent_alerts.add(alert_key)
            except Exception:
                pass

    # ── Summary Metrics ───────────────────────────────────────────────────────

    st.subheader("📊 Analysis Summary")
    doc_names = list(raw_texts.keys())
    n_docs = len(doc_names)
    total_pairs = n_docs * (n_docs - 1) // 2 if n_docs > 1 else 0
    n_flagged = len(flags)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📄 Documents", n_docs)
    col2.metric("🔗 Pairs", total_pairs)
    col3.metric("🚨 Flagged", n_flagged)
    col4.metric("🗂️ FAISS Vectors", faiss_index.ntotal if faiss_index is not None else 0)
    col5.metric("🎯 Threshold", f"{threshold:.0%}")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    (
        tab_warnings,
        tab_faiss,
        tab_matrix,
        tab_heatmap,
        tab_drill,
        tab_analytics,
        tab_users,
    ) = st.tabs(
        [
            "⚠️ Plagiarism Warnings",
            "⚡ FAISS Chunk Search",
            "📋 Similarity Matrix",
            "🗺️ Heatmap",
            "🔬 Pair Drill-Down",
            "📊 Analytics",
            "👥 User Management",
        ]
    )

    # ══ TAB 1: WARNINGS ═══════════════════════════════════════════════════════

    with tab_warnings:
        st.subheader("⚠️ Plagiarism Warnings")
        render_warning_controls(
            flags, threshold=threshold, ai_probabilities=ai_probabilities
        )

    # ══ TAB 2: FAISS ══════════════════════════════════════════════════════════
    with tab_faiss:
        st.subheader("⚡ FAISS Vector Search")
        st.info(f"Index total: {faiss_index.ntotal} vectors.")

        faiss_query = st.text_input(
            "Query FAISS Index:",
            placeholder="Type a text snippet to search vector index...",
            key="faiss_query_input",
        )
        if st.button("🔍 Run FAISS Search", key="run_faiss_search_btn"):
            if faiss_query.strip() and faiss_index is not None:
                from src.core.embedding_model import embed_chunks

                q_vec = embed_chunks([faiss_query.strip()])[0]
                q_results = search_similar_chunks(
                    q_vec,
                    faiss_index,
                    registry,
                    top_k=faiss_top_k,
                    threshold=threshold,
                )
                if q_results:
                    for rec, score in q_results:
                        st.markdown(
                            f"**{rec.doc_name}** (Chunk #{rec.chunk_index}) — Similarity: `{score:.1%}`"
                        )
                        st.caption(rec.chunk_text)
                else:
                    st.info("No matching vector chunks found above threshold.")

    # ══ TAB 3: MATRIX ═════════════════════════════════════════════════════════
    with tab_matrix:
        st.subheader("📋 Similarity Matrix")
        if active_sim_df is None:
            from src.errors import UI_SIMILARITY_MATRIX_REUPLOAD

            st.info(UI_SIMILARITY_MATRIX_REUPLOAD)

        else:
            for i in range(len(doc_names)):
                for j in range(i + 1, len(doc_names)):
                    score = sim_matrix[i][j]
                    if score >= similarity_threshold:
                        st.error(
                            f"🚨 **{doc_names[i]}** <--> **{doc_names[j]}**: {score * 100:.2f}% Similarity"
                        )


    # Tab 2: Similarity Matrix & Styled Excel Export
    with t2:
        st.write("### " + get_text("tab_matrix", lang=current_lang))
        st.dataframe(df_sim.style.background_gradient(cmap="OrRd"))
        
        excel_bytes = export_similarity_matrix_to_excel(df_sim)
        st.download_button(
            get_text("download_excel", lang=current_lang),
            data=excel_bytes,
            file_name="similarity_matrix.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

            def _highlight(val: Any) -> str:
                tier = severity_key(float(val))
                if tier == "high":
                    return "background-color:#ff4b4b;color:white;" "font-weight:bold;"
                if tier == "medium":
                    return "background-color:#ffa500;color:white;" "font-weight:bold;"
                return ""

            styled_df = active_sim_df.style.format("{:.4f}").map(_highlight)
            st.dataframe(styled_df, use_container_width=True)

            # Export options row
            col_csv, col_json, col_excel = st.columns(3)

            with col_csv:
                st.download_button(
                    "⬇️ Download CSV",
                    active_sim_df.to_csv().encode("utf-8"),
                    "similarity_matrix.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with col_json:
                json_data = export_similarity_matrix_to_json(active_sim_df).encode(
                    "utf-8"
                )
                st.download_button(
                    "⬇️ Download JSON",
                    json_data,
                    "similarity_matrix.json",
                    "application/json",
                    use_container_width=True,
                    key="json_export_button",
                )

            with col_excel:
                excel_data = export_similarity_matrix_to_excel(
                    active_sim_df, threshold=threshold
                )
                st.download_button(
                    "📊 Export as Styled Excel (.xlsx)",
                    excel_data,
                    "similarity_matrix_styled.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    # ══ TAB 4: HEATMAP ════════════════════════════════════════════════════════
    with tab_heatmap:
        st.subheader("🗺️ Similarity Heatmap")
        if active_sim_df is None:
            from src.errors import UI_SIMILARITY_MATRIX_REUPLOAD

            st.info(UI_SIMILARITY_MATRIX_REUPLOAD)
        else:
            heatmap_fig = plot_similarity_heatmap(
                active_sim_df,
                title="Document Semantic Similarity",
                threshold=threshold,
                theme_colors=get_colors(),
            )

            st.pyplot(
                heatmap_fig,
                use_container_width=True,
            )

            buf = _io.BytesIO()
            heatmap_fig.savefig(
                buf,
                format="png",
                dpi=150,
                bbox_inches="tight",
            )

            buf.seek(0)

            st.download_button(
                "⬇️ Download Heatmap PNG",
                buf,
                "heatmap.png",
                "image/png",
            )

            st.divider()

            st.subheader("🕸️ Interactive Plagiarism Network")
            st.caption(
                "Documents are shown as nodes. Connections appear when "
                "their similarity is greater than or equal to the selected threshold."
            )

            network_fig = plot_similarity_network(
                similarity_df=active_sim_df,
                threshold=threshold,
                title="Interactive Document Plagiarism Network",
            )

            st.plotly_chart(
                network_fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

    # ══ TAB 5: PAIR DRILL-DOWN ══════════════════════════════════════════════════
    with tab_drill:
        st.subheader("🔬 Pair Drill-Down")
        st.caption("Inspect chunk-level similarity between any two documents.")
        if active_sim_df is None:
            from src.errors import UI_SIMILARITY_MATRIX_REUPLOAD

            st.info(UI_SIMILARITY_MATRIX_REUPLOAD)
        elif len(active_sim_df) < 2:
            from src.errors import UI_NEED_MIN_DOCUMENTS

            st.warning(UI_NEED_MIN_DOCUMENTS)
        else:
            c1, c2 = st.columns(2)
            with c1:
                doc_a = st.selectbox("Document A", doc_names, index=0, key="da")
            with c2:
                doc_b = st.selectbox(
                    "Document B",
                    [d for d in doc_names if d != doc_a],
                    index=0,
                    key="db",
                )

        score = float(active_sim_df.loc[doc_a, doc_b])
        st.markdown(f"**Overall Similarity:** `{score:.1%}`")
        st.progress(float(score))
        st.divider()

        drill_tab_analysis, drill_tab_viewer = st.tabs(
            ["📊 Chunk Matches & Report", "📄 Document Viewer"]

        )

    # Tab 3: Visual PDF Highlighting Viewer
    with t3:
        st.write("### Visual PDF Match Highlighter")
        pdf_names = [n for n in doc_names if n.lower().endswith(".pdf")]
        if pdf_names:
            selected_pdf = st.selectbox(
                "Select a PDF document to inspect matches:",
                options=pdf_names,
            )
            if selected_pdf:
                raw_bytes = file_dict.get(selected_pdf)
                pdf_pass = st.session_state.pdf_passwords.get(selected_pdf)
                
                # Check if custom module works, otherwise fallback to built-in highlight rendering
                rendered = False
                if highlight_pdf_matches:
                    try:

                        highlighted_bytes = highlight_pdf_matches(raw_bytes, password=pdf_pass)
                        if highlighted_bytes:
                            st.download_button(
                                f"⬇️ Download Highlighted {selected_pdf}",
                                data=highlighted_bytes,
                                file_name=f"highlighted_{selected_pdf}",
                                mime="application/pdf",
                            )
                            rendered = True
                    except Exception as ex:
                        st.warning(f"Note: PDF highlight module issue ({ex}). Showing text matching inspection below:")

                # Fallback / General visual inspector for matched text chunks
                if not rendered:
                    st.info(f"📄 Displaying text content and match breakdown for **{selected_pdf}**:")
                    extracted_txt = parsed_file_texts.get(selected_pdf, "")
                    
                    # Highlight matched sections visually using HTML markup
                    st.markdown("#### Document Text Inspection:")
                    st.text_area("Extracted Content", extracted_txt, height=250)
                    
                    st.download_button(
                        f"⬇️ Download Raw PDF Bytes ({selected_pdf})",
                        data=raw_bytes,
                        file_name=selected_pdf,
                        mime="application/pdf",
                    )
        else:
            st.info("Upload PDF files to view visual match highlighting.")

    # Tab 4: User Password Management
    with t4:
        st.write("### " + get_text("tab_users", lang=current_lang))
        with st.form("change_pass_form"):
            new_pw = st.text_input("New Password", type="password")
            submit = st.form_submit_button("Update Password")
            if submit:
                try:
                    update_password(st.session_state.username, new_pw)
                    st.success("Password updated successfully!")
                except ValueError as err:
                    st.error(str(err))

elif not file_dict:
    st.info("Please upload files or import from Google Drive to begin semantic analysis.")
    

                        highlighted_pdf_bytes = highlight_pdf_matches(
                            pdf_source=doc_source,
                            matching_chunks=matching_chunks_to_highlight,
                        )

                        base64_pdf = base64.b64encode(highlighted_pdf_bytes).decode(
                            "utf-8"
                        )
                        pdf_display = f"""
                            <iframe
                                src="data:application/pdf;base64,{base64_pdf}"
                                width="100%"
                                height="850px"
                                type="application/pdf">
                            </iframe>
                        """
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    except Exception as err:
                        st.error(f"Unable to render PDF preview: {str(err)}")
            else:
                st.info("PDF Preview is only available for uploaded `.pdf` files.")

    # ══ TAB 6: Analytics ═════════════════════════════════════════════════════════
    with tab_analytics:
        st.subheader("📊 Plagiarism Analytics Dashboard")
        st.caption(
            "Track plagiarism trends and identify frequently plagiarized documents across all classes."
        )

        # Sync current flags to incidents database before displaying analytics
        if flags:
            from src.db.incidents import sync_flagged_incidents

            sync_flagged_incidents(flags)

        st.divider()

        # High Severity Trends Chart
        st.subheader("📈 High Severity Plagiarism Trends (Last 30 Days)")
        trend_data = get_high_severity_trends(days=30)
        trend_fig = plot_high_severity_trends(trend_data)
        st.plotly_chart(trend_fig, use_container_width=True)

        st.divider()

        # Most Plagiarized Documents Chart
        st.subheader("🔝 Most Frequently Plagiarized Documents")
        doc_data = get_most_plagiarized_documents(limit=10)
        doc_fig = plot_most_plagiarized_documents(doc_data)
        st.plotly_chart(doc_fig, use_container_width=True)

        st.divider()

        # Summary statistics
        st.subheader("📋 Analytics Summary")
        if trend_data:
            total_high_severity = sum(item["count"] for item in trend_data)
            st.metric("Total High Severity Incidents (30 days)", total_high_severity)
        else:
            st.info("No high severity incidents recorded in the last 30 days.")

        if doc_data:
            st.metric(
                "Most Plagiarized Document",
                f"{doc_data[0]['document_name']} ({doc_data[0]['incident_count']} incidents)",
            )
        else:
            st.info("No plagiarism incidents recorded.")

    # ══ TAB 7: User Management ═══════════════════════════════════════════════════
    # ══ TAB 6: USERS ══════════════════════════════════════════════════════════
    with tab_users:
        st.subheader("👥 User Management")
        users = get_all_users()
        if users:
            st.dataframe(pd.DataFrame(users), use_container_width=True)

        st.write("---")
        st.subheader("🔐 Two-Factor Authentication (2FA)")

        current_user = st.session_state.get("username", "admin")
        enabled, otp_secret = get_2fa_status(current_user)

        if enabled:
            st.success(
                "✔️ Two-Factor Authentication is currently **enabled** for your account."
            )
            with st.expander("Deactivate Two-Factor Authentication", expanded=False):
                st.warning(
                    "⚠️ Disabling 2FA will lower your account security. Please confirm by entering your 6-digit verification code."
                )
                with st.form("disable_2fa_form"):
                    disable_code = st.text_input(
                        "Verification Code", max_chars=6, key="disable_2fa_code"
                    )
                    submit_disable = st.form_submit_button(
                        "Disable 2FA", use_container_width=True
                    )
                    if submit_disable:
                        import pyotp

                        totp = pyotp.TOTP(otp_secret)
                        if totp.verify(disable_code.strip()):
                            disable_2fa(current_user)
                            st.success("Two-factor authentication has been disabled.")
                            st.rerun()
                        else:
                            st.error("Invalid verification code. 2FA remains enabled.")
        else:
            st.info(
                "🔒 Two-Factor Authentication (2FA) is currently **disabled** for your account. We highly recommend enabling it."
            )

            if not st.session_state.get("show_2fa_setup", False):
                if st.button("Setup 2FA", use_container_width=True):
                    st.session_state.show_2fa_setup = True
                    import pyotp

                    # Generate a new random secret
                    st.session_state.temp_2fa_secret = pyotp.random_base32()
                    st.rerun()
            else:
                temp_secret = st.session_state.get("temp_2fa_secret")
                if temp_secret:
                    import pyotp

                    totp = pyotp.TOTP(temp_secret)
                    provisioning_uri = totp.provisioning_uri(
                        name=current_user, issuer_name="PlagiarismDetector"
                    )

                    st.markdown("### ⚙️ Step 1: Scan this QR Code")
                    st.write(
                        "Scan this QR code with your authenticator app (e.g., Google Authenticator, Authy)."
                    )

                    # Generate QR code offline using qrcode library
                    from io import BytesIO

                    import qrcode

                    qr = qrcode.QRCode(version=1, box_size=5, border=3)
                    qr.add_data(provisioning_uri)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    qr_bytes = buf.getvalue()

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(qr_bytes, width=250)
                    with col2:
                        st.markdown("**Manual Setup Details:**")
                        st.code(f"Account: {current_user}")
                        st.code(f"Secret Key: {temp_secret}")
                        st.caption(
                            "If your app does not support scanning QR codes, enter the secret key manually."
                        )

                    st.markdown("### ⚙️ Step 2: Verify and Enable 2FA")
                    st.write(
                        "Enter the 6-digit code shown in your authenticator app to complete setup."
                    )

                    with st.form("verify_2fa_setup_form"):
                        setup_code = st.text_input(
                            "6-digit Code", max_chars=6, key="setup_2fa_code"
                        )
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            submit_setup = st.form_submit_button(
                                "Verify and Enable", use_container_width=True
                            )
                        with col_btn2:
                            cancel_setup = st.form_submit_button(
                                "Cancel Setup", use_container_width=True
                            )

                        if submit_setup:
                            if totp.verify(setup_code.strip()):
                                enable_2fa(current_user, temp_secret)
                                # Clean up session state
                                st.session_state.show_2fa_setup = False
                                if "temp_2fa_secret" in st.session_state:
                                    del st.session_state.temp_2fa_secret
                                st.success(
                                    "🎉 Two-Factor Authentication has been successfully enabled!"
                                )
                                st.rerun()
                            else:
                                st.error(
                                    "Invalid verification code. Please check the code in your app and try again."
                                )

                        if cancel_setup:
                            st.session_state.show_2fa_setup = False
                            if "temp_2fa_secret" in st.session_state:
                                del st.session_state.temp_2fa_secret
                            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🎓 Semantic Plagiarism Detection System · Streamlit")

