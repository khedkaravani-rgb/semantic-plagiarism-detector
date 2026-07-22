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

init_db()
# Safe import for PDF Highlighting
try:
    from src.utils.pdf_highlighter import highlight_pdf_matches  # type: ignore
except Exception:
    highlight_pdf_matches = None

# Safe import for Google Drive integration
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
    