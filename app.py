import tempfile
import streamlit as st
from parser import extract_text
from utils.chunker import chunk_text
from utils.embeddings import create_embeddings
from ingest import store_chunks
from rag import generate_answer

st.title("Research Assistant")
st.caption("Handles up to 100 pages · Scanned PDFs ❌ · Photo-based PDFs ❌")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "db_path" not in st.session_state:
    st.session_state.db_path = tempfile.mkdtemp()

def reset_app():
    st.session_state.uploader_key += 1
    st.session_state.db_path = tempfile.mkdtemp()
    st.rerun()

if st.button("Reset / Upload New PDF"):
    reset_app()

uploaded_file = st.file_uploader(
    "Upload Research Paper",
    type="pdf",
    key=f"pdf_uploader_{st.session_state.uploader_key}"
)

if uploaded_file:
    text = extract_text(uploaded_file)
    st.success("PDF Loaded Successfully")

    chunks = chunk_text(text)
    st.write("Total Chunks:", len(chunks))

    embeddings = create_embeddings(chunks)
    st.write("Embeddings Created Successfully")
    st.write("Embedding Shape:", embeddings.shape)

    st.subheader("First Chunk")
    st.write(chunks[0])

    stored = store_chunks(chunks, embeddings, st.session_state.db_path)
    st.success(f"{stored} chunks stored in ChromaDB")

    st.divider()

    question = st.text_input(
        "Ask a question about the document",
        key=f"question_input_{st.session_state.uploader_key}"
    )

    if question:
        answer = generate_answer(question, st.session_state.db_path)
        st.subheader("Answer")
        st.write(answer)