import streamlit as st
from parser import extract_text
from utils.chunker import chunk_text
from utils.embeddings import create_embeddings
from ingest import store_chunks
from rag import generate_answer

st.title("Research Assistant")
st.caption("Handles up to 100 pages · Scanned PDFs ❌ · Photo-based PDFs ❌")

uploaded_file = st.file_uploader(
    "Upload Research Paper",
    type="pdf"
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

    stored = store_chunks(
        chunks,
        embeddings
    )

    st.success(
        f"{stored} chunks stored in ChromaDB"
    )

    st.divider()

    question = st.text_input(
        "Ask a question about the document"
    )

    if question:
        answer = generate_answer(question)
        st.subheader("Answer")
        st.write(answer)