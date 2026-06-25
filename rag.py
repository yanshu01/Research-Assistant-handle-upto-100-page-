import os
import chromadb
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from utils.embeddings import model

load_dotenv()

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=api_key)


def search_chunks(question, db_path, top_k=5):
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name="research_papers"
    )

    count = collection.count()

    if count == 0:
        return []

    question_embedding = model.encode([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, count)
    )

    retrieved_chunks = results["documents"][0]

    first_chunk_data = collection.get(ids=["chunk_0"])
    first_chunk = first_chunk_data["documents"][0]

    return list(dict.fromkeys([first_chunk] + retrieved_chunks))


def generate_answer(question, db_path):
    chunks = search_chunks(question, db_path)

    if not chunks:
        return "Please upload and process a PDF first."

    context = "\n\n".join(chunks)

    prompt = f"""
You are a research assistant.

Answer only from the provided document context.
If the answer is not available, say:
"I don't know based on this document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content