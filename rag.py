import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from utils.embeddings import model

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

client = chromadb.PersistentClient(path="vector_db")


def search_chunks(question, top_k=5):
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

    # Always include the first chunk for document context
    first_chunk_data = collection.get(ids=["chunk_0"])
    first_chunk = first_chunk_data["documents"][0]

    final_chunks = [first_chunk] + retrieved_chunks

    # Remove duplicates while preserving order
    return list(dict.fromkeys(final_chunks))


def generate_answer(question):
    chunks = search_chunks(question)

    if not chunks:
        return "Please upload and process a PDF first. No document data found in ChromaDB."

    context = "\n\n".join(chunks)

    prompt = f"""
You are a research assistant.

Answer the question using only the provided context.
If the answer is not present in the context, say: "I don't know based on this document."

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