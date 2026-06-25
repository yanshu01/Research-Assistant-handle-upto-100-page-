import chromadb

client = chromadb.PersistentClient(path="vector_db")

def store_chunks(chunks, embeddings):
    try:
        client.delete_collection(name="research_papers")
    except:
        pass

    collection = client.get_or_create_collection(
        name="research_papers"
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return len(chunks)