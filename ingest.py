import chromadb

def store_chunks(chunks, embeddings, db_path):
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection(name="research_papers")
    except Exception:
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