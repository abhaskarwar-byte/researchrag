import chromadb


def reset_database():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    try:
        client.delete_collection(
            "research_documents"
        )
    except:
        pass

    client.create_collection(
        "research_documents"
    )

    return True