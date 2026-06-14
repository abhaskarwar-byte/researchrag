import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("research_documents")

results = collection.query(
    query_texts=["What analyzes data?"],
    n_results=2
)

print(results["documents"])