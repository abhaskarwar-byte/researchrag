import chromadb
import ollama

# =====================================
# CONNECT TO CHROMADB
# =====================================

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="research_documents"
)

# =====================================
# USER QUESTION
# =====================================

question = input("Ask a question: ")

# =====================================
# HYDE
# =====================================

hyde_response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": f"""
Generate a short hypothetical answer.

Question:
{question}

Return only the answer.
"""
        }
    ]
)

hyde_text = hyde_response["message"]["content"]

# =====================================
# EMBED HYPOTHETICAL ANSWER
# =====================================

embedding = ollama.embeddings(
    model="nomic-embed-text",
    prompt=hyde_text
)["embedding"]

# =====================================
# RETRIEVE
# =====================================

results = collection.query(
    query_embeddings=[embedding],
    n_results=3
)

context = "\n\n".join(results["documents"][0])

# =====================================
# FINAL RAG PROMPT
# =====================================

prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Be concise and factual.
- If the answer is not present in the context, respond exactly:
  "I could not find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

response = ollama.chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]

# =====================================
# ANSWER
# =====================================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(answer)

# =====================================
# SOURCES
# =====================================

if "could not find that information" not in answer.lower():

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    shown = set()

    for metadata in results["metadatas"][0]:

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")
        chunk = metadata.get("chunk", "N/A")

        key = (source, page, chunk)

        if key not in shown:
            shown.add(key)

            print(
                f"{source} | "
                f"Page: {page} | "
                f"Chunk: {chunk}"
            )

# =====================================
# DEVELOPER DEBUG
# =====================================

print("\n" + "=" * 60)
print("DEVELOPER DEBUG")
print("=" * 60)

print(f"\nQuestion: {question}")

print("\nHyDE Answer:")
print(hyde_text)

print("\nRetrieved References:")

for metadata in results["metadatas"][0]:

    source = metadata.get("source", "Unknown")
    page = metadata.get("page", "N/A")
    chunk = metadata.get("chunk", "N/A")

    print(
        f"Source: {source} | "
        f"Page: {page} | "
        f"Chunk: {chunk}"
    )