import chromadb
import ollama


def query_document(question):

    # =====================================
    # CONNECT TO CHROMADB
    # =====================================

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="research_documents"
    )

    # =====================================
    # HYDE
    # =====================================

    hyde_response = ollama.chat(
        model="gemma2:2b",
        messages=[
            {
                "role": "user",
                "content": f"""
You are helping retrieve information from a research paper.

Write a short hypothetical passage that might appear in a research paper answering the question.

Rules:
- Stay within the domain implied by the question.
- Do not invent exact numbers.
- Do not invent statistics.
- Do not invent dates.
- Keep it under 3 sentences.

Question:
{question}

Hypothetical Passage:
"""
            }
        ]
    )

    hyde_text = hyde_response["message"]["content"]

    # =====================================
    # EMBEDDING
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
        n_results=5,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print("\nDISTANCES:")
    print(results["distances"][0])

    print("\nTOP RETRIEVED CHUNKS:")

    for i, doc in enumerate(results["documents"][0]):
        print(f"\n----- CHUNK {i+1} -----")
        print(doc[:500])
    # =====================================
    # EMPTY DATABASE
    # =====================================

    if (
        len(results["documents"]) == 0
        or len(results["documents"][0]) == 0
    ):

        return {
            "question": question,
            "answer": "No documents have been ingested.",
            "sources": [],
            "hyde_answer": hyde_text
        }

    # =====================================
    # USE ALL 5 RETRIEVED CHUNKS
    # =====================================

    filtered_docs = results["documents"][0]

    filtered_meta = results["metadatas"][0]

    filtered_scores = results["distances"][0]

    # =====================================
    # NO VALID DOCUMENTS
    # =====================================

    if len(filtered_docs) == 0:

        return {
            "question": question,
            "answer": "I could not find that information in the document.",
            "sources": [],
            "hyde_answer": hyde_text
        }

    context = "\n\n".join(
        filtered_docs
    )

    # =====================================
    # FINAL RAG PROMPT
    # =====================================

    prompt = f"""
You are a research-paper question answering assistant.

Use ONLY the provided context.

Rules:

1. Every statement must be directly supported by the context.

2. Do NOT use outside knowledge.

3. Do NOT guess.

4. Do NOT invent facts, numbers, dates, statistics, or explanations.

5. If multiple values are present, include ALL relevant values.

6. If the answer is not explicitly present in the context, respond EXACTLY:

I could not find that information in the document.

7. Never provide both an answer and the fallback response.

8. Answer clearly and directly.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="gemma2:2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"].strip()

    # =====================================
    # CLEAN FALLBACK MIXUPS
    # =====================================

    fallback = "I could not find that information in the document."

    if fallback.lower() in answer.lower():

        cleaned = answer.replace(
            fallback,
            ""
        ).strip()

        if len(cleaned) > 20:
            answer = cleaned
        else:
            answer = fallback

    # =====================================
    # SOURCES
    # =====================================

    sources = []

    shown = set()

    for metadata, distance in zip(
        filtered_meta,
        filtered_scores
    ):

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "N/A"
        )

        chunk = metadata.get(
            "chunk",
            "N/A"
        )

        key = (
            source,
            page,
            chunk
        )

        if key not in shown:

            shown.add(key)

            sources.append({
                "source": source,
                "page": page,
                "chunk": chunk,
                "distance": round(
                    float(distance),
                    4
                )
            })

    # =====================================
    # RETURN JSON
    # =====================================

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "hyde_answer": hyde_text
    }