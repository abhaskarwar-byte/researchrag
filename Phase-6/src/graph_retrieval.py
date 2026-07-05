import ollama

from src.neo4j_store import (
    vector_search,
    fulltext_search
)


EMBED_MODEL = "nomic-embed-text"


# =====================================
# QUERY EMBEDDING
# =====================================

def generate_query_embedding(

    query

):

    response = ollama.embeddings(

        model=EMBED_MODEL,

        prompt=query

    )

    return response["embedding"]


# =====================================
# VECTOR RETRIEVAL
# =====================================

def vector_retrieval(

    query,

    top_k=5

):

    print(

        "\nRunning Vector Retrieval..."

    )

    embedding = generate_query_embedding(

        query

    )

    results = vector_search(

        embedding,

        top_k
    )

    retrieved = []

    for record in results:

        node = record["node"]

        score = record["score"]

        retrieved.append(

            {

                "type": "vector",

                "score": score,

                "chunk_id": node["chunk_id"],

                "text": node["text"],

                "page": node["page"],

                "source": node["source"],

                "parent_id": node["parent_id"]

            }

        )

    return retrieved


# =====================================
# FULL TEXT RETRIEVAL
# =====================================

def fulltext_retrieval(

    query,

    top_k=5

):

    print(

        "\nRunning Full Text Retrieval..."

    )

    results = fulltext_search(

        query,

        top_k
    )

    retrieved = []

    for record in results:

        node = record["node"]

        score = record["score"]

        retrieved.append(

            {

                "type": "fulltext",

                "score": score,

                "chunk_id": node["chunk_id"],

                "text": node["text"],

                "page": node["page"],

                "source": node["source"],

                "parent_id": node["parent_id"]

            }

        )

    return retrieved

# =====================================
# HYBRID RETRIEVAL
# =====================================

def hybrid_retrieval(

    query,

    top_k=5

):

    print(

        "\nRunning Hybrid Retrieval..."

    )

    vector_results = vector_retrieval(

        query,

        top_k
    )

    fulltext_results = fulltext_retrieval(

        query,

        top_k
    )

    combined = (

        vector_results +

        fulltext_results
    )

    unique_results = {}

    for result in combined:

        chunk_id = result["chunk_id"]

        if chunk_id not in unique_results:

            unique_results[chunk_id] = result

        else:

            if result["score"] > unique_results[chunk_id]["score"]:

                unique_results[chunk_id] = result

    final_results = list(

        unique_results.values()

    )

    final_results.sort(

        key=lambda x: x["score"],

        reverse=True

    )

    return final_results[:top_k]

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    query = input("Enter your question: ")

    print("\n==============================")
    print("VECTOR RESULTS")
    print("==============================")

    vector = vector_retrieval(query)

    for result in vector:

        print(result)

    print("\n==============================")
    print("FULL TEXT RESULTS")
    print("==============================")

    fulltext = fulltext_retrieval(query)

    for result in fulltext:

        print(result)

    print("\n==============================")
    print("HYBRID RESULTS")
    print("==============================")

    hybrid = hybrid_retrieval(query)

    for result in hybrid:

        print(result)