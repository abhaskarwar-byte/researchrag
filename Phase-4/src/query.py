from src.retrieval import (
    retrieve_candidate_children
)

from src.cross_encoder import (
    rerank_chunks
)

from src.answer_generator import (
    generate_answer
)


def query_document(question):

    # ==========================
    # RETRIEVAL
    # ==========================

    candidate_children = (
        retrieve_candidate_children(
            question
        )
    )

    # ==========================
    # CROSS ENCODER RERANKING
    # ==========================

    reranked_chunks = (
        rerank_chunks(
            question,
            candidate_children,
            top_k=3
        )
    )

    # ==========================
    # ANSWER GENERATION
    # ==========================

    answer = (
        generate_answer(
            question,
            reranked_chunks
        )
    )

    # ==========================
    # SOURCES
    # ==========================

    sources = []

    for chunk in reranked_chunks:

        sources.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "parent_id": chunk["parent_id"]
            }
        )

    # ==========================
    # RETURN
    # ==========================

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "pipeline": [
            "hyde",
            "multi_query",
            "bm25",
            "rrf",
            "parent_retrieval",
            "child_retrieval",
            "cross_encoder",
            "answer_generation"
        ]
    }


if __name__ == "__main__":

    question = (
        "What is positional encoding?"
    )

    result = query_document(
        question
    )

    print(
        "\nQUESTION:\n"
    )

    print(
        result["question"]
    )

    print(
        "\nANSWER:\n"
    )

    print(
        result["answer"]
    )

    print(
        "\nSOURCES:\n"
    )

    for source in result["sources"]:

        print(source)

    print(
        "\nPIPELINE:\n"
    )

    print(
        " -> ".join(
            result["pipeline"]
        )
    )