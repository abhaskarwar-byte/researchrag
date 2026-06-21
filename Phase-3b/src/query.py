from src.retrieval import (
    retrieve_candidate_children
)

from src.reranker import (
    rerank_children
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
    # RERANKING
    # ==========================

    reranked_chunks = (
        rerank_children(
            question,
            candidate_children
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
            "parent_retrieval",
            "bm25",
            "child_retrieval",
            "llm_reranking",
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