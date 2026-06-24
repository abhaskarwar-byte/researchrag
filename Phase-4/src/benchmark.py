from src.query_expansion import (
    generate_queries
)

from src.perspective_queries import (
    generate_perspective_queries
)

from src.step_back import (
    generate_step_back_query
)

from src.query_decomposition import (
    generate_subquestions
)

from src.hyde import (
    generate_hypothetical_answer
)

from src.retrieval import (
    retrieve_children_from_queries
)

from src.cross_encoder import (
    rerank_chunks
)

from src.answer_generator import (
    generate_answer
)


def evaluate_method(
    method_name,
    question,
    queries
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        method_name
    )

    print(
        "=" * 70
    )

    print(
        "\nGENERATED QUERIES:\n"
    )

    for query in queries:

        print(
            "-",
            query[:150]
        )

    candidate_children = (
        retrieve_children_from_queries(
            queries
        )
    )

    print(
        f"\nRetrieved "
        f"{len(candidate_children)} "
        f"candidate children"
    )

    if not candidate_children:

        print(
            "\nNo retrieval results."
        )

        return

    reranked_chunks = (
        rerank_chunks(
            question,
            candidate_children,
            top_k=3
        )
    )

    print(
        "\nTOP CHUNKS:\n"
    )

    for chunk in reranked_chunks:

        print(
            f"Source: "
            f"{chunk['source']}"
        )

        print(
            f"Page: "
            f"{chunk['page']}"
        )

        print(
            f"Parent: "
            f"{chunk['parent_id']}"
        )

        print(
            "-" * 40
        )

    answer = (
        generate_answer(
            question,
            reranked_chunks
        )
    )

    print(
        "\nANSWER:\n"
    )

    print(
        answer
    )


if __name__ == "__main__":

    questions = [

        "What is positional encoding?",

        "What is multi-head attention?",

        "How do latent diffusion models reduce computational cost?"

    ]

    for question in questions:

        print(
            "\n\n"
            + "#" * 80
        )

        print(
            f"\nQUESTION:\n{question}"
        )

        print(
            "\n"
            + "#" * 80
        )

        # ==========================
        # QUERY REWRITE
        # ==========================

        evaluate_method(
            "QUERY REWRITE",
            question,
            generate_queries(
                question
            )
        )

        # ==========================
        # PERSPECTIVE
        # ==========================

        evaluate_method(
            "PERSPECTIVE QUERIES",
            question,
            generate_perspective_queries(
                question
            )
        )

        # ==========================
        # STEP BACK
        # ==========================

        evaluate_method(
            "STEP BACK",
            question,
            [
                generate_step_back_query(
                    question
                )
            ]
        )

        # ==========================
        # DECOMPOSITION
        # ==========================

        evaluate_method(
            "QUERY DECOMPOSITION",
            question,
            generate_subquestions(
                question
            )
        )

        # ==========================
        # HYDE
        # ==========================

        evaluate_method(
            "HYDE",
            question,
            [
                generate_hypothetical_answer(
                    question
                )
            ]
        )