from langgraph.graph import (
    StateGraph,
    END
)

from src.state import (
    RAGState
)

from src.nodes import (
    retrieve_node,
    rerank_node,
    grade_node,
    rewrite_node,
    web_search_node,
    answer_node
)

from src.config import (
    MAX_RETRIES
)


# ==========================
# ROUTING
# ==========================

def route_after_grading(
    state: RAGState
):

    if state["grade"] == "GOOD":

        return "answer"

    if state["grade"] == "MIXED":

        if (
            state["retry_count"]
            < MAX_RETRIES
        ):

            return "rewrite"

        return "answer"

    return "web"


# ==========================
# BUILD GRAPH
# ==========================

graph = StateGraph(
    RAGState
)

graph.add_node(
    "retrieve",
    retrieve_node
)

graph.add_node(
    "rerank",
    rerank_node
)

graph.add_node(
    "grade",
    grade_node
)

graph.add_node(
    "rewrite",
    rewrite_node
)

graph.add_node(
    "web",
    web_search_node
)

graph.add_node(
    "answer",
    answer_node
)


graph.set_entry_point(
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "rerank"
)

graph.add_edge(
    "rerank",
    "grade"
)

graph.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "answer": "answer",
        "rewrite": "rewrite",
        "web": "web"
    }
)

graph.add_edge(
    "rewrite",
    "retrieve"
)

graph.add_edge(
    "web",
    "answer"
)

graph.add_edge(
    "answer",
    END
)


agent = graph.compile()


# ==========================
# TEST
# ==========================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    initial_state = {

        "question":
        question,

        "rewritten_question":
        "",

        "candidate_chunks":
        [],

        "reranked_chunks":
        [],

        "grade":
        "",

        "answer":
        "",

        "retry_count":
        0
    }

    result = agent.invoke(
        initial_state
    )

    print(
        "\n========================"
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "========================\n"
    )

    print(
        result["answer"]
    )

    print(
        "\nGRADE:",
        result["grade"]
    )

    print(
        "RETRIES:",
        result["retry_count"]
    )