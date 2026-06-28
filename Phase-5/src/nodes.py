from src.state import (
    RAGState
)

from src.retrieval import (
    retrieve_candidate_children
)

from src.cross_encoder import (
    rerank_chunks
)

from src.relevance_grader import (
    grade_retrieval
)

from src.query_rewriter import (
    rewrite_query
)

from src.answer_generator import (
    generate_answer
)

from src.web_search import (
    web_search
)


# ==========================
# RETRIEVAL NODE
# ==========================

def retrieve_node(
    state: RAGState
):

    question = (
        state["rewritten_question"]
        if state["rewritten_question"]
        else state["question"]
    )

    candidate_chunks = (
        retrieve_candidate_children(
            question
        )
    )

    state["candidate_chunks"] = (
        candidate_chunks
    )

    return state


# ==========================
# RERANK NODE
# ==========================

def rerank_node(
    state: RAGState
):

    question = (
        state["rewritten_question"]
        if state["rewritten_question"]
        else state["question"]
    )

    reranked_chunks = (
        rerank_chunks(
            question,
            state["candidate_chunks"]
        )
    )

    state["reranked_chunks"] = (
        reranked_chunks
    )

    return state


# ==========================
# GRADE NODE
# ==========================

def grade_node(
    state: RAGState
):

    question = (
        state["rewritten_question"]
        if state["rewritten_question"]
        else state["question"]
    )

    grade = (
        grade_retrieval(
            question,
            state["reranked_chunks"]
        )
    )

    state["grade"] = grade

    return state


# ==========================
# REWRITE NODE
# ==========================

def rewrite_node(
    state: RAGState
):

    rewritten = (
        rewrite_query(
            state["question"],
            state["reranked_chunks"]
        )
    )

    state["rewritten_question"] = (
        rewritten
    )

    state["retry_count"] += 1

    return state


# ==========================
# WEB SEARCH NODE
# ==========================

def web_search_node(
    state: RAGState
):

    print(
        "\nWEB SEARCH FALLBACK\n"
    )

    web_chunks = (
        web_search(
            state["question"]
        )
    )

    reranked = (
        rerank_chunks(
            state["question"],
            web_chunks
        )
    )

    state["reranked_chunks"] = (
        reranked
    )

    return state


# ==========================
# ANSWER NODE
# ==========================

def answer_node(
    state: RAGState
):

    question = (
        state["rewritten_question"]
        if state["rewritten_question"]
        else state["question"]
    )

    answer = (
        generate_answer(
            question,
            state["reranked_chunks"]
        )
    )

    state["answer"] = answer

    return state