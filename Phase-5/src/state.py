from typing import TypedDict


class RAGState(TypedDict):

    question: str

    rewritten_question: str

    candidate_chunks: list

    reranked_chunks: list

    grade: str

    answer: str

    retry_count: int