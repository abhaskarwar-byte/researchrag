from src.retrieval import retrieve_candidate_children
from src.reranker import rerank_children
from src.answer_generator import generate_answer


question = (
    "What factors are considered when comparing self-attention with recurrent and convolutional layers?"
)

children = retrieve_candidate_children(
    question
)

reranked = rerank_children(
    question,
    children
)

answer = generate_answer(
    question,
    reranked
)

print("\nANSWER:\n")

print(answer)