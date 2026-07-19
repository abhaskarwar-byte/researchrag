from src.answer_metrics import (
    exact_match,
    token_f1,
    entity_precision,
    entity_recall,
    entity_f1,
    relationship_precision,
    relationship_recall,
    relationship_f1
)


def evaluate_answer(
    question,
    reference_answer,
    generated_answer,
    expected_entities,
    retrieved_entities,
    expected_relationships,
    retrieved_relationships
):
    """
    Evaluate a generated answer against the benchmark answer.

    Returns a dictionary containing all answer-level metrics.
    """

    token_scores = token_f1(
        reference_answer,
        generated_answer
    )

    return {

        "question": question,

        "reference_answer": reference_answer,

        "generated_answer": generated_answer,

        "exact_match": exact_match(
            reference_answer,
            generated_answer
        ),

        "token_precision": token_scores["precision"],
        "token_recall": token_scores["recall"],
        "token_f1": token_scores["f1"],

        "entity_precision": entity_precision(
            expected_entities,
            retrieved_entities
        ),

        "entity_recall": entity_recall(
            expected_entities,
            retrieved_entities
        ),

        "entity_f1": entity_f1(
            expected_entities,
            retrieved_entities
        ),

        "relationship_precision": relationship_precision(
            expected_relationships,
            retrieved_relationships
        ),

        "relationship_recall": relationship_recall(
            expected_relationships,
            retrieved_relationships
        ),

        "relationship_f1": relationship_f1(
            expected_relationships,
            retrieved_relationships
        )

    }

# =====================================
# EVALUATE ALL ANSWERS
# =====================================

def evaluate_answers(answer_results):
    """
    Computes average metrics across all evaluated answers.
    """

    if not answer_results:
        return {}

    summary = {
        "exact_match": 0.0,
        "token_precision": 0.0,
        "token_recall": 0.0,
        "token_f1": 0.0,
        "entity_precision": 0.0,
        "entity_recall": 0.0,
        "entity_f1": 0.0,
        "relationship_precision": 0.0,
        "relationship_recall": 0.0,
        "relationship_f1": 0.0
    }

    total = len(answer_results)

    for result in answer_results:

        summary["exact_match"] += result["exact_match"]

        summary["token_precision"] += result["token_precision"]
        summary["token_recall"] += result["token_recall"]
        summary["token_f1"] += result["token_f1"]

        summary["entity_precision"] += result["entity_precision"]
        summary["entity_recall"] += result["entity_recall"]
        summary["entity_f1"] += result["entity_f1"]

        summary["relationship_precision"] += result["relationship_precision"]
        summary["relationship_recall"] += result["relationship_recall"]
        summary["relationship_f1"] += result["relationship_f1"]

    for key in summary:
        summary[key] /= total

    return summary