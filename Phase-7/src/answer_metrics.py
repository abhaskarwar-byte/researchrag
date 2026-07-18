import re
from collections import Counter


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_text(text):
    """
    Normalize text for answer comparison.

    Steps:
    - Lowercase
    - Remove punctuation
    - Collapse multiple spaces
    """

    if text is None:
        return ""

    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# ANSWER METRICS
# =====================================================

def exact_match(reference_answer, generated_answer):
    """
    Returns:
        1 if answers match exactly after normalization.
        0 otherwise.
    """

    return int(
        normalize_text(reference_answer)
        ==
        normalize_text(generated_answer)
    )


def token_f1(reference_answer, generated_answer):
    """
    Computes token-level Precision, Recall and F1.

    Returns:
        {
            "precision": float,
            "recall": float,
            "f1": float
        }
    """

    reference_tokens = normalize_text(reference_answer).split()
    generated_tokens = normalize_text(generated_answer).split()

    if not reference_tokens and not generated_tokens:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0
        }

    if not reference_tokens or not generated_tokens:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    reference_counter = Counter(reference_tokens)
    generated_counter = Counter(generated_tokens)

    common = reference_counter & generated_counter

    overlap = sum(common.values())

    precision = overlap / len(generated_tokens)

    recall = overlap / len(reference_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            /
            (precision + recall)
        )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


# =====================================================
# ENTITY METRICS
# =====================================================

def _entity_set(entity_list):
    """
    Converts entity dictionaries into a normalized set.

    Input:
    [
        {"name":"ControlNet","type":"Model"},
        ...
    ]
    """

    return {
        normalize_text(entity["name"])
        for entity in entity_list
    }


def entity_precision(expected_entities, retrieved_entities):

    expected = _entity_set(expected_entities)
    retrieved = _entity_set(retrieved_entities)

    if not retrieved:
        return 0.0

    true_positive = len(expected & retrieved)

    return round(
        true_positive / len(retrieved),
        4
    )


def entity_recall(expected_entities, retrieved_entities):

    expected = _entity_set(expected_entities)
    retrieved = _entity_set(retrieved_entities)

    if not expected:
        return 0.0

    true_positive = len(expected & retrieved)

    return round(
        true_positive / len(expected),
        4
    )


def entity_f1(expected_entities, retrieved_entities):

    precision = entity_precision(
        expected_entities,
        retrieved_entities
    )

    recall = entity_recall(
        expected_entities,
        retrieved_entities
    )

    if precision + recall == 0:
        return 0.0

    return round(
        2 * precision * recall
        /
        (precision + recall),
        4
    )


# =====================================================
# RELATIONSHIP METRICS
# =====================================================

def _relationship_set(relationship_list):
    """
    Converts relationships into tuples.

    Example:

    {
        "source":"ControlNet",
        "relation":"USES",
        "target":"Stable Diffusion"
    }

    becomes

    (
        "controlnet",
        "uses",
        "stable diffusion"
    )
    """

    relationship_set = set()

    for relation in relationship_list:

        relationship_set.add(

            (
                normalize_text(relation["source"]),
                normalize_text(relation["relation"]),
                normalize_text(relation["target"])
            )

        )

    return relationship_set


def relationship_precision(
    expected_relationships,
    retrieved_relationships
):

    expected = _relationship_set(expected_relationships)

    retrieved = _relationship_set(retrieved_relationships)

    if not retrieved:
        return 0.0

    true_positive = len(
        expected & retrieved
    )

    return round(
        true_positive / len(retrieved),
        4
    )


def relationship_recall(
    expected_relationships,
    retrieved_relationships
):

    expected = _relationship_set(expected_relationships)

    retrieved = _relationship_set(retrieved_relationships)

    if not expected:
        return 0.0

    true_positive = len(
        expected & retrieved
    )

    return round(
        true_positive / len(expected),
        4
    )


def relationship_f1(
    expected_relationships,
    retrieved_relationships
):

    precision = relationship_precision(
        expected_relationships,
        retrieved_relationships
    )

    recall = relationship_recall(
        expected_relationships,
        retrieved_relationships
    )

    if precision + recall == 0:
        return 0.0

    return round(
        2 * precision * recall
        /
        (precision + recall),
        4
    )