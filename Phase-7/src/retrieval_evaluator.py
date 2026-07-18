from src.graph_evaluator import evaluate_dataset


# =====================================
# PRECISION
# =====================================

def precision(expected, retrieved):

    expected = set(expected)
    retrieved = set(retrieved)

    if len(retrieved) == 0:
        return 0.0

    correct = len(expected.intersection(retrieved))

    return correct / len(retrieved)


# =====================================
# RECALL
# =====================================

def recall(expected, retrieved):

    expected = set(expected)
    retrieved = set(retrieved)

    if len(expected) == 0:
        return 0.0

    correct = len(expected.intersection(retrieved))

    return correct / len(expected)


# =====================================
# F1 SCORE
# =====================================

def f1_score(precision_value, recall_value):

    if precision_value + recall_value == 0:
        return 0.0

    return (

        2 * precision_value * recall_value

    ) / (

        precision_value + recall_value

    )


# =====================================
# ENTITY RETRIEVAL
# =====================================

def evaluate_entity_retrieval(benchmark):

    expected_entities = [

        entity["name"]

        for entity in benchmark["expected_entities"]

    ]

    retrieved_entities = [

        entity["entity"]

        for entity in benchmark["retrieved_graph"]

    ]

    p = precision(

        expected_entities,

        retrieved_entities

    )

    r = recall(

        expected_entities,

        retrieved_entities

    )

    f1 = f1_score(

        p,

        r

    )

    return {

        "entity_precision": p,

        "entity_recall": r,

        "entity_f1": f1

    }


# =====================================
# RELATIONSHIP RETRIEVAL
# =====================================

def evaluate_relationship_retrieval(benchmark):

    expected_relationships = [

        (

            relationship["source"],

            relationship["relation"],

            relationship["target"]

        )

        for relationship in benchmark["expected_relationships"]

    ]

    retrieved_relationships = [

        (

            relationship["source"],

            relationship["relation"],

            relationship["target"]

        )

        for relationship in benchmark.get(

            "retrieved_relationships",

            []

        )

    ]

    p = precision(

        expected_relationships,

        retrieved_relationships

    )

    r = recall(

        expected_relationships,

        retrieved_relationships

    )

    f1 = f1_score(

        p,

        r

    )

    return {

        "relationship_precision": p,

        "relationship_recall": r,

        "relationship_f1": f1

    }


# =====================================
# EVALUATE RETRIEVAL
# =====================================

def evaluate_retrieval(graph_results=None):

    """
    Evaluates retrieval quality.

    Parameters
    ----------
    graph_results : list, optional

        Output from graph_evaluator.evaluate_dataset().

        If None, graph evaluation is executed automatically.
    """

    if graph_results is None:

        graph_results = evaluate_dataset()

    retrieval_results = []

    print("\n======================================")

    print("RETRIEVAL EVALUATION")

    print("======================================")

    entity_precision = 0
    entity_recall = 0
    entity_f1 = 0

    relationship_precision = 0
    relationship_recall = 0
    relationship_f1 = 0

    for benchmark in graph_results:

        entity_metrics = evaluate_entity_retrieval(

            benchmark

        )

        relationship_metrics = (

            evaluate_relationship_retrieval(

                benchmark

            )

        )

        result = {

            **benchmark,

            **entity_metrics,

            **relationship_metrics

        }

        retrieval_results.append(

            result

        )

        entity_precision += entity_metrics[

            "entity_precision"

        ]

        entity_recall += entity_metrics[

            "entity_recall"

        ]

        entity_f1 += entity_metrics[

            "entity_f1"

        ]

        relationship_precision += relationship_metrics[

            "relationship_precision"

        ]

        relationship_recall += relationship_metrics[

            "relationship_recall"

        ]

        relationship_f1 += relationship_metrics[

            "relationship_f1"

        ]

    total = len(retrieval_results)

    if total:

        entity_precision /= total
        entity_recall /= total
        entity_f1 /= total

        relationship_precision /= total
        relationship_recall /= total
        relationship_f1 /= total

    print("\n======================================")

    print("AVERAGE RETRIEVAL METRICS")

    print("======================================")

    print(f"Entity Precision       : {entity_precision:.3f}")

    print(f"Entity Recall          : {entity_recall:.3f}")

    print(f"Entity F1              : {entity_f1:.3f}")

    print()

    print(f"Relationship Precision : {relationship_precision:.3f}")

    print(f"Relationship Recall    : {relationship_recall:.3f}")

    print(f"Relationship F1        : {relationship_f1:.3f}")

    return retrieval_results


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    evaluate_retrieval()