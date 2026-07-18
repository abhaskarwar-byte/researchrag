import ollama

from src.benchmark_questions import (
    get_all_questions
)

from src.neo4j_store import (
    vector_search,
    expand_neighbors,
    get_parent_entities
)


# =====================================
# CONFIGURATION
# =====================================

EMBED_MODEL = "nomic-embed-text"

TOP_K = 5

GRAPH_HOPS = 1


# =====================================
# EMBEDDING
# =====================================

def generate_embedding(

    text

):

    response = ollama.embeddings(

        model=EMBED_MODEL,

        prompt=text

    )

    return response["embedding"]


# =====================================
# VECTOR RETRIEVAL
# =====================================

def retrieve_chunks(

    question,

    top_k=TOP_K

):

    embedding = generate_embedding(

        question

    )

    return vector_search(

        embedding,

        top_k

    )


# =====================================
# BUILD VECTOR CONTEXT
# =====================================

def build_vector_context(

    retrieved_chunks

):

    vector_context = []

    for record in retrieved_chunks:

        node = record["node"]

        score = record["score"]

        vector_context.append(

            {

                "chunk_id": node.get(

                    "chunk_id"

                ),

                "parent_id": node.get(

                    "parent_id"

                ),

                "page": node.get(

                    "page"

                ),

                "source": node.get(

                    "source"

                ),

                "text": node.get(

                    "text"

                ),

                "score": score

            }

        )

    return vector_context


# =====================================
# BUILD GRAPH CONTEXT
# =====================================

def build_graph_context(

    vector_context

):

    graph_context = []

    visited_entities = set()

    for chunk in vector_context:

        parent_id = chunk["parent_id"]

        entities = get_parent_entities(

            parent_id

        )

        for entity in entities:

            entity_name = entity["name"]

            if entity_name in visited_entities:

                continue

            visited_entities.add(

                entity_name

            )

            neighbors = expand_neighbors(

                entity_name,

                GRAPH_HOPS

            )

            relationship_triples = []

            seen_relationships = set()

            for record in neighbors:

                source_node = record["e"]

                target_node = record["n"]

                relationships = record["r"]

                source_name = source_node.get(

                    "name",

                    source_node.element_id

                )

                target_name = target_node.get(

                    "name",

                    target_node.element_id

                )

                for relationship in relationships:

                    triple = (

                        source_name,

                        relationship.type,

                        target_name

                    )

                    if triple in seen_relationships:

                        continue

                    seen_relationships.add(

                        triple

                    )

                    relationship_triples.append(

                        {

                            "source": source_name,

                            "relation": relationship.type,

                            "target": target_name

                        }

                    )

            graph_context.append(

                {

                    "entity": entity,

                    "neighbors": neighbors,

                    "relationships": relationship_triples

                }

            )

    return graph_context

# =====================================
# RETRIEVE KNOWLEDGE
# =====================================

def retrieve_knowledge(

    question

):

    retrieved_chunks = retrieve_chunks(

        question

    )

    vector_context = build_vector_context(

        retrieved_chunks

    )

    graph_context = build_graph_context(

        vector_context

    )

    return {

        "vector_context": vector_context,

        "graph_context": graph_context

    }


# =====================================
# EVALUATE ONE QUESTION
# =====================================

def evaluate_question(

    benchmark

):

    retrieval = retrieve_knowledge(

        benchmark["question"]

    )

    vector_results = []

    for chunk in retrieval["vector_context"]:

        vector_results.append(

            {

                "chunk_id": chunk["chunk_id"],

                "parent_id": chunk["parent_id"],

                "page": chunk["page"],

                "source": chunk["source"],

                "score": round(

                    chunk["score"],

                    4

                ),

                "text": chunk["text"]

            }

        )

    graph_results = []

    retrieved_relationships = []

    seen_entities = set()

    seen_relationships = set()

    for item in retrieval["graph_context"]:

        entity_name = item["entity"]["name"]

        entity_type = item["entity"]["type"]

        if entity_name not in seen_entities:

            graph_results.append(

                {

                    "entity": entity_name,

                    "entity_type": entity_type,

                    "neighbor_count": len(

                        item["neighbors"]

                    )

                }

            )

            seen_entities.add(

                entity_name

            )

        for relationship in item["relationships"]:

            triple = (

                relationship["source"],

                relationship["relation"],

                relationship["target"]

            )

            if triple in seen_relationships:

                continue

            seen_relationships.add(

                triple

            )

            retrieved_relationships.append(

                {

                    "source": relationship["source"],

                    "relation": relationship["relation"],

                    "target": relationship["target"]

                }

            )

    return {

        "id": benchmark["id"],

        "paper": benchmark["paper"],

        "page": benchmark["page"],

        "question": benchmark["question"],

        "question_type": benchmark["question_type"],

        "difficulty": benchmark["difficulty"],

        "reference_answer": benchmark["reference_answer"],

        "expected_entities":

            benchmark["expected_entities"],

        "expected_relationships":

            benchmark["expected_relationships"],

        "retrieved_vector":

            vector_results,

        "retrieved_graph":

            graph_results,

        "retrieved_relationships":

            retrieved_relationships

    }

# =====================================
# EVALUATE COMPLETE DATASET
# =====================================

def evaluate_dataset():

    benchmark_questions = get_all_questions()

    results = []

    print("\n======================================")

    print("GRAPH RETRIEVAL EVALUATION")

    print("======================================")

    print(

        f"Questions : {len(benchmark_questions)}"

    )

    for benchmark in benchmark_questions:

        print(

            f"Evaluating Question {benchmark['id']}"

        )

        result = evaluate_question(

            benchmark

        )

        results.append(

            result

        )

    print("\n======================================")

    print("GRAPH EVALUATION COMPLETE")

    print("======================================")

    print(

        f"Questions Evaluated : {len(results)}"

    )

    return results


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    results = evaluate_dataset()

    print("\n======================================")

    print("SAMPLE RESULT")

    print("======================================")

    if results:

        sample = results[0]

        print(

            f"Question : {sample['question']}"

        )

        print(

            f"Retrieved Vector Chunks : "

            f"{len(sample['retrieved_vector'])}"

        )

        print(

            f"Retrieved Graph Entities : "

            f"{len(sample['retrieved_graph'])}"

        )

        print(

            f"Retrieved Relationship Triples : "

            f"{len(sample['retrieved_relationships'])}"

        )

        print("\nSample Relationships:\n")

        for relationship in sample["retrieved_relationships"][:10]:

            print(

                f"{relationship['source']} "

                f"-[{relationship['relation']}]-> "

                f"{relationship['target']}"

            )

# =====================================
# EVALUATE COMPLETE DATASET
# =====================================

def evaluate_dataset():

    benchmark_questions = get_all_questions()

    results = []

    print("\n======================================")

    print("GRAPH RETRIEVAL EVALUATION")

    print("======================================")

    print(

        f"Questions : {len(benchmark_questions)}"

    )

    for benchmark in benchmark_questions:

        print(

            f"Evaluating Question {benchmark['id']}"

        )

        result = evaluate_question(

            benchmark

        )

        results.append(

            result

        )

    print("\n======================================")

    print("GRAPH EVALUATION COMPLETE")

    print("======================================")

    print(

        f"Questions Evaluated : {len(results)}"

    )

    return results


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    results = evaluate_dataset()

    print("\n======================================")

    print("SAMPLE RESULT")

    print("======================================")

    if results:

        sample = results[0]

        print(

            f"Question : {sample['question']}"

        )

        print(

            f"Retrieved Vector Chunks : "

            f"{len(sample['retrieved_vector'])}"

        )

        print(

            f"Retrieved Graph Entities : "

            f"{len(sample['retrieved_graph'])}"

        )

        print(

            f"Retrieved Relationship Triples : "

            f"{len(sample['retrieved_relationships'])}"

        )

        print("\nSample Relationships:\n")

        for relationship in sample["retrieved_relationships"][:10]:

            print(

                f"{relationship['source']} "

                f"-[{relationship['relation']}]-> "

                f"{relationship['target']}"

            )