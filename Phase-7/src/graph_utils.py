from src.neo4j_store import (
    expand_neighbors
)


# =====================================
# EXPAND SINGLE ENTITY
# =====================================

def expand_entity(

    entity_name,

    hops=1

):

    print(

        f"\nExpanding entity: {entity_name}"

    )

    results = expand_neighbors(

        entity_name,

        hops
    )

    return results


# =====================================
# EXPAND MULTIPLE ENTITIES
# =====================================

def expand_entities(

    entity_names,

    hops=1

):

    graph_results = []

    for entity in entity_names:

        expanded = expand_entity(

            entity,

            hops
        )

        graph_results.extend(

            expanded
        )

    return graph_results


# =====================================
# REMOVE DUPLICATES
# =====================================

def remove_duplicates(

    graph_results

):

    unique = []

    seen = set()

    for record in graph_results:

        node = record["n"]

        node_id = node.element_id

        if node_id not in seen:

            seen.add(

                node_id
            )

            unique.append(

                record
            )

    return unique
# =====================================
# BUILD GRAPH CONTEXT
# =====================================

def build_graph_context(

    graph_results

):

    if len(graph_results) == 0:

        return "No graph context found."

    context = ""

    for record in graph_results:

        entity = record["e"]

        neighbor = record["n"]

        relations = record["r"]
        relation = relations[0] 

        context += (

            f"{entity['name']} "

            f"--{relation.type}--> "

            f"{neighbor.get('name', neighbor.get('parent_id', 'Node'))}\n"

        )

    return context


# =====================================
# PRINT GRAPH CONTEXT
# =====================================

def print_graph_context(

    graph_results

):

    print(

        "\n=============================="

    )

    print(

        "GRAPH CONTEXT"

    )

    print(

        "==============================\n"

    )

    if len(graph_results) == 0:

        print(

            "No neighboring entities found."

        )

        return

    for record in graph_results:

        entity = record["e"]

        neighbor = record["n"]

        relations = record["r"]
        relation = relations[0]

        print(

            f"{entity['name']} "

            f"--{relation.type}--> "

            f"{neighbor.get('name', neighbor.get('parent_id', 'Node'))}"

        )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    entity = input(

        "Enter entity name: "

    )

    results = expand_entity(

        entity,

        hops=1

    )

    results = remove_duplicates(

        results

    )

    print_graph_context(

        results

    )

    print(

        "\n=============================="

    )

    print(

        "FORMATTED CONTEXT"

    )

    print(

        "==============================\n"

    )

    print(

        build_graph_context(

            results

        )

    )