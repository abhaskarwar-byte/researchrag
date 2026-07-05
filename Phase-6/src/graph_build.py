import os
import ollama

from src.entity_extractor import extract_entities

from src.ingest import (
    ingest_document
)

from src.neo4j_store import (
    clear_graph,
    create_indexes,
    store_paper,
    store_parent_chunk,
    store_chunk,
    store_entity,
    store_relationship,
    link_paper_parent,
    link_parent_child,
    link_child_parent,
    link_parent_entity,
    graph_stats
)


EMBED_MODEL = "nomic-embed-text"


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
# BUILD GRAPH
# =====================================

def build_graph(

    file_path

):

    chunks = ingest_document(

        file_path

    )

    clear_graph()

    create_indexes()

    papers = set()

    print("\n==============================")

    print("PASS 1 : BUILDING KNOWLEDGE GRAPH")

    print("==============================\n")

    # =====================================
    # PASS 1
    # Parent Chunks
    # =====================================

    for chunk in chunks:

        if chunk["chunk_type"] != "parent":

            continue

        paper_id = chunk["source"]

        title = os.path.basename(

            paper_id

        )

        if paper_id not in papers:

            store_paper(

                paper_id,

                title

            )

            papers.add(

                paper_id

            )

        parent_id = (

            f"{paper_id}_parent_"

            f"{chunk['parent_id']}"

        )

        print(

            f"Processing Parent {chunk['parent_id']}"

        )

        store_parent_chunk(

            parent_id=parent_id,

            summary=chunk["summary"],

            page=chunk["page"],

            source=paper_id

        )

        link_paper_parent(

            paper_id,

            parent_id

        )

        print(

            "Extracting entities..."

        )

        extracted = extract_entities(

            chunk["summary"]

        )

        for entity in extracted["entities"]:

            store_entity(

                entity["name"],

                entity["type"]

            )

            link_parent_entity(

                parent_id,

                entity["name"]

            )

        for relation in extracted["relationships"]:

            store_relationship(

                relation["source"],

                relation["target"],

                relation["relation"]

            )

    print("\n==============================")

    print("PASS 2 : BUILDING VECTOR INDEX")

    print("==============================\n")


    # =====================================
    # PASS 2
    # Child Chunks
    # =====================================

    for chunk in chunks:

        if chunk["chunk_type"] != "child":

            continue

        paper_id = chunk["source"]

        parent_id = (

            f"{paper_id}_parent_"

            f"{chunk['parent_id']}"

        )

        chunk_id = (

            f"{paper_id}_"

            f"{chunk['page']}_"

            f"{chunk['chunk']}"

        )

        print(

            f"Embedding Child Chunk {chunk['chunk']}"

        )

        embedding = generate_embedding(

            chunk["text"]

        )

        store_chunk(

            chunk_id=chunk_id,

            text=chunk["text"],

            page=chunk["page"],

            source=paper_id,

            parent_id=parent_id,

            embedding=embedding

        )

        link_parent_child(

            parent_id,

            chunk_id

        )

        link_child_parent(

            chunk_id,

            parent_id

        )

    print("\n======================================")

    print("GRAPH BUILD COMPLETE")

    print("======================================\n")

    stats = graph_stats()

    print(

        f"Papers         : {stats['papers']}"

    )

    print(

        f"Parent Chunks  : {stats['parent_chunks']}"

    )

    print(

        f"Child Chunks   : {stats['chunks']}"

    )

    print(

        f"Entities       : {stats['entities']}"

    )

    print(

        f"Relationships  : {stats['relationships']}"

    )

    print(

        "\nNeo4j Knowledge Graph successfully built."

    )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    build_graph(

        "documents/sample.pdf"

    )