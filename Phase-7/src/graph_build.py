import os
import ollama

from src.entity_extractor import extract_entities

from src.ingest import (
    ingest_document
)

from src.neo4j_store import (
    create_indexes,
    store_paper,
    paper_complete,
    start_paper_build,
    finish_paper_build,
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

    paper_id = file_path

    title = os.path.basename(

        file_path

    )

    # Skip already-built papers

    if paper_complete(

        paper_id

    ):

        print(

            f"\nSkipping {title} (already indexed)."

        )

        return

    chunks = ingest_document(

        file_path

    )

    create_indexes()

    store_paper(

        paper_id,

        title

    )

    start_paper_build(

        paper_id

    )

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

    # ---------------------------------
    # Mark paper as completely built
    # ---------------------------------

    finish_paper_build(

        paper_id

    )

    print("\n======================================")

    print("GRAPH BUILD COMPLETE")

    print("======================================\n")

    stats = graph_stats()

    print(

        f"Papers              : {stats['papers']}"

    )

    print(

        f"Completed Papers    : {stats['completed_papers']}"

    )

    print(

        f"Building Papers     : {stats['building_papers']}"

    )

    print(

        f"Parent Chunks       : {stats['parent_chunks']}"

    )

    print(

        f"Child Chunks        : {stats['chunks']}"

    )

    print(

        f"Entities            : {stats['entities']}"

    )

    print(

        f"Relationships       : {stats['relationships']}"

    )

    print(

        "\nNeo4j Knowledge Graph successfully built."

    )


# =====================================
# BUILD GRAPH FOR ALL DOCUMENTS
# =====================================

def build_all_graphs():

    print("\n======================================")

    print("BUILDING KNOWLEDGE GRAPH")

    print("======================================")

    create_indexes()

    documents = sorted(

        [

            file

            for file in os.listdir(

                "documents"

            )

            if file.lower().endswith(

                (

                    ".pdf",

                    ".docx",

                    ".txt"

                )

            )

        ]

    )

    for document in documents:

        build_graph(

            os.path.join(

                "documents",

                document

            )

        )

    print("\n======================================")

    print("ALL DOCUMENTS PROCESSED")

    print("======================================")

    stats = graph_stats()

    print(

        f"Papers              : {stats['papers']}"

    )

    print(

        f"Completed Papers    : {stats['completed_papers']}"

    )

    print(

        f"Building Papers     : {stats['building_papers']}"

    )

    print(

        f"Parent Chunks       : {stats['parent_chunks']}"

    )

    print(

        f"Child Chunks        : {stats['chunks']}"

    )

    print(

        f"Entities            : {stats['entities']}"

    )

    print(

        f"Relationships       : {stats['relationships']}"

    )


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    build_all_graphs()