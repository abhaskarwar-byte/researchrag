from neo4j import GraphDatabase

from src.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD
)


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)


# =====================================
# CONNECTION
# =====================================

def get_session():

    return driver.session()


def close():

    driver.close()


# =====================================
# CLEAR GRAPH
# =====================================

def clear_graph():

    with get_session() as session:

        session.run(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )

    print("Graph cleared.")


# =====================================
# CREATE INDEXES
# =====================================

def create_indexes():

    with get_session() as session:

        session.run("""
        CREATE INDEX paper_index
        IF NOT EXISTS
        FOR (p:Paper)
        ON (p.paper_id)
        """)

        session.run("""
        CREATE INDEX parent_chunk_index
        IF NOT EXISTS
        FOR (p:ParentChunk)
        ON (p.parent_id)
        """)

        session.run("""
        CREATE INDEX chunk_index
        IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.chunk_id)
        """)

        session.run("""
        CREATE INDEX entity_index
        IF NOT EXISTS
        FOR (e:Entity)
        ON (e.name)
        """)

        session.run("""
        CREATE FULLTEXT INDEX chunk_text_index
        IF NOT EXISTS
        FOR (c:Chunk)
        ON EACH [c.text]
        """)

        session.run("""
        CREATE VECTOR INDEX chunk_vector_index
        IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`:768,
                `vector.similarity_function`:'cosine'
            }
        }
        """)

    print("Indexes created.")


# =====================================
# PAPER
# =====================================

def store_paper(

    paper_id,

    title

):

    with get_session() as session:

        session.run(
            """
            MERGE (p:Paper {paper_id:$paper_id})

            SET

                p.title = $title,

                p.status = coalesce(
                    p.status,
                    "BUILDING"
                )
            """,

            paper_id=paper_id,

            title=title
        )


# =====================================
# PAPER COMPLETE?
# =====================================

def paper_complete(

    paper_id

):

    with get_session() as session:

        result = session.run(
            """
            MATCH (p:Paper {paper_id:$paper_id})

            RETURN p.status AS status
            """,

            paper_id=paper_id
        ).single()

        if result is None:

            return False

        return result["status"] == "COMPLETE"


# =====================================
# START PAPER BUILD
# =====================================

def start_paper_build(

    paper_id

):

    with get_session() as session:

        session.run(
            """
            MATCH (p:Paper {paper_id:$paper_id})

            SET p.status = "BUILDING"
            """,

            paper_id=paper_id
        )


# =====================================
# FINISH PAPER BUILD
# =====================================

def finish_paper_build(

    paper_id

):

    with get_session() as session:

        session.run(
            """
            MATCH (p:Paper {paper_id:$paper_id})

            SET p.status = "COMPLETE"
            """,

            paper_id=paper_id
        )

# =====================================
# PARENT CHUNK
# =====================================

def store_parent_chunk(

    parent_id,

    summary,

    page,

    source

):

    with get_session() as session:

        session.run(
            """
            MERGE (p:ParentChunk {parent_id:$parent_id})

            SET

                p.summary = $summary,

                p.page = $page,

                p.source = $source,

                p.display_name = split($parent_id, "_parent_")[1]
            """,

            parent_id=parent_id,

            summary=summary,

            page=page,

            source=source
        )


# =====================================
# CHILD CHUNK
# =====================================

def store_chunk(

    chunk_id,

    text,

    page,

    source,

    parent_id,

    embedding

):

    with get_session() as session:

        session.run(
            """
            MERGE (c:Chunk {chunk_id:$chunk_id})

            SET

                c.text = $text,

                c.page = $page,

                c.source = $source,

                c.parent_id = $parent_id,

                c.embedding = $embedding
            """,

            chunk_id=chunk_id,

            text=text,

            page=page,

            source=source,

            parent_id=parent_id,

            embedding=embedding
        )


# =====================================
# ENTITY
# =====================================

def store_entity(

    entity_name,

    entity_type

):

    with get_session() as session:

        session.run(
            """
            MERGE (e:Entity {name:$entity_name})

            SET

                e.type = $entity_type
            """,

            entity_name=entity_name,

            entity_type=entity_type
        )


# =====================================
# ENTITY RELATIONSHIP
# =====================================

def store_relationship(

    source,

    target,

    relation

):

    relation = (

        relation

        .upper()

        .replace(

            " ",

            "_"

        )

    )

    with get_session() as session:

        query = f"""
        MATCH

            (a:Entity {{name:$source}}),

            (b:Entity {{name:$target}})

        MERGE

            (a)-[:{relation}]->(b)
        """

        session.run(

            query,

            source=source,

            target=target

        )

# =====================================
# PAPER -> PARENT
# =====================================

def link_paper_parent(

    paper_id,

    parent_id

):

    with get_session() as session:

        session.run(
            """
            MATCH

                (p:Paper {paper_id:$paper_id}),

                (pc:ParentChunk {parent_id:$parent_id})

            MERGE

                (p)-[:HAS_PARENT]->(pc)
            """,

            paper_id=paper_id,

            parent_id=parent_id
        )


# =====================================
# PARENT -> CHILD
# =====================================

def link_parent_child(

    parent_id,

    chunk_id

):

    with get_session() as session:

        session.run(
            """
            MATCH

                (p:ParentChunk {parent_id:$parent_id}),

                (c:Chunk {chunk_id:$chunk_id})

            MERGE

                (p)-[:HAS_CHILD]->(c)
            """,

            parent_id=parent_id,

            chunk_id=chunk_id
        )


# =====================================
# CHILD -> PARENT
# =====================================

def link_child_parent(

    chunk_id,

    parent_id

):

    with get_session() as session:

        session.run(
            """
            MATCH

                (c:Chunk {chunk_id:$chunk_id}),

                (p:ParentChunk {parent_id:$parent_id})

            MERGE

                (c)-[:BELONGS_TO]->(p)
            """,

            chunk_id=chunk_id,

            parent_id=parent_id
        )


# =====================================
# PARENT -> ENTITY
# =====================================

def link_parent_entity(

    parent_id,

    entity_name

):

    with get_session() as session:

        session.run(
            """
            MATCH

                (p:ParentChunk {parent_id:$parent_id}),

                (e:Entity {name:$entity_name})

            MERGE

                (p)-[:MENTIONS]->(e)
            """,

            parent_id=parent_id,

            entity_name=entity_name
        )


# =====================================
# VECTOR SEARCH
# =====================================

def vector_search(

    embedding,

    top_k=5

):

    with get_session() as session:

        result = session.run(
            """
            CALL db.index.vector.queryNodes(

                'chunk_vector_index',

                $top_k,

                $embedding

            )

            YIELD node, score

            RETURN

                node,

                score
            """,

            embedding=embedding,

            top_k=top_k
        )

        return list(result)


# =====================================
# FULL-TEXT SEARCH
# =====================================

def fulltext_search(

    search_text,

    top_k=5

):

    with get_session() as session:

        result = session.run(
            """
            CALL db.index.fulltext.queryNodes(

                'chunk_text_index',

                $search_text

            )

            YIELD node, score

            RETURN

                node,

                score

            LIMIT $top_k
            """,

            search_text=search_text,

            top_k=top_k
        )

        return list(result)


# =====================================
# GRAPH EXPANSION
# =====================================

def expand_neighbors(

    entity_name,

    hops=1

):

    query = f"""
    MATCH

        (e:Entity {{name:$entity_name}})

        -[r*1..{hops}]-

        (n)

    RETURN

        e,

        r,

        n
    """

    with get_session() as session:

        result = session.run(

            query,

            entity_name=entity_name

        )

        return list(result)

# =====================================
# GET ENTITIES FROM PARENT CHUNK
# =====================================

def get_parent_entities(

    parent_id

):

    with get_session() as session:

        result = session.run(
            """
            MATCH

                (p:ParentChunk {parent_id:$parent_id})

                -[:MENTIONS]->

                (e:Entity)

            RETURN

                e.name AS name,

                e.type AS type
            """,

            parent_id=parent_id
        )

        return [

            {

                "name": record["name"],

                "type": record["type"]

            }

            for record in result

        ]

# =====================================
# GRAPH STATS
# =====================================

def graph_stats():

    with get_session() as session:

        papers = session.run(
            """
            MATCH (p:Paper)

            RETURN count(p) AS total
            """
        ).single()["total"]

        parents = session.run(
            """
            MATCH (p:ParentChunk)

            RETURN count(p) AS total
            """
        ).single()["total"]

        chunks = session.run(
            """
            MATCH (c:Chunk)

            RETURN count(c) AS total
            """
        ).single()["total"]

        entities = session.run(
            """
            MATCH (e:Entity)

            RETURN count(e) AS total
            """
        ).single()["total"]

        relationships = session.run(
            """
            MATCH ()-[r]->()

            RETURN count(r) AS total
            """
        ).single()["total"]

        completed = session.run(
            """
            MATCH (p:Paper)

            WHERE p.status = "COMPLETE"

            RETURN count(p) AS total
            """
        ).single()["total"]

        building = session.run(
            """
            MATCH (p:Paper)

            WHERE p.status = "BUILDING"

            RETURN count(p) AS total
            """
        ).single()["total"]

        return {

            "papers": papers,

            "completed_papers": completed,

            "building_papers": building,

            "parent_chunks": parents,

            "chunks": chunks,

            "entities": entities,

            "relationships": relationships

        }


# =====================================
# CLOSE DRIVER
# =====================================

def close():

    driver.close()