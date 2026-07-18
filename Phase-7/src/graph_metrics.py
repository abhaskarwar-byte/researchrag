from src.neo4j_store import driver


# =====================================================
# GRAPH METRICS
# =====================================================

class GraphMetrics:

    def __init__(self):
        self.driver = driver

    # =================================================
    # EXECUTE SCALAR QUERY
    # =================================================

    def execute_scalar(self, query):

        with self.driver.session() as session:

            result = session.run(query)

            return result.single().value()

    # =================================================
    # TOTAL PAPERS
    # =================================================

    def total_papers(self):

        return self.execute_scalar("""

            MATCH (p:Paper)

            RETURN count(p)

        """)

    # =================================================
    # TOTAL PARENT CHUNKS
    # =================================================

    def total_parent_chunks(self):

        return self.execute_scalar("""

            MATCH (p:ParentChunk)

            RETURN count(p)

        """)

    # =================================================
    # TOTAL CHILD CHUNKS
    # =================================================

    def total_child_chunks(self):

        return self.execute_scalar("""

            MATCH (c:Chunk)

            RETURN count(c)

        """)

    # =================================================
    # TOTAL ENTITIES
    # =================================================

    def total_entities(self):

        return self.execute_scalar("""

            MATCH (e:Entity)

            RETURN count(e)

        """)

    # =================================================
    # TOTAL RELATIONSHIPS
    # =================================================

    def total_relationships(self):

        return self.execute_scalar("""

            MATCH ()-[r]->()

            RETURN count(r)

        """)

    # =================================================
    # ORPHAN NODES
    # =================================================

    def orphan_nodes(self):

        return self.execute_scalar("""

            MATCH (n)

            WHERE NOT (n)--()

            RETURN count(n)

        """)

    # =================================================
    # DUPLICATE ENTITIES
    # =================================================

    def duplicate_entities(self):

        return self.execute_scalar("""

            MATCH (e:Entity)

            WITH toLower(e.name) AS name,
                 count(*) AS cnt

            WHERE cnt > 1

            RETURN count(name)

        """)

    # =================================================
    # ENTITY DISTRIBUTION
    # =================================================

    def entity_distribution(self):

        distribution = {}

        query = """

        MATCH (e:Entity)

        RETURN
            e.type AS type,
            count(*) AS count

        ORDER BY count DESC

        """

        with self.driver.session() as session:

            result = session.run(query)

            for record in result:

                distribution[record["type"]] = record["count"]

        return distribution

    # =================================================
    # RELATIONSHIP DISTRIBUTION
    # =================================================

    def relationship_distribution(self):

        distribution = {}

        query = """

        MATCH ()-[r]->()

        RETURN
            type(r) AS relationship,
            count(*) AS count

        ORDER BY count DESC

        """

        with self.driver.session() as session:

            result = session.run(query)

            for record in result:

                distribution[record["relationship"]] = record["count"]

        return distribution

    # =================================================
    # GRAPH SUMMARY
    # =================================================

    def graph_summary(self):

        return {

            "papers": self.total_papers(),

            "parent_chunks": self.total_parent_chunks(),

            "child_chunks": self.total_child_chunks(),

            "entities": self.total_entities(),

            "relationships": self.total_relationships(),

            "duplicate_entities": self.duplicate_entities(),

            "orphan_nodes": self.orphan_nodes(),

            "entity_distribution": self.entity_distribution(),

            "relationship_distribution": self.relationship_distribution()

        }


# =====================================================
# FUNCTION FOR PIPELINE
# =====================================================

def graph_statistics():
    """
    Returns graph statistics dictionary.
    Used by evaluation_pipeline.py
    """

    metrics = GraphMetrics()

    return metrics.graph_summary()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    summary = graph_statistics()

    print("\n========== GRAPH METRICS ==========\n")

    for key, value in summary.items():

        print(f"{key} :")

        print(value)

        print()