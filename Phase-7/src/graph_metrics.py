from src.neo4j_store import driver


# =====================================================
# GRAPH METRICS
# =====================================================

class GraphMetrics:

    def __init__(self):

        self.driver = driver


    # =================================================
    # EXECUTE CYPHER
    # =================================================

    def execute_query(self, query):

        with self.driver.session() as session:

            result = session.run(query)

            return result.single().value()


    # =================================================
    # TOTAL PAPERS
    # =================================================

    def total_papers(self):

        query = """

        MATCH (p:Paper)

        RETURN count(p)

        """

        return self.execute_query(query)


    # =================================================
    # TOTAL PARENT CHUNKS
    # =================================================

    def total_parent_chunks(self):

        query = """

        MATCH (p:ParentChunk)

        RETURN count(p)

        """

        return self.execute_query(query)


    # =================================================
    # TOTAL CHILD CHUNKS
    # =================================================

    def total_child_chunks(self):

        query = """

        MATCH (c:Chunk)

        RETURN count(c)

        """

        return self.execute_query(query)


    # =================================================
    # TOTAL ENTITIES
    # =================================================

    def total_entities(self):

        query = """

        MATCH (e:Entity)

        RETURN count(e)

        """

        return self.execute_query(query)


    # =================================================
    # TOTAL RELATIONSHIPS
    # =================================================

    def total_relationships(self):

        query = """

        MATCH ()-[r]->()

        RETURN count(r)

        """

        return self.execute_query(query)


    # =================================================
    # ORPHAN NODES
    # =================================================

    def orphan_nodes(self):

        query = """

        MATCH (n)

        WHERE NOT (n)--()

        RETURN count(n)

        """

        return self.execute_query(query)


    # =================================================
    # DUPLICATE ENTITIES
    # =================================================

    def duplicate_entities(self):

        query = """

        MATCH (e:Entity)

        WITH toLower(e.name) AS name,
             count(*) AS cnt

        WHERE cnt > 1

        RETURN count(name)

        """

        return self.execute_query(query)


    # =================================================
    # ENTITY TYPE DISTRIBUTION
    # =================================================

    def entity_distribution(self):

        query = """

        MATCH (e:Entity)

        RETURN
            e.type AS type,
            count(*) AS count

        ORDER BY count DESC

        """

        distribution = {}

        with self.driver.session() as session:

            result = session.run(query)

            for record in result:

                distribution[
                    record["type"]
                ] = record["count"]

        return distribution


    # =================================================
    # RELATIONSHIP DISTRIBUTION
    # =================================================

    def relationship_distribution(self):

        query = """

        MATCH ()-[r]->()

        RETURN
            type(r) AS relationship,
            count(*) AS count

        ORDER BY count DESC

        """

        distribution = {}

        with self.driver.session() as session:

            result = session.run(query)

            for record in result:

                distribution[
                    record["relationship"]
                ] = record["count"]

        return distribution


    # =================================================
    # GRAPH SUMMARY
    # =================================================

    def graph_summary(self):

        return {

            "papers":
                self.total_papers(),

            "parent_chunks":
                self.total_parent_chunks(),

            "child_chunks":
                self.total_child_chunks(),

            "entities":
                self.total_entities(),

            "relationships":
                self.total_relationships(),

            "duplicate_entities":
                self.duplicate_entities(),

            "orphan_nodes":
                self.orphan_nodes(),

            "entity_distribution":
                self.entity_distribution(),

            "relationship_distribution":
                self.relationship_distribution()

        }


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    metrics = GraphMetrics()

    summary = metrics.graph_summary()

    print("\n========== GRAPH METRICS ==========\n")

    for key, value in summary.items():

        print(f"{key} :")

        print(value)

        print()

from collections import defaultdict, deque

from src.neo4j_store import driver


class GraphMetrics:

    def __init__(self):

        self.driver = driver


    # ==========================================
    # Execute Cypher
    # ==========================================

    def execute_scalar(self, query):

        with self.driver.session() as session:

            result = session.run(query)

            return result.single().value()


    # ==========================================
    # Node Counts
    # ==========================================

    def total_papers(self):

        return self.execute_scalar("""

            MATCH (p:Paper)

            RETURN count(p)

        """)


    def total_parent_chunks(self):

        return self.execute_scalar("""

            MATCH (p:ParentChunk)

            RETURN count(p)

        """)


    def total_child_chunks(self):

        return self.execute_scalar("""

            MATCH (c:Chunk)

            RETURN count(c)

        """)


    def total_entities(self):

        return self.execute_scalar("""

            MATCH (e:Entity)

            RETURN count(e)

        """)


    def total_relationships(self):

        return self.execute_scalar("""

            MATCH ()-[r]->()

            RETURN count(r)

        """)


    # ==========================================
    # Duplicate Entities
    # ==========================================

    def duplicate_entities(self):

        return self.execute_scalar("""

            MATCH (e:Entity)

            WITH toLower(e.name) AS name,
                 count(*) AS cnt

            WHERE cnt > 1

            RETURN count(name)

        """)


    # ==========================================
    # Orphan Nodes
    # ==========================================

    def orphan_nodes(self):

        return self.execute_scalar("""

            MATCH (n)

            WHERE NOT (n)--()

            RETURN count(n)

        """)


    # ==========================================
    # Entity Distribution
    # ==========================================

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

                distribution[

                    record["type"]

                ] = record["count"]

        return distribution


    # ==========================================
    # Relationship Distribution
    # ==========================================

    def relationship_distribution(self):

        distribution = {}

        query = """

        MATCH ()-[r]->()

        RETURN

            type(r) AS relation,

            count(*) AS count

        ORDER BY count DESC

        """

        with self.driver.session() as session:

            result = session.run(query)

            for record in result:

                distribution[

                    record["relation"]

                ] = record["count"]

        return distribution