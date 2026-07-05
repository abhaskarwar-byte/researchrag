from groq import Groq

from src.config import (
    GROQ_API_KEY
)

client = Groq(
    api_key=GROQ_API_KEY
)


# =====================================
# CYPHER GENERATION
# =====================================

def generate_cypher(

    question

):

    prompt = f"""
You are an expert Neo4j Cypher generator.

The graph contains:

(:Paper)

(:ParentChunk)

(:Chunk)

(:Entity)

Relationships:

(Paper)-[:HAS_PARENT]->(ParentChunk)

(ParentChunk)-[:HAS_CHILD]->(Chunk)

(Chunk)-[:BELONGS_TO]->(ParentChunk)

(ParentChunk)-[:MENTIONS]->(Entity)

Entity nodes may have many different relationship types such as:

USES
CONTAINS
PROPOSES
TRAINED_ON
BASED_ON
EVALUATES
IMPROVES
DEPENDS_ON

Do NOT assume a relationship called RELATION.

If the relationship type is unknown, use:

MATCH (e:Entity {{name:"..."}})-[r]-(n)

RETURN e,r,n

Rules:

Generate ONLY valid Cypher.

Do NOT explain.

Do NOT use markdown.

Return only the query.

Question:

{question}
"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {

                "role": "user",

                "content": prompt

            }

        ],

        temperature=0

    )

    cypher = response.choices[0].message.content.strip()

    return cypher
from src.neo4j_store import (
    get_session
)


# =====================================
# EXECUTE CYPHER
# =====================================

def execute_cypher(

    cypher

):

    print(

        "\nExecuting Cypher...\n"

    )

    with get_session() as session:

        result = session.run(

            cypher

        )

        return [

            record.data()

            for record in result

        ]


# =====================================
# GRAPH QUERY
# =====================================

def graph_query(

    question

):

    print(

        "\nGenerating Cypher...\n"

    )

    cypher = generate_cypher(

        question

    )

    print(

        "Generated Query:\n"

    )

    print(

        cypher

    )

    results = execute_cypher(

        cypher

    )

    return {

        "cypher": cypher,

        "results": results

    }


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    question = input(

        "Ask a graph question: "

    )

    response = graph_query(

        question

    )

    print(

        "\n=============================="

    )

    print(

        "GRAPH RESULTS"

    )

    print(

        "==============================\n"

    )

    if len(

        response["results"]

    ) == 0:

        print(

            "No results found."

        )

    else:

        for row in response["results"]:

            print(

                row

            )