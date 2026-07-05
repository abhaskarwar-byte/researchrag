from groq import Groq

from src.config import (
    GROQ_API_KEY
)

from src.graph_retrieval import (
    hybrid_retrieval,
    final_hybrid_retrieval
)

from src.cypher_generator import (
    graph_query
)


client = Groq(
    api_key=GROQ_API_KEY
)


# =====================================
# ROUTE CLASSIFICATION
# =====================================

def classify_question(

    question

):

    prompt = f"""
You are a Graph RAG router.

Choose ONLY ONE option.

vector
graph
hybrid

Rules:

Use vector when the question asks for:

- explanation
- summary
- definition
- describe
- how does something work

Use graph when the question asks for:

- relationships
- connected entities
- neighbours
- linked concepts
- graph traversal
- which entities
- show connections

Use hybrid when BOTH semantic retrieval
and graph relationships are needed.

Return ONLY ONE WORD.

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

    route = (

        response

        .choices[0]

        .message

        .content

        .strip()

        .lower()

    )

    if route not in [

        "vector",

        "graph",

        "hybrid"

    ]:

        route = "hybrid"

    return route


# =====================================
# ROUTER
# =====================================

def route_query(

    question

):

    route = classify_question(

        question

    )

    print(

        f"\nSelected Route : {route}\n"

    )

    if route == "vector":

        return hybrid_retrieval(

            question

        )

    elif route == "graph":

        return graph_query(

            question

        )

    else:

        return final_hybrid_retrieval(

            question

        )
# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    question = input(

        "Ask a question: "

    )

    response = route_query(

        question

    )

    print(

        "\n=============================="

    )

    print(

        "ROUTER OUTPUT"

    )

    print(

        "==============================\n"

    )

    if isinstance(

        response,

        dict

    ):

        if "results" in response:

            print(

                "Generated Cypher:\n"

            )

            print(

                response["cypher"]

            )

            print(

                "\nGraph Results:\n"

            )

            if len(

                response["results"]

            ) == 0:

                print(

                    "No graph results found."

                )

            else:

                for row in response["results"]:

                    print(

                        row

                    )

        else:

            print(

                response

            )

    elif isinstance(

        response,

        list

    ):

        if len(

            response

        ) == 0:

            print(

                "No retrieval results found."

            )

        else:

            for item in response:

                print(

                    item

                )

    else:

        print(

            response

        )