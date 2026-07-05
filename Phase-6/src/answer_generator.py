from groq import Groq

from src.config import (
    GROQ_API_KEY
)


client = Groq(
    api_key=GROQ_API_KEY
)


# =====================================
# FORMAT VECTOR/HYBRID CONTEXT
# =====================================

def format_retrieval_context(

    retrieval_results

):

    if len(retrieval_results) == 0:

        return "No relevant context found."

    context = ""

    for i, chunk in enumerate(

        retrieval_results,

        start=1

    ):

        context += (

            f"\nChunk {i}\n"

            f"Source : {chunk['source']}\n"

            f"Page   : {chunk['page']}\n\n"

            f"{chunk['text']}\n"

        )

    return context


# =====================================
# FORMAT GRAPH CONTEXT
# =====================================

def format_graph_context(

    graph_results

):

    if len(graph_results) == 0:

        return "No graph results found."

    context = ""

    for row in graph_results:

        context += str(row)

        context += "\n\n"

    return context
# =====================================
# GENERATE ANSWER
# =====================================

def generate_answer(

    question,

    context

):

    prompt = f"""
You are an expert AI research assistant.

Answer the user's question ONLY using the
provided context.

Rules:

- Do NOT invent facts.
- Do NOT use outside knowledge.
- If the answer is not present, say:
  "The provided context does not contain enough information."

Question:

{question}

Context:

{context}

Answer:
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

    return (

        response

        .choices[0]

        .message

        .content

        .strip()

    )


# =====================================
# ANSWER QUESTION
# =====================================

def answer_question(

    question,

    retrieval_result,

    retrieval_type="vector"

):

    if retrieval_type == "graph":

        context = format_graph_context(

            retrieval_result["results"]

        )

    else:

        context = format_retrieval_context(

            retrieval_result

        )

    answer = generate_answer(

        question,

        context

    )

    return {

        "question": question,

        "context": context,

        "answer": answer

    }


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    from src.router import route_query

    question = input(

        "Ask a question: "

    )

    retrieval = route_query(

        question

    )

    if isinstance(

        retrieval,

        dict

    ):

        response = answer_question(

            question,

            retrieval,

            retrieval_type="graph"

        )

    else:

        response = answer_question(

            question,

            retrieval,

            retrieval_type="vector"

        )

    print(

        "\n=============================="

    )

    print(

        "FINAL ANSWER"

    )

    print(

        "==============================\n"

    )

    print(

        response["answer"]

    )