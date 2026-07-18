from groq import Groq

from src.config import GROQ_API_KEY


client = Groq(
    api_key=GROQ_API_KEY
)


# =====================================
# FORMAT VECTOR CONTEXT
# =====================================

def format_vector_context(vector_context):
    """
    Formats retrieved vector chunks into
    a readable prompt.
    """

    if not vector_context:
        return "No semantic context found."

    context = ""

    for i, chunk in enumerate(vector_context, start=1):

        context += (
            f"\nChunk {i}\n"
            f"Source : {chunk.get('source','Unknown')}\n"
            f"Page   : {chunk.get('page','-')}\n"
            f"Score  : {round(chunk.get('score',0),4)}\n\n"
            f"{chunk.get('text','')}\n\n"
        )

    return context


# =====================================
# FORMAT GRAPH CONTEXT
# =====================================

def format_graph_context(graph_context):
    """
    Formats graph entities into text.
    """

    if not graph_context:
        return "No graph context found."

    context = ""

    for item in graph_context:

        entity = item.get("entity", "Unknown")
        entity_type = item.get("entity_type", "Unknown")

        context += (
            f"Entity : {entity}\n"
            f"Type   : {entity_type}\n\n"
        )

    return context


# =====================================
# BUILD FINAL CONTEXT
# =====================================

def build_context(vector_context, graph_context):
    """
    Combines semantic and graph context.
    """

    semantic = format_vector_context(vector_context)

    graph = format_graph_context(graph_context)

    return (
        "===== SEMANTIC CONTEXT =====\n\n"
        + semantic
        + "\n\n"
        + "===== GRAPH CONTEXT =====\n\n"
        + graph
    )


# =====================================
# GENERATE ANSWER
# =====================================

def generate_answer(question, context):
    """
    Calls Groq LLM to generate an answer.
    """

    prompt = f"""
You are an expert AI research assistant.

Answer the user's question ONLY using the provided context.

Rules:

- Use ONLY the supplied context.
- Do NOT invent facts.
- Do NOT use outside knowledge.
- If the answer cannot be found, reply:

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
# GENERATE FROM RETRIEVAL
# =====================================

def generate_answer_from_retrieval(
    question,
    vector_context,
    graph_context
):
    """
    Used by evaluation_pipeline.py.

    Inputs are the outputs from graph_evaluator.py.
    """

    context = build_context(
        vector_context,
        graph_context
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
# BACKWARD COMPATIBILITY
# =====================================

def answer_question(
    question,
    retrieval_result,
    retrieval_type="vector"
):
    """
    Existing interface used by api.py.
    """

    if retrieval_type == "graph":

        context = format_graph_context(
            retrieval_result["results"]
        )

    elif retrieval_type == "hybrid":

        semantic = format_vector_context(
            retrieval_result["semantic_results"]
        )

        graph = format_graph_context(
            retrieval_result["graph_results"]["results"]
        )

        context = (
            "===== SEMANTIC CONTEXT =====\n\n"
            + semantic
            + "\n\n"
            + "===== GRAPH CONTEXT =====\n\n"
            + graph
        )

    else:

        context = format_vector_context(
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

    question = input("Ask a question: ")

    retrieval = route_query(question)

    if isinstance(retrieval, dict):

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

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================\n")
    print(response["answer"])