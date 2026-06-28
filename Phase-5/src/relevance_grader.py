import ollama

from src.config import (
    MODEL_NAME
)


def grade_retrieval(
    question,
    chunks
):

    if not chunks:

        return "BAD"

    context = "\n\n".join(

        chunk["text"]

        for chunk in chunks
    )

    prompt = f"""
You are evaluating whether the retrieved passages are sufficient to answer a user's question.

Question:
{question}

Retrieved Context:
{context}

Instructions:

Determine whether the retrieved context contains enough information to answer the question.

Return ONLY one of the following labels:

GOOD
- The retrieved passages clearly contain the information needed to answer the question.

MIXED
- The retrieved passages contain some relevant information, but the answer is incomplete or only partially supported.

BAD
- The retrieved passages do not contain the information needed to answer the question.

Do not explain your reasoning.

Return only one word:
GOOD
MIXED
or
BAD
"""

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    grade = (
        response["message"]["content"]
        .strip()
        .upper()
    )

    if "GOOD" in grade:

        return "GOOD"

    if "MIXED" in grade:

        return "MIXED"

    return "BAD"


if __name__ == "__main__":

    sample_chunks = [

        {
            "text":
            (
                "Positional encoding uses sinusoidal functions "
                "to represent token positions."
            )
        }

    ]

    result = grade_retrieval(

        "What is positional encoding?",

        sample_chunks
    )

    print(
        "\nGRADE:\n"
    )

    print(
        result
    )