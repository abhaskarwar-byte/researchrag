import ollama

from src.config import (
    MODEL_NAME
)


def rewrite_query(
    question,
    chunks=None
):

    prompt = f"""
You are improving a search query for retrieving passages from a research paper.

Your task is ONLY to improve retrieval.

Rules:

- Preserve the original meaning exactly.
- Do NOT answer the question.
- Do NOT broaden the scope.
- Do NOT introduce new concepts, examples, or constraints.
- Keep important technical terms unchanged.
- If the question is already clear, return it unchanged.
- Only rewrite if wording is ambiguous.
- Return ONE sentence only.

Original Question:

{question}

Rewritten Question:
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

    rewritten = (
        response["message"]["content"]
        .strip()
        .replace('"', "")
    )

    print(
        "\nREWRITTEN QUERY:\n"
    )

    print(
        rewritten
    )

    return rewritten


if __name__ == "__main__":

    question = (
        "What is positional encoding?"
    )

    print(
        rewrite_query(question)
    )