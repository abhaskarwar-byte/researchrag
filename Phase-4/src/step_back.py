import ollama


def generate_step_back_query(
    question
):

    prompt = f"""
Generate one broader conceptual question
that would help answer the original question.

Rules:

- Return only one question
- Make it more general
- Do not answer the question
- No explanations
- No numbering

Original Question:
{question}

Broader Question:
"""

    response = ollama.chat(
        model="gemma2:2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response["message"]["content"]
        .strip()
    )


if __name__ == "__main__":

    result = generate_step_back_query(
        "What is positional encoding?"
    )

    print(
        "\nSTEP BACK QUERY:\n"
    )

    print(result)