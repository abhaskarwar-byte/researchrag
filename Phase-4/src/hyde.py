import ollama


def generate_hypothetical_answer(
    question
):

    prompt = f"""
Generate keywords and concepts that may
appear in a research paper answering the question.

Rules:

- Maximum 3 sentences.
- No emojis.
- No bullet points.
- Focus on terminology.
- Focus on concepts.
- Do not introduce unrelated topics.
- Keep the response concise.

Question:
{question}

Keywords and Concepts:
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

    hypothetical_answer = (
        response["message"]["content"]
        .strip()
    )

    return hypothetical_answer


if __name__ == "__main__":

    result = generate_hypothetical_answer(
        "What is positional encoding?"
    )

    print(
        "\nHYDE OUTPUT:\n"
    )

    print(
        result
    )