import ollama


def generate_queries(
    question
):

    prompt = f"""
Generate 4 alternative search queries
for document retrieval.

Rules:

- Keep the same meaning.
- Use different wording.
- One query per line.
- No numbering.
- No bullet points.
- Maximum 12 words per query.

Question:
{question}

Queries:
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

    output = (
        response["message"]["content"]
        .strip()
    )

    queries = [
        question
    ]

    for line in output.splitlines():

        query = line.strip()

        if (
            query
            and query not in queries
        ):
            queries.append(
                query
            )

    return queries[:5]


if __name__ == "__main__":

    queries = generate_queries(
        "What is positional encoding?"
    )

    print("\nEXPANDED QUERIES:\n")

    for i, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"{i}. {query}"
        )