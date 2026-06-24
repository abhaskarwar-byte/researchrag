import ollama


def generate_perspective_queries(
    question
):

    prompt = f"""
Generate 5 retrieval queries about the topic
from different perspectives.

Perspectives:

1. Definition
2. Working mechanism
3. Advantages
4. Applications
5. Limitations

Rules:

- One query per line
- No numbering
- No bullet points
- Keep each query short
- Queries should target different aspects

Question:
{question}
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

    text = (
        response["message"]["content"]
        .strip()
    )

    queries = []

    for line in text.split("\n"):

        line = line.strip()

        if line:

            line = (
                line.replace("-", "")
                .replace("*", "")
                .strip()
            )

            queries.append(
                line
            )

    return queries


if __name__ == "__main__":

    queries = (
        generate_perspective_queries(
            "What is positional encoding?"
        )
    )

    print(
        "\nPERSPECTIVE QUERIES:\n"
    )

    for i, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"{i}. {query}"
        )