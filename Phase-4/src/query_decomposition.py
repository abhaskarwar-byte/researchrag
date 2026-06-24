import ollama
import re


def generate_subquestions(
    question
):

    prompt = f"""
Break the question into 3 smaller
retrieval-focused questions.

Rules:

- Return exactly 3 questions
- Every line must be a question
- Do not return fewer than 3
- Do not return explanations
- Do not return answers
- One question per line

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

    subquestions = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^\d+\.\s*",
            "",
            line
        )

        line = (
            line.replace("-", "")
            .replace("*", "")
            .strip()
        )

        if line.endswith("?"):

            subquestions.append(
                line
            )

    return subquestions


if __name__ == "__main__":

    questions = (
        generate_subquestions(
            "How do latent diffusion models reduce computational cost?"
        )
    )

    print(
        "\nSUBQUESTIONS:\n"
    )

    for i, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"{i}. {question}"
        )