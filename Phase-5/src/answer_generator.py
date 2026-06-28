import ollama


FALLBACK_MESSAGE = (
    "I could not find that information in the document."
)


def generate_answer(
    question,
    chunks
):

    if not chunks:

        return FALLBACK_MESSAGE

    context = "\n\n".join(
        chunk["text"]
        for chunk in chunks
    )

    prompt = f"""
You are answering questions about a research paper.

Use ONLY the retrieved context.

Rules

- Never use outside knowledge.
- Never infer missing information.
- Never combine retrieved text with your own memory.
- If the answer is only partially supported, answer only the supported part.
- If the answer is not contained in the context, reply exactly:

"I could not find that information in the document."

Answer in 2-4 concise sentences.

Context:
{context}

Question:
{question}

Answer:
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

    answer = (
        response["message"]["content"]
        .strip()
    )

    return answer