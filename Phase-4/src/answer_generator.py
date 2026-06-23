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
You are a research paper question answering assistant.

Use ONLY the provided context.

Rules:

1. Use only information found in the context.

2. Do not use outside knowledge.

3. Do not invent facts.

4. If the answer is not present in the context, respond exactly:

I could not find that information in the document.

5. If the question starts with "What is",
give a short definition using only wording
and concepts present in the context.

6. Do not create your own definition.

7. If the context contains a formula,
explain it using nearby text from the context.

8. Every sentence in the answer must be
traceable to the context.

9. Keep the answer between 2 and 4 sentences.

10. Do not repeat the question.

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