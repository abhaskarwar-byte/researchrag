import re
import ollama


def parse_rerank_indices(
    ranking,
    max_index
):

    selected_indices = []

    numbers = re.findall(
        r"C([0-9]+)",
        ranking,
        flags=re.IGNORECASE
    )

    for item in numbers:

        index = int(item)

        if (
            1 <= index <= max_index
        ):

            index -= 1

            if index not in selected_indices:

                selected_indices.append(
                    index
                )

    return selected_indices


def rerank_children(
    question,
    children
):

    if not children:
        return []

    chunk_text = ""

    for i, child in enumerate(
        children
    ):

        chunk_text += (
            f"\nC{i+1}:\n"
            f"{child['text'][:1200]}\n"
        )

    prompt = f"""
You are a research paper retrieval assistant.

Question:
{question}

Candidate Chunks:

{chunk_text}

Your task:

Evaluate each chunk based on:

1. Does it directly answer the question?
2. Does it explain the concept?
3. Does it contain useful supporting details?
4. Is it more informative than the other chunks?

Rank chunks by relevance, not by their order.

Return ONLY the identifiers of the
three most relevant chunks.

Example format:
C2,C5,C1

Choose the actual best chunks.
Do NOT always return C1,C2,C3.
Do NOT explain your answer.
Do NOT use bullet points.
Return only chunk IDs.

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

    ranking = (
        response["message"]["content"]
        .strip()
    )

    print(
        "\nRERANK RESULT:"
    )

    print(
        ranking
    )

    indices = parse_rerank_indices(
        ranking,
        len(children)
    )

    if not indices:

        indices = [
            0,
            1,
            2
        ]

    reranked = []

    for index in indices[:3]:

        reranked.append(
            children[index]
        )

    return reranked