from sentence_transformers import (
    CrossEncoder
)

MIN_RELEVANCE_SCORE = -1.0

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_chunks(
    question,
    chunks,
    top_k=3
):

    if not chunks:
        return []

    pairs = []

    for chunk in chunks:

        pairs.append(
            [
                question,
                chunk["text"]
            ]
        )

    scores = model.predict(
        pairs
    )

    ranked = []

    for chunk, score in zip(
        chunks,
        scores
    ):

        ranked.append(
            (
                float(score),
                chunk
            )
        )

    ranked.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    print(
        "\nCROSS ENCODER SCORES:\n"
    )

    for score, chunk in ranked:

        print(
            f"Score={score:.4f}"
        )

        print(
            chunk["text"][:200]
        )

        print(
            "-" * 50
        )

    filtered_chunks = []

    for score, chunk in ranked:

        if score >= MIN_RELEVANCE_SCORE:

            filtered_chunks.append(
                chunk
            )

        if len(filtered_chunks) >= top_k:

            break

    print(
        "\nAFTER THRESHOLD:\n"
    )

    if not filtered_chunks:

        print(
            "No chunks passed the relevance threshold."
        )

    else:

        for chunk in filtered_chunks:

            print(
                chunk["text"][:200]
            )

            print(
                "-" * 50
            )

    return filtered_chunks


if __name__ == "__main__":

    sample_chunks = [
        {
            "text":
            "Positional encodings inject information about relative positions."
        },
        {
            "text":
            "Convolutional networks process local features."
        }
    ]

    results = rerank_chunks(
        "What is positional encoding?",
        sample_chunks
    )

    print(
        "\nFINAL CHUNKS:\n"
    )

    for chunk in results:

        print(
            chunk["text"]
        )