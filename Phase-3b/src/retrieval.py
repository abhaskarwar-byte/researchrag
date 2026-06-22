import json
import math
import re
from collections import Counter

from nltk.corpus import stopwords


STOPWORDS = set(
    stopwords.words("english")
)


def extract_keywords(text):

    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def tokenize(text):

    return re.findall(
        r"[a-z0-9]+",
        text.lower()
    )


def score_chunk(
    text,
    keywords,
    question
):

    score = 0

    text_lower = text.lower()

    for keyword in keywords:

        matches = re.findall(
            rf"\b{re.escape(keyword)}\b",
            text_lower
        )

        score += len(matches)

    phrase = " ".join(
        keywords
    )

    if phrase:

        score += (
            text_lower.count(
                phrase
            ) * 2
        )

    return score


def bm25_rank(
    chunks,
    keywords,
    top_k=5
):

    if not chunks:
        return []

    documents = []

    document_lengths = []

    document_frequency = Counter()

    for keyword_score, chunk in chunks:

        text = (
            chunk.get(
                "summary",
                ""
            )
            .lower()
        )

        tokens = tokenize(
            text
        )

        token_counts = Counter(
            tokens
        )

        documents.append(
            {
                "chunk": chunk,
                "counts": token_counts,
                "length": len(tokens),
                "keyword_score": keyword_score
            }
        )

        document_lengths.append(
            len(tokens)
        )

        for term in set(tokens):

            document_frequency[
                term
            ] += 1

    total_documents = len(
        documents
    )

    average_length = (
        sum(document_lengths)
        / total_documents
    )

    k1 = 1.5
    b = 0.75

    ranked = []

    for document in documents:

        bm25_score = 0.0

        for keyword in keywords:

            tf = document[
                "counts"
            ].get(
                keyword,
                0
            )

            if tf == 0:
                continue

            df = document_frequency.get(
                keyword,
                0
            )

            idf = math.log(
                1
                + (
                    total_documents
                    - df
                    + 0.5
                )
                /
                (
                    df
                    + 0.5
                )
            )

            denominator = (
                tf
                + k1
                * (
                    1
                    - b
                    + b
                    * document["length"]
                    / average_length
                )
            )

            bm25_score += (
                idf
                * tf
                * (k1 + 1)
                / denominator
            )

        ranked.append(
            (
                bm25_score,
                document["chunk"]
            )
        )

    ranked.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return ranked[:top_k]


def retrieve_candidate_children(
    question
):

    with open(
        "data/chunks.json",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    parent_chunks = [
        chunk
        for chunk in chunks
        if chunk["chunk_type"]
        == "parent"
    ]

    child_chunks = [
        chunk
        for chunk in chunks
        if chunk["chunk_type"]
        == "child"
    ]

    keywords = (
        extract_keywords(
            question
        )
    )

    print(
        "\nKEYWORDS:"
    )

    print(
        keywords
    )

    scored_parents = []

    for parent in parent_chunks:

        score = score_chunk(
            parent.get(
                "summary",
                ""
            ),
            keywords,
            question
        )

        if score > 0:

            scored_parents.append(
                (
                    score,
                    parent
                )
            )

    top_parents = bm25_rank(
        scored_parents,
        keywords,
        top_k=5
    )

    print(
        "\nTOP PARENTS:"
    )

    parent_ids = set()

    for bm25_score, parent in top_parents:

        print(
            f"Parent {parent['parent_id']} | "
            f"BM25={bm25_score:.4f}"
        )

        parent_ids.add(
            parent["parent_id"]
        )

    candidate_children = []

    for child in child_chunks:

        if (
            child["parent_id"]
            in parent_ids
        ):

            candidate_children.append(
                child
            )

    print(
        f"\nCandidate Children: "
        f"{len(candidate_children)}"
    )

    return candidate_children


if __name__ == "__main__":

    children = (
        retrieve_candidate_children(
            "What is positional encoding?"
        )
    )

    print(
        f"\nRetrieved "
        f"{len(children)} children"
    )