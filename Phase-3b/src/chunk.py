import re


def normalize_text(text):

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"-\n",
        "",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n +",
        "\n",
        text
    )

    return text.strip()


def split_into_paragraphs(text):

    cleaned = normalize_text(
        text
    )

    return [
        p.strip()
        for p in cleaned.split("\n\n")
        if p.strip()
    ]


def split_into_sentences(text):

    text = normalize_text(
        text
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text,
    chunk_size=2,
    chunk_overlap=0
):

    paragraphs = split_into_paragraphs(
        text
    )

    if not paragraphs:
        return []

    chunks = []

    start = 0

    while start < len(paragraphs):

        end = min(
            start + chunk_size,
            len(paragraphs)
        )

        chunk = "\n\n".join(
            paragraphs[start:end]
        )

        chunks.append(
            chunk
        )

        if end == len(paragraphs):
            break

        start += max(
            1,
            chunk_size - chunk_overlap
        )

    return chunks


def summarize_text(
    text,
    max_chars=300
):

    cleaned = normalize_text(
        text
    )

    if not cleaned:
        return ""

    sentences = split_into_sentences(
        cleaned
    )

    if not sentences:
        return cleaned[:max_chars]

    summary = ""

    for sentence in sentences:

        if (
            len(summary)
            + len(sentence)
            <= max_chars
        ):

            summary = (
                f"{summary} {sentence}"
            ).strip()

        else:
            break

    if not summary:
        summary = cleaned[:max_chars]

    if len(summary) > max_chars:

        summary = (
            summary[: max_chars - 3]
            .rstrip()
            + "..."
        )

    return summary


def build_parent_child_chunks(
    text,
    chunk_size=2,
    chunk_overlap=0,
    parent_chunk_size=3
):

    children = chunk_text(
        text,
        chunk_size,
        chunk_overlap
    )

    parents = []

    parent_id = 1

    for i in range(
        0,
        len(children),
        parent_chunk_size
    ):

        parents.append(
            {
                "parent_id": parent_id,
                "child_range": [
                    i,
                    min(
                        i + parent_chunk_size,
                        len(children)
                    )
                ]
            }
        )

        parent_id += 1

    return children, parents