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
        paragraph.strip()
        for paragraph in cleaned.split("\n\n")
        if paragraph.strip()
    ]


def split_into_sentences(text):

    cleaned = normalize_text(
        text
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# =====================================
# RECURSIVE SEMANTIC CHUNKING
# =====================================

def recursive_chunk(
    text,
    max_chars=700
):

    text = normalize_text(
        text
    )

    if not text:
        return []

    if len(text) <= max_chars:

        return [text]

    paragraphs = split_into_paragraphs(
        text
    )

    # ---------------------------------
    # SPLIT BY PARAGRAPHS
    # ---------------------------------

    if len(paragraphs) > 1:

        chunks = []

        for paragraph in paragraphs:

            chunks.extend(
                recursive_chunk(
                    paragraph,
                    max_chars
                )
            )

        return chunks

    # ---------------------------------
    # SPLIT BY SENTENCES
    # ---------------------------------

    sentences = split_into_sentences(
        text
    )

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        if (
            len(current_chunk)
            + len(sentence)
            <= max_chars
        ):

            current_chunk += (
                sentence + " "
            )

        else:

            chunks.append(
                current_chunk.strip()
            )

            current_chunk = (
                sentence + " "
            )

    if current_chunk:

        chunks.append(
            current_chunk.strip()
        )

    return chunks


def summarize_text(
    text,
    max_chars=500
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
    parent_chunk_size=3
):

    children = recursive_chunk(
        text,
        max_chars=700
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


if __name__ == "__main__":

    sample_text = """
    Paragraph one.

    Paragraph two.

    Paragraph three.

    Paragraph four.
    """

    children, parents = (
        build_parent_child_chunks(
            sample_text
        )
    )

    print(
        f"Children: {len(children)}"
    )

    print(
        f"Parents: {len(parents)}"
    )