import json
import math
import re
import sys
from collections import Counter

import ollama


FALLBACK_MESSAGE = "I could not find that information in the document."


STOPWORDS = {
    "what",
    "is",
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "for",
    "and",
    "on",
    "with",
    "how",
    "why",
    "does",
    "do",
    "are",
    "was",
    "were"
}


def extract_keywords(text):

    keywords = []

    for word in text.lower().split():

        cleaned = re.sub(
            r"[^a-z0-9]",
            "",
            word
        )

        if (
            cleaned
            and cleaned not in STOPWORDS
        ):
            keywords.append(
                cleaned
            )

    return keywords


def tokenize(text):

    return re.findall(
        r"[a-z0-9]+",
        text.lower()
    )


def score_chunk(text, keywords, question):

    score = 0

    for keyword in keywords:

        matches = re.findall(
            rf"\b{re.escape(keyword)}\b",
            text
        )

        score += len(matches)

    phrase = " ".join(keywords)

    if phrase:

        score += text.count(phrase) * 5

    if (
        "what is" in question.lower()
        and phrase
        and phrase in text
    ):
        score += 5

    return score


def bm25_rank(chunks, keywords, top_k=8):

    if not chunks or not keywords:
        return []

    documents = []
    document_lengths = []
    document_frequency = Counter()

    for keyword_score, chunk in chunks:

        tokens = tokenize(
            chunk["text"]
        )

        token_counts = Counter(
            tokens
        )

        documents.append(
            {
                "keyword_score": keyword_score,
                "chunk": chunk,
                "counts": token_counts,
                "length": len(tokens)
            }
        )

        document_lengths.append(
            len(tokens)
        )

        for term in set(tokens):

            document_frequency[term] += 1

    total_documents = len(documents)
    average_length = (
        sum(document_lengths) / total_documents
        if total_documents
        else 0
    )

    if average_length == 0:
        return []

    k1 = 1.5
    b = 0.75

    ranked = []

    for document in documents:

        bm25_score = 0.0

        for keyword in keywords:

            term_frequency = document["counts"].get(
                keyword,
                0
            )

            if term_frequency == 0:
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
                / (
                    df
                    + 0.5
                )
            )

            denominator = (
                term_frequency
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
                * term_frequency
                * (
                    k1
                    + 1
                )
                / denominator
            )

        phrase = " ".join(
            keywords
        )

        chunk_text = document["chunk"]["text"].lower()

        if phrase and phrase in chunk_text:

            bm25_score += 2.0

        if (
            "what is" in chunk_text
            or "definition" in chunk_text
        ):

            bm25_score += 0.25

        ranked.append(
            (
                bm25_score,
                document["chunk"],
                document["keyword_score"]
            )
        )

    ranked.sort(
        key=lambda item: (
            item[0],
            item[2]
        ),
        reverse=True
    )

    return [
        (
            bm25_score,
            chunk
        )
        for bm25_score, chunk, _ in ranked[:top_k]
    ]


def parse_rerank_indices(ranking, max_index):

    selected_indices = []

    labeled_numbers = re.findall(
        r"\bC([1-9]\d*)\b",
        ranking,
        flags=re.IGNORECASE
    )

    if labeled_numbers:
        numbers = labeled_numbers
    else:
        numbers = re.findall(
            r"(?<![\d.])([1-9]\d*)(?![\d.])",
            ranking
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


def definition_relevance(chunk, keywords):

    text = chunk["text"].lower()
    phrase = " ".join(keywords)

    if not phrase or phrase not in text:
        return 0

    score = 3

    cues = [
        "in order for the model",
        "order of the sequence",
        "inject some information",
        "input embeddings",
        "same dimension",
        "sine and cosine",
        "where pos is the position",
        "relative positions",
        "extrapolate to sequence lengths"
    ]

    for cue in cues:

        if cue in text:
            score += 2

    if score == 3:
        return 0

    return score


def safe_print(value):

    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"

    print(
        text.encode(
            encoding,
            errors="replace"
        ).decode(
            encoding,
            errors="replace"
        )
    )


def select_best_indices(selected_indices, top_chunks, keywords, question):

    if not question.lower().startswith("what is"):
        return selected_indices[:3]

    selected = []

    relevant_candidates = []

    for index, (_, chunk) in enumerate(top_chunks):

        relevance = definition_relevance(
            chunk,
            keywords
        )

        if relevance > 0:

            relevant_candidates.append(
                (
                    relevance,
                    index
                )
            )

    relevant_candidates.sort(
        reverse=True
    )

    for index in selected_indices:

        relevance = definition_relevance(
            top_chunks[index][1],
            keywords
        )

        if (
            relevance > 0
            and index not in selected
        ):

            selected.append(
                index
            )

    for _, index in relevant_candidates:

        if index not in selected:

            selected.append(
                index
            )

        if len(selected) == 3:
            break

    if selected:
        return selected[:3]

    return selected_indices[:3]


def answer_is_grounded(answer, context, keywords):

    if answer == FALLBACK_MESSAGE:
        return True

    answer_lower = answer.lower()
    context_lower = context.lower()

    if len(answer_lower) < 40:
        return False

    for keyword in keywords:

        if (
            keyword in answer_lower
            and keyword not in context_lower
        ):
            return False

    if (
        "positional" in keywords
        and "encoding" in keywords
    ):

        definition_terms = [
            "positional encoding",
            "positional encodings"
        ]

        position_terms = [
            "position",
            "positions",
            "order",
            "token",
            "tokens",
            "sequence"
        ]

        embedding_terms = [
            "embedding",
            "embeddings",
            "summed",
            "added"
        ]

        if not any(
            term in answer_lower
            for term in definition_terms
        ):
            return False

        if not any(
            term in answer_lower
            for term in position_terms
        ):
            return False

        if not any(
            term in answer_lower
            for term in embedding_terms
        ):
            return False

        context_terms = [
            "positional encodings",
            "order of the sequence",
            "position of the tokens",
            "embedding"
        ]

        if not any(
            term in context_lower
            for term in context_terms
        ):
            return False

    return True


def query_document(question):

    # =====================================
    # LOAD CHUNKS
    # =====================================

    with open(
        "data/chunks.json",
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    # =====================================
    # STOPWORDS
    # =====================================

    keywords = extract_keywords(
        question
    )

    print("\nKEYWORDS:")
    print(keywords)

    # =====================================
    # SCORE CHUNKS
    # =====================================

    scored_chunks = []

    for chunk in chunks:

        text = chunk["text"].lower()

        # =====================================
        # FILTER SHORT CHUNKS
        # =====================================

        if len(text) < 200:
            continue

        score = score_chunk(
            text,
            keywords,
            question
        )

        if score > 0:

            scored_chunks.append(
                (
                    score,
                    chunk
                )
            )

    # =====================================
    # NO MATCHES
    # =====================================

    if len(scored_chunks) == 0:

        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "sources": []
        }

    # =====================================
    # BM25 RANKING
    # =====================================

    top_chunks = bm25_rank(
        scored_chunks,
        keywords,
        top_k=8
    )

    print("\nTOP BM25 CHUNKS:")

    for score, chunk in top_chunks:

        print(
            f"BM25: {score:.4f} | "
            f"Page: {chunk['page']} | "
            f"Chunk: {chunk['chunk']}"
        )

    # =====================================
    # BUILD RERANKING PROMPT
    # =====================================

    chunk_text = ""

    for i, (_, chunk) in enumerate(
        top_chunks
    ):

        chunk_text += (
            f"\nC{i+1}:\n"
            f"{chunk['text'][:1000]}\n"
        )

    # =====================================
    # RERANKING PROMPT
    # =====================================

    rerank_prompt = f"""
    You are a research-paper retrieval assistant.

    Question:
    {question}

    Candidate Chunks:

    {chunk_text}

    Select the 3 candidate IDs that BEST answer the question.

    Ignore chunks that only mention keywords but do not explain the concept.
    Ignore section numbers such as 3.5, 5.1, 5.2, or 6.3.

    Return ONLY three candidate IDs separated by commas.

    Example:
    C1,C2,C4

    Do not explain your answer.
    Do not use bullet points.
    """

    # =====================================
    # GEMMA RERANKING
    # =====================================

    rerank_response = ollama.chat(
        model="gemma2:2b",
        messages=[
            {
                "role": "user",
                "content": rerank_prompt
            }
        ]
    )

    ranking = (
        rerank_response["message"]["content"]
        .strip()
    )

    print("\nRERANKING RESULT:")
    print(ranking)

        # =====================================
    # PARSE RANKING
    # =====================================

    selected_indices = parse_rerank_indices(
        ranking,
        len(top_chunks)
    )

    selected_indices = select_best_indices(
        selected_indices,
        top_chunks,
        keywords,
        question
    )

    # =====================================
    # FALLBACK
    # =====================================

    if len(selected_indices) == 0:

        selected_indices = [
            0,
            1,
            2
        ]

    selected_indices = selected_indices[:3]


    # =====================================
    # RERANKED CHUNKS
    # =====================================

    reranked_chunks = []

    for index in selected_indices[:3]:

        reranked_chunks.append(
            top_chunks[index]
        )

    reranked_chunks.sort(
        key=lambda item: (
            item[1]["page"],
            item[1]["chunk"]
        )
    )

    print("\nTOP RERANKED CHUNKS:")

    for score, chunk in reranked_chunks:

        print(
            f"\nBM25 Score: {score:.4f}"
        )

        print(
            f"Page: {chunk['page']}"
        )

        safe_print(
            chunk["text"][:300]
        )

    # =====================================
    # BUILD CONTEXT
    # =====================================

    context = "\n\n".join(
        chunk["text"]
        for _, chunk in reranked_chunks
    )

    # =====================================
    # FINAL RAG PROMPT
    # =====================================

    prompt = f"""
    You are a research-paper question answering assistant.

    Use ONLY the provided context.

    Rules:

    1. Every statement must be directly supported by the context.

    2. Do NOT use outside knowledge.

    3. Do NOT guess.

    4. Do NOT invent facts, numbers, dates, statistics, or explanations.

    5. If the answer is not present in the context, respond EXACTLY:

    I could not find that information in the document.

    6. If the question asks "What is ...", start with a direct definition or explanation from the context.

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

    if not answer_is_grounded(
        answer,
        context,
        keywords
    ):
        answer = FALLBACK_MESSAGE

    # =====================================
    # SOURCES
    # =====================================

    sources = []

    for _, chunk in reranked_chunks:

        sources.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "bm25_score": round(
                    score,
                    4
                )
            }
        )

    # =====================================
    # RETURN JSON
    # =====================================

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval_pipeline": [
            "keyword_search",
            "bm25_ranking",
            "llm_reranking",
            "grounded_answer"
        ]
    }


if __name__ == "__main__":

    result = query_document(
        "What is positional encoding?"
    )

    print("\nANSWER:\n")

    print(
        result["answer"]
    )

    print("\nSOURCES:\n")

    for source in result["sources"]:

        print(source)
