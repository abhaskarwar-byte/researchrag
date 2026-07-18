import json
import os
import re
import time

from groq import (
    Groq,
    RateLimitError
)

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL
)

from src.ingest import ingest_document
from src.entity_extractor import extract_entities


# =====================================================
# CONFIGURATION
# =====================================================

DOCUMENTS_DIR = "documents"

OUTPUT_FILE = os.path.join(
    "benchmarks",
    "benchmark_dataset.json"
)

MIN_SUMMARY_WORDS = 20

SKIP_SECTIONS = [

    "references",

    "bibliography",

    "acknowledgements",

    "acknowledgments",

    "appendix",

    "author information",

    "copyright",

    "license"

]


client = Groq(
    api_key=GROQ_API_KEY
)


# =====================================================
# BENCHMARK PROMPT
# =====================================================

BENCHMARK_PROMPT = """
You are an API.

Return ONLY valid JSON.

Do NOT explain anything.

Do NOT introduce the answer.

Do NOT conclude the answer.

Do NOT use markdown.

Do NOT use triple backticks.

Do NOT output anything before the JSON.

Do NOT output anything after the JSON.

The output MUST begin with '['.

The output MUST end with ']'.

-----------------------------------------------------

You are given

1. Research Paper Name

2. Parent Summary

3. Extracted Entities

4. Extracted Relationships

Generate between 0 and 2 benchmark questions.

The questions should evaluate understanding of the
research paper.

Use ONLY information explicitly present.

Never invent facts.

Never infer information.

Avoid yes/no questions.

Question Types

Conceptual

Factual

Methodology

Comparison

Application

Difficulty

Easy

Medium

Hard

If no meaningful questions can be created return

[]

Return ONLY

[
    {
        "question":"",
        "question_type":"",
        "difficulty":"",
        "reference_answer":""
    }
]
"""


# =====================================================
# PARSE JSON
# =====================================================

def parse_json(response):

    response = response.strip()

    response = re.sub(
        r"^```json",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"^```",
        "",
        response
    )

    response = re.sub(
        r"```$",
        "",
        response
    )

    response = response.strip()

    start = response.find("[")

    end = response.rfind("]")

    if start != -1 and end != -1:

        response = response[start:end + 1]

    try:

        return json.loads(response)

    except Exception:

        print("\nInvalid JSON Returned\n")

        print(response)

        return []


# =====================================================
# REFERENCE FILTER
# =====================================================

def is_reference_section(summary):

    summary_lower = summary.lower()

    for section in SKIP_SECTIONS:

        if section in summary_lower:

            return True

    bibliography_keywords = [

        "et al.",

        "proceedings",

        "conference",

        "journal",

        "transactions",

        "vol.",

        "volume",

        "pages",

        "pp.",

        "ieee",

        "acm",

        "springer",

        "elsevier",

        "arxiv"

    ]

    keyword_matches = sum(

        keyword in summary_lower

        for keyword in bibliography_keywords

    )

    citation_count = len(

        re.findall(

            r"\[\d+\]",

            summary

        )

    )

    if citation_count >= 5:

        return True

    if keyword_matches >= 3:

        return True

    return False


# =====================================================
# GENERATE QUESTIONS
# =====================================================

def generate_questions(
    paper,
    summary,
    entities,
    relationships
):

    prompt = f"""
Research Paper

{paper}

Parent Summary

{summary}

Entities

{json.dumps(entities, indent=2)}

Relationships

{json.dumps(relationships, indent=2)}

{BENCHMARK_PROMPT}
"""

    retries = 5

    for attempt in range(retries):

        try:

            response = client.chat.completions.create(

                model=GROQ_MODEL,

                temperature=0,

                messages=[

                    {
                        "role": "system",
                        "content": "Return ONLY valid JSON."
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ]

            )

            return parse_json(
                response.choices[0].message.content
            )

        except RateLimitError:

            wait = 2 ** attempt

            print(
                f"\nRate limit reached. Waiting {wait} seconds..."
            )

            time.sleep(wait)

    print("\nSkipping after repeated rate limits.")

    return []

# =====================================================
# GENERATE BENCHMARK
# =====================================================

def generate_parent_benchmark(

    parent,

    benchmark_id

):

    summary = parent["summary"]

    if is_reference_section(summary):

        return [], benchmark_id

    if len(summary.split()) < MIN_SUMMARY_WORDS:

        return [], benchmark_id

    extracted = extract_entities(summary)

    entities = extracted["entities"]

    relationships = extracted["relationships"]

    if not entities:

        return [], benchmark_id

    questions = generate_questions(

        os.path.basename(parent["source"]),

        summary,

        entities,

        relationships

    )

    benchmark = []

    for item in questions:

        question = item.get(
            "question",
            ""
        ).strip()

        if not question:

            continue

        benchmark.append(

            {

                "id": benchmark_id,

                "paper": os.path.basename(
                    parent["source"]
                ),

                "page": parent["page"],

                "parent_chunk": parent["parent_id"],

                "summary": summary,

                "question": question,

                "question_type": item.get(
                    "question_type",
                    "Conceptual"
                ),

                "difficulty": item.get(
                    "difficulty",
                    "Medium"
                ),

                "reference_answer": item.get(
                    "reference_answer",
                    ""
                ),

                "expected_entities": entities,

                "expected_relationships": relationships

            }

        )

        benchmark_id += 1

    return benchmark, benchmark_id

# =====================================================
# RESUME HELPERS
# =====================================================

def load_existing_dataset():

    if os.path.exists(OUTPUT_FILE):

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    return []


def get_next_benchmark_id(dataset):

    if not dataset:
        return 1

    return max(item["id"] for item in dataset) + 1


def build_completed_parent_lookup(dataset):

    completed = set()

    for item in dataset:

        completed.add(
            (
                item["paper"],
                item["parent_chunk"]
            )
        )

    return completed

# =====================================================
# GENERATE COMPLETE BENCHMARK DATASET
# =====================================================

def generate_benchmark_dataset():

    dataset = load_existing_dataset()

    print(
    f"Loaded {len(dataset)} existing benchmark questions."
    )
    
    benchmark_id = get_next_benchmark_id(dataset)

    completed_parents = build_completed_parent_lookup(dataset)

    processed_papers = 0

    skipped_parents = 0

    print("\n======================================")
    print("BENCHMARK DATASET GENERATION")
    print("======================================")

    files = sorted(os.listdir(DOCUMENTS_DIR))

    for file in files:

        if not file.lower().endswith(
            (".pdf", ".docx", ".txt")
        ):
            continue

        processed_papers += 1

        file_path = os.path.join(
            DOCUMENTS_DIR,
            file
        )

        print(f"\nProcessing Paper : {file}")

        chunks = ingest_document(
            file_path
        )

        parent_chunks = [

            chunk

            for chunk in chunks

            if chunk["chunk_type"] == "parent"

        ]

        print(
            f"Parent Chunks Found : {len(parent_chunks)}"
        )

        paper_questions = 0

        paper_skipped = 0

        for parent in parent_chunks:

            
            key = (
                file,
                parent["parent_id"]
            )

            if key in completed_parents:

                print(
                    f"Skipping Parent {parent['parent_id']} (Already Completed)"
                )

                continue

            benchmark, benchmark_id = (

                generate_parent_benchmark(

                    parent,

                    benchmark_id

                )

            )

            if not benchmark:

                skipped_parents += 1
                paper_skipped += 1

                continue

            dataset.extend(
                benchmark
            )

            completed_parents.add(key)

            save_dataset(dataset)
            
            paper_questions += len(
                benchmark
            )

        print(
            f"Benchmark Questions Generated : {paper_questions}"
        )

        print(
            f"Skipped Parent Summaries      : {paper_skipped}"
        )


    print("\n======================================")
    print("BENCHMARK GENERATION COMPLETE")
    print("======================================")

    print(
        f"Papers Processed          : {processed_papers}"
    )

    print(
        f"Skipped Parent Summaries  : {skipped_parents}"
    )

    print(
        f"Total Benchmark Questions : {len(dataset)}"
    )

    return dataset


# =====================================================
# SAVE DATASET
# =====================================================

def save_dataset(dataset):

    os.makedirs(

        os.path.dirname(
            OUTPUT_FILE
        ),

        exist_ok=True

    )

    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            dataset,

            file,

            indent=4,

            ensure_ascii=False

        )

    print("\n======================================")
    print("BENCHMARK DATASET SAVED")
    print("======================================")

    print(
        f"Location : {OUTPUT_FILE}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    generate_benchmark_dataset()



if __name__ == "__main__":

    main()