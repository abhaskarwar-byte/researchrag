import json
import re

from groq import Groq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL
)


client = Groq(
    api_key=GROQ_API_KEY
)


PROMPT = """
You are an expert information extraction system.

Extract ONLY the entities and relationships that are EXPLICITLY present in the given text.

Return ONLY valid JSON.

Format:

{
    "entities":[
        {
            "name":"",
            "type":""
        }
    ],

    "relationships":[
        {
            "source":"",
            "target":"",
            "relation":""
        }
    ]
}

Rules:

1. Do NOT explain anything.

2. Do NOT return markdown.

3. Do NOT surround the JSON with ```.

4. Do NOT invent entities.

5. Do NOT infer relationships.

6. If nothing is found return:

{
    "entities":[],
    "relationships":[]
}

Entity Types:

Person
Organization
Model
Method
Concept
Dataset
Task
Metric
Component
Conference
Location

Relationship labels must be UPPER_CASE.

Examples:

USES
PROPOSES
TRAINED_ON
EVALUATED_ON
MENTIONS
CONTAINS
OUTPERFORMS
AFFILIATED_WITH
USED_FOR
USED_IN
APPLIED_TO
COMPUTED_AS
IMPROVES
"""
# =====================================
# EXTRACT ENTITIES
# =====================================

def extract_entities(
    text
):

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        temperature=0,

        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    answer = response.choices[0].message.content.strip()

    # ---------------------------------
    # Remove markdown if Groq adds it
    # ---------------------------------

    answer = re.sub(
        r"^```json",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = re.sub(
        r"^```",
        "",
        answer
    )

    answer = re.sub(
        r"```$",
        "",
        answer
    )

    answer = answer.strip()

    # ---------------------------------
    # Parse JSON
    # ---------------------------------

    try:

        data = json.loads(
            answer
        )

    except Exception:

        print("\nInvalid JSON returned by Groq\n")

        print(answer)

        data = {}

    # ---------------------------------
    # Always return same structure
    # ---------------------------------

    entities = data.get(
        "entities",
        []
    )

    relationships = data.get(
        "relationships",
        []
    )

    # ---------------------------------
    # Remove duplicate entities
    # ---------------------------------

    unique = {}

    for entity in entities:

        name = entity.get(
            "name",
            ""
        ).strip()

        if not name:
            continue

        unique[name.lower()] = {

            "name": name,

            "type": entity.get(
                "type",
                "Concept"
            )
        }

    entities = list(
        unique.values()
    )

    # ---------------------------------
    # Normalize relationship labels
    # ---------------------------------

    cleaned_relationships = []

    for relation in relationships:

        source = relation.get(
            "source",
            ""
        ).strip()

        target = relation.get(
            "target",
            ""
        ).strip()

        rel = relation.get(
            "relation",
            ""
        ).upper().replace(
            " ",
            "_"
        )

        if source and target and rel:

            cleaned_relationships.append(

                {

                    "source": source,

                    "target": target,

                    "relation": rel

                }

            )

    return {

        "entities": entities,

        "relationships": cleaned_relationships

    }


# =====================================
# TEST
# =====================================

if __name__ == "__main__":

    sample = """
    Transformer uses Multi-Head Attention
    and Positional Encoding.
    """

    result = extract_entities(
        sample
    )

    print(

        json.dumps(

            result,

            indent=4

        )

    )