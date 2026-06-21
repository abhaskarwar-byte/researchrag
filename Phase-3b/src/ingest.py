import json
import os

from src.extract import extract_pdf_pages
from src.chunk import (
    build_parent_child_chunks,
    summarize_text
)


DATA_DIR = "data"

CHUNKS_PATH = os.path.join(
    DATA_DIR,
    "chunks.json"
)


def ingest_document(file_path):

    pages = extract_pdf_pages(
        file_path
    )

    all_chunks = []

    print(
        f"\nTOTAL PAGES: {len(pages)}"
    )

    global_parent_id = 1
    global_chunk_id = 1

    for page_data in pages:

        page_number = page_data["page"]

        print(
            f"\nProcessing Page {page_number}"
        )

        children, parents = (
            build_parent_child_chunks(
                page_data["text"]
            )
        )

        for parent in parents:

            start_index = (
                parent["child_range"][0]
            )

            end_index = (
                parent["child_range"][1]
            )

            parent_children = children[
                start_index:end_index
            ]

            parent_text = "\n\n".join(
                parent_children
            )

            parent_summary = (
                summarize_text(
                    parent_text,
                    max_chars=300
                )
            )

            # -------------------------
            # STORE PARENT
            # -------------------------

            all_chunks.append(
                {
                    "source": file_path,
                    "page": page_number,
                    "chunk": global_chunk_id,
                    "parent_id": global_parent_id,
                    "chunk_type": "parent",
                    "summary": parent_summary
                }
            )

            global_chunk_id += 1

            # -------------------------
            # STORE CHILDREN
            # -------------------------

            for child_text in parent_children:

                all_chunks.append(
                    {
                        "source": file_path,
                        "page": page_number,
                        "chunk": global_chunk_id,
                        "parent_id": global_parent_id,
                        "chunk_type": "child",
                        "text": child_text
                    }
                )

                global_chunk_id += 1

            print(
                f"Parent {global_parent_id} stored"
            )

            global_parent_id += 1

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            indent=4,
            ensure_ascii=False
        )

    return len(all_chunks)


if __name__ == "__main__":

    total_chunks = ingest_document(
        "documents/sample.pdf"
    )

    print(
        f"\nStored {total_chunks} records"
    )