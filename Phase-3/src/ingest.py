import json
import os

from src.extract import extract_pdf_pages
from src.chunk import chunk_text


DATA_DIR = "data"
CHUNKS_PATH = os.path.join(
    DATA_DIR,
    "chunks.json"
)


def ingest_document(file_path):

    pages = extract_pdf_pages(file_path)

    all_chunks = []

    chunk_counter = 1

    print(f"\nTOTAL PAGES: {len(pages)}")

    for page in pages:

        print(
            f"Page {page['page']} -> "
            f"{len(page['text'])} chars"
        )

    for page_data in pages:

        page_number = page_data["page"]

        chunks = chunk_text(
            page_data["text"]
        )

        for chunk in chunks:

            all_chunks.append(
                {
                    "source": file_path,
                    "page": page_number,
                    "chunk": chunk_counter,
                    "text": chunk
                }
            )

            chunk_counter += 1

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
        f"Stored {total_chunks} chunks"
    )
