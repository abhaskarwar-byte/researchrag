import chromadb
import ollama

from src.extract import extract_pdf_pages
from src.chunk import chunk_text


def ingest_document(file_path):

    # =====================================
    # CONNECT TO CHROMADB
    # =====================================

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="research_documents"
    )

    # =====================================
    # EXTRACT PAGES
    # =====================================

    pages = extract_pdf_pages(
        file_path
    )

    print(f"\nTOTAL PAGES: {len(pages)}")

    for page in pages:
     print(
        f"Page {page['page']} -> "
        f"{len(page['text'])} chars"
    )

    chunk_counter = 1

    # =====================================
    # PROCESS PAGES
    # =====================================

    for page_data in pages:

        page_number = page_data["page"]

        chunks = chunk_text(
            page_data["text"]
        )

        for chunk in chunks:

            embedding = ollama.embeddings(
                model="nomic-embed-text",
                prompt=chunk
            )["embedding"]

            collection.add(
                ids=[
                    f"chunk_{chunk_counter}"
                ],
                documents=[
                    chunk
                ],
                embeddings=[
                    embedding
                ],
                metadatas=[
                    {
                        "source": file_path,
                        "page": page_number,
                        "chunk": chunk_counter
                    }
                ]
            )

            chunk_counter += 1

    return chunk_counter - 1


if __name__ == "__main__":

    total_chunks = ingest_document(
        "documents/sample.pdf"
    )

    print(
        f"Stored {total_chunks} chunks"
    )