import chromadb
import ollama

from extract import extract_pdf_pages
from chunk import chunk_text

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="research_documents"
)

file_path = "documents/sample.pdf"

pages = extract_pdf_pages(file_path)

chunk_counter = 1

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
            ids=[f"chunk_{chunk_counter}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": file_path,
                "page": page_number,
                "chunk": chunk_counter
            }]
        )

        chunk_counter += 1

print(
    f"Stored {chunk_counter - 1} chunks"
)