from fastapi import FastAPI, UploadFile, File
import os

from src.ingest import ingest_document

from src.query import query_document
app = FastAPI()

from src.reset import reset_database

DOCUMENTS_DIR = "documents"

os.makedirs(
    DOCUMENTS_DIR,
    exist_ok=True
)


# =====================================
# HOME
# =====================================

@app.get("/")
def home():

    return {
        "message": "Research RAG API Running"
    }


# =====================================
# UPLOAD FILE
# =====================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        DOCUMENTS_DIR,
        file.filename
    )

    with open(file_path, "wb") as f:
        f.write(
            await file.read()
        )

    return {
        "message": "File uploaded successfully",
        "filename": file.filename
    }


# =====================================
# INGEST DOCUMENT
# =====================================

@app.post("/ingest")
def ingest(file_name: str):

    file_path = os.path.join(
        DOCUMENTS_DIR,
        file_name
    )

    if not os.path.exists(file_path):

        return {
            "error": f"{file_name} not found"
        }

    total_chunks = ingest_document(
        file_path
    )

    return {
        "message": "Document ingested successfully",
        "chunks": total_chunks
    }

@app.post("/query")
def query(question: str):

    result = query_document(
        question
    )

    return result

@app.post("/reset")
def reset():

    reset_database()

    return {
        "message": "Vector database reset successfully"
    }