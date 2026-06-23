from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File
)

import os

from src.ingest import (
    ingest_document
)

from src.query import (
    query_document
)

from src.reset import (
    reset_database
)


app = FastAPI()

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
        "message":
        "Hierarchical Research RAG API Running",

        "retrieval":
        "parent_summary_bm25_child_retrieval_llm_reranking"
    }


# =====================================
# UPLOAD FILE
# =====================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF uploads are supported"
            )
        )

    file_path = os.path.join(
        DOCUMENTS_DIR,
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    return {
        "message":
        "File uploaded successfully",

        "filename":
        file.filename
    }


# =====================================
# INGEST DOCUMENT
# =====================================

@app.post("/ingest")
def ingest(
    file_name: str
):

    file_path = os.path.join(
        DOCUMENTS_DIR,
        file_name
    )

    if not os.path.exists(
        file_path
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                f"{file_name} not found"
            )
        )

    total_chunks = ingest_document(
        file_path
    )

    return {
        "message":
        "Document ingested successfully",

        "records":
        total_chunks
    }


# =====================================
# QUERY DOCUMENT
# =====================================

@app.post("/query")
def query(
    question: str
):

    if not question.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "Question cannot be empty"
            )
        )

    if not os.path.exists(
        "data/chunks.json"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "No chunks found. "
                "Upload and ingest a document first."
            )
        )

    result = query_document(
        question
    )

    return result


# =====================================
# RESET DATABASE
# =====================================

@app.post("/reset")
def reset():

    reset_database()

    return {
        "message":
        "Chunk store reset successfully"
    }