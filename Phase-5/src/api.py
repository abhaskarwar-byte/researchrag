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

from src.agent import (
    agent
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
        "Research RAG API Running",

        "retrieval":
        "LangGraph + BM25 + Parent-Child Retrieval + Cross Encoder + CRAG"
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
            detail="Only PDF uploads are supported."
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
        "File uploaded successfully.",

        "filename":
        file.filename
    }


# =====================================
# INGEST
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
            detail=f"{file_name} not found."
        )

    total_chunks = ingest_document(
        file_path
    )

    return {
        "message":
        "Document ingested successfully.",

        "chunks":
        total_chunks
    }


# =====================================
# QUERY
# =====================================

@app.post("/query")
def query(
    question: str
):

    if not question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not os.path.exists(
        "data/chunks.json"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "No chunks found. "
                "Please upload and ingest a document first."
            )
        )

    initial_state = {

        "question":
        question,

        "rewritten_question":
        "",

        "candidate_chunks":
        [],

        "reranked_chunks":
        [],

        "grade":
        "",

        "answer":
        "",

        "retry_count":
        0
    }

    result = agent.invoke(
        initial_state
    )

    sources = []

    for chunk in result["reranked_chunks"]:

        sources.append(
            {
                "source":
                chunk.get("source"),

                "page":
                chunk.get("page"),

                "chunk":
                chunk.get("chunk")
            }
        )

    return {

        "answer":
        result["answer"],

        "grade":
        result["grade"],

        "retries":
        result["retry_count"],

        "sources":
        sources
    }


# =====================================
# RESET
# =====================================

@app.post("/reset")
def reset():

    reset_database()

    return {
        "message":
        "Chunk store reset successfully."
    }