from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel

import os
import shutil

from src.graph_build import (
    build_graph
)

from src.router import (
    route_query
)

from src.answer_generator import (
    answer_question
)


app = FastAPI(

    title="Graph RAG API",

    version="1.0"

)


DOCUMENT_FOLDER = "documents"

os.makedirs(

    DOCUMENT_FOLDER,

    exist_ok=True

)


# =====================================
# REQUEST MODEL
# =====================================

class QueryRequest(

    BaseModel

):

    question: str


# =====================================
# HOME
# =====================================

@app.get("/")

def home():

    return {

        "message": "Graph RAG API Running"

    }


# =====================================
# UPLOAD DOCUMENT
# =====================================

@app.post("/upload")

async def upload_document(

    file: UploadFile = File(...)

):

    if not (

        file.filename.endswith(".pdf")

        or file.filename.endswith(".txt")

        or file.filename.endswith(".docx")

    ):

        raise HTTPException(

            status_code=400,

            detail="Unsupported file type."

        )

    file_path = os.path.join(

        DOCUMENT_FOLDER,

        file.filename

    )

    with open(

        file_path,

        "wb"

    ) as buffer:

        shutil.copyfileobj(

            file.file,

            buffer

        )

    return {

        "message": "Upload successful.",

        "filename": file.filename,

        "file_path": file_path

    }


# =====================================
# INGEST DOCUMENT
# =====================================

@app.post("/ingest")

def ingest(

    filename: str

):

    file_path = os.path.join(

        DOCUMENT_FOLDER,

        filename

    )

    if not os.path.exists(

        file_path

    ):

        raise HTTPException(

            status_code=404,

            detail="Document not found."

        )

    build_graph(

        file_path

    )

    return {

        "message": "Knowledge Graph built successfully."

    }


# =====================================
# QUERY
# =====================================

@app.post("/query")

def query(

    request: QueryRequest

):

    retrieval = route_query(

        request.question

    )

    # -----------------------------
    # Final Hybrid Retrieval
    # -----------------------------

    if (

        isinstance(retrieval, dict)

        and "semantic_results" in retrieval

    ):

        response = answer_question(

            request.question,

            retrieval,

            retrieval_type="hybrid"

        )

    # -----------------------------
    # Graph Retrieval
    # -----------------------------

    elif (

        isinstance(retrieval, dict)

        and "results" in retrieval

    ):

        response = answer_question(

            request.question,

            retrieval,

            retrieval_type="graph"

        )

    # -----------------------------
    # Semantic Retrieval
    # (Vector + Full Text)
    # -----------------------------

    else:

        response = answer_question(

            request.question,

            retrieval,

            retrieval_type="vector"

        )

    return {

        "question": response["question"],

        "answer": response["answer"],

        "context": response["context"]

    }


# =====================================
# RUN
# =====================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "src.api:app",

        host="127.0.0.1",

        port=8000,

        reload=True

    )