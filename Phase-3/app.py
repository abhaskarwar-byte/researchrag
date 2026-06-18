import os
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


def post_api(path, **kwargs):

    response = requests.post(
        f"{API_URL}{path}",
        timeout=180,
        **kwargs
    )

    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        return {}

    if not response.ok:
        message = data.get(
            "detail",
            "Request failed"
        )
        raise RuntimeError(message)

    return data


st.set_page_config(
    page_title="Research RAG",
    page_icon="R",
    layout="wide"
)

# =====================================
# SESSION STATE
# =====================================

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "query_result" not in st.session_state:
    st.session_state.query_result = None

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp {
    background-color: #141820;
    color: #b8c0cc;
}

section[data-testid="stSidebar"] {
    background-color: #0d1017;
}

h1,h2,h3 {
    color: #e2c97e;
}

.stButton button {
    background-color: #b48f2a;
    color: black;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}

.stTextInput input {
    background-color: #1f2535;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("Research RAG")

    st.divider()

    if st.session_state.uploaded_file_name:
        st.success(
            st.session_state.uploaded_file_name
        )
    else:
        st.info("No document uploaded")

    st.divider()

    if st.button("Reset Chunks"):

        try:

            data = post_api("/reset")

            st.success(
                data["message"]
            )

            st.session_state.query_result = None
            st.session_state.uploaded_file_name = None

        except Exception as e:

            st.error(str(e))

# =====================================
# HEADER
# =====================================

st.title("Research Paper RAG Assistant")

st.caption(
    "Upload a paper, store text chunks in JSON, and ask questions with keyword search, BM25 ranking, and LLM reranking."
)

# =====================================
# UPLOAD
# =====================================

st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

manual_path = st.text_input(
    "Or enter PDF path manually",
    placeholder=r"Example: C:\papers\paper.pdf"
)

if st.button("Upload"):

    file_to_send = None
    file_name = None

    if uploaded_file is not None:

        file_to_send = uploaded_file.getvalue()
        file_name = uploaded_file.name

    elif manual_path.strip():

        path = manual_path.strip().strip('"')

        if not path.lower().endswith(".pdf"):

            st.error(
                "Manual path must point to a PDF file."
            )
            st.stop()

        if not os.path.exists(path):

            st.error(
                "Path does not exist. Please check the file location."
            )
            st.stop()

        with open(
            path,
            "rb"
        ) as f:

            file_to_send = f.read()

        file_name = os.path.basename(path)

    else:

        st.warning(
            "Upload a PDF or enter a valid PDF path."
        )
        st.stop()

    if file_to_send is not None:

        try:

            files = {
                "file": (
                    file_name,
                    file_to_send,
                    "application/pdf"
                )
            }

            data = post_api(
                "/upload",
                files=files
            )

            st.session_state.uploaded_file_name = data["filename"]
            st.session_state.query_result = None

            st.success(
                data["message"]
            )

        except Exception as e:

            st.error(str(e))

# =====================================
# INGEST
# =====================================

st.subheader("Ingest Document")

if st.button("Ingest Document"):

    if st.session_state.uploaded_file_name:

        try:

            data = post_api(
                "/ingest",
                params={
                    "file_name":
                    st.session_state.uploaded_file_name
                }
            )

            st.success(
                data["message"]
            )

            st.info(
                f"Chunks stored: {data['chunks']}"
            )

        except Exception as e:

            st.error(str(e))

    else:

        st.warning(
            "Upload a document first."
        )

# =====================================
# QUERY
# =====================================

st.subheader("Ask Question")

question = st.text_input(
    "Question",
    placeholder="What is positional encoding?"
)

if st.button("Ask"):

    if question.strip():

        try:

            with st.spinner(
                "Searching chunks and reranking with the LLM..."
            ):

                data = post_api(
                    "/query",
                    params={
                        "question":
                        question
                    }
                )

                st.session_state.query_result = data

        except Exception as e:

            st.error(str(e))

    else:

        st.warning(
            "Enter a question first."
        )

# =====================================
# RESULTS
# =====================================

result = st.session_state.query_result

if result:

    st.divider()

    st.subheader("Answer")

    st.write(
        result["answer"]
    )

    st.divider()

    st.subheader("Sources")

    if result["sources"]:

        for source in result["sources"]:

            st.write(
                source["source"]
            )

            st.write(
                f"Page: {source['page']} | "
                f"Chunk: {source['chunk']} | "
                f"BM25: {source.get('bm25_score', 'n/a')}"
            )

            st.write("---")

    else:

        st.info(
            "No sources available."
        )

    with st.expander(
        "Raw JSON"
    ):

        st.json(
            result
        )
