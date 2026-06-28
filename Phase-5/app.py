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

        raise RuntimeError(
            data.get(
                "detail",
                "Request failed"
            )
        )

    return data


st.set_page_config(
    page_title="Research RAG",
    page_icon="📄",
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
# CSS
# =====================================

st.markdown("""
<style>

.stApp{
    background:#141820;
    color:#d7d7d7;
}

section[data-testid="stSidebar"]{
    background:#0d1017;
}

h1,h2,h3{
    color:#e2c97e;
}

.stButton button{
    background:#b48f2a;
    color:black;
    border:none;
    border-radius:8px;
    font-weight:bold;
}

.stTextInput input{
    background:#1f2535;
    color:white;
}

</style>
""",
unsafe_allow_html=True)


# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title(
        "Research RAG"
    )

    st.divider()

    if st.session_state.uploaded_file_name:

        st.success(
            st.session_state.uploaded_file_name
        )

    else:

        st.info(
            "No document uploaded"
        )

    st.divider()

    if st.button(
        "Reset Chunks"
    ):

        try:

            data = post_api(
                "/reset"
            )

            st.success(
                data["message"]
            )

            st.session_state.uploaded_file_name = None

            st.session_state.query_result = None

        except Exception as e:

            st.error(str(e))


# =====================================
# HEADER
# =====================================

st.title(
    "Research Paper RAG Assistant"
)

st.caption(
    "LangGraph • BM25 • Parent-Child Retrieval • Cross Encoder • CRAG • Query Rewrite"
)


# =====================================
# UPLOAD
# =====================================

st.subheader(
    "Upload Document"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

manual_path = st.text_input(
    "Or enter PDF path manually",
    placeholder=r"C:\papers\paper.pdf"
)

if st.button(
    "Upload"
):

    file_data = None

    filename = None

    if uploaded_file:

        file_data = uploaded_file.getvalue()

        filename = uploaded_file.name

    elif manual_path.strip():

        path = manual_path.strip().strip('"')

        if not os.path.exists(path):

            st.error(
                "File not found."
            )

            st.stop()

        filename = os.path.basename(path)

        with open(
            path,
            "rb"
        ) as f:

            file_data = f.read()

    else:

        st.warning(
            "Please upload a PDF."
        )

        st.stop()

    try:

        data = post_api(

            "/upload",

            files={
                "file":
                (
                    filename,
                    file_data,
                    "application/pdf"
                )
            }
        )

        st.session_state.uploaded_file_name = data["filename"]

        st.success(
            data["message"]
        )

    except Exception as e:

        st.error(str(e))


# =====================================
# INGEST
# =====================================

st.subheader(
    "Ingest Document"
)

if st.button(
    "Ingest Document"
):

    if not st.session_state.uploaded_file_name:

        st.warning(
            "Upload a PDF first."
        )

    else:

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


# =====================================
# QUERY
# =====================================

st.subheader(
    "Ask Question"
)

question = st.text_input(
    "Question",
    placeholder="What is positional encoding?"
)

if st.button(
    "Ask"
):

    if question.strip():

        try:

            with st.spinner(

                "Running LangGraph pipeline..."
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
            "Please enter a question."
        )


# =====================================
# RESULTS
# =====================================

result = st.session_state.query_result

if result:

    st.divider()

    st.subheader(
        "Answer"
    )

    st.write(
        result["answer"]
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Retrieval Grade",
            result["grade"]
        )

    with col2:

        st.metric(
            "Rewrite Attempts",
            result["retries"]
        )

    st.divider()

    st.subheader(
        "Retrieved Sources"
    )

    if result["sources"]:

        for source in result["sources"]:

            st.markdown(
                f"""
**{source['source']}**

Page: **{source['page']}**

Chunk: **{source['chunk']}**
"""
            )

            st.write("---")

    else:

        st.info(
            "No sources returned."
        )

    with st.expander(
        "Raw JSON"
    ):

        st.json(
            result
        )