import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Research RAG",
    page_icon="📚",
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

    st.title("📚 Research RAG")

    st.divider()

    if st.session_state.uploaded_file_name:
        st.success(
            st.session_state.uploaded_file_name
        )
    else:
        st.info("No document uploaded")

    st.divider()

    if st.button("Reset Database"):

        try:

            response = requests.post(
                f"{API_URL}/reset"
            )

            st.success(
                response.json()["message"]
            )

            st.session_state.query_result = None

        except Exception as e:

            st.error(str(e))

# =====================================
# HEADER
# =====================================

st.title("Research Paper RAG Assistant")

st.caption(
    "Upload papers, ingest them into ChromaDB, and ask questions."
)

# =====================================
# UPLOAD
# =====================================

st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if st.button("Upload"):

    if uploaded_file is not None:

        try:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            response = requests.post(
                f"{API_URL}/upload",
                files=files
            )

            data = response.json()

            st.session_state.uploaded_file_name = data["filename"]

            st.success(
                data["message"]
            )

        except Exception as e:

            st.error(str(e))

    else:

        st.warning(
            "Select a PDF first."
        )

# =====================================
# INGEST
# =====================================

st.subheader("Ingest Document")

if st.button("Ingest Document"):

    if st.session_state.uploaded_file_name:

        try:

            response = requests.post(
                f"{API_URL}/ingest",
                params={
                    "file_name":
                    st.session_state.uploaded_file_name
                }
            )

            data = response.json()

            if "error" in data:

                st.error(
                    data["error"]
                )

            else:

                st.success(
                    data["message"]
                )

                st.info(
                    f"Chunks Stored: {data['chunks']}"
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
    placeholder="What are latent diffusion models?"
)

if st.button("Ask"):

    if question.strip():

        try:

            with st.spinner(
                "Generating answer..."
            ):

                response = requests.post(
                    f"{API_URL}/query",
                    params={
                        "question":
                        question
                    }
                )

                st.session_state.query_result = (
                    response.json()
                )

        except Exception as e:

            st.error(str(e))

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
                f"📄 {source['source']}"
            )

            st.write(
                f"Page: {source['page']} | "
                f"Chunk: {source['chunk']} | "
                f"Distance: {source['distance']}"
            )

            st.write("---")

    else:

        st.info(
            "No sources available."
        )

    with st.expander(
        "HyDE Answer"
    ):

        st.write(
            result["hyde_answer"]
        )

    with st.expander(
        "Raw JSON"
    ):

        st.json(
            result
        )