import streamlit as st

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚"
)

st.title("📚 Research Paper Assistant")

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX or TXT",
    type=["pdf", "docx", "txt"]
)

question = st.text_input(
    "Ask a question"
)

if st.button("Ask"):
    st.write("Backend coming next...")