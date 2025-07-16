import streamlit as st
import os
from src.extractor import extract_text_from_pdf
from src.embedder import create_vectorstore
from src.qa_chain import build_qa_chain

# 🌐 Load Groq API key
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Ask Your PDF", layout="wide")
st.title("📖 Ask Questions From Your PDF Book")

uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
query = st.text_input("Ask a question from your book:")

if uploaded_pdf:
    with st.spinner("Reading and embedding your book..."):
        raw_text = extract_text_from_pdf(uploaded_pdf)
        vectorstore = create_vectorstore(raw_text)
        qa_chain = build_qa_chain(vectorstore, api_key=api_key)

    st.success("Your book has been indexed!")

    if query:
        response = qa_chain.run(query)
        st.markdown("### 📌 Answer:")
        st.write(response)

