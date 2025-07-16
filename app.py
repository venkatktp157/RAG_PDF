import streamlit as st
import os
from src.extractor import extract_text_from_pdf
from src.embedder import create_vectorstore
from src.qa_chain import build_qa_chain

st.set_page_config(page_title="Ask Your PDF", layout="wide")
st.title("📖 Ask Questions From Your PDF Book")

# Load Groq API key
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

# Question input
query = st.text_input("Ask a question based on the uploaded book:")

# If PDF is uploaded and a query is asked
if uploaded_pdf and query:
    with st.spinner("Reading and embedding your book..."):
        raw_text = extract_text_from_pdf(uploaded_pdf)
        vectorstore = create_vectorstore(raw_text)
        qa_chain = build_qa_chain(vectorstore, api_key=api_key)
        response = qa_chain.invoke(query)

    st.success("✅ Your book has been indexed.")
    st.markdown("### 📌 Answer:")
    st.write(response["result"])

if "source_documents" in response:
    st.markdown("### 🧾 Context Used:")
    for i, doc in enumerate(response["source_documents"], 1):
        st.markdown(f"**Chunk {i}:**")
        st.write(doc.page_content[:500])  # Show first 500 characters of each chunk
