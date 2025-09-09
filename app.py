import streamlit as st
import os
from src.extractor import extract_text_from_pdf
from src.embedder import create_vectorstore
from src.qa_chain import build_qa_chain

st.set_page_config(page_title="Ask Your PDF", layout="wide")
st.title("📖 Ask Questions From Your PDF Book")

api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
query = st.text_input("Ask a question based on the uploaded book:")

if uploaded_pdf and query:
    with st.spinner("Reading and embedding your book..."):
        raw_text = extract_text_from_pdf(uploaded_pdf)
        print("📄 Raw text extracted from PDF:")
        print(raw_text[:1000])

        vectorstore = create_vectorstore(raw_text)
        qa_chain = build_qa_chain(vectorstore, api_key=api_key)
        response = qa_chain.invoke(query)
        docs = vectorstore.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(query)

    response_text = response.content if hasattr(response, "content") else str(response)
    valid_docs = [doc for doc in docs if doc.page_content.strip()]

    if not valid_docs or not response_text.strip() or "i'm not sure" in response_text.lower():
        st.warning("🚫 Sorry, I couldn't find an answer in your uploaded document. Please ask something based on its content.")
    else:
        st.success("✅ Your book has been indexed.")
        st.markdown("### 📌 Answer:")
        st.write(response_text)

        with st.expander("📄 Show context chunks used in answer"):
            for i, doc in enumerate(valid_docs, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content[:500])

    # # 💬 Context chunks preview (optional)
    # with st.expander("📄 Show context chunks used in answer"):
    #     for i, doc in enumerate(docs, start=1):
    #         chunk = getattr(doc, "page_content", "")
    #         st.markdown(f"**Chunk {i}:**")
    #         st.write(chunk[:500])
