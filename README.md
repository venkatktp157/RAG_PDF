# 📖 PDF-Based RAG Q&A App with Streamlit

This Streamlit app allows users to upload a PDF book and ask questions using Retrieval-Augmented Generation (RAG). Answers are generated solely from the uploaded content using embeddings and a language model.

---

## 🔧 Features

- Extract text from PDF documents
- Generate embeddings using SentenceTransformers
- Retrieve relevant passages using FAISS
- Answer queries using LangChain + OpenAI (or Groq)
- Fully modular code structure
- API key management via Streamlit Secrets

---

## 🗂️ Folder Structure

```plaintext
pdf_rag_app/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── assets/
├── src/
│   ├── extractor.py
│   ├── embedder.py
│   ├── qa_chain.py
│   └── utils.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
