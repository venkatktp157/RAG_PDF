from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def create_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"🧩 Number of chunks generated: {len(chunks)}")
    if chunks:
        print(f"🔍 First chunk preview:\n{chunks[0][:300]}")

    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = embedder.embed_documents(chunks)
    print(f"📊 Number of embeddings generated: {len(embeddings)}")

    if not embeddings:
        raise ValueError("❌ No embeddings were generated. Check input format or model compatibility.")

    return FAISS.from_texts(chunks, embedding=embedder)
