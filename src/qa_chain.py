from langchain_community.llms import Groq
from langchain.chains import RetrievalQA

def build_qa_chain(vectorstore, model="llama3-8b-8192", api_key=None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = Groq(model=model, api_key=api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
