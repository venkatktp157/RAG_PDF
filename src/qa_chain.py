from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

def build_qa_chain(vectorstore, model="llama3-70b-8192", api_key=None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(groq_api_key=api_key, model_name=model)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain
