from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

def build_qa_chain(vectorstore, model="llama3-8b-8192", api_key=None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(groq_api_key=api_key, model_name=model)

    prompt = PromptTemplate.from_template(
        """
        Use the following context to answer the question:
        {context}
        Question: {question}
        """
    )

    # Custom document stuffing chain
    document_chain = StuffDocumentsChain(llm=llm, prompt=prompt)

    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=document_chain,
        return_source_documents=True
    )

    return qa_chain

