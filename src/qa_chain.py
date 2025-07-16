from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

def build_qa_chain(vectorstore, model="llama3-8b-8192", api_key=None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(groq_api_key=api_key, model_name=model)

    prompt = PromptTemplate.from_template("""
        Use the following context to answer the question:
        {context}

        Question: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain
