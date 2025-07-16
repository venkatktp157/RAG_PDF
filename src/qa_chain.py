from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_qa_chain(vectorstore, model="llama3-8b-8192", api_key=None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(groq_api_key=api_key, model_name=model)

    prompt = PromptTemplate.from_template("""
        Use the following context to answer the question:
        {context}

        Question: {question}
    """)

    qa_chain = RetrievalQA(
        retriever=retriever,
        llm=llm,
        prompt=prompt,
        return_source_documents=True,
        output_key="result"  # This WILL be honored
    )

    return qa_chain
