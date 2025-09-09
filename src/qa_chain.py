from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def build_qa_chain(vectorstore, model="llama-3.3-70b-versatile", api_key=None):
    # Use retriever to get chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Set up Groq language model
    llm = ChatGroq(groq_api_key=api_key, model_name=model)

    # Prompt structure for LLM
    prompt = PromptTemplate.from_template("""
        Use the following context to answer the question.
        {context}

        Question: {question}
    """)

    # Helper to format chunks as a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain assembly
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain
