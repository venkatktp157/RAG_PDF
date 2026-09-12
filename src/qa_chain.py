from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def build_qa_chain(vectorstore, model="openai/gpt-oss-120b", api_key=None):
    # Use retriever to get chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Set up Groq language model
    llm = ChatGroq(groq_api_key=api_key, model_name=model)

    # Prompt structure for LLM
    prompt = PromptTemplate.from_template("""
        You are a helpful assistant. Use ONLY the following context to answer the question.
        If the context does not contain the answer, say "I couldn't find relevant information in the document."
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
