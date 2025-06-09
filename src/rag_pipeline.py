# src/rag_pipeline.py
from langchain.chains import RetrievalQA
from .retriever import get_retriever
from .llm import get_llm

def create_rag_chain(vectorstore, model_name="gemini-1.5-flash", temperature=0.1, k=1):
    llm = get_llm(model_name, temperature)
    retriever = get_retriever(vectorstore, k)
    chain = RetrievalQA.from_chain_type(
<<<<<<< HEAD
        llm=llm,             
=======
        llm=llm,              # giờ đây là một BaseChatModel
>>>>>>> e10216675b09a9e6e5fe87885184f76e4f60b7a1
        chain_type="stuff",
        retriever=retriever
    )
    return chain
