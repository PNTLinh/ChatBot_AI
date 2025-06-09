# src/rag_pipeline.py
from langchain.chains import RetrievalQA
from .retriever import get_retriever
from .llm import get_llm

def create_rag_chain(vectorstore, model_name="gemini-1.5-flash", temperature=0.1, k=1):
    llm = get_llm(model_name, temperature)
    retriever = get_retriever(vectorstore, k)
    chain = RetrievalQA.from_chain_type(
        llm=llm,             
        chain_type="stuff",
        retriever=retriever
    )
    return chain
