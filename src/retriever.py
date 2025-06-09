from langchain.vectorstores import FAISS

def get_retriever(vectorstore, k: int = 5):
    """
    Trả về một Retriever để RAG chain dùng.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})
