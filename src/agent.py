from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import StreamlitChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.document import document

def get_retriever() -> EnsembleRetriever :
    vectorstore = connect_to_milvus('http://localhost:19530', 'data_test')
    milvus_retriever = vectorstore.as_retriever(search_type="similariry", search_kwargs={"k": 5})
    documents = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in vectorstore.similarity_search("", k=100)]
    bm25_retriever = BM25Retriever.from_documents(documents, search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[milvus_retriever, bm25_retriever],
        weights=[0.7, 0.3],
    )
    return ensemble_retriever

tool = create_retriever_tool(
    get_retriever(),
    name="find",
    description="Use this tool to find information in the documents. You can also use it to find information in the knowledge base.",
)

def get_llm_agent(retriever) -> AgentExecutor:
    tools = [tool]
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    system = "You are a helpful assistant. You can answer questions and provide information based on the documents you have access to. You can also use the find tool to search for information in the documents."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        verbose=True,
    )
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    
    
retriever = get_retriever()
agent_executor = get_llm_agent(retriever)