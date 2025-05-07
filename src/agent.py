# Import các thư viện cần thiết
from langchain.tools.retriever import create_retriever_tool  
from langchain_openai import ChatOpenAI 
from langchain.agents import AgentExecutor, create_openai_functions_agent 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from seed_data import seed_milvus, connect_to_milvus  
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  
from langchain.retrievers import EnsembleRetriever  
from langchain_community.retrievers import BM25Retriever 
from langchain_core.documents import Document  
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("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")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def get_retriever(collection_name: str = "C:\Users\ntlinh\OneDrive - Hanoi University of Science and Technology\Documents\20242\ChatBot_AI\dataset\TTHC.csv") -> EnsembleRetriever:
    try:
        vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
        milvus_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
        ]
        
        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")
            
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)

tool = create_retriever_tool(
    get_retriever(),
    "find",
    "Search for information of Stack AI."
)

def get_llm_and_agent(_retriever, model_choice="gpt4") -> AgentExecutor:
    llm = ChatOpenAI(
        temperature=0,
        streaming=True,
        model='gpt-4',
        api_key=OPENAI_API_KEY
    )
    
    tools = [tool]
    
    system = """You are an expert at AI. Your name is ChatchatAI."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)