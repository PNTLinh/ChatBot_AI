import streamlit as st  
from dotenv import load_dotenv  
from seed_data import seed_milvus, seed_milvus_live
from agent import get_retriever as get_openai_retriever, get_llm_and_agent as get_openai_agent
from local_ollama import get_retriever as get_ollama_retriever, get_llm_and_agent as get_ollama_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

def setup_page():
    st.set_page_config(
        page_title="AI Assistant",  
        page_icon="💬",  
        layout="wide"  
    )

def init_app():
    load_dotenv() 
    setup_page()  
    setup_sidebar()
    
def setup_sidebar():
    with st.sidebar:
        st.title("Options")
        st.subheader("Settings")
        data_src = st.radio("Select data source", ("Local file", "Web URL"))
        if data_src == "Local file":
            filename  = st.text_input("Enter file name")
            directory = st.text_input("Enter directory")
            if st.button("Load file"):
                with st.spinner("Loading file..."):
                    seed_milvus('https://localhost:19530', 'data_test',filename, directory)
                st.success("File loaded successfully")
        else:
            url = st.text_input("Enter URL")
            if st.button("Crawl URL"):
                with st.spinner("Crawling URL..."):
                    seed_milvus('https://localhost:19530', 'data_test',url)
                st.success("URL crawled successfully")

def handle_local_file(use_ollama_embeddings: bool):
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    filename = st.text_input("Tên file JSON:", "stack.json")
    directory = st.text_input("Thư mục chứa file:", "data")
    
    if st.button("Tải dữ liệu từ file"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang tải dữ liệu..."):
            try:
                seed_milvus(
                    'http://localhost:19530', 
                    collection_name, 
                    filename, 
                    directory, 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"Đã tải dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {str(e)}")

def handle_url_input(use_ollama_embeddings: bool):
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test_live",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    url = st.text_input("Nhập URL:", "https://www.stack-ai.com/docs")
    
    if st.button("Crawl dữ liệu"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang crawl dữ liệu..."):
            try:
                seed_milvus_live(
                    url, 
                    'http://localhost:19530', 
                    collection_name, 
                    'stack-ai', 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi crawl dữ liệu: {str(e)}")

def setup_chat_interface(model_choice):
    st.title("💬 AI Assistant")
    st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và OpenAI GPT-4")
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

def handle_user_input(msgs, agent_executor):
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về Stack AI!"):
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]}
            )
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

def main():
    init_app()
    model_choice, collection_to_query = setup_sidebar()
    msgs = setup_chat_interface(model_choice)
    retriever = get_openai_retriever(collection_to_query)
    agent_executor = get_openai_agent(retriever, "gpt4")
    handle_user_input(msgs, agent_executor)

if __name__ == "__main__":
    main() 