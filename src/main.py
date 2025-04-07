def init_app():
    load_to_env()
    setup_page()
    setup_sidebar()

def setup_chat_interface():
    st.title("Chat Interface")
    st.caption("Ask me anything!")
    #khoi tao bo nho chat
    msgs = StreamlitChatMessageHistory( key="chat_history")
    # khoi tao tin nhan chao mung neu la doan chat moi
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
        msgs.add_message("Hello! How can I help you today?")
    # hien thi lich su chat
    for msg in st.session_state.chat_history:
        role = "assistant" if msg["role"] == "assistant" else "user":
        st.chat_message(role).write(msg["content"])  
    return msgs
    
def handle_user_input(agent_executor, msgs):
    # xu ly tin nhan cua nguoi dung
    if prompt := st.chat_input("Ask me anything!"):
        # hien thi tin nhan nguoi dung
        st.chat_message("user").write(prompt)
        # luu tin nhan nguoi dung vao lich su
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        msgs.add_user_message(prompt)
        
        # xu ly va hien thi cau tra loi
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), key="assistant")
            # goi AI 
            response = agent_executor.invoke(
                {"input": prompt,
                 "chat_history": st.session_state.chat_history},
                callbacks=[st_callback],
            )
            # hien thi cau tra loi
            output = response["output"]
            st.session_state.chat_history.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)    
def setup_page():
    st.set_page_config(page_title="ChatGPT", page_icon=":robot:", layout="wide")
    st.title("ChatGPT with Langchain")
    st.subheader("Chatbot using Langchain and OpenAI API")
    
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
        
def main():
    init_app()
    msgs = setup_chat_interface()
    
    #khoi tao AI
    retriver = get_retriever()
    agent_executor = get_llm_agent(retriver)
    
    #xu ly chat
    handle_user_input(agent_executor, msgs)
    
if __name__ == "__main__":
    main()