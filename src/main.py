import os
import streamlit as st  
from dotenv import load_dotenv  
from typing import Dict, List, Any, Optional
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import (
    HarmCategory, HarmBlockThreshold
)
from langchain.chains import RetrievalQA
from langchain.chains import ConversationChain
import os
from langchain.schema import HumanMessage, AIMessage

# Import từ module seed_data
from seed_data import seed_faiss, seed_faiss_live

# Thiết lập cache cho các hàm tốn tài nguyên
@st.cache_resource(ttl=3600)
def get_embeddings(api_key):
    """Cache embeddings để tránh tạo lại mỗi khi refresh"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

@st.cache_resource(ttl=3600)
def load_faiss_index(collection_name, _embeddings):
    """Cache FAISS index để tăng tốc độ"""
    index_path = f"faiss_indexes/{collection_name}"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Không tìm thấy chỉ mục FAISS tại: {index_path}")
    
    return FAISS.load_local(
        folder_path=index_path,
        embeddings=_embeddings,
        allow_dangerous_deserialization=True,
        index_name=collection_name
    )

# Cấu hình trang
def setup_page():
    """Cấu hình các thông số cho trang Streamlit"""
    st.set_page_config(
        page_title="Hành Chính Công AI",  
        page_icon="🏛️",  
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Tùy chỉnh CSS để cải thiện giao diện
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .element-container img {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        padding: 0.7rem;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
    }
    .chat-message.bot {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
    }
    .source-doc {
        background-color: #F7F7F7;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 1rem;
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Kiểm tra và thiết lập API key
def check_api_key() -> bool:
    """Kiểm tra xem đã có API key hợp lệ chưa"""
    api_key = st.session_state.get("gemini_api_key", os.getenv("GOOGLE_API_KEY", ""))
    if not api_key:
        st.sidebar.error("⚠️ Vui lòng nhập Gemini API Key để sử dụng ứng dụng")
        return False
    
    # Lưu API key vào session state nếu chưa có
    if "gemini_api_key" not in st.session_state:
        st.session_state["gemini_api_key"] = api_key
    
    # Kiểm tra API key có hợp lệ không
    try:
        # Thử tạo embeddings để kiểm tra API key
        _ = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        return True
    except Exception as e:
        st.sidebar.error(f"❌ API Key không hợp lệ: {str(e)}")
        return False

# Thiết lập sidebar
def setup_sidebar() -> str:
    """Thiết lập sidebar với các tùy chọn và trả về tên collection để truy vấn"""
    with st.sidebar:
        st.title("🔧 Tùy chọn")
        
        # Tabs cho sidebar
        data_tab, settings_tab, about_tab = st.tabs(["📚 Dữ liệu", "⚙️ Cài đặt", "ℹ️ Thông tin"])
        
        with data_tab:
            st.markdown("<div class='sidebar-title'>Cấu hình dữ liệu</div>", unsafe_allow_html=True)
            collection_to_query = st.text_input(
                "Tên dữ liệu để truy vấn:", 
                value=st.session_state.get("collection_name", "tthc_index"),
                help="Nhập tên của chỉ mục FAISS bạn muốn truy vấn",
                key="collection_input"
            )
            
            # Lưu collection name vào session state
            if collection_to_query != st.session_state.get("collection_name", ""):
                st.session_state["collection_name"] = collection_to_query
            
            st.divider()
            
            # Phần tải dữ liệu từ file JSON
            st.markdown("<div class='sidebar-title'>Tải dữ liệu</div>", unsafe_allow_html=True)
            st.info("Sử dụng dữ liệu TTHC.json từ Google Drive hoặc máy tính")
            
            # Xử lý file từ Google Drive hoặc upload
            col1, col2 = st.columns(2)
            with col1:
                source_option = st.radio(
                    "Nguồn dữ liệu:",
                    ["Google Drive", "Upload File"],
                    index=0
                )
            
            index_name = st.text_input(
                "Tên chỉ mục FAISS:",
                value=st.session_state.get("index_name", "tthc_index"),
                help="Nhập tên cho chỉ mục FAISS"
            )
            
            # Lưu index name vào session state
            if index_name != st.session_state.get("index_name", ""):
                st.session_state["index_name"] = index_name
            
            if source_option == "Google Drive":
                drive_path = st.text_input(
                    "Đường dẫn đến file TTHC.json:",
                    value=st.session_state.get("drive_path", "/content/drive/MyDrive/DS_6/Datasets/TTHC.json"),
                    help="Đường dẫn đầy đủ đến file TTHC.json"
                )
                
                # Lưu drive path vào session state
                if drive_path != st.session_state.get("drive_path", ""):
                    st.session_state["drive_path"] = drive_path
                
                # Nút tải dữ liệu từ Google Drive
                if st.button("📥 Tải từ Google Drive"):
                    if not index_name:
                        st.error("Vui lòng nhập tên chỉ mục!")
                    else:
                        with st.spinner("Đang tải dữ liệu từ Google Drive..."):
                            try:
                                directory, filename = os.path.split(drive_path)
                                seed_faiss(
                                    index_name, 
                                    filename, 
                                    directory, 
                                    google_api_key=api_key
                                )
                                st.success(f"✅ Đã tải dữ liệu thành công vào chỉ mục '{index_name}'!")
                                
                                # Cập nhật collection name nếu người dùng muốn sử dụng chỉ mục vừa tạo
                                st.session_state["collection_name"] = index_name
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
            else:
                # Upload file
                uploaded_file = st.file_uploader("Upload file TTHC.json", type=["json"])
                if uploaded_file is not None:
                    if st.button("📥 Xử lý file đã upload"):
                        if not index_name:
                            st.error("Vui lòng nhập tên chỉ mục!")
                        else:
                            with st.spinner("Đang xử lý file đã upload..."):
                                try:
                                    # Lưu file tạm thời
                                    temp_dir = "temp_uploads"
                                    os.makedirs(temp_dir, exist_ok=True)
                                    file_path = os.path.join(temp_dir, "temp_tthc.json")
                                    
                                    with open(file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                    
                                    # Xử lý file đã upload
                                    seed_faiss(
                                        index_name,
                                        "temp_tthc.json",
                                        temp_dir,
                                        google_api_key=api_key
                                    )
                                    st.success(f"✅ Đã tải dữ liệu thành công vào chỉ mục '{index_name}'!")
                                    
                                    # Cập nhật collection name
                                    st.session_state["collection_name"] = index_name
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"❌ Lỗi khi xử lý file: {str(e)}")
        
        with settings_tab:
            st.markdown("<div class='sidebar-title'>API và Mô hình</div>", unsafe_allow_html=True)
            api_key = st.text_input(
                "Gemini API Key", 
                type="password", 
                value=st.session_state.get("gemini_api_key", os.getenv("GOOGLE_API_KEY", "")),
                help="Nhập Google API Key để sử dụng Gemini"
            )
            
            # Lưu API key vào session state
            if api_key != st.session_state.get("gemini_api_key", ""):
                st.session_state["gemini_api_key"] = api_key
            
            st.divider()
            
            st.markdown("<div class='sidebar-title'>Thông số RAG</div>", unsafe_allow_html=True)
            # Số lượng tài liệu trả về từ retriever
            k_documents = st.slider(
                "Số lượng tài liệu kết quả:", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.get("k_documents", 5),
                help="Số lượng tài liệu được truy xuất để tạo câu trả lời"
            )
            
            # Lưu k_documents vào session state
            if k_documents != st.session_state.get("k_documents", 5):
                st.session_state["k_documents"] = k_documents
            
            # Nhiệt độ (temperature) của mô hình
            temperature = st.slider(
                "Độ sáng tạo (Temperature):", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.get("temperature", 0.7),
                step=0.1,
                help="Giá trị thấp hơn cho câu trả lời nhất quán, giá trị cao hơn cho câu trả lời sáng tạo"
            )
            
            # Lưu temperature vào session state
            if temperature != st.session_state.get("temperature", 0.7):
                st.session_state["temperature"] = temperature
            
            # Thêm tùy chọn cho mô hình ngôn ngữ
            model_options = ["gemini-pro", "gemini-1.5-pro"]
            selected_model = st.selectbox(
                "Mô hình ngôn ngữ:",
                options=model_options,
                index=model_options.index(st.session_state.get("selected_model", "gemini-pro")),
                help="Chọn mô hình ngôn ngữ để sử dụng"
            )
            
            # Lưu lựa chọn mô hình vào session state
            if selected_model != st.session_state.get("selected_model", "gemini-pro"):
                st.session_state["selected_model"] = selected_model
            
            # Nút xóa lịch sử chat
            st.divider()
            if st.button("🗑️ Xóa lịch sử chat"):
                st.session_state.messages = [
                    {"role": "assistant", "content": "Xin chào! Tôi là trợ lý AI hành chính công. Bạn cần giúp đỡ gì về thủ tục hành chính?"}
                ]
                if "langchain_messages" in st.session_state:
                    st.session_state.langchain_messages = []
                st.success("Đã xóa lịch sử chat!")
        
        with about_tab:
            st.markdown("<div class='sidebar-title'>Giới thiệu</div>", unsafe_allow_html=True)
            st.markdown("""
            **Trợ lý Hành Chính Công AI** là một chatbot thông minh sử dụng công nghệ RAG (Retrieval-Augmented Generation) để cung cấp thông tin chính xác về thủ tục hành chính tại Việt Nam.
            
            **Công nghệ sử dụng:**
            - 🧠 Google Gemini Pro
            - 🔍 FAISS Vector Database
            - 🔄 LangChain Framework
            - 🌐 Streamlit UI
            
            **Tính năng chính:**
            - Tìm kiếm thông tin về thủ tục hành chính
            - Giải đáp các thắc mắc về quy trình, hồ sơ, lệ phí
            - Trả lời dựa trên dữ liệu chính thống
            - Trích dẫn nguồn thông tin rõ ràng
            
            **Liên hệ**: support@hanhchinhcong.ai
            """)
            
            # Thêm phần thống kê
            if "query_count" not in st.session_state:
                st.session_state.query_count = 0
                
            st.divider()
            st.markdown("<div class='sidebar-title'>Thống kê</div>", unsafe_allow_html=True)
            st.metric("Số câu hỏi đã xử lý", st.session_state.query_count)
            
            # Hiển thị thời gian phiên làm việc
            if "session_start" not in st.session_state:
                st.session_state.session_start = time.time()
            
            session_duration = int(time.time() - st.session_state.session_start)
            minutes, seconds = divmod(session_duration, 60)
            hours, minutes = divmod(minutes, 60)
            
            if hours > 0:
                session_time = f"{hours} giờ {minutes} phút"
            else:
                session_time = f"{minutes} phút {seconds} giây"
                
            st.metric("Thời gian phiên làm việc", session_time)
    
    return collection_to_query

# Khởi tạo retriever cho RAG
def get_gemini_retriever(collection_name: str) -> Any:
    """Khởi tạo retriever sử dụng Gemini và FAISS"""
    try:
        # Tạo embeddings
        api_key = st.session_state.get("gemini_api_key", os.getenv("GOOGLE_API_KEY"))
        embeddings = get_embeddings(api_key)
        
        # Tải vectorstore
        vectorstore = FAISS.load_local(
            folder_path=f"/content/faiss_indexes/{collection_name}",
            embeddings=embeddings,
            index_name=collection_name, 
            allow_dangerous_deserialization=True  # Cho phép load pickle
        )
        
        # Tạo và trả về retriever
        k_documents = st.session_state.get("k_documents", 5)
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_documents}
        )
    except Exception as e:
        raise ValueError(f"Lỗi khi tạo retriever: {str(e)}")

# Khởi tạo agent RAG sử dụng Gemini
# Khởi tạo agent RAG sử dụng Gemini
def get_gemini_agent(retriever): # <--- Bỏ tham số msgs ở đây
    """Tạo agent RAG sử dụng Gemini và retriever đã cho"""
    try:
        # Lấy thông số cấu hình từ session state
        api_key = st.session_state.get("gemini_api_key", os.getenv("GOOGLE_API_KEY"))
        temperature = st.session_state.get("temperature", 0.7)
        model_name = st.session_state.get("selected_model", "gemini-1.5-flash")

        # Khởi tạo mô hình Gemini
        llm = GoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )

        # Bỏ khởi tạo ConversationBufferMemory ở đây

        # Định nghĩa template cho prompt hệ thống
        # Vẫn giữ input variables chat_history, context, query vì ta sẽ truyền chúng thủ công
        system_template = """
        Bạn là trợ lý AI hành chính công của Việt Nam, chuyên giải đáp thắc mắc về thủ tục hành chính,
        giấy tờ, và các quy trình liên quan đến dịch vụ công. Hãy sử dụng thông tin được cung cấp để trả lời câu hỏi.
        Lịch sử cuộc hội thoại:
        {chat_history}
        Hãy trả lời theo các nguyên tắc sau:
        1. Trả lời đúng về nội dung thủ tục hành chính dựa trên thông tin được cung cấp
        2. Giải thích các bước thực hiện một cách rõ ràng và dễ hiểu
        3. Liệt kê đầy đủ thành phần hồ sơ cần thiết nếu được hỏi
        4. Cung cấp thông tin về thời gian giải quyết và lệ phí nếu có
        5. Nêu rõ cơ quan thực hiện thủ tục
        6. Nếu thông tin được cung cấp không đủ, hãy gợi ý người dùng cung cấp thêm thông tin
        7. Nếu không biết câu trả lời, hãy thừa nhận điều đó và đề xuất người dùng liên hệ trực tiếp với cơ quan hành chính có thẩm quyền
        8. Trả lời ngắn gọn và súc tích, sử dụng định dạng markdown để dễ đọc
        9. Thông tin liệt kê nên sử dụng dạng gạch đầu dòng cho dễ đọc

        Thông tin liên quan từ cơ sở dữ liệu thủ tục hành chính:
        {context}

        Câu hỏi của người dùng: {question} # Biến này khớp với key "query" ta sẽ truyền vào

        Trả lời (sử dụng định dạng markdown cho dễ đọc):
        """

        # Tạo prompt template
        prompt = PromptTemplate(
            template=system_template,
            # Các biến này phải có vì chúng được dùng trong template
            input_variables=["chat_history","context", "question"]
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )


        # Trả về chain, không cần trả về memory nữa
        return qa_chain
    except Exception as e:
        raise ValueError(f"Lỗi khi khởi tạo mô hình: {str(e)}")

# Hiển thị liên kết nhanh
def display_quick_links():
    """Hiển thị các liên kết nhanh đến các thủ tục phổ biến"""
    st.subheader("💡 Thủ tục phổ biến")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Đăng ký kết hôn"):
            st.session_state.pending_question = "Thủ tục đăng ký kết hôn cần những giấy tờ gì?"
            st.rerun()
    
    with col2:
        if st.button("Cấp căn cước công dân"):
            st.session_state.pending_question = "Thủ tục cấp căn cước công dân gồm những bước nào?"
            st.rerun()
    
    with col3:
        if st.button("Đăng ký kinh doanh"):
            st.session_state.pending_question = "Thủ tục đăng ký kinh doanh hộ cá thể cần những giấy tờ gì?"
            st.rerun()
# Thiết lập giao diện chat
def setup_chat_interface():
    """Thiết lập giao diện chat với lịch sử tin nhắn"""
    # Container chính
    main_container = st.container()
    with main_container:
        # Header
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image("https://raw.githubusercontent.com/vanhung4499/temp/main/icons/government.png", width=80)
        with col2:
            st.title("🏛️ Trợ lý Hành chính công AI")
            st.caption("Trợ lý AI được trang bị công nghệ RAG (Retrieval-Augmented Generation) và Google Gemini")
        
        st.divider()
        
        # Hiển thị các liên kết nhanh
        display_quick_links()
    
    # Khởi tạo lịch sử tin nhắn nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi là trợ lý AI hành chính công. Tôi có thể giúp bạn tìm hiểu về các thủ tục hành chính, hồ sơ cần thiết, quy trình thực hiện, thời gian và lệ phí. Bạn cần hỗ trợ vấn đề gì?"}
        ]
    
    # THÊM VÀO ĐÂY: Xử lý pending_question nếu có
    if "pending_question" in st.session_state and st.session_state.pending_question:
        # Thêm câu hỏi vào lịch sử tin nhắn
        st.session_state.messages.append({"role": "human", "content": st.session_state.pending_question})
        
        # Xóa pending_question sau khi đã xử lý để tránh vòng lặp
        question = st.session_state.pending_question
        st.session_state.pending_question = ""
        
        # QUAN TRỌNG: Đặt một cờ để xử lý câu trả lời ở phần khác của code
        # Bạn cần xử lý câu trả lời ở main() hoặc nơi khác sau khi setup_chat_interface() được gọi
        st.session_state.needs_answer = True
        st.session_state.current_question = question
    
    # Hiển thị tin nhắn hiện có với CSS được cải thiện
    messages_container = st.container(height=500)
    with messages_container:
        for msg in st.session_state.messages:
            if msg["role"] == "human":
                with st.chat_message("user"):
                    st.markdown(f"<div class='chat-message user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"<div class='chat-message bot'>{msg['content']}</div>", unsafe_allow_html=True)
    
    # Trả về đối tượng lịch sử tin nhắn cho LangChain
    return StreamlitChatMessageHistory(key="langchain_messages")

# Xử lý đầu vào từ người dùng
# Xử lý đầu vào từ người dùng
# Xử lý đầu vào từ người dùng
def handle_user_input(msgs, agent_executor):
    """Xử lý đầu vào từ người dùng và tạo phản hồi từ trợ lý"""
    # Kiểm tra câu hỏi từ nút liên kết nhanh
    prompt = None
    if "pending_question" in st.session_state and st.session_state.pending_question:
        prompt = st.session_state.pending_question
        # Xóa câu hỏi đang chờ để tránh lặp lại
        del st.session_state.pending_question
    else:
        # Trường nhập liệu cho người dùng
        prompt = st.chat_input("Nhập câu hỏi của bạn về thủ tục hành chính...", key="user_input")

    if prompt:
        # Tăng bộ đếm câu hỏi
        if "query_count" in st.session_state:
            st.session_state.query_count += 1

        # Thêm tin nhắn của người dùng vào lịch sử Streamlit (để hiển thị và lưu state)
        st.session_state.messages.append({"role": "human", "content": prompt})

        # Quan trọng: Thêm tin nhắn người dùng vào history object để có thể format nó sau này
        msgs.add_user_message(prompt)


        # Hiển thị tin nhắn của người dùng
        with st.chat_message("human"):
            st.markdown(f"<div class='chat-message user'>{prompt}</div>", unsafe_allow_html=True)

        # Hiển thị tin nhắn của trợ lý
        with st.chat_message("assistant"):
            response_container = st.container()

            with response_container:
                st_callback = StreamlitCallbackHandler(st.container())

                try:
                    with st.spinner("Đang tìm kiếm thông tin..."):
                        # --- BẮT ĐẦU LOGIC FORMAT LỊCH SỬ CHAT THỦ CÔNG ---
                        chat_history_str = ""
                        for msg in msgs.messages[:-1]:
                            if getattr(msg, "type", None) == "human":
                                chat_history_str += f"Human: {msg.content}\n"
                            elif getattr(msg, "type", None) == "ai":
                                chat_history_str += f"AI: {msg.content}\n"
                        # --- KẾT THÚC LOGIC FORMAT LỊCH SỬ CHAT THỦ CÔNG ---

                        # --- DEBUG PRINTS ---
                        print("\n--- DEBUG INFO ---")
                        print(f"Prompt received: {prompt}")
                        print(f"Length of msgs.messages: {len(msgs.messages)}")
                        # Print history messages (optional, can be long)
                        # print("History messages before formatting:")
                        # for msg in msgs.messages[:-1]:
                        #     print(f"  - {msg.role}: {msg.content[:50]}...") # Print truncated
                        print(f"Formatted chat_history_str (first 200 chars): {chat_history_str[:200]}...")
                        print(f"Input dictionary being sent to invoke: {{'query': '{prompt}', 'chat_history': '...'}}") # Avoid printing full history string again
                        print("--- END DEBUG INFO ---\n")
                        # --- END DEBUG PRINTS ---


                        # Gọi agent để xử lý câu hỏi
                        # TRUYỀN RÕ RÀNG CẢ "query" và "chat_history" vào input dictionary
                        response = agent_executor.invoke(
                            {
                                "question": prompt,  # Đảm bảo key đúng với prompt.input_variables
                                "chat_history": chat_history_str
                            },
                            {"callbacks": [st_callback]}
                        )


                    # Lấy câu trả lời từ response
                    output = response.get("answer") or response.get("result") or "Không tìm thấy câu trả lời."



                    # Thêm thông tin về tài liệu nguồn nếu có
                    if "source_documents" in response and response["source_documents"]:
                        sources = []
                        for i, doc in enumerate(response["source_documents"]):
                            if i >= 3:
                                break
                            metadata = doc.metadata
                            source_info = f"**Nguồn {i+1}:** "
                            if "ten_tthc" in metadata:
                                source_info += f"{metadata['ten_tthc']}"
                            if "co_quan_thuc_hien" in metadata:
                                source_info += f" - {metadata['co_quan_thuc_hien']}"
                            sources.append(source_info)
                        if sources:
                            output += "\n\n---\n\n**Tài liệu tham khảo:**\n" + "\n".join(sources)


                    # Hiển thị output với styling
                    st.markdown(f"<div class='chat-message bot'>{output}</div>", unsafe_allow_html=True)

                    # Quan trọng: Thêm câu trả lời của AI vào history object để lượt sau có lịch sử
                    msgs.add_ai_message(output)

                except Exception as e:
                    error_msg = f"❌ Đã xảy ra lỗi: {str(e)}"
                    st.error(error_msg)
                    # Thêm lỗi vào history object (tùy chọn)
                    # msgs.add_ai_message(error_msg)
# Hàm chính
def main():
    """Hàm chính điều khiển toàn bộ ứng dụng"""
    # Thiết lập trang
    setup_page()

    # Khởi tạo môi trường
    load_dotenv()

    # Thiết lập sidebar và lấy tên collection
    collection_name = setup_sidebar()

    # Thiết lập giao diện chat và lấy đối tượng lịch sử tin nhắn Streamlit
    msgs = setup_chat_interface() # Vẫn cần msgs để lưu lịch sử Streamlit

    # Kiểm tra API key
    if check_api_key():
        try:
            # Khởi tạo retriever
            retriever = get_gemini_retriever(collection_name)

            # Khởi tạo agent, CHỈ truyền retriever (msgs sẽ được dùng trong handle_user_input)
            agent_executor = get_gemini_agent(retriever) # <--- SỬA LẠI DÒNG NÀY

            # Xử lý đầu vào từ người dùng
            handle_user_input(msgs, agent_executor) # Vẫn truyền msgs vào đây

        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo hệ thống: {str(e)}")
    else:
        st.info("🔑 Vui lòng nhập Google API Key trong phần Cài đặt để bắt đầu.")

if __name__ == "__main__":
    main()