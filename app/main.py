import streamlit as st
from components import render_header, render_answer
<<<<<<< HEAD
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
import re
import time

# Cấu hình trang
st.set_page_config(
    page_title="TTHC RAG System - Gemini Flash",
    layout="wide",
)

render_header()

# Sidebar config
with st.sidebar:
    st.header("🤖 Gemini 1.5 Flash")
    gemini_key = st.text_input(
        "Google API Key:", 
        type="password",
        help="Lấy API key tại: https://aistudio.google.com/app/apikey"
    )
    
    # Advanced settings
    with st.expander("⚙️ Cài đặt nâng cao"):
        temperature = st.slider("Temperature (Độ sáng tạo)", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("Max tokens", 100, 2000, 800, 100)
        top_k = st.slider("Số tài liệu tham khảo", 2, 8, 4, 1)

# 1) Form nhập câu hỏi
with st.form("query_form"):
    query = st.text_input("Nhập câu hỏi về thủ tục hành chính")
    col1, col2 = st.columns(2)
    with col1:
        submitted = st.form_submit_button("Tìm kiếm", use_container_width=True)
    with col2:
        detailed = st.form_submit_button("Trả lời chi tiết", use_container_width=True)

if not (submitted or detailed) or not query:
    st.stop()

# Kiểm tra API key
if not gemini_key:
    st.error(" Vui lòng nhập Google API Key trong sidebar")
    st.stop()

# 2) Setup Gemini
@st.cache_resource
def setup_gemini(api_key):
    """Setup Gemini model with caching"""
    genai.configure(api_key=api_key)
    
    # Cấu hình generation
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    
    # Safety settings (tùy chọn)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

# 3) Load vectorstore
@st.cache_resource(ttl=86400)
def load_vectorstore():
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
=======
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Cấu hình trang
st.set_page_config(
    page_title="TTHC RAG System",
    layout="wide",
)

# Header
render_header()

# 1) Form nhập câu hỏi
with st.form("query_form"):
    query = st.text_input("🔎 Nhập câu hỏi của bạn")
    submitted = st.form_submit_button("Tìm kiếm")

# Dừng nếu chưa submit hoặc query rỗng
if not submitted or not query:
    st.stop()

# 2) Lazy-load vectorstore từ disk
@st.cache_resource(ttl=86400)
def load_vectorstore():
    embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
>>>>>>> e10216675b09a9e6e5fe87885184f76e4f60b7a1
    return FAISS.load_local(
        "data/faiss_store",
        embed,
        allow_dangerous_deserialization=True,
    )

<<<<<<< HEAD
# 4) Optimized prompts cho Gemini Flash
def get_prompt_template(is_detailed=False):
    """Get optimized prompt template for Gemini Flash"""
    
    if is_detailed:
        return """
Bạn là chuyên gia tư vấn thủ tục hành chính Việt Nam với kinh nghiệm 10 năm.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}

HƯỚNG DẪN TRẢ LỜI CHI TIẾT:
1. **Tóm tắt thủ tục**: Mô tả ngắn gọn thủ tục cần thực hiện
2. **Điều kiện**: Liệt kê rõ ràng các điều kiện cần đáp ứng
3. **Hồ sơ cần thiết**: Danh sách đầy đủ giấy tờ, số lượng bản
4. **Quy trình thực hiện**: Các bước cụ thể theo thứ tự
5. **Thời gian và phí**: Thời gian xử lý và mức phí (nếu có)

YÊU CẦU:
- Sử dụng ngôn ngữ rõ ràng, dễ hiểu
- Cấu trúc thông tin có thứ tự logic
- Chỉ dựa vào thông tin trong tài liệu
- Nếu thiếu thông tin, nói rõ phần nào chưa có

TRẢ LỜI:
"""
    else:
        return """
Bạn là chuyên gia tư vấn thủ tục hành chính. Hãy trả lời câu hỏi một cách súc tích nhưng đầy đủ.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}

YÊU CẦU TRẢ LỜI:
- Trả lời trực tiếp, ngắn gọn
- Bao gồm thông tin chính: điều kiện, hồ sơ, quy trình
- Sử dụng bullet points khi cần thiết
- Chỉ dựa trên thông tin có trong tài liệu

TRẢ LỜI:
"""

# 5) Generate answer với Gemini Flash
@st.cache_data(ttl=1800)  # Cache 30 phút
def generate_gemini_answer(query, context, is_detailed, _api_key):
    """Generate answer using Gemini Flash with caching"""
    
    try:
        model = setup_gemini(_api_key)
        
        # Chuẩn bị prompt
        prompt_template = get_prompt_template(is_detailed)
        prompt = prompt_template.format(context=context, question=query)
        
        # Generate với retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = model.generate_content(prompt)
                end_time = time.time()
                
                # Log performance
                st.sidebar.success(f"⚡ Thời gian: {end_time - start_time:.2f}s")
                
                return response.text
                
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Thử lại lần {attempt + 2}...")
                    time.sleep(1)
                else:
                    raise e
                    
    except Exception as e:
        return f"❌ Lỗi khi tạo câu trả lời: {str(e)}\n\nVui lòng kiểm tra API key hoặc thử lại sau."

# 6) Format và improve answer
def post_process_answer(answer):
    """Post-process Gemini answer for better formatting"""
    
    # Loại bỏ markdown thừa
    answer = re.sub(r'\*\*(.*?)\*\*', r'**\1**', answer)
    
    # Cải thiện format danh sách
    answer = re.sub(r'^\s*[-•]\s*', '• ', answer, flags=re.MULTILINE)
    answer = re.sub(r'^\s*(\d+)[\.)]\s*', r'\1. ', answer, flags=re.MULTILINE)
    
    # Chuẩn hóa line breaks
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    
    # Thêm icons cho các section
    answer = re.sub(r'^(#+ .*)', r' \1', answer, flags=re.MULTILINE)
    answer = re.sub(r'\*\*(Điều kiện|Yêu cầu)([^*]*)\*\*', r' **\1\2**', answer)
    answer = re.sub(r'\*\*(Hồ sơ|Giấy tờ)([^*]*)\*\*', r' **\1\2**', answer)
    answer = re.sub(r'\*\*(Quy trình|Thủ tục)([^*]*)\*\*', r' **\1\2**', answer)
    answer = re.sub(r'\*\*(Thời gian|Phí)([^*]*)\*\*', r' **\1\2**', answer)
    
    return answer.strip()

# 7) Main execution
with st.spinner("Đang tìm kiếm thông tin..."):
    vs = load_vectorstore()
    docs = vs.similarity_search(query, k=top_k)

# 8) Prepare context cho Gemini
if docs:
    context_parts = []
    for i, doc in enumerate(docs, 1):
        # Giới hạn độ dài context để tối ưu token
        content = doc.page_content[:1000] if not detailed else doc.page_content[:1500]
        source = doc.metadata.get("source", f"Tài liệu {i}")
        context_parts.append(f"=== {source} ===\n{content}")
    
    context = "\n\n".join(context_parts)
else:
    context = "Không tìm thấy tài liệu liên quan."

# 9) Hiển thị tài liệu tham khảo
with st.expander("📚 Tài liệu tham khảo", expanded=False):
    if docs:
        for i, d in enumerate(docs, 1):
            snippet = d.page_content[:200].strip().replace("\n", " ")
            st.markdown(f"**📄 Tài liệu {i}:** {snippet}...")
            
            # Metadata
            if d.metadata:
                meta_info = []
                for key, value in d.metadata.items():
                    if key == "url" and value:
                        meta_info.append(f"[🔗 Nguồn]({value})")
                    elif key != "url":
                        meta_info.append(f"**{key}:** {value}")
                
                if meta_info:
                    st.markdown(" | ".join(meta_info))
            
            st.divider()
    else:
        st.info("Không tìm thấy tài liệu liên quan.")

# 10) Generate answer với Gemini Flash
answer_type = "chi tiết" if detailed else "nhanh"
with st.spinner(f"🤖 Gemini đang tạo câu trả lời {answer_type}..."):
    
    if docs:
        raw_answer = generate_gemini_answer(
            query, 
            context, 
            detailed, 
            gemini_key
        )
        
        # Post-process
        final_answer = post_process_answer(raw_answer)
        
        # Thêm metadata
        final_answer += f"\n\n---\n🔬 **Phân tích từ {len(docs)} tài liệu** • 🤖 **Gemini 1.5 Flash** • ⚡ **Cached 30 phút**"
        
    else:
        final_answer = """
❌ **Không tìm thấy thông tin phù hợp**

💡 **Gợi ý:**
• Thử từ khóa khác cụ thể hơn
• Kiểm tra chính tả
• Đặt câu hỏi về thủ tục cụ thể

🔍 **Ví dụ câu hỏi tốt:**
• "Thủ tục làm CMND mới"
• "Đăng ký kết hôn cần giấy tờ gì"
• "Quy trình xin giấy phép kinh doanh"
"""

# 11) Hiển thị kết quả
render_answer(final_answer)

# 12) Action buttons
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("👍 Hữu ích", use_container_width=True):
        st.success("Cảm ơn feedback!")

with col2:
    if st.button("👎 Cần cải thiện", use_container_width=True):
        st.info("Đã ghi nhận feedback!")

with col3:
    if st.button("🔄 Tạo lại", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col4:
    if st.button("📋 Chế độ khác", use_container_width=True):
        st.info("Thử nút 'Trả lời chi tiết' hoặc 'Tìm kiếm' để thay đổi độ chi tiết!")

# 13) Sidebar stats
with st.sidebar:
    st.markdown("---")
    st.markdown("### Thống kê")
    
    if docs:
        st.metric("Tài liệu tìm thấy", len(docs))
        
        # Tính độ liên quan
        query_words = set(query.lower().split())
        total_matches = 0
        for doc in docs:
            doc_words = set(doc.page_content.lower().split())
            matches = len(query_words.intersection(doc_words))
            total_matches += matches
        
        relevance = min(total_matches / len(query_words) / len(docs), 1.0) if query_words else 0
        st.metric("Độ liên quan", f"{relevance:.1%}")
        
        # Token estimation
        context_tokens = len(context.split()) * 1.3  # Rough estimate
        st.metric("Tokens ước tính", f"{int(context_tokens)}")
    
    st.markdown("### Gemini Flash")
    st.info("""
    **Ưu điểm:**
    • ⚡ Tốc độ nhanh
    • 💰 Chi phí thấp  
    • 🎯 Chất lượng cao
    • 🧠 Context dài (1M tokens)
    """)
    
    st.markdown("### 💡 Tips")
    st.success("""
    **Câu hỏi hiệu quả:**
    • Cụ thể và rõ ràng
    • Một thủ tục/vấn đề
    • Bao gồm ngữ cảnh
    
    **Ví dụ:** 
    "Thủ tục xin visa du lịch Hàn Quốc cho người Việt Nam"
    """)
=======
# 3) Tìm tài liệu tham khảo
with st.spinner("Tìm tài liệu tham khảo…"):
    vs = load_vectorstore()
    docs = vs.similarity_search(query, k=3)

# 4) Hiển thị expander tài liệu tham khảo
with st.expander("📄 Tài liệu tham khảo", expanded=False):
    for i, d in enumerate(docs, 1):
        snippet_preview = d.page_content[:200].strip().replace("\n", " ")
        st.markdown(f"**Tài liệu {i}:** {snippet_preview}…")
        if url := d.metadata.get("url"):
            st.markdown(f"[Nguồn]({url})")
        st.divider()

# 5) Lấy toàn bộ nội dung của doc thỏa mãn query làm câu trả lời
with st.spinner("Đang trích xuất câu trả lời…"):
    answer_parts = []
    q_lower = query.lower()
    for i, d in enumerate(docs, 1):
        content = d.page_content.strip()
        if q_lower in content.lower():
            # Thêm toàn bộ nội dung tài liệu vào câu trả lời
            answer_parts.append(f"**Theo tài liệu {i}:**\n{content}")
    if not answer_parts:
        answer = "Không tìm thấy thông tin liên quan trong các tài liệu tham khảo."
    else:
        answer = "\n\n".join(answer_parts)

# 6) Hiển thị câu trả lời
render_answer(answer)
>>>>>>> e10216675b09a9e6e5fe87885184f76e4f60b7a1
