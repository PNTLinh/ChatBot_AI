import streamlit as st
from components import render_header, render_answer
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
    return FAISS.load_local(
        "data/faiss_store",
        embed,
        allow_dangerous_deserialization=True,
    )

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
