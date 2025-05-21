import streamlit as st

def render_header():
    st.title("🤖 Chatbot Hành Chính Công")
    st.markdown("> Sử dụng RAG + Gemini API để trả lời câu hỏi về thủ tục Hành chính công")

def input_form():
    return st.text_input("Nhập câu hỏi của bạn:")

def render_answer(answer: str):
    st.markdown("### Trả lời:")
    st.write(answer)
