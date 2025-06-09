# ChatBot_AI
ChatBot  AI xử lý hành chính công
# Cau tru thu muc
CHATBOT_AI/
├── colab_notebook.ipynb     # Tập tin Colab chính để thực thi
├── requirements.txt         # Các thư viện cần thiết
├── data/
│   └── procedures.json      # Dữ liệu thủ tục hành chính
├── src/
│   ├── __init__.py          # File khởi tạo package
│   ├── data_loader.py       # Xử lý và nạp dữ liệu
│   ├── embeddings.py        # Tạo và lưu embeddings 
│   ├── retriever.py         # Module truy xuất thông tin
│   ├── llm.py               # Cấu hình và tương tác với LLM
│   ├── rag_pipeline.py      # Pipeline kết hợp retriever và LLM
│   └── utils.py             # Các hàm tiện ích
└── app/
    ├── __init__.py          # File khởi tạo package
    ├── components.py        # Các thành phần giao diện người dùng
    └── main.py              # Ứng dụng Streamlit chính