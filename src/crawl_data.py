import os
import re
import json
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_to_env()
def bs4_extractor(html: str) -> str:
    """
    Hàm trích xuất và làm sạch nội dung từ HTML
    Args:
        html: Chuỗi HTML cần xử lý
    Returns:
        str: Văn bản đã được làm sạch, loại bỏ các thẻ HTML và khoảng trắng thừa
    """
    soup = BeautifulSoup(html, "html.parser")  # Phân tích cú pháp HTML
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()  # Xóa khoảng trắng và dòng trống thừa

def crawl_web(url_data):
    loader = RecursiveUrlLoader(url=url_data, extractor=bs4_extractor, max_depth=4)
    docs = loader.load()  
    print('length: ', len(docs)) 
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    print('length_all_splits: ', len(all_splits))  # In số lượng đoạn văn bản sau khi chia
    return all_splits

def web_base_loader(url_data):
    loader = WebBaseLoader(url_data)  
    docs = loader.load()  
    print('length: ', len(docs))  
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def save_data_locally(documents, filename, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, filename)  

    data_to_save = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    with open(file_path, 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print(f'Data saved to {file_path}')  # In thông báo lưu thành công

def main():
    data = crawl_web('https://www.stack-ai.com/docs')
    save_data_locally(data, 'stack.json', 'data')
    print('data: ', data)  

if __name__ == "__main__":
    main()