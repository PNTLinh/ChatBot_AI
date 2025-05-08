import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    DirectoryLoader
)
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("seed_data")

def load_json_data(filename: str, directory: str) -> List[Dict[str, Any]]:
    """
    Tải dữ liệu từ file JSON
    
    Args:
        filename: Tên file JSON
        directory: Thư mục chứa file
        
    Returns:
        List các dictionary chứa dữ liệu
        
    Raises:
        FileNotFoundError: Khi không tìm thấy file
        ValueError: Khi file không phải là JSON hợp lệ
    """
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"Không tìm thấy tệp: {file_path}")
        raise FileNotFoundError(f"Không tìm thấy tệp: {file_path}")
    
    try:
        logger.info(f"Đang tải dữ liệu từ: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Xử lý các cấu trúc JSON khác nhau
        # Nếu data là dictionary với key chứa records, trích xuất records
        if isinstance(data, dict):
            for key in ["records", "data", "items", "results"]:
                if key in data and isinstance(data[key], list):
                    logger.info(f"Đã tìm thấy dữ liệu trong khóa '{key}' với {len(data[key])} bản ghi")
                    return data[key]
            
            # Nếu không có key nào tồn tại, chuyển đổi dictionary thành list có một phần tử
            if not any(isinstance(data[key], list) for key in data):
                logger.info("Chuyển đổi dictionary đơn lẻ thành danh sách có một phần tử")
                return [data]
        
        # Nếu data đã là list
        if isinstance(data, list):
            logger.info(f"Dữ liệu đã là danh sách với {len(data)} phần tử")
            return data
        
        # Fallback: trả về dưới dạng một phần tử duy nhất
        logger.warning("Cấu trúc dữ liệu không rõ ràng, trả về dưới dạng một phần tử")
        return [data]
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi phân tích JSON: {str(e)}")
        raise ValueError(f"Tệp {file_path} không phải là JSON hợp lệ: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi không xác định khi tải JSON: {str(e)}")
        raise

def _normalize_key(key: str) -> str:
    """
    Chuẩn hóa tên khóa từ camelCase/snake_case thành dạng hiển thị
    
    Args:
        key: Tên khóa cần chuẩn hóa
        
    Returns:
        Tên khóa đã chuẩn hóa
    """
    # Map tên trường tiếng Anh sang tiếng Việt
    vietnamese_names = {
        "ten_thu_tuc": "Tên thủ tục",
        "linh_vuc": "Lĩnh vực",
        "co_quan": "Cơ quan thực hiện", 
        "thanh_phan_ho_so": "Thành phần hồ sơ",
        "trinh_tu_thuc_hien": "Trình tự thực hiện",
        "thoi_han_giai_quyet": "Thời hạn giải quyết",
        "le_phi": "Lệ phí",
        "phi": "Phí",
        "ket_qua_thuc_hien": "Kết quả thực hiện",
        "can_cu_phap_ly": "Căn cứ pháp lý",
        "yeu_cau": "Yêu cầu điều kiện",
        "ma_thu_tuc": "Mã thủ tục",
        "muc_do": "Mức độ",
        "cach_thuc_thuc_hien": "Cách thức thực hiện",
        "doi_tuong_thuc_hien": "Đối tượng thực hiện"
    }
    
    # Nếu tên trường đã được định nghĩa trong map, sử dụng giá trị từ map
    if key in vietnamese_names:
        return vietnamese_names[key]
    
    # Chuyển đổi camelCase thành snake_case
    if any(c.isupper() for c in key) and '_' not in key:
        s1 = ''.join(['_' + c.lower() if c.isupper() else c for c in key])
        if s1.startswith('_'):
            s1 = s1[1:]
    else:
        s1 = key
    
    # Chuyển đổi snake_case thành dạng tiêu đề (Title Case)
    words = s1.split('_')
    return ' '.join(word.capitalize() for word in words)

def convert_json_to_documents(json_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Chuyển đổi dữ liệu JSON thành các Document LangChain - Chuyên biệt cho định dạng TTHC.json
    
    Args:
        json_data: List các dictionary chứa dữ liệu
        
    Returns:
        List các đối tượng Document
    """
    documents = []
    logger.info(f"Bắt đầu chuyển đổi {len(json_data)} bản ghi thành Document")
    
    # Ánh xạ tên trường chuẩn hóa
    field_mapping = {
        "tenThuTuc": "ten_thu_tuc",
        "linhVuc": "linh_vuc",
        "coQuanThucHien": "co_quan",
        "thanhPhanHoSo": "thanh_phan_ho_so",
        "trinhTuThucHien": "trinh_tu_thuc_hien",
        "thoiHanGiaiQuyet": "thoi_han_giai_quyet",
        "lePhi": "le_phi",
        "ketQuaThucHien": "ket_qua_thuc_hien",
        "canCuPhapLy": "can_cu_phap_ly",
        "yeuCau": "yeu_cau",
        "maThuTuc": "ma_thu_tuc"
    }
    
    # Các trường quan trọng cần hiển thị đầu tiên
    important_fields = [
        "ten_thu_tuc", "linh_vuc", "co_quan", "thanh_phan_ho_so", 
        "trinh_tu_thuc_hien", "thoi_han_giai_quyet", "le_phi", 
        "ket_qua_thuc_hien", "can_cu_phap_ly"
    ]
    
    for idx, item in enumerate(json_data):
        try:
            # Khởi tạo nội dung và metadata
            content = ""
            metadata = {}
            
            # Tiền xử lý: chuẩn hóa tên trường
            normalized_item = {}
            for key, value in item.items():
                # Áp dụng ánh xạ tên trường nếu có
                normalized_key = field_mapping.get(key, key)
                normalized_item[normalized_key] = value
                
                # Thêm vào metadata
                metadata[normalized_key] = value
            
            # Xử lý các trường quan trọng trước
            for field in important_fields:
                if field in normalized_item:
                    display_name = _normalize_key(field)
                    value = normalized_item[field]
                    
                    # Xử lý cho danh sách
                    if isinstance(value, list):
                        if value:  # Chỉ xử lý khi danh sách không rỗng
                            items_text = "\n- " + "\n- ".join(str(item) for item in value if item)
                            content += f"**{display_name}**: {items_text}\n\n"
                    else:
                        if value:  # Chỉ thêm khi giá trị không rỗng
                            content += f"**{display_name}**: {value}\n\n"
            
            # Xử lý các trường khác
            for key, value in normalized_item.items():
                if key not in important_fields and value:
                    display_name = _normalize_key(key)
                    
                    # Xử lý danh sách
                    if isinstance(value, list):
                        if value:
                            items_text = "\n- " + "\n- ".join(str(item) for item in value if item)
                            content += f"**{display_name}**: {items_text}\n\n"
                    else:
                        content += f"**{display_name}**: {value}\n\n"
            
            # Đặt identifier cho nguồn
            source_id = normalized_item.get("ma_thu_tuc", None)
            if not source_id:
                source_id = f"TTHC-{idx+1}"
            
            metadata["source"] = source_id
            
            # Chỉ thêm vào khi có nội dung
            if content.strip():
                documents.append(Document(page_content=content.strip(), metadata=metadata))
        
        except Exception as e:
            logger.warning(f"Lỗi khi xử lý bản ghi thứ {idx+1}: {str(e)}")
            continue
    
    logger.info(f"Đã chuyển đổi thành công {len(documents)} Document")
    return documents

def create_documents_chunks(documents: List[Document]) -> List[Document]:
    """
    Chia nhỏ các Document thành các đoạn nhỏ để xử lý hiệu quả hơn
    
    Args:
        documents: Danh sách các đối tượng Document ban đầu
        
    Returns:
        Danh sách các Document đã được chia nhỏ
    """
    try:
        # Khởi tạo text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"Bắt đầu phân đoạn {len(documents)} tài liệu")
        # Phân chia văn bản
        chunks = text_splitter.split_documents(documents)
        
        # Đảm bảo metdata được giữ nguyên
        for i, chunk in enumerate(chunks):
            # Giữ nguyên metadata từ tài liệu gốc
            if "source" not in chunk.metadata and hasattr(chunk, "source"):
                chunk.metadata["source"] = chunk.source
                
            # Thêm ID chunk để dễ theo dõi
            chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}-chunk-{i+1}"
            
            # Đổi tên các trường metadata cho rõ ràng
            if "ten_thu_tuc" in chunk.metadata:
                chunk.metadata["ten_tthc"] = chunk.metadata["ten_thu_tuc"]
            if "co_quan" in chunk.metadata:
                chunk.metadata["co_quan_thuc_hien"] = chunk.metadata["co_quan"]
                
        logger.info(f"Đã tạo {len(chunks)} đoạn văn bản từ {len(documents)} tài liệu")
        return chunks
    except Exception as e:
        logger.error(f"Lỗi khi chia nhỏ tài liệu: {str(e)}")
        # Trả về tài liệu gốc nếu có lỗi
        return documents

def create_faiss_index(
    documents: List[Document], 
    collection_name: str,
    embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
    google_api_key: Optional[str] = None
) -> FAISS:
    """
    Tạo chỉ mục FAISS từ các Document
    
    Args:
        documents: Danh sách các đối tượng Document
        collection_name: Tên của bộ sưu tập/chỉ mục
        embeddings: Đối tượng embeddings (tùy chọn)
        google_api_key: API key của Google (nếu không cung cấp embeddings)
        
    Returns:
        Đối tượng FAISS đã được khởi tạo
        
    Raises:
        ValueError: Khi không thể tạo embeddings
    """
    try:
        # Khởi tạo embeddings nếu không được cung cấp
        if embeddings is None:
            if not google_api_key: # Kiểm tra xem key có được truyền vào không
                  # Nếu không có key được truyền vào hàm này, thử lấy từ env (fallback cuối cùng)
                  google_api_key = os.environ.get("GOOGLE_API_KEY")

            if not google_api_key:
                 logger.error("Không tìm thấy Google API Key để tạo embeddings")
                 raise ValueError("Cần cung cấp Google API Key để tạo embeddings")

            logger.info("Khởi tạo embeddings với Google API trong create_faiss_index")
            embeddings = GoogleGenerativeAIEmbeddings(
                 model="models/embedding-001",
                 google_api_key=google_api_key
             )
        elif not isinstance(embeddings, GoogleGenerativeAIEmbeddings):
              # Xử lý trường hợp embeddings được truyền vào nhưng không đúng kiểu
              logger.error(f"Đối tượng embeddings được truyền vào không đúng kiểu: {type(embeddings)}")
              raise TypeError("Đối tượng embeddings được truyền vào không đúng kiểu.")

        db = FAISS.from_documents(documents, embeddings)
        
        # Lưu chỉ mục vào ổ đĩa
        logger.info(f"Lưu chỉ mục FAISS vào {index_path}")
        db.save_local(index_path, collection_name)
        
        return db
    except Exception as e:
        logger.error(f"Lỗi khi tạo chỉ mục FAISS: {str(e)}")
        raise

def seed_faiss(
    collection_name: str,
    json_file: str,
    json_dir: str,
    google_api_key: Optional[str] = None,
) -> None:
    """
    Tạo và lưu trữ chỉ mục FAISS từ dữ liệu JSON
    
    Args:
        collection_name: Tên cho chỉ mục FAISS
        json_file: Tên file JSON
        json_dir: Thư mục chứa file JSON
        use_gemini: Sử dụng Gemini API cho embeddings
        
    Returns:
        None
    """
    try:
        # Tải dữ liệu từ JSON
        json_data = load_json_data(json_file, json_dir)
        
        # Chuyển JSON thành Document
        documents = convert_json_to_documents(json_data)
        
        # Chia nhỏ Document
        chunks = create_documents_chunks(documents)
        
        # Khởi tạo embeddings nếu dùng Gemini
        embeddings = None
        if google_api_key: # <-- Chỉ kiểm tra xem key có tồn tại không
            logger.info("Khởi tạo embeddings với Google API")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
            )
        else:
            logger.warning("Không có Google API Key được cung cấp, không thể tạo embeddings.")
            # Tùy chọn: raise error hoặc tiếp tục mà không tạo index

            
        # Tạo chỉ mục FAISS
        create_faiss_index(chunks, collection_name, embeddings)
        logger.info(f"Đã tạo thành công chỉ mục FAISS '{collection_name}'")
    except Exception as e:
        logger.error(f"Lỗi khi seed dữ liệu: {str(e)}")
        raise

def fetch_url_content(url: str) -> Optional[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        logger.info(f"Đang tải nội dung từ: {url}") # <-- Thêm log bắt đầu tải
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"Đã tải nội dung thành công từ: {url}") # <-- Thêm log tải thành công
        return response.text
    except requests.exceptions.RequestException as e: # <-- Bắt lỗi requests cụ thể hơn
        logger.error(f"Lỗi request khi tải nội dung từ {url}: {str(e)}") # <-- Log lỗi chi tiết hơn
        return None
    except Exception as e:
        logger.error(f"Lỗi không xác định khi tải nội dung từ {url}: {str(e)}") # <-- Log lỗi chi tiết hơn
        return None

def extract_tthc_data(html_content: str) -> Dict[str, Any]:
    """
    Trích xuất dữ liệu thủ tục hành chính từ trang web
    
    Args:
        html_content: Nội dung HTML của trang
        
    Returns:
        Dictionary chứa thông tin về thủ tục hành chính
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        tthc_data = {}
        
        # Tìm tiêu đề
        title_element = soup.find('h1', class_='detail-title')
        if title_element:
            tthc_data['ten_thu_tuc'] = title_element.get_text(strip=True)
        
        # Tìm các trường thông tin khác
        content_sections = soup.find_all('div', class_='detail-content')
        for section in content_sections:
            header = section.find('h3')
            if header:
                field_name = header.get_text(strip=True).lower()
                
                # Ánh xạ tiêu đề HTML sang tên trường
                field_mapping = {
                    'trình tự thực hiện': 'trinh_tu_thuc_hien',
                    'cách thức thực hiện': 'cach_thuc_thuc_hien',
                    'thành phần hồ sơ': 'thanh_phan_ho_so',
                    'số lượng hồ sơ': 'so_luong_ho_so',
                    'thời hạn giải quyết': 'thoi_han_giai_quyet',
                    'đối tượng thực hiện': 'doi_tuong_thuc_hien',
                    'cơ quan thực hiện': 'co_quan',
                    'kết quả thực hiện': 'ket_qua_thuc_hien',
                    'lệ phí': 'le_phi',
                    'tên mẫu đơn': 'ten_mau_don',
                    'yêu cầu điều kiện': 'yeu_cau',
                    'căn cứ pháp lý': 'can_cu_phap_ly',
                    'lĩnh vực': 'linh_vuc'
                }
                
                # Tìm trường tương ứng
                for key, value in field_mapping.items():
                    if key in field_name:
                        content_div = section.find('div', class_='content')
                        if content_div:
                            tthc_data[value] = content_div.get_text(strip=True)
                            break
        
        return tthc_data
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất dữ liệu từ HTML: {str(e)}")
        return {}

def process_urls_to_documents(urls: List[str]) -> List[Document]:
    """
    Xử lý danh sách URL để tạo Document
    
    Args:
        urls: Danh sách các URL cần xử lý
        
    Returns:
        Danh sách các Document đã được tạo
    """
    documents = []
    
    def process_url(url):
        try:
            html_content = fetch_url_content(url)
            if not html_content:
                logger.warning(f"Không lấy được nội dung HTML từ {url}. Bỏ qua.") # <-- Thêm log cảnh báo
                return None

            tthc_data = extract_tthc_data(html_content)
            if not tthc_data:
                logger.warning(f"Không trích xuất được dữ liệu TTHC từ {url}. Bỏ qua.") # <-- Thêm log cảnh báo
                return None

            # Tạo Document từ dữ liệu
            doc = convert_json_to_documents([tthc_data])
            logger.info(f"Đã xử lý thành công URL {url} và tạo Document.") # <-- Thêm log thành công
            return doc[0] if doc else None
        except Exception as e:
            logger.error(f"Lỗi khi xử lý URL {url}: {str(e)}") # <-- Log lỗi cụ thể URL và lỗi
            return None
    
    # Sử dụng ThreadPoolExecutor để xử lý song song
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_url, urls))
    
    # Lọc kết quả None
    documents = [doc for doc in results if doc is not None]
    return documents

def seed_faiss_live(
    collection_name: str,
    urls: List[str],
    google_api_key: Optional[str] = None,
) -> None:
    """
    Tạo và lưu trữ chỉ mục FAISS từ các URL
    
    Args:
        collection_name: Tên cho chỉ mục FAISS
        urls: Danh sách các URL chứa thông tin TTHC
        use_gemini: Sử dụng Gemini API cho embeddings
        
    Returns:
        None
    """
    try:
        # Xử lý các URL thành Document
        documents = process_urls_to_documents(urls)
        if not documents:
            logger.error("Không trích xuất được dữ liệu từ các URL")
            return
            
        # Chia nhỏ Document
        chunks = create_documents_chunks(documents)
        
        # Khởi tạo embeddings nếu dùng Gemini
        embeddings = None
        if google_api_key: # <-- Chỉ kiểm tra xem key có tồn tại không
            logger.info("Khởi tạo embeddings với Google API")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
            )
        else:
            logger.warning("Không có Google API Key được cung cấp, không thể tạo embeddings.")
            # Tùy chọn: raise error hoặc tiếp tục mà không tạo index

        
        # Tạo chỉ mục FAISS
        create_faiss_index(chunks, collection_name, embeddings)
        logger.info(f"Đã tạo thành công chỉ mục FAISS '{collection_name}' từ URL")
    except Exception as e:
        logger.error(f"Lỗi khi seed dữ liệu từ URL: {str(e)}")
        raise

if __name__ == "__main__":
    # Test các hàm khi chạy trực tiếp file này
    try:
        # Kiểm tra môi trường
        from dotenv import load_dotenv
        load_dotenv()
        
        # Kiểm tra API key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("Vui lòng đặt GOOGLE_API_KEY trong file .env để test")
        
        # Thử load dữ liệu từ file TTHC_test.json nếu có
        test_file = "TTHC_test.json"
        if os.path.exists(test_file):
            logger.info(f"Thử nghiệm với file {test_file}")
            seed_faiss("test_index", test_file, ".", use_gemini=bool(api_key))
            logger.info("Test thành công!")
        else:
            logger.info(f"Không tìm thấy file {test_file} để test")
            
    except Exception as e:
        logger.error(f"Test không thành công: {str(e)}")