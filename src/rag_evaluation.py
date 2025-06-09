import pandas as pd
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import google.generativeai as genai
from bert_score import score as bert_score
import time
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class RAGEvaluator:
    def __init__(self, gemini_api_key, faiss_path="data/faiss_store"):
        """
        Initialize RAG Evaluator
        
        Args:
            gemini_api_key (str): Google API key for Gemini
            faiss_path (str): Path to FAISS vector store
        """
        self.gemini_api_key = gemini_api_key
        self.faiss_path = faiss_path
        self.setup_models()
        
    def setup_models(self):
        """Setup Gemini and embeddings models"""
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 800,
            }
        )
        
        # Setup embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = FAISS.load_local(
            self.faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        
    def get_prompt_template(self):
        """Get prompt template for Gemini"""
        return """Bạn là chuyên gia tư vấn thủ tục hành chính Việt Nam.

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {question}

YÊU CẦU TRẢ LỜI:
- Trả lời trực tiếp, ngắn gọn
- Bao gồm thông tin chính: điều kiện, hồ sơ, quy trình
- Chỉ dựa trên thông tin có trong tài liệu

TRẢ LỜI:"""
    
    def generate_answer(self, query, top_k=6):
        """
        Generate optimized answer using enhanced RAG pipeline
        
        Args:
            query (str): User query
            top_k (int): Number of documents to retrieve
            
        Returns:
            tuple: (answer, response_time)
        """
        start_time = time.time()
        
        # Enhanced retrieval with MMR for diversity
        docs = self.vectorstore.max_marginal_relevance_search(
            query, k=top_k, fetch_k=top_k*2
        )
        
        if not docs:
            return "Không tìm thấy thông tin liên quan.", time.time() - start_time
        
        # Smart context preparation with relevance scoring
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Use more content for better context
            content = doc.page_content[:1500] 
            source = doc.metadata.get("source", f"Tài liệu {i}")
            
            # Add relevance indicators
            context_parts.append(f"[Tài liệu {i}] {source}\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt with specific structure matching
        enhanced_prompt = f"""Bạn là chuyên gia tư vấn thủ tục hành chính Việt Nam.

NGUYÊN TẮC TRẢ LỜI:
- Sử dụng CHÍNH XÁC các thuật ngữ và cụm từ có trong tài liệu tham khảo
- Giữ nguyên tên thủ tục, mã số, tên cơ quan như trong tài liệu
- Cấu trúc câu trả lời theo format chuẩn

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {query}

CÁCH TRẢ LỜI:
1. Bắt đầu với "Tên thủ tục:" nếu có
2. Ghi rõ "Mã thủ tục:" nếu có  
3. Liệt kê "Cấp thực hiện:" và "Cơ quan thực hiện:"
4. Sử dụng chính xác các từ ngữ trong tài liệu

TRẢ LỜI:"""
        
        try:
            # Multiple attempts for better results
            best_answer = None
            best_length = 0
            
            for attempt in range(2):  # Try twice, pick better answer
                response = self.model.generate_content(enhanced_prompt)
                answer = response.text.strip()
                
                # Prefer longer, more detailed answers
                if len(answer) > best_length and "Tên thủ tục" in answer:
                    best_answer = answer
                    best_length = len(answer)
                elif best_answer is None:
                    best_answer = answer
            
            answer = best_answer or "Không thể tạo câu trả lời phù hợp."
            
        except Exception as e:
            answer = f"Lỗi khi tạo câu trả lời: {str(e)}"
        
        response_time = time.time() - start_time
        return answer, response_time
    
    def calculate_bert_score(self, references, candidates):
        """
        Calculate BERT score between reference and candidate texts
        
        Args:
            references (list): List of ground truth texts
            candidates (list): List of generated texts
            
        Returns:
            dict: BERT scores (Precision, Recall, F1)
        """
        try:
            P, R, F1 = bert_score(candidates, references, lang='vi', verbose=False)
            
            return {
                'BERT_Precision': P.mean().item(),
                'BERT_Recall': R.mean().item(),
                'BERT_F1': F1.mean().item()
            }
        except Exception as e:
            print(f"Error calculating BERT score: {e}")
            return {
                'BERT_Precision': 0.0,
                'BERT_Recall': 0.0,
                'BERT_F1': 0.0
            }
    
    def evaluate_dataset(self, test_data, output_file="rag_evaluation_results.csv"):
        """
        Evaluate RAG model on test dataset
        
        Args:
            test_data (list): List of dictionaries with 'query' and 'reference_answer' keys
            output_file (str): Output CSV file name
            
        Returns:
            pandas.DataFrame: Evaluation results
        """
        results = []
        
        print(f"Bắt đầu đánh giá {len(test_data)} câu hỏi...")
        print("=" * 60)
        
        for i, item in enumerate(test_data, 1):
            query = item['query']
            reference = item['reference_answer']
            
            print(f"\n[{i}/{len(test_data)}] Đang xử lý: {query[:50]}...")
            
            # Generate answer
            answer, response_time = self.generate_answer(query)
            
            # Calculate BERT scores
            bert_scores = self.calculate_bert_score([reference], [answer])
            
            # Store result
            result = {
                'query': query,
                'reference_answer': reference,
                'generated_answer': answer,
                'response_time': response_time,
                'BERT_Precision': bert_scores['BERT_Precision'],
                'BERT_Recall': bert_scores['BERT_Recall'],
                'BERT_F1': bert_scores['BERT_F1'],
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Print progress
            print(f"  BERT F1: {bert_scores['BERT_F1']:.4f}")
            print(f"  Response time: {response_time:.2f}s")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ Kết quả đã được lưu vào: {output_file}")
        
        # Print summary statistics
        self.print_summary_stats(df)
        
        return df
    
    def print_summary_stats(self, df):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("TỔNG KẾT KẾT QUẢ ĐÁNH GIÁ")
        print("="*60)
        
        print(f"Số câu hỏi: {len(df)}")
        print(f"BERT Precision: {df['BERT_Precision'].mean():.4f} ± {df['BERT_Precision'].std():.4f}")
        print(f"BERT Recall: {df['BERT_Recall'].mean():.4f} ± {df['BERT_Recall'].std():.4f}")
        print(f"BERT F1: {df['BERT_F1'].mean():.4f} ± {df['BERT_F1'].std():.4f}")
        print(f"Thời gian phản hồi trung bình: {df['response_time'].mean():.2f}s ± {df['response_time'].std():.2f}s")

def create_sample_test_data():
    """Tạo tập dữ liệu mẫu"""
    sample_data = [
        {
            'query': 'Nhận chăm sóc, nuôi dưỡng đối tượng cần bảo vệ khẩn cấp',
            'reference_answer': '''Tên thủ tục: Nhận chăm sóc, nuôi dưỡng đối tượng cần bảo vệ khẩn cấp
Mã thủ tục: 1.001739
Cấp thực hiện: Cấp Huyện, Cấp xã
Cơ quan thực hiện: Ủy ban Nhân dân huyện, quận, thành phố trực thuộc tỉnh, thị xã'''
        },
        {
            'query': 'Quyết định trợ cấp xã hội hàng tháng, hỗ trợ kinh phí chăm sóc, nuôi dưỡng hàng tháng khi đối tượng thay đổi nơi cư trú giữa các quận, huyện, thị xã, thành phố thuộc tỉnh, trong và ngoài tỉnh, thành phố trực thuộc trung ương',
            'reference_answer': '''Tên thủ tục: Quyết định trợ cấp xã hội hàng tháng, hỗ trợ kinh phí chăm sóc, nuôi dưỡng hàng tháng
Mã thủ tục: 1.000145
Cấp thực hiện: Cấp Huyện, Cấp Xã
Cơ quan thực hiện: Ủy ban Nhân dân huyện, quận, thành phố trực thuộc tỉnh, thị xã., Ủy ban Nhân dân xã, phường, thị trấn'''
        },
        {
            'query': 'Cấp lại giấy chứng sinh đối với trường hợp bị mất hoặc hư hỏng',
            'reference_answer': '''Tên thủ tục: Cấp lại giấy chứng sinh đối với trường hợp bị mất hoặc hư hỏng
Mã thủ tục: 1.002341
Cấp thực hiện: Cơ quan khác
Cơ quan thực hiện: Các cơ sở khám chữa bệnh Trung ương và địa phương'''
        },
        {
            'query': 'Thủ tục cấp Giấy phép thành lập và hoạt động của tổ chức tín dụng phi ngân hàng',
            'reference_answer': '''Thủ tục cấp Giấy phép thành lập và hoạt động của tổ chức tín dụng phi ngân hàng
Mã thủ tục: 1.006245
Cấp thực hiện: Cấp Bộ
Cơ quan thực hiện: Ngân hàng Nhà nước Việt Nam'''
        }
    ]
    return sample_data

if __name__ == "__main__":
    # Configuration
    GEMINI_API_KEY = "AIzaSyBmahRQr-CKQwQqvH_3_Eayu9itHmWcJng"  # Thay bằng API key thực
    FAISS_PATH = "data/faiss_store"
    OUTPUT_FILE = "rag_evaluation_results.csv"
    
    # Check if API key is provided
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("⚠️  Vui lòng thay thế GEMINI_API_KEY bằng API key thực của bạn!")
        exit(1)
    
    # Check if FAISS store exists
    if not os.path.exists(FAISS_PATH):
        print(f"⚠️  Không tìm thấy FAISS vector store tại: {FAISS_PATH}")
        exit(1)
    
    try:
        # Initialize evaluator
        print("🚀 Khởi tạo RAG Evaluator...")
        evaluator = RAGEvaluator(GEMINI_API_KEY, FAISS_PATH)
        
        # Create test data
        print("📝 Tạo dữ liệu test...")
        test_data = create_sample_test_data()
        
        # Run evaluation
        print("🔍 Bắt đầu đánh giá mô hình...")
        results_df = evaluator.evaluate_dataset(test_data, OUTPUT_FILE)
        
        print(f"\n🎉 Hoàn thành! Kiểm tra file {OUTPUT_FILE} để xem chi tiết kết quả.")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        print("Vui lòng kiểm tra lại cấu hình và thử lại.")