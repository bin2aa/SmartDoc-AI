# SmartDoc AI - Intelligent Document Q&A System

[cite_start]Dự án này là một hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh, cho phép người dùng tải lên tài liệu PDF và đặt câu hỏi về nội dung tài liệu. [cite_start]Hệ thống được tối ưu hóa đặc biệt cho tiếng Việt và hỗ trợ hơn 50 ngôn ngữ khác.

## Kiến trúc Công nghệ

[cite_start]Hệ thống được xây dựng dựa trên kiến trúc Multi-layer:
* [cite_start]**Giao diện (Frontend):** Xây dựng bằng Streamlit framework.
* [cite_start]**Application Layer:** Sử dụng LangChain framework để quản lý pipeline.
* [cite_start]**Vector Database:** Sử dụng FAISS để lưu trữ và tìm kiếm vector hiệu quả.
* [cite_start]**Embedding Model:** Sử dụng HuggingFace `paraphrase-multilingual-mpnet-base-v2` .
* [cite_start]**LLM Engine:** Chạy local model Qwen2.5:7b thông qua Ollama.

## Hướng dẫn Cài đặt

[cite_start]Yêu cầu hệ thống: Python 3.8+, Ollama runtime và pip package manager.

**Bước 1: Thiết lập môi trường ảo**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc venv\Scripts\activate cho Windows
```

**Bước 2: Cài đặt các thư viện phụ thuộc**
```bash
pip install -r requirements.txt
```

**Bước 3: Tải mô hình AI cục bộ**
(Đảm bảo máy chủ Ollama đang chạy ngầm trên máy)
```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b

```
**Bước 4: Cài đặt phần mềm hệ thống (Tesseract & Poppler)**
```bash
Tùy theo hệ điều hành bạn đang sử dụng, hãy thực hiện các bước cài đặt tương ứng dưới đây:

+ Trên macOS (sử dụng Homebrew)
Mở Terminal và chạy lệnh sau:
brew install tesseract tesseract-lang poppler

+ Trên Linux (Ubuntu / Debian)
Mở Terminal và thực hiện lần lượt các lệnh:
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-vie poppler-utils

+ Trên Windows 
a. Cài đặt Tesseract OCR (Công cụ nhận dạng ký tự)
•	Tải bộ cài đặt tại:
https://github.com/UB-Mannheim/tesseract/wiki (chọn phiên bản 64-bit) 
•	Trong quá trình cài đặt: 
o	Tại mục Additional language data, bắt buộc chọn Vietnamese để hỗ trợ nhận dạng tiếng Việt. 
•	Sau khi cài đặt, đường dẫn mặc định của Tesseract sẽ là: 
C:\Program Files\Tesseract-OCR\tesseract.exe

b. Cài đặt Poppler (Công cụ xử lý PDF sang ảnh)
•	Tải bản Poppler dành cho Windows tại:
Release poppler-windows 
•	Sau khi tải về: 
1.	Giải nén file .zip 
2.	Đổi tên thư mục vừa giải nén thành poppler 
3.	Sao chép thư mục này vào ổ C, ví dụ: 
C:\poppler
```
**Bước 5: Khởi chạy ứng dụng**
```bash
pip install streamlit
streamlit run app.py
```