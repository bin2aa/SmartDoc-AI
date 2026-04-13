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

**Bước 4: Khởi chạy ứng dụng**
```bash
pip install streamlit
streamlit run app.py
```