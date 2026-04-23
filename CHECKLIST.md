# ✅ CHECKLIST - SmartDoc AI — Đánh giá đủ 10 câu hỏi bài tập lớn OSSD 2026

> **Last updated:** 2026-04-12  
> **Project:** SmartDoc AI - Intelligent Document Q&A System  
> **Course:** OSSD Spring 2026

---

## 8.2.1 Câu hỏi 1: Thêm hỗ trợ file DOCX

### Yêu cầu
- [x] Mở rộng hệ thống để hỗ trợ tải lên và xử lý file DOCX
- [x] Sử dụng thư viện phù hợp (python-docx hoặc DocxLoader từ LangChain)
- [x] Đảm bảo text extraction chính xác

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| Document Loader Factory | `src/services/document_service.py` | `DocumentLoaderFactory` sử dụng `Docx2txtLoader` từ LangChain |
| File type validation | `src/utils/constants.py` | `ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt']` |
| Upload UI | `src/views/document_screen.py` | File uploader chấp nhận `.docx` |

### Code Evidence
```python
# src/services/document_service.py — DocumentLoaderFactory
class DocumentLoaderFactory:
    LOADERS = {
        '.pdf': PDFPlumberLoader,
        '.docx': Docx2txtLoader,    # ✅ DOCX support
        '.txt': TextLoader,
    }
```

### Test Steps
1. Chuẩn bị file `.docx` test
2. Mở tab **📄 Documents** → Upload file DOCX
3. Xác nhận processing thành công, sidebar hiện "🟢 Documents Loaded"
4. Đặt câu hỏi về nội dung DOCX → Kiểm tra câu trả lời chính xác

---

## 8.2.2 Câu hỏi 2: Lưu trữ lịch sử hội thoại

### Yêu cầu
- [x] Lưu trữ các câu hỏi và câu trả lời trong session
- [x] Hiển thị lịch sử chat trong sidebar
- [x] Cho phép người dùng xem lại các câu hỏi đã hỏi

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| ChatMessage model | `src/models/chat_model.py` | `@dataclass ChatMessage` với role, content, timestamp, metadata |
| ChatHistory model | `src/models/chat_model.py` | `@dataclass ChatHistory` quản lý danh sách messages, `max_history=50` |
| ChatSession model | `src/models/chat_model.py` | `@dataclass ChatSession` hỗ trợ multi-chat (giống Gemini) |
| Session state | `app.py` | `SessionStateManager` khởi tạo `chat_sessions`, `active_chat_id` |
| Sidebar history | `app.py` | Danh sách chat sessions trong sidebar, click để chuyển |

### Code Evidence
```python
# src/models/chat_model.py
@dataclass
class ChatSession:
    id: str
    name: str
    history: ChatHistory
    created_at: datetime

@dataclass
class ChatHistory:
    messages: List[ChatMessage]
    max_history: int = 50
    
    def add_message(self, role, content, metadata=None): ...
    def get_recent(self, n=10): ...
    def clear(self): ...
```

### Test Steps
1. Upload tài liệu → Đặt câu hỏi → Xác nhận Q&A hiển thị trong chat
2. Đặt thêm câu hỏi → Xác nhận lịch sử đầy đủ
3. Tạo chat mới bằng nút **✚ New Chat** → Xác nhận session riêng biệt
4. Chuyển giữa các chat sessions → Xác nhận lịch sử đúng session

---

## 8.2.3 Câu hỏi 3: Thêm nút xóa lịch sử

### Yêu cầu
- [x] Thêm button "Clear History" để xóa toàn bộ lịch sử chat
- [x] Thêm button "Clear Vector Store" để xóa tài liệu đã upload
- [x] Có confirmation dialog trước khi xóa

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| Clear History | `src/controllers/chat_controller.py` | `clear_history()` xóa active session |
| Clear Vector Store | `src/services/vector_store_service.py` | `clear_store()` xóa FAISS index + BM25 |
| Confirmation UI | `src/views/document_screen.py` | Checkbox confirm trước khi xóa |
| Chat session delete | `app.py` | Nút 🗑️ xóa từng chat session |

### Code Evidence
```python
# src/controllers/chat_controller.py
def clear_history(self) -> None:
    history = self._get_active_history()
    if history:
        history.clear()

# src/services/vector_store_service.py
def clear_store(self) -> None:
    self.vector_store = None
    self._bm25_retriever = None
```

### Test Steps
1. Đặt vài câu hỏi → Nhấn **🗑️** bên cạnh chat session → Xác nhận xóa
2. Upload tài liệu → Nhấn **Clear Vector Store** (có confirm) → Xác nhận sidebar hiện "🟡 No documents"
3. Thử xóa session cuối cùng → Xác nhận chỉ clear messages, không xóa session

---

## 8.2.4 Câu hỏi 4: Cải thiện chunk strategy

### Yêu cầu
- [x] Thử nghiệm các giá trị chunk_size khác nhau (500, 1000, 1500, 2000)
- [x] Thử nghiệm chunk_overlap khác nhau (50, 100, 200)
- [x] So sánh và báo cáo kết quả về độ chính xác
- [x] Cho phép người dùng tùy chỉnh chunk parameters

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| Chunk config UI | `src/views/settings_screen.py` | Sliders chunk_size (500-2000), chunk_overlap (50-300) |
| Benchmark UI | `src/views/settings_screen.py` | "Run Chunk Benchmark" so sánh 12 configs |
| Benchmark logic | `src/controllers/document_controller.py` | `benchmark_chunk_configs()` |
| Text splitter | `src/services/document_service.py` | `SimpleTextSplitter` với chunk_size/overlap |
| Config update | `src/services/document_service.py` | `update_chunk_config()` |

### Code Evidence
```python
# src/views/settings_screen.py — Benchmark
chunk_sizes = [500, 1000, 1500, 2000]
chunk_overlaps = [50, 100, 200]
configs = [(size, overlap) for size in chunk_sizes for overlap in chunk_overlaps]
results = self.document_controller.benchmark_chunk_configs(query, configs)
```

### Test Steps
1. Mở **⚙️ Settings** → Điều chỉnh Chunk Size và Chunk Overlap
2. Nhấn **💾 Apply Chunk Settings** → Upload tài liệu mới
3. Nhập benchmark query → Nhấn **Run Chunk Benchmark**
4. Xem bảng kết quả so sánh 12 configs → Xác nhận best config được highlight

---

## 8.2.5 Câu hỏi 5: Thêm citation/source tracking

### Yêu cầu
- [x] Hiển thị nguồn gốc của thông tin (trang số, vị trí trong PDF)
- [x] Cho phép người dùng click để xem context gốc
- [x] Highlight các đoạn văn được sử dụng để trả lời

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| Citation model | `src/models/document_model.py` | `get_citation()` trả về `[source, page X]` |
| Source display | `src/views/chat_screen.py` | `st.expander("📚 Xem nguồn tham khảo")` |
| Used chunk marking | `src/controllers/chat_controller.py` | `_mark_used_chunks()` tính term overlap |
| Highlight | `src/views/chat_screen.py` | `<mark>` HTML cho used context |
| Metadata | `src/services/document_service.py` | chunk_index, source_file, file_type, uploaded_at |

### Code Evidence
```python
# src/controllers/chat_controller.py
@staticmethod
def _mark_used_chunks(answer, sources):
    answer_terms = {term for term in answer.lower().split() if len(term) > 3}
    for source in sources:
        overlap = sum(1 for term in answer_terms if term in source.content.lower())
        source.metadata["used_in_answer"] = overlap > 0
        source.metadata["used_term_overlap"] = overlap

# src/views/chat_screen.py
if is_used:
    st.caption(f"✅ Được sử dụng trong câu trả lời (term overlap: {overlap})")
    preview = f"<mark>{preview}</mark>"
```

### Test Steps
1. Upload PDF → Đặt câu hỏi → Xác nhận có "📚 Xem nguồn tham khảo"
2. Mở expander → Xác nhận hiển thị source file, page number
3. Kiểm tra ✅ highlight cho chunks được sử dụng
4. Kiểm tra term overlap score

---

## 8.2.6 Câu hỏi 6: Implement Conversational RAG

### Yêu cầu
- [x] Thêm memory để theo dõi ngữ cảnh cuộc hội thoại
- [x] LLM có thể tham chiếu câu hỏi và trả lời trước đó
- [x] Xử lý follow-up questions (câu hỏi tiếp theo)

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| Conversation context | `src/controllers/chat_controller.py` | `_conversation_context(max_turns=4)` |
| Query rewriting | `src/controllers/chat_controller.py` | `_rewrite_query()` phát hiện follow-up |
| Prompt template | `src/controllers/chat_controller.py` | `_build_prompt()` chứa RECENT CONVERSATION |
| Follow-up markers | `src/controllers/chat_controller.py` | Vietnamese + English markers |

### Code Evidence
```python
# src/controllers/chat_controller.py
def _conversation_context(self, max_turns=4):
    chat_history = self._get_active_history()
    recent = chat_history.get_recent(n=max_turns * 2)
    return "\n".join([f"{msg.role}: {msg.content}" for msg in recent])

def _rewrite_query(self, query):
    followup_markers = ["nó", "cái đó", "điều đó", "thế còn", "it", "that", "those"]
    is_followup = len(query.split()) <= 8 or any(m in query.lower() for m in followup_markers)
    if is_followup:
        return f"{query} (follow-up to: {previous_question})"
```

### Test Steps
1. Upload tài liệu → Hỏi "What is machine learning?"
2. Tiếp tục hỏi "Nó có ứng dụng gì?" (follow-up với "nó")
3. Xác nhận LLM hiểu "nó" = machine learning
4. Kiểm tra status hiển thị "✏️ Viết lại câu hỏi"

---

## 8.2.7 Câu hỏi 7: Thêm hybrid search

### Yêu cầu
- [x] Kết hợp semantic search (vector) với keyword search (BM25)
- [x] Implement ensemble retriever
- [x] So sánh performance với pure vector search

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| BM25 retriever | `src/services/vector_store_service.py` | `BM25Retriever` từ LangChain |
| Hybrid search | `src/services/vector_store_service.py` | `_bm25_search()`, `_merge_results()` |
| Result merging | `src/services/vector_store_service.py` | Deduplication + rank-preserving merge |
| Comparison | `src/controllers/chat_controller.py` | Vector-only vs Hybrid metrics |
| Toggle UI | `src/views/settings_screen.py` | Toggle "Hybrid Search (Vector + BM25)" |
| Metrics display | `src/views/chat_screen.py` | `st.expander("📈 Retrieval Metrics")` |

### Code Evidence
```python
# src/services/vector_store_service.py
def _bm25_search(self, query, k):
    self._bm25_retriever.k = k
    results = self._bm25_retriever.invoke(query)
    return [Document(content=doc.page_content, metadata=doc.metadata) for doc in results]

def _merge_results(self, vector_docs, bm25_docs):
    # Deduplicate by content hash, preserve rank order
    ...
```

### Test Steps
1. Mở **⚙️ Settings** → Bật **Hybrid Search (Vector + BM25)**
2. Đặt câu hỏi → Mở **📈 Retrieval Metrics**
3. Kiểm tra `bm25_time_ms`, `overlap_count`, `hybrid_results` vs `vector_results`
4. So sánh chất lượng câu trả lời hybrid vs pure vector

---

## 8.2.8 Câu hỏi 8: Multi-document RAG với metadata filtering

### Yêu cầu
- [x] Hỗ trợ upload nhiều PDF cùng lúc
- [x] Lưu trữ metadata cho mỗi document (tên file, ngày upload, loại)
- [x] Cho phép filter theo metadata khi search
- [x] Hiển thị câu trả lời từ document nào

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| Multi-upload | `src/views/document_screen.py` | Upload nhiều files cùng lúc |
| Metadata storage | `src/services/document_service.py` | source, source_file, file_type, uploaded_at, chunk_index |
| Metadata filtering | `src/services/vector_store_service.py` | `_apply_metadata_filters()` |
| Source filters UI | `src/views/document_screen.py` | Filter theo source file, file type |
| Source display | `src/views/chat_screen.py` | Citation hiển thị source file name |

### Code Evidence
```python
# src/services/document_service.py — Metadata
metadata={
    'source': file_path,
    'source_file': source_path.name,
    'file_type': source_path.suffix.lower(),
    'uploaded_at': upload_time,
    'chunk_index': idx,
}

# src/services/vector_store_service.py — Filtering
def _apply_metadata_filters(documents, metadata_filters):
    source_files = set(metadata_filters.get("source_files", []))
    file_types = set(metadata_filters.get("file_types", []))
    # Filter by source name and file type
```

### Test Steps
1. Upload 2-3 tài liệu khác nhau (PDF + DOCX)
2. Mở **📄 Documents** → Kiểm tra danh sách files + filter options
3. Chọn filter theo source file → Đặt câu hỏi → Xác nhận chỉ trả lời từ file đã chọn
4. Kiểm tra citation hiển thị đúng source file

---

## 8.2.9 Câu hỏi 9: Implement Re-ranking với Cross-Encoder

### Yêu cầu
- [x] Thêm bước re-ranking sau retrieval
- [x] Sử dụng cross-encoder model để đánh giá lại relevance
- [x] So sánh với bi-encoder (current approach)
- [x] Tối ưu hóa latency

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| Cross-encoder | `src/services/vector_store_service.py` | `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` |
| Lazy loading | `src/services/vector_store_service.py` | `_load_cross_encoder()` load khi cần |
| Reranking | `src/services/vector_store_service.py` | `_rerank_documents()` với score + rank metadata |
| Toggle UI | `src/views/settings_screen.py` | Toggle "Cross-Encoder Re-ranking" |
| Latency tracking | `src/services/vector_store_service.py` | `rerank_time_ms` trong stats |

### Code Evidence
```python
# src/services/vector_store_service.py
def _rerank_documents(self, query, documents):
    cross_encoder = self._load_cross_encoder()
    pairs = [[query, doc.content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
    # Add rerank_score and rerank_rank to metadata
```

### Test Steps
1. Mở **⚙️ Settings** → Bật **Cross-Encoder Re-ranking**
2. Đặt câu hỏi → Mở **📈 Retrieval Metrics**
3. Kiểm tra `rerank_time_ms` và `rerank_rank` trong source metadata
4. So sánh chất lượng câu trả lời có/không reranking

---

## 8.2.10 Câu hỏi 10: Advanced RAG với Self-RAG

### Yêu cầu
- [x] Implement Self-RAG: LLM tự đánh giá câu trả lời
- [x] Query rewriting: Tự động cải thiện câu hỏi
- [x] Multi-hop reasoning
- [x] Confidence scoring

### Implementation
| Thành phần | File | Chi tiết |
|-----------|------|----------|
| Query rewriting | `src/controllers/chat_controller.py` | `_rewrite_query()` phát hiện follow-up, viết lại query |
| Self-evaluation | `src/controllers/chat_controller.py` | `_self_evaluate()` LLM tự chấm điểm câu trả lời |
| Confidence scoring | `src/controllers/chat_controller.py` | `_compute_confidence()` dựa trên retrieval + self-eval |
| Multi-hop reasoning | `src/controllers/chat_controller.py` | `_multi_hop_reasoning()` chia câu hỏi phức tạp thành sub-queries |
| Confidence display | `src/views/chat_screen.py` | Hiển thị confidence badge sau câu trả lời |

### Code Evidence
```python
# src/controllers/chat_controller.py
def _self_evaluate(self, question, answer, context):
    """LLM tự đánh giá câu trả lời trên thang 1-5."""
    eval_prompt = f"""Rate this answer 1-5 based on accuracy and completeness..."""
    return self.llm_service.generate(eval_prompt)

def _compute_confidence(self, sources, self_eval_score):
    """Tính confidence score từ retrieval quality + self-evaluation."""
    ...

def _multi_hop_reasoning(self, query, k):
    """Chia câu hỏi phức tạp thành sub-queries."""
    ...
```

### Test Steps
1. Upload tài liệu → Đặt câu hỏi phức tạp
2. Kiểm tra status hiển thị "✏️ Viết lại câu hỏi" (query rewriting)
3. Kiểm tra confidence score hiển thị sau câu trả lời
4. Đặt follow-up question → Xác nhận multi-hop reasoning hoạt động

---

## 📊 Tổng kết

| # | Yêu cầu | Status | Files chính |
|---|---------|--------|-------------|
| 1 | DOCX Support | ✅ Hoàn thành | `document_service.py` |
| 2 | Lưu lịch sử hội thoại | ✅ Hoàn thành | `chat_model.py`, `app.py` |
| 3 | Nút xóa lịch sử | ✅ Hoàn thành | `chat_controller.py`, `vector_store_service.py` |
| 4 | Chunk strategy | ✅ Hoàn thành | `settings_screen.py`, `document_service.py` |
| 5 | Citation/source tracking | ✅ Hoàn thành | `chat_screen.py`, `chat_controller.py` |
| 6 | Conversational RAG | ✅ Hoàn thành | `chat_controller.py` |
| 7 | Hybrid search | ✅ Hoàn thành | `vector_store_service.py` |
| 8 | Multi-document + metadata | ✅ Hoàn thành | `vector_store_service.py`, `document_screen.py` |
| 9 | Re-ranking Cross-Encoder | ✅ Hoàn thành | `vector_store_service.py` |
| 10 | Self-RAG | ✅ Hoàn thành | `chat_controller.py`, `chat_screen.py` |

### Kiến trúc dự án
```
src/
├── models/          # Domain entities (Document, ChatMessage, ChatSession)
├── services/        # Business logic (RAG, VectorStore, LLM, Embedding)
├── controllers/     # Orchestration (Chat, Document, Config)
├── views/           # Streamlit UI (Chat, Document, Settings screens)
└── utils/           # Logger, exceptions, constants, validators
```

### Design Patterns sử dụng
- **Factory Pattern**: `DocumentLoaderFactory` cho PDF/DOCX/TXT
- **Repository Pattern**: `AbstractVectorStoreService` → `FAISSVectorStoreService`
- **Strategy Pattern**: Text splitting, hybrid search, reranking
- **Singleton Pattern**: `SessionStateManager`, cached services
- **Dependency Injection**: Controllers nhận services qua constructor

---

**🎉 Dự án đã đáp ứng đầy đủ 10 yêu cầu của bài tập lớn OSSD 2026!**