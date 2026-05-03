"""Document domain models for SmartDoc AI.

MOT SO QUAN HE GIUA CAC MODEL:

  Document (document_model.py)
    ^
    | luu tru nhu
    |
  LCDocument (LangChain, trong FAISS vector store)
    |
    | duoc embed thanh
    |
  Vector Embeddings (768 chieu, HuggingFace)

  metadata (di cung Document) chua thong tin phu:
    - source_file: ten file -> dung de filter
    - file_type: loai file -> dung de filter
    - page, chunk_index -> dung de citation
    - uploaded_at, file_size_mb -> dung de hien thi

VÍ DỤ LUỒNG ĐI:
  1. User upload "report.pdf"
  2. DocumentService.load_document() tao 15 Document objects
  3. Moi Document co metadata: {source_file: "report.pdf", ...}
  4. Document objects duoc embed -> vector embeddings
  5. Vector embeddings + LCDocument (voi metadata) duoc luu vao FAISS
  6. BM25Retriever duoc xay dung tu LCDocument
  7. User hoi: "tom tat bai"
  8. Vector search -> 3 Document objects
  9. Metadata filter (neu co) -> loc con lai 2 Document
  10. Cross-encoder rerank (neu co) -> sap xep 2 Document
  11. Tra ve 2 Document cho ChatController
  12. ChatController goi LLM -> tra loi
  13. ChatScreen._render_sources() hien thi metadata cua 2 Document
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class Document:
    """
    DOMAIN ENTITY CHO MỘT TÀI LIỆU/CHUNK.

    Đây là "nguyên tắc" (entity) của domain, đại diện cho một phần
    nội dung tài liệu đã được chunking.

    SO VỚI LCDocument (LangChain Document):
      - LCDocument: format của LangChain, sử dụng trong FAISS
      - Document: format domain, sử dụng trong toàn bộ ứng dụng

    CÁC THUỘC TÍNH:
      - content: nội dung văn bản của chunk
      - metadata: từ điển chứa thông tin phụ (file, trang, chunk, ...)
      - id: UUID duy nhất để nhận biết (auto-generated)
      - created_at: thời gian tạo (auto-generated)

    METADATA QUAN TRỌNG (được bổ sung khi load):
      - source: đường dẫn đầy đủ đến file gốc
      - source_file: chỉ tên file (VD: "report.pdf")
      - file_type: phần mở rộng (VD: ".pdf")
      - uploaded_at: thời gian upload (định dạng ISO timestamp)
      - chunk_index: vị trí chunk trong file (0-based)
      - page: số trang (nếu có)
      - file_size_bytes, file_size_mb: kích thước file
      - title: tiêu đề rút gọn từ tên file
      - page_count: tổng số trang của file
      - total_chunks: tổng số chunks từ file này
      - rerank_score: điểm cross-encoder (nếu có rerank)
      - rerank_rank: thứ tự sau rerank (nếu có)
      - used_in_answer: True/False (tài liệu có được sử dụng trong câu trả lời không)
      - used_term_overlap: số từ trùng nhau giữa câu trả lời và tài liệu

    VÍ DỤ:
      doc = Document(
          content="Nội dung trang 3...",
          metadata={
              "source": "/uploads/report.pdf",
              "source_file": "report.pdf",
              "file_type": ".pdf",
              "page": 3,
              "chunk_index": 5,
              "uploaded_at": "2026-05-03T23:30:00",
          }
      )

    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validation after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Document content cannot be empty")
    
    @property
    def page_number(self) -> Optional[int]:
        """Get page number from metadata if available."""
        return self.metadata.get('page')
    
    @property
    def source_file(self) -> str:
        """Get source file from metadata."""
        return self.metadata.get('source', 'Unknown')
    
    def get_citation(self) -> str:
        """
        Format citation string.
        
        Returns:
            Formatted citation like "[filename.pdf, page 5]"
        """
        from pathlib import Path
        page = self.page_number
        source = Path(self.source_file).name if self.source_file else 'Unknown'
        
        chunk_start = self.metadata.get('chunk_start')
        chunk_end = self.metadata.get('chunk_end')

        if page and chunk_start is not None and chunk_end is not None:
            return f"[{source}, page {page}, chars {chunk_start}-{chunk_end}]"
        if page:
            return f"[{source}, page {page}]"
        if chunk_start is not None and chunk_end is not None:
            return f"[{source}, chars {chunk_start}-{chunk_end}]"
        return f"[{source}]"
