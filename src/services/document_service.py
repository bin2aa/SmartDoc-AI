"""Document service for loading and processing documents."""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import DocumentLoadError
from src.utils.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = setup_logger(__name__)


class DocumentLoaderFactory:
    """Factory pattern for document loaders."""
    
    LOADERS = {
        '.pdf': PDFPlumberLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
    }
    
    @classmethod
    def create_loader(cls, file_path: str):
        """
        Create appropriate loader based on file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document loader instance
            
        Raises:
            DocumentLoadError: If file type not supported
        """
        extension = Path(file_path).suffix.lower()
        loader_class = cls.LOADERS.get(extension)
        
        if not loader_class:
            raise DocumentLoadError(
                f"File type {extension} not supported. "
                f"Supported types: {list(cls.LOADERS.keys())}"
            )
        
        logger.info(f"Creating {loader_class.__name__} for {file_path}")
        return loader_class(file_path)


# Define a simple text splitter as a replacement for RecursiveCharacterTextSplitter
class SimpleTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        chunks = []
        for doc in documents:
            text = doc.page_content
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                chunk_text = text[i:i + self.chunk_size]
                if not chunk_text.strip():
                    continue
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_start": i,
                    "chunk_end": i + len(chunk_text),
                }
                chunks.append(LCDocument(page_content=chunk_text, metadata=chunk_metadata))
        return chunks


class DocumentService:
    """Service for document loading and processing."""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        Initialize document service.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = SimpleTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"DocumentService initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        LOAD MỘT TÀI LIỆU, TRÍCH XUẤT METADATA GIÀU, VÀ CHIA THÀNH CHUNKS.

        ĐÂY LÀ HÀM DÙNG ĐỂ XỬ LÝ MỘT FILE PDF/DOCX/TXT SAU KHI UPLOAD.

        QUY TRÌNH 4 BƯỚC:

          BƯỚC 1: ĐỌC FILE
            - Sử dụng loader phù hợp (PDFPlumberLoader, Docx2txtLoader, TextLoader)
            - Trả về danh sách LCDocument (định dạng LangChain Document)

          BƯỚC 2: CHIA THÀNH CHUNKS
            - Sử dụng SimpleTextSplitter để chia văn bản thành các đoạn nhỏ
            - Mỗi chunk có độ dài = chunk_size (VD: 1000 ký tự)
            - Các chunk có overlap = chunk_overlap (VD: 100 ký tự) để đảm bảo tính liên tục

          BƯỚC 3: BỔ SUNG METADATA GIÀU
            - Mỗi chunk được thêm các trường metadata:
              * source: đường dẫn đầy đủ đến file (VD: "/uploads/report.pdf")
              * source_file: chỉ tên file (VD: "report.pdf")
              * file_type: phần mở rộng (VD: ".pdf")
              * uploaded_at: thời gian upload (định dạng ISO)
              * chunk_index: vị trí chunk trong file (0, 1, 2, ...)
              * chunk_start, chunk_end: vị trí bắt đầu/kết thúc trong văn bản gốc
              * file_size_bytes, file_size_mb: kích thước file
              * title: tiêu đề document (được rút gọn từ tên file)
              * page_count: số trang/section
              * total_chunks: tổng số chunks tạo ra từ file này

          BƯỚC 4: TRẢ VỀ DANH SÁCH DOCUMENT
            - Trả về List[Document] (domain model, không phải LCDocument)
            - Mỗi Document chứa: content (nội dung chunk) + metadata (thông tin phong phú)

        TẠI SAO METADATA QUAN TRỌNG?
          Metadata được sử dụng để:
            1. LỌC (filter): lọc tài liệu theo tên file hoặc loại file
            2. HIỂN THỊ: hiển thị thông tin tài liệu nguồn trong chat
            3. CITATION: tạo trích dẫn như "[report.pdf, page 3, chunk 2]"
            4. DEBUG: kiểm tra xem tài liệu nào được upload khi nào, kích thước bao nhiêu

        VÍ DỤ METADATA:
          Document chunk 3 của file "report.pdf":
          {
            "content": "Nội dung trang 3...",
            "metadata": {
              "source": "/uploads/report.pdf",
              "source_file": "report.pdf",
              "file_type": ".pdf",
              "uploaded_at": "2026-05-03T23:30:00",
              "chunk_index": 3,
              "chunk_start": 3000,
              "chunk_end": 4000,
              "file_size_bytes": 1048576,
              "file_size_mb": 1.0,
              "title": "report",
              "page_count": 10,
              "total_chunks": 15
            }
          }

        Args:
            file_path: Đường dẫn đầy đủ đến file (VD: "/uploads/report.pdf")

        Returns:
            List[Document]: Danh sách Document objects (mỗi object = 1 chunk)

        Raises:
            DocumentLoadError: Nếu load thất bại (file không tồn tại, format không hỗ trợ, ...)
        """
        try:
            logger.info(f"Loading document: {file_path}")

            source_path = Path(file_path)
            upload_time = datetime.now().isoformat()
            file_size_bytes = source_path.stat().st_size
            file_name = source_path.name
            file_ext = source_path.suffix.lower()

            # Load document content
            loader = DocumentLoaderFactory.create_loader(file_path)
            lc_docs = loader.load()
            num_pages = len(lc_docs)

            # Split into chunks
            chunks = self.text_splitter.split_documents(lc_docs)
            total_chunks = len(chunks)
            logger.info(f"Split into {total_chunks} chunks")

            # Derive a readable title from the filename
            title = file_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")

            # Build enriched document chunks
            documents = [
                Document(
                    content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "source": file_path,
                        "source_file": file_name,
                        "file_type": file_ext,
                        "uploaded_at": upload_time,
                        "chunk_index": idx,
                        "file_size_bytes": file_size_bytes,
                        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                        "title": title,
                        "page_count": num_pages,
                        "total_chunks": total_chunks,
                    }
                )
                for idx, chunk in enumerate(chunks)
            ]

            logger.info(f"Successfully processed {file_path}: {len(documents)} chunks, "
                        f"{num_pages} pages, {file_size_bytes} bytes")
            return documents

        except DocumentLoadError:
            raise
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise DocumentLoadError(f"Cannot find file: {file_path}")
        except Exception as e:
            logger.exception(f"Unexpected error loading {file_path}")
            raise DocumentLoadError(f"Failed to load document: {str(e)}")
    
    def update_chunk_config(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Update chunking configuration.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = SimpleTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"Updated chunk config: size={chunk_size}, overlap={chunk_overlap}")
