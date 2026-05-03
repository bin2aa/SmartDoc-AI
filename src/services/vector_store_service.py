"""Vector store service cho SmartDoc AI, sử dụng FAISS.

CÁC THƯ VIỆN CHÍNH:
  - FAISS: Thư viện vector database của Facebook, dùng để lưu trữ và tìm kiếm
            vector embeddings. Mô tả: Tìm kiếm theo nghĩa semantic (văn bản).
  - BM25Retriever (langchain_community): Thư viện tìm kiếm theo từ khóa (keyword).
            BM25 là thuật toán xếp hạng tài liệu theo độ tương đồng bởi tần suất xuất hiện
            của các từ trong câu hỏi và nội dung tài liệu. Mô tả: Tìm kiếm theo
            ký tự chính xác (như Google search).
  - HuggingFaceEmbeddings: Mô hình embedding đa ngôn ngữ, chuyển văn bản thành vector
            768 chiều. Ví dụ: "chế biến" -> [0.1, -0.2, 0.5, ...]
  - CrossEncoder: Mô hình deep learning đóng vai trò re-ranker, chấm điểm độ liên quan
            giữa câu hỏi và từng tài liệu candidate, trả về điểm số chính xác hơn vector search.

VÍ DỤ THỰC TẾ CÁCH HOẠT ĐỘNG:
  Câu hỏi: "làm thế nào để cài đặt python"
  
  1. VECTOR SEARCH (Semantic): 
     - Embed câu hỏi: "làm thế nào để cài đặt python" -> [0.2, -0.1, ...]
     - Tìm kiếm FAISS: tìm các vector gần nhất trong database
     - Kết quả: Tài liệu nhận biết được "setup", "install", "python" 
                (ngay cả khi không có từ "cài đặt" nhưng vẫn cùng nghĩa)

  2. BM25 SEARCH (Keyword):
     - Phân tích câu hỏi: tách thành ["làm", "thế", "nào", "để", "cài", "đặt", "python"]
     - Tính điểm BM25 cho mỗi tài liệu trong database
     - Điểm = f(từ) * (k1+1) / (f(từ) + k1*(1-b+b*|doc|/|avg_doc|))
     - Kết quả: Tài liệu chứa đúng từ "cài đặt" + "python" sẽ có điểm cao nhất

  3. HYBRID MERGE (Ensemble):
     - Lấy kết quả từ 1 + kết quả từ 2, loại bỏ trùng lặp
     - Tài liệu xuất hiện trong cả 2 sẽ được ưu tiên cao hơn

  4. CROSS-ENCODER RERANK:
     - Gọi CrossEncoder chuẩn bị cặp [câu hỏi, tài liệu]
     - Mô hình deep learning đánh giá độ liên quan thật sự
     - Sắp xếp lại kết quả theo điểm mới
     - Kết quả chính xác hơn nhưng mất thời gian thêm

TẠI SAO CẦN HYBRID SEARCH?
  - Vector search: tốt cho các câu hỏi nghĩa, nhưng bỏ qua từ khóa chính xác
  - BM25: tốt cho tìm kiếm chính xác, nhưng không hiểu ngôn ngữ tự nhiên
  - Hybrid: kết hợp ưu điểm của cả hai, trả về kết quả tốt nhất của cả hai thế giới
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable
import os
import re
import time
import hashlib
from pathlib import Path
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import VectorStoreError
from src.utils.constants import EMBEDDING_MODEL

logger = setup_logger(__name__)

# Configure model cache directory and ensure it is writable.
def _resolve_cache_dir() -> Path:
    """Return a writable cache directory for HuggingFace models."""
    project_cache = Path(__file__).resolve().parents[2] / "models_cache"

    try:
        project_cache.mkdir(parents=True, exist_ok=True)
        test_file = project_cache / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return project_cache
    except Exception as permission_error:
        fallback = Path.home() / ".cache" / "smartdocai" / "models_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Project cache directory is not writable (%s). Falling back to %s",
            permission_error,
            fallback,
        )
        return fallback


_MODEL_CACHE_DIR = _resolve_cache_dir()
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_MODEL_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_MODEL_CACHE_DIR))


class AbstractVectorStoreService(ABC):
    """Abstract interface (hợp đồng) cho tất cả các vector store.

    Một abstract class là "hợp đồng" chỉ định các phương thức mà subclass
    bắt buộc phải implement. Nó chỉ định WHAT cần làm, không chỉ định HOW làm.

    VÍ DỤ:
        class Dog(ABC):
            @abstractmethod
            def speak(self):  # Bắt buộc implement trong subclass
                pass

        class Pug(Dog):
            def speak(self):  # Implement cụ thể
                print("Gâu gâu!")

    Trong dự án này:
        AbstractVectorStoreService chỉ định các phương thức:
          - add_documents(): thêm tài liệu vào vector store
          - similarity_search(): tìm tài liệu tương tự
          - search(): tìm kiếm nâng cao (hybrid, filter, rerank)
          - clear_store(): xóa vector store

        FAISSVectorStoreService là implement cụ thể của các phương thức này.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[LCDocument]) -> None:
        """Add documents to vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> List[LCDocument]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 3,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = False,
        use_bm25_only: bool = False,
        rerank: bool = False,
        fetch_k: int = 20,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Advanced retrieval with optional hybrid and reranking."""
        pass
    
    @abstractmethod
    def clear_store(self) -> None:
        """Clear the vector store."""
        pass


class LocalHashEmbeddings:
    """Deterministic local embeddings fallback that works fully offline."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, byteorder="big") % self.dimension
            vec[bucket] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class FAISSVectorStoreService(AbstractVectorStoreService):
    """
    FAISS vector store implementation.

    ĐÂY LÀ TRUNG TÂM CỦA HỆ THỐNG RAG - NÓ LƯU TRỮ TOÀN BỘ VECTOR EMBEDDINGS
    CỦA CÁC TÀI LIỆU ĐÃ UPLOAD.

    CÁC THÀNH PHẦN CHÍNH:
      1. self.vector_store (FAISS object):
         - Lưu trữ tất cả các vector embeddings của tài liệu
         - Hỗ trợ tìm kiếm nhanh bằng chỉ số của Facebook
         - Khỏe: 10,000 vectors vẫn còn rất nhanh (<100ms)

      2. self.embeddings (HuggingFaceEmbeddings):
         - Chuyển văn bản thành vector 768 chiều
         - Sử dụng model: paraphrase-multilingual-mpnet-base-v2
         - Chạy trên CPU (không cần GPU)

      3. self._bm25_retriever (BM25Retriever):
         - Chỉ số BM25 xây dựng từ toàn bộ tài liệu hiện tại
         - Hỗ trợ tìm kiếm keyword (không cần embedding)
         - Mỗi khi thêm tài liệu mới, phải rebuild chỉ số BM25

      4. self._cross_encoder (CrossEncoder):
         - Mô hình deep learning để re-rank kết quả
         - Lazy-load: chỉ tải khi cần sử dụng
         - Model: ms-marco-MiniLM-L-6-v2

    VÒNG ĐỜI CỦA MỘT TÀI LIỆU TRONG HỆ THỐNG:
      1. User upload PDF "report.pdf"
      2. DocumentService load và chunk thành 20 chunks
      3. Mỗi chunk được embed = [0.1, -0.2, ...] (768 chiều)
      4. Các vector được lưu vào FAISS index
      5. BM25 index được rebuild từ 20 chunks
      6. User hỏi: "tóm tắt bài báo này"
      7. Câu hỏi được embed, tìm kiếm FAISS + BM25
      8. Kết quả được merge và trả về
    """
    
    # Dù có lỗi gì thì hệ thống vẫn chạy được"
    """
    INIT
    ↓
    try load embedding (online)
    ↓ fail?
    ↓
    retry (reset client)
    ↓ fail?
        ↓
        load local cache
        ↓ fail?
            ↓
            fallback fake embedding
    """
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model: HuggingFace embedding model name
            
        Raises:
            VectorStoreError: If embedding model cannot be loaded
        """
        logger.info(f"Initializing FAISS with embedding model: {embedding_model}")
        self.embedding_model = embedding_model
        self.embeddings: Optional[Any] = None
        self.vector_store: Optional[FAISS] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._cross_encoder: Optional[Any] = None
        self._cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._init_error: Optional[str] = None
        self.using_offline_fallback: bool = False
        
        try:
            self.embeddings = self._create_embeddings(local_files_only=False)
            logger.info("FAISS vector store initialized successfully")
            
        except Exception as online_error:
            logger.warning(
                "Online embedding initialization failed for '%s': %s",
                embedding_model,
                online_error,
            )

            if "client has been closed" in str(online_error).lower():
                self._reset_hf_client()
                try:
                    self.embeddings = self._create_embeddings(local_files_only=False)
                    self.vector_store = None
                    logger.info("FAISS vector store initialized after resetting HuggingFace client")
                    return
                except Exception as retry_error:
                    logger.warning("Embedding init retry failed: %s", retry_error)

            try:
                self.embeddings = self._create_embeddings(local_files_only=True)
                logger.info("FAISS vector store initialized from local cache only")
                return
            except Exception as local_error:
                local_error_msg = str(local_error)
                logger.error(
                    "Failed to initialize FAISS vector store (online and local): %s | %s",
                    online_error,
                    local_error,
                )
                self._init_error = (
                    f"Failed to initialize embedding model '{embedding_model}'.\n"
                    f"Online init error: {online_error}\n"
                    f"Local-cache init error: {local_error_msg}\n"
                    f"Model cache path: {_MODEL_CACHE_DIR}\n"
                    "Please ensure internet is available for first download, or pre-download the model into the local cache."
                )
                logger.error(self._init_error)
                self.embeddings = LocalHashEmbeddings(dimension=768)
                self.using_offline_fallback = True
                logger.warning(
                    "Using LocalHashEmbeddings fallback (offline mode). Retrieval quality may be lower until HuggingFace model is available."
                )


    def _create_embeddings(self, local_files_only: bool) -> HuggingFaceEmbeddings:
        """Create embedding client with optional offline-only loading."""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            cache_folder=str(_MODEL_CACHE_DIR),
            model_kwargs={
                'device': 'cpu',
                'local_files_only': local_files_only,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32,
            },
        )

# Reset lại kết nối HTTP của HuggingFace khi client bị “chết” (stale/closed).
    @staticmethod
    def _reset_hf_client() -> None:
        """Reset HuggingFace HTTP sessions when a stale closed client is detected."""
        try:
            from huggingface_hub.utils._http import reset_sessions

            reset_sessions()
            logger.info("Reset HuggingFace Hub HTTP sessions")
        except Exception as reset_error:
            logger.debug("Unable to reset HuggingFace Hub sessions: %s", reset_error)
    
    def add_documents(self, documents: List[LCDocument]) -> None:
        """
        THÊM TÀI LIỆU VÀO VECTOR STORE.

        QUY TRÌNH CHI TIẾT:
          1. Nhận danh sách Document objects (mỗi Document = 1 chunk đã được embed)
          2. Chuyển đổi Document -> LCDocument (định dạng LangChain)
          3. Nếu FAISS index chưa có: tạo index mới từ đầu
          4. Nếu FAISS index đã có: thêm vector mới vào index cũ
          5. REBUILD BM25 INDEX: Quan trọng! Mỗi khi thêm vector,
             BM25 phải được xây dựng lại vì nó chứa toàn bộ tài liệu.

        TẠI SAO PHẢI REBUILD BM25?
          BM25 không lưu vector mà lưu chỉ số từ. Khi thêm tài liệu mới,
          chỉ số BM25 cũ phải được cập nhật với tài liệu mới, nếu không
          tài liệu mới sẽ không thể tìm kiếm bằng keyword được.

        Args:
            documents: Danh sách Document objects (mỗi object = 1 chunk)

        Raises:
            VectorStoreError: Nếu thêm documents thất bại
        """
        if not documents:
            logger.warning("No documents to add")
            return

        if self.embeddings is None:
            error_message = self._init_error or "Embedding model is not initialized"
            raise VectorStoreError(error_message)

        try:
            logger.info(f"Adding {len(documents)} documents to vector store")

            # Chuyển đổi từ domain Document (của tao) sang LangChain Document
            # Lý do: FAISS.from_documents() của LangChain yêu cầu LCDocument
            lc_docs = [
                LCDocument(page_content=doc.content, metadata=doc.metadata)
                for doc in documents
            ]

            if self.vector_store is None:
                # Lần đầu tiên: tạo FAISS index mới
                # from_documents() sẽ:
                #   1. Embed từng document = vector 768chiều
                #   2. Lưu các vector vào FAISS index
                self.vector_store = FAISS.from_documents(lc_docs, self.embeddings)
                logger.info("Created new FAISS index")
            else:
                # Thêm vào index cũ: chỉ cần embed documents mới
                self.vector_store.add_documents(lc_docs)
                logger.info("Added documents to existing FAISS index")

            # REBUILD BM25: Bắt buộc phải gọi sau khi add documents!
            # BM25 chỉ số từ vựng, phải cập nhật khi có tài liệu mới
            self._rebuild_bm25_retriever()

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[LCDocument]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar Document objects
        """
        docs, _ = self.search(query=query, k=k, use_hybrid=False, rerank=False)
        return docs

    def search(
        self,
        query: str,
        k: int = 3,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = False,
        use_bm25_only: bool = False,
        rerank: bool = False,
        fetch_k: int = 20,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        TÌM KIẾM NÂNG CAO - TRUNG TÂM CỦA HYBRID SEARCH.

        === 3 CHIẾN LƯỢC RETRIEVAL (chọn 1 trong 3) ===

        1. PURE VECTOR (use_hybrid=False, use_bm25_only=False):
           - Chỉ sử dụng embedding vector
           - Câu hỏi -> embed -> tìm vector gần nhất trong FAISS
           - Ưu điểm: Hiểu nghĩa semantic, nhận biết từ đồng nghĩa
           - Nhược: Bỏ qua từ khóa chính xác

        2. PURE BM25 (use_bm25_only=True):
           - Chỉ sử dụng chỉ số BM25
           - Tách câu hỏi thành từ, tính điểm BM25 cho mỗi tài liệu
           - Ưu điểm: Tìm kiếm từ khóa chính xác như Google
           - Nhược: Không hiểu nghĩa, phân biệt "tôi" vs "mùa"

        3. HYBRID (use_hybrid=True):
           - KHUYẾN NGHỊ - sử dụng cái này!
           - Chạy cả Vector và BM25, lấy kết quả của cả hai
           - Merge và loại bỏ trùng lặp
           - Kết quả tốt nhất của cả hai thế giới
           - Chi phí: gấp đôi thời gian (nếu là máy cùi)

        === 2 KỸ THUẬT PHỤ ===

        4. CROSS-ENCODER RERANK (rerank=True):
           - Sau khi có kết quả candidate, gọi CrossEncoder
           - Mô hình deep learning chấm điểm từng cặp [câu hỏi, tài liệu]
           - Sắp xếp lại kết quả theo điểm mới (chính xác hơn)
           - Chi phí: thêm 200-500ms nhưng chắc chắn hơn

        5. METADATA FILTER:
           - Lọc kết quả theo tên file hoặc loại file
           - Thực hiện SAU KHI retrieve, TRƯỚC KHI trả về
           - Không ảnh hưởng tới chủng loại tài liệu nào được tìm

        VÍ DỤ GỌI HÀM:
          # Tìm kiếm bình thường (vector only)
          docs, stats = service.search("làm thế nào để cài đặt python", k=5)

          # Tìm kiếm hybrid (vector + BM25)
          docs, stats = service.search(
              query="làm thế nào để cài đặt python",
              k=5,
              use_hybrid=True
          )

          # Tìm kiếm hybrid + rerank + filter
          docs, stats = service.search(
              query="làm thế nào để cài đặt python",
              k=5,
              use_hybrid=True,
              rerank=True,
              metadata_filters={"source_files": ["report.pdf"], "file_types": [".pdf"]}
          )

        Args:
            query: Câu hỏi người dùng (VD: "làm thế nào để cài đặt python")
            k: Số lượng tài liệu muốn lấy về (mặc định 3)
            metadata_filters: Lọc theo tên file hoặc loại file
            use_hybrid: True = kết hợp Vector + BM25
            use_bm25_only: True = chỉ BM25 (dùng cho benchmark)
            rerank: True = sử dụng CrossEncoder để re-rank kết quả
            fetch_k: Số lượng candidate lấy trước khi lọc/rerank (mặc định 20)

        Returns:
            Tuple: (danh sách Document tìm được, thống kê retrieval)
            Thống kê bao gồm:
              - vector_time_ms: thời gian vector search (ms)
              - bm25_time_ms: thời gian BM25 search (ms)
              - rerank_time_ms: thời gian re-rank (ms)
              - total_time_ms: tổng thời gian (ms)
              - overlap_count: số tài liệu trùng nhau giữa Vector và BM25
        """
        if self.vector_store is None:
            if self.embeddings is None:
                error_message = self._init_error or "Embedding model is not initialized"
                raise VectorStoreError(error_message)
            logger.warning("Vector store not initialized, returning empty results")
            return [], self._build_empty_stats(use_hybrid=use_hybrid, rerank=rerank)

        # === KHỞI TẠO STATS: Theo dõi thời gian của mỗi bước ===
        stats: Dict[str, Any] = {
            "use_hybrid": use_hybrid,
            "rerank": rerank,
            "vector_time_ms": 0.0,    # Thời gian vector search
            "bm25_time_ms": 0.0,       # Thời gian BM25 search
            "rerank_time_ms": 0.0,     # Thời gian re-rank
            "total_time_ms": 0.0,      # Tổng thời gian
            "vector_candidates": 0,      # Số lượng candidate từ vector
            "bm25_candidates": 0,       # Số lượng candidate từ BM25
            "merged_candidates": 0,     # Số lượng candidate sau merge
            "overlap_count": 0,         # Số tài liệu trùng nhau
            "results": 0,               # Số lượng trả về cuối cùng
        }

        try:
            # === BƯỚC 1: VECTOR SEARCH (luôn chạy đầu tiên) ===
            start = time.perf_counter()
            # Tìm kiếm k vector gần nhất trong FAISS
            vector_candidates = self._vector_search(query=query, k=max(k, fetch_k))
            stats["vector_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
            stats["vector_candidates"] = len(vector_candidates)

            # Khởi tạo kết quả merge = kết quả vector (sẽ thêm BM25 nếu cần)
            merged_candidates = vector_candidates
            bm25_candidates: List[Document] = []

            # === BƯỚC 2: BM25 SEARCH (nếu cần) ===
            if use_bm25_only:
                # Chỉ BM25: bỏ qua vector, chỉ lấy BM25
                # Dùng để so sánh với hybrid trong benchmark
                bm25_start = time.perf_counter()
                bm25_candidates = self._bm25_search(query=query, k=max(k, fetch_k))
                stats["bm25_time_ms"] = round((time.perf_counter() - bm25_start) * 1000, 2)
                stats["bm25_candidates"] = len(bm25_candidates)
                merged_candidates = bm25_candidates
                stats["overlap_count"] = 0
            elif use_hybrid:
                # HYBRID: Chạy BM25 rồi merge với vector
                bm25_start = time.perf_counter()
                bm25_candidates = self._bm25_search(query=query, k=max(k, fetch_k))
                stats["bm25_time_ms"] = round((time.perf_counter() - bm25_start) * 1000, 2)
                stats["bm25_candidates"] = len(bm25_candidates)

                # Merge: kết hợp 2 danh sách, loại bỏ trùng lặp
                merged_candidates = self._merge_results(vector_candidates, bm25_candidates)
                # Đếm số tài liệu trùng nhau giữa 2 phương pháp
                stats["overlap_count"] = self._count_overlap(vector_candidates, bm25_candidates)

            # === BƯỚC 3: METADATA FILTER (lọc theo file) ===
            if metadata_filters:
                # Lọc những tài liệu không thuộc file được chọn
                merged_candidates = self._apply_metadata_filters(merged_candidates, metadata_filters)

            stats["merged_candidates"] = len(merged_candidates)

            # === BƯỚC 4: CROSS-ENCODER RERANK (nếu cần) ===
            if rerank:
                rerank_start = time.perf_counter()
                # Sắp xếp lại candidate theo điểm CrossEncoder
                merged_candidates = self._rerank_documents(query, merged_candidates)
                stats["rerank_time_ms"] = round((time.perf_counter() - rerank_start) * 1000, 2)

            # === BƯỚC 5: TRẢ VỀ KẾT QUẢ CUỐI CÙNG ===
            # Chỉ lấy k tài liệu đầu tiên (đã được sort/re-rank)
            final_docs = merged_candidates[:k]
            stats["results"] = len(final_docs)
            # Tổng thời gian = vector + bm25 + rerank
            stats["total_time_ms"] = round(
                stats["vector_time_ms"] + stats["bm25_time_ms"] + stats["rerank_time_ms"],
                2,
            )

            logger.info(
                "Retrieval done | hybrid=%s bm25_only=%s rerank=%s results=%s vector_ms=%.2f bm25_ms=%.2f rerank_ms=%.2f",
                use_hybrid,
                use_bm25_only,
                rerank,
                len(final_docs),
                stats["vector_time_ms"],
                stats["bm25_time_ms"],
                stats["rerank_time_ms"],
            )
            return final_docs, stats
        except Exception as e:
            logger.error("Advanced search failed: %s", e)
            raise VectorStoreError(f"Search failed: {str(e)}")

    def _vector_search(self, query: str, k: int) -> List[Document]:
        """TÌM KIẾM VECTOR (SEMANTIC SEARCH).

        Bước 1 trong quy trình retrieval.

        CƠ CHẾ HOẠT ĐỘNG:
          1. Nhận câu hỏi người dùng: "làm thế nào để cài đặt python"
          2. Chuyển câu hỏi thành vector 768chiều bằng HuggingFace model
             embed("làm thế nào để cài đặt python") -> [0.2, -0.1, 0.7, ...]
          3. Tìm trong FAISS index những vector GẦN NHẤT với vector câu hỏi
             (gần nhất = tương tự nhất về nghĩa)
          4. Trả về k tài liệu có vector gần nhất

        VÍ DỤ:
          Câu hỏi: "hướng dẫn cài đặt python"
          Embedding: [0.123, -0.456, ...]
          FAISS tìm: [0.125, -0.458, ...] -> "Cài đặt Python trên Windows" (điểm 0.98)
                     [0.200, -0.300, ...] -> "Hướng dẫn lập trình Python" (điểm 0.85)
                     [0.150, -0.400, ...] -> "Python cơ bản" (điểm 0.82)

        ƯU ĐIỂM:
          - Hiểu nghĩa semantic (nhận biết "cài đặt" ~ "install" ~ "setup")
          - Không phân biệt ngữ pháp, từ đặng

        NHƯỢC ĐIỂM:
          - Có thể bỏ qua từ khóa chính xác
          - Gặp vấn đề với các thuật ngữ đặc thù, tên riêng
        """
        if self.using_offline_fallback:
            # Chế độ offline: sử dụng LocalHashEmbeddings thay vì HuggingFace
            # Tạo vector đơn giản từ hash của từ (không cần download model)
            query_embedding = self.embeddings.embed_query(query)
            results = self.vector_store.similarity_search_by_vector(query_embedding, k=k)
        else:
            # Chế độ bình thường: sử dụng HuggingFaceEmbeddings
            results = self.vector_store.similarity_search(query, k=k)
        docs = [Document(content=doc.page_content, metadata=doc.metadata) for doc in results]
        return docs

    def _all_lc_documents(self) -> List[LCDocument]:
        """LẤY TOÀN BỘ TÀI LIỆU TỪ FAISS DOCSTORE.

        FAISS lưu trữ 2 thứ:
          1. Index (vector embeddings): dùng để tìm kiếm nhanh
          2. Docstore (documents gốc): lưu nội dung văn bản gốc

        Hàm này lấy NỘI DUNG GỐC (chưa được embed) từ docstore.

        TẠI SAO CẦN DÙNG HÀM NÀY?
          BM25 cần danh sách văn bản gốc để xây dựng chỉ số từ vựng.
          Không thể embed lại vì BM25 làm việc với văn bản, không phải vector.
        """
        if self.vector_store is None:
            return []
        # Truy cập docstore._dict để lấy toàn bộ LangChain Document
        docstore_dict = getattr(getattr(self.vector_store, "docstore", None), "_dict", {})
        return list(docstore_dict.values()) if docstore_dict else []

    def _rebuild_bm25_retriever(self) -> None:
        """XÂY DỰNG / TÁI TẠO CHỈ SỐ BM25.

        BM25 (Best Matching 25) là thuật toán xếp hạng tài liệu theo từ khóa.

        CƠ CHẾ HOẠT ĐỘNG CỦA BM25:
          BM25 điểm = IDF(từ) * f(từ, Q) * (k1 + 1) / (f(từ, Q) + k1 * (1 - b + b * |D| / avgD))

        Trong đó:
          - IDF(từ): Inverse Document Frequency - từ hiếm gặp ở ít tài liệu = quan trọng hơn
          - f(từ, Q): Tần số xuất hiện của từ trong tài liệu Q
          - |D|: Độ dài tài liệu
          - avgD: Độ dài trung bình của tất cả tài liệu
          - k1, b: Hằng số (thường k1=1.5, b=0.75)

        VÍ DỤ:
          Câu hỏi: "cài đặt python"
          Tài liệu 1: "Hướng dẫn cài đặt Python nhanh" -> BM25 = 8.5 (nhiều từ khớp)
          Tài liệu 2: "Hướng dẫn lập trình" -> BM25 = 0.2 (không có từ nào khớp)
          Tài liệu 3: "Cài đặt Windows" -> BM25 = 4.0 (chỉ "cài đặt" khớp)

        TẠI SAO PHẢI REBUILD (TÁI TẠO)?
          Khi thêm tài liệu mới vào FAISS, chỉ số BM25 cũ không còn chứa tài liệu mới.
          Ví dụ: có 100 tài liệu, add thêm 10 tài liệu mới, BM25 chỉ chứa 100 tài liệu cũ.
          -> Phải tái tạo chỉ số BM25 từ 110 tài liệu.

        QUY TRÌNH REBUILD:
          1. Lấy toàn bộ tài liệu từ FAISS docstore (_all_lc_documents)
          2. Tạo BM25Retriever từ danh sách tài liệu đó
          3. Lưu vào self._bm25_retriever để sử dụng sau
        """
        try:
            all_docs = self._all_lc_documents()
            if not all_docs:
                self._bm25_retriever = None
                return
            # BM25Retriever.from_documents() tự động:
            #   1. Tokenize mỗi tài liệu (tách từ)
            #   2. Tính IDF cho mỗi từ
            #   3. Lưu chỉ số vào bộ nhớ
            retriever = BM25Retriever.from_documents(all_docs)
            retriever.k = min(20, len(all_docs))  # Số lượng kết quả tối đa
            self._bm25_retriever = retriever
            logger.info("BM25 retriever rebuilt with %s docs", len(all_docs))
        except Exception as bm25_error:
            self._bm25_retriever = None
            logger.warning("Could not rebuild BM25 retriever: %s", bm25_error)

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """TÌM KIẾM BM25 (KEYWORD SEARCH).

        Bước 2 trong quy trình hybrid retrieval.

        CƠ CHẾ HOẠT ĐỘNG:
          1. Nhận câu hỏi: "làm thế nào để cài đặt python"
          2. Tokenize câu hỏi: ["làm", "thế", "nào", "để", "cài", "đặt", "python"]
          3. Duyệt qua chỉ số BM25 đã xây dựng, tính điểm cho mỗi tài liệu:
             - Tài liệu có từ "python" + "cài đặt" -> điểm cao
             - Tài liệu có từ "python" nhưng không có "cài đặt" -> điểm thấp hơn
             - Tài liệu không có từ nào -> điểm = 0
          4. Trả về k tài liệu có điểm cao nhất

        KHÁC BIỆT VỚI VECTOR SEARCH:
          - Vector: tìm "nghĩa" tương tự ("cài đặt" ~ "install" ~ "setup")
          - BM25: tìm "từ khóa" chính xác ("cài đặt" != "install")

        VÍ DỤ:
          Câu hỏi: "install python"

          Vector tìm: "Cài đặt Python" (0.95), "Setup Python" (0.92), "Hướng dẫn Python" (0.88)
                     -> Không cần từ "install" nhưng vẫn tìm được vì nghĩa tương tự

          BM25 tìm: "Install Python here" (10.5), "How to install Python" (9.2), "Setup Python" (3.1)
                    -> Từ "install" chính xác có điểm cao nhất, "setup" điểm thấp hơn
                    (vì "setup" != "install" trong BM25)
        """
        if self._bm25_retriever is None:
            self._rebuild_bm25_retriever()
        if self._bm25_retriever is None:
            return []

        self._bm25_retriever.k = k
        results = self._bm25_retriever.invoke(query)
        return [Document(content=doc.page_content, metadata=doc.metadata) for doc in results]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """TẠO KEY DUY NHẤT CHO MỘT TÀI LIỆU (để loại bỏ trùng lặp).

        Khi merge kết quả từ Vector và BM25, cần một cách để nhận biết
        2 tài liệu có phải là cùng 1 tài liệu hay không.

        Key được tạo từ:
          - source: đường dẫn file (VD: "/uploads/report.pdf")
          - page: số trang (VD: "3")
          - chunk_index: vị trí chunk (VD: "5")
          - content_hash: MD5 hash của nội dung (VD: "a1b2c3d4...")

        VÍ DỤ:
          Tài liệu 1: source="/uploads/a.pdf", page="3", chunk="5", hash="abc123"
          Key 1 = "/uploads/a.pdf|3|5|abc123"

          Tài liệu 2: source="/uploads/a.pdf", page="3", chunk="5", hash="abc123"
          Key 2 = "/uploads/a.pdf|3|5|abc123"
          -> Key giống nhau -> là cùng 1 tài liệu -> loại bỏ

          Tài liệu 3: source="/uploads/a.pdf", page="3", chunk="6", hash="def456"
          Key 3 = "/uploads/a.pdf|3|6|def456"
          -> Key khác -> là tài liệu khác -> giữ lại
        """
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        content_hash = hashlib.md5(doc.content.encode("utf-8")).hexdigest()
        return f"{source}|{page}|{chunk_idx}|{content_hash}"

    def _merge_results(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        """GỘP KẾT QUẢ TỪ VECTOR VÀ BM25, LOẠI BỎ TRÙNG LẶP.

        Đây là bước CHÍNH của hybrid search - kết hợp 2 danh sách kết quả.

        CƠ CHẾ HOẠT ĐỘNG:
          1. Tạo một set để track đã thấy tài liệu nào
          2. Duyệt qua danh sách vector (ưu tiên trước)
          3. Duyệt qua danh sách BM25
          4. Nếu tài liệu chưa từng xuất hiện -> thêm vào danh sách gộp
          5. Trả về danh sách gộp (thứ tự: vector trước, BM25 bổ sung sau)

        TẠI SAO KHÔNG TÍNH ĐIỂM TRUNG BÌNH?
          Thực ra có 2 cách để gộp kết quả:
            1. Weighted average: điểm = 0.5*vector + 0.5*bm25 (phổ biến hơn)
            2. Union + Deduplicate: lấy cả 2 danh sách, loại trùng (dùng cách này)

          Cách này được chọn vì:
            - Đơn giản, dễ hiểu
            - Không cần normalize giữa 2 hệ thống chấm điểm khác nhau
            - Vẫn đảm bảo cả 2 phương pháp đều có tiếng nói

        ƯU ĐIỂM:
          - Tài liệu chỉ xuất hiện trong Vector sẽ có trong kết quả
          - Tài liệu chỉ xuất hiện trong BM25 sẽ có trong kết quả
          - Tài liệu xuất hiện trong cả 2 (overlap) sẽ xuất hiện 1 lần

        VÍ DỤ:
          Vector: [A, B, C, D]  (A, B, C cùng một tài liệu; D riêng)
          BM25:  [B, D, E, F]  (B, D cùng một tài liệu; E, F riêng)

          Merge:
            seen = {}
            -> Thêm A: seen={A}, merged=[A]
            -> Thêm B: seen={A,B}, merged=[A,B]
            -> Thêm C: seen={A,B,C}, merged=[A,B,C]
            -> Thêm D: seen={A,B,C,D}, merged=[A,B,C,D]
            -> Thêm E: seen={A,B,C,D,E}, merged=[A,B,C,D,E]
            -> Thêm F: seen={A,B,C,D,E,F}, merged=[A,B,C,D,E,F]

          Kết quả: [A, B, C, D, E, F] (6 tài liệu duy nhất)
        """
        merged: List[Document] = []
        seen: set = set()
        for doc in vector_docs + bm25_docs:
            key = self._doc_key(doc)
            if key in seen:
                continue  # Trùng lặp -> bỏ qua
            seen.add(key)
            merged.append(doc)
        return merged

    def _count_overlap(self, vector_docs: List[Document], bm25_docs: List[Document]) -> int:
        """ĐẾM SỐ TÀI LIỆU TRÙNG NHAU GIỮA VECTOR VÀ BM25.

        Overlap là chỉ số quan trọng để đo chênh lệch giữa 2 phương pháp:
          - Overlap cao (VD: 3/3 trong 3 kết quả) = 2 phương pháp đồng nhất
          - Overlap thấp (VD: 1/3 trong 3 kết quả) = 2 phương pháp bổ sung nhau

        VÍ DỤ:
          Vector: [A, B, C]  (A, B cùng; C riêng)
          BM25:   [A, B, D]  (A, B cùng; D riêng)

          vector_keys = {A, B, C}
          bm25_keys = {A, B, D}
          overlap = {A, B} = 2

        TẠI SAO CẦN STATS NÀY?
          Nếu overlap rất thấp, hybrid search có thể tốt hơn nhiều so với chỉ
          dùng vector hoặc chỉ dùng BM25. Nếu overlap rất cao, có thể chỉ
          cần dùng 1 phương pháp.
        """
        vector_keys = {self._doc_key(doc) for doc in vector_docs}
        bm25_keys = {self._doc_key(doc) for doc in bm25_docs}
        return len(vector_keys.intersection(bm25_keys))

    @staticmethod
    def _apply_metadata_filters(
        documents: List[Document],
        metadata_filters: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """LỌC KẾT QUẢ THEO METADATA (tên file, loại file).

        Lọc được thực hiện SAU KHI retrieve, TRƯỚC KHI trả về kết quả cuối.
        Không thay đổi số lượng tài liệu gốc trong FAISS.

        CƠ CHẾ HOẠT ĐỘNG:
          1. Nhận danh sách tài liệu từ retrieval (vector, BM25, hoặc hybrid)
          2. Kiểm tra từng tài liệu:
             - source_file: tên file có trong danh sách được lọc không?
             - file_type: loại file (.pdf, .docx, .txt) có trong danh sách không?
          3. Nếu không có filter nào -> trả về nguyên danh sách
          4. Nếu có filter -> chỉ giữ lại những tài liệu thỏa mãn

        LƯU Ý VỀ LOGIC LỌC:
          - source_files empty = không lọc theo tên file (giữ tất cả)
          - file_types empty = không lọc theo loại file (giữ tất cả)
          - Tất cả các filter đều là AND (phải thỏa mãn cả 2)

        VÍ DỤ:
          Tài liệu: [A.pdf, B.docx, C.pdf, D.txt]
          Filter: source_files=["A.pdf", "B.docx"], file_types=[".pdf"]
          Kết quả: [A.pdf] (thỏa mãn cả 2 filter)

          Tài liệu: [A.pdf, B.pdf, C.pdf]
          Filter: source_files=[], file_types=[".pdf"]  (không lọc theo tên)
          Kết quả: [A.pdf, B.pdf, C.pdf] (chỉ lọc theo loại file)
        """
        if not metadata_filters:
            return documents

        source_files = set(metadata_filters.get("source_files", []))
        file_types = set(metadata_filters.get("file_types", []))

        filtered: List[Document] = []
        for doc in documents:
            # Lấy tên file từ đường dẫn: "/uploads/report.pdf" -> "report.pdf"
            source = str(doc.metadata.get("source", ""))
            source_name = Path(source).name
            # Lấy loại file: ".pdf"
            file_type = str(doc.metadata.get("file_type", Path(source).suffix.lower()))

            # AND logic: phải thỏa mãn cả 2 filter
            if source_files and source_name not in source_files:
                continue  # Không thuộc file được chọn -> bỏ qua
            if file_types and file_type not in file_types:
                continue  # Không thuộc loại file được chọn -> bỏ qua
            filtered.append(doc)
        return filtered

    def _load_cross_encoder(self) -> Optional[Any]:
        """TẢI CROSS-ENCODER MODEL (LAZY LOAD).

        Cross-Encoder là mô hình deep learning có khả năng đánh giá
        độ liên quan thật sự giữa câu hỏi và tài liệu.

        TẠI SAO DÙNG LAZY LOAD?
          - CrossEncoder là model nhỏ nhưng vẫn tốn thời gian download
          - Nếu người dùng không bao giờ bật rerank, không cần tải model
          - Chỉ tải khi được gọi lần đầu tiên

        MODEL SỬ DỤNG: cross-encoder/ms-marco-MiniLM-L-6-v2
          - Được fine-tune trên Microsoft MARCO dataset (real search queries)
          - Nhận diện được độ liên quan thật sự giữa câu hỏi và tài liệu
          - Nhanh: chỉ cần một forward pass là có điểm (không cần search)

        VÍ DỤ:
          Câu hỏi: "làm thế nào để cài đặt python"
          Tài liệu 1: "Cài đặt Python nhanh" -> CrossEncoder điểm: 0.95
          Tài liệu 2: "Hướng dẫn lập trình Python" -> CrossEncoder điểm: 0.72
          Tài liệu 3: "Cài đặt Windows" -> CrossEncoder điểm: 0.31
        """
        if self._cross_encoder is not None:
            return self._cross_encoder  # Đã tải rồi -> trả về cache
        try:
            from sentence_transformers import CrossEncoder

            # Tải CrossEncoder model từ HuggingFace
            # Nếu đã có trong cache -> sử dụng local
            self._cross_encoder = CrossEncoder(self._cross_encoder_name)
            logger.info("Cross-encoder loaded: %s", self._cross_encoder_name)
        except Exception as error:
            logger.warning("Cross-encoder unavailable, skipping rerank: %s", error)
            self._cross_encoder = None
        return self._cross_encoder

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """SẮP XẾP LẠI KẾT QUẢ BẰNG CROSS-ENCODER (RE-RANKING).

        Bước cuối cùng của retrieval pipeline khi rerank=True.

        CƠ CHẾ HOẠT ĐỘNG:
          1. Tạo cặp [câu hỏi, tài liệu] cho từng tài liệu candidate
          2. Gọi CrossEncoder.predict() để lấy điểm liên quan
          3. Sắp xếp lại danh sách theo điểm mới (cao -> thấp)

        TẠI SAO CẦN RE-RANK?
          Vector search + BM25 tìm nhanh nhưng chỉ là "presort" (sắp xếp sơ bộ).
          Cross-Encoder chấm điểm chính xác hơn vì:
            - Sử dụng mô hình deep learning đã được train
            - Hiểu được ngôn ngữ tự nhiên tốt hơn
            - Có thể nhận biết "python cài đặt" vs "python là ngôn ngữ"

        VÒNG ĐỜI MỘT CROSS-ENCODER CALL:
          pairs = [
            ["làm thế nào để cài đặt python", "Cài đặt Python nhanh nhất..."],
            ["làm thế nào để cài đặt python", "Hướng dẫn lập trình Python cơ bản..."],
          ]
          scores = cross_encoder.predict(pairs)
          # scores = [0.95, 0.72]
          # Kết quả: ["Cài đặt Python nhanh nhất...", "Hướng dẫn lập trình Python cơ bản..."]
        """
        if not documents:
            return []

        cross_encoder = self._load_cross_encoder()
        if cross_encoder is None:
            return documents  # Không có CrossEncoder -> trả về như cũ

        # Tạo cặp [câu hỏi, tài liệu] cho CrossEncoder
        pairs = [[query, doc.content] for doc in documents]
        # CrossEncoder trả về điểm cho từng cặp
        scores = cross_encoder.predict(pairs)
        # Sắp xếp theo điểm giảm dần (điểm cao = liên quan hơn)
        ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)

        # Tạo danh sách tài liệu đã sắp xếp, thêm metadata rerank
        reranked_docs: List[Document] = []
        for rank, (doc, score) in enumerate(ranked, start=1):
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = float(score)   # Điểm CrossEncoder
            metadata["rerank_rank"] = rank             # Thứ tự sau re-rank
            reranked_docs.append(Document(content=doc.content, metadata=metadata))
        return reranked_docs

    @staticmethod
    def _build_empty_stats(use_hybrid: bool, rerank: bool) -> Dict[str, Any]:
        """Build empty retrieval statistics payload."""
        return {
            "use_hybrid": use_hybrid,
            "rerank": rerank,
            "vector_time_ms": 0.0,
            "bm25_time_ms": 0.0,
            "rerank_time_ms": 0.0,
            "total_time_ms": 0.0,
            "vector_candidates": 0,
            "bm25_candidates": 0,
            "merged_candidates": 0,
            "overlap_count": 0,
            "results": 0,
        }
    
    def clear_store(self) -> None:
        """Clear the vector store."""
        logger.warning("Clearing vector store")
        self.vector_store = None
        self._bm25_retriever = None
        logger.info("Vector store cleared")
    
    @property
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self.vector_store is not None


# ---------------------------------------------------------------------------
# Retrieval Benchmark
# ---------------------------------------------------------------------------

class RetrievalBenchmark:
    """
    SO SÁNH 3 CHIẾN LƯỢC RETRIEVAL.

    Đây là class dùng để so sánh hiệu suất giữa 3 phương pháp retrieval:
      1. Pure Vector (semantic search): Tìm theo nghĩa, sử dụng vector embedding
      2. Pure BM25 (keyword search): Tìm theo từ khóa chính xác
      3. Hybrid (Vector + BM25 ensemble): Kết hợp cả 2

    === 3 METRICS DÙNG ĐỂ SO SÁNH ===

    1. Recall@K:
       - Tỷ lệ tài liệu liên quan được recall trong top-K kết quả
       - Giá trị: 0.0 -> 1.0 (cao hơn tốt hơn)
       - Công thức: Recall@K = (số tài liệu liên quan trong top-K) / (tổng số tài liệu liên quan)
       - Ví dụ: Có 10 tài liệu liên quan trong 100 tài liệu.
                  Top-5 trả về 3 tài liệu liên quan.
                  Recall@5 = 3/10 = 0.3 = 30%

    2. Speed (Thời gian - ms):
       - Thời gian phản hồi của mỗi phương pháp
       - Giá trị: ms (thấp hơn nhanh hơn)
       - Vector thường nhanh hơn BM25, hybrid chậm nhất

    3. Coverage (Số tài liệu duy nhất):
       - Số nguồn tài liệu duy nhất trong top-K kết quả
       - Giá trị: 1 -> K (cao hơn tốt hơn)
       - Ví dụ: Top-3 trả về 3 tài liệu từ 3 file khác nhau = coverage 3
                  Top-3 trả về 3 tài liệu từ 1 file = coverage 1

    === CÁCH TÍNH GROUND TRUTH ===

    "Tài liệu liên quan" được xác định bằng cách:
      1. Loại bỏ stop words (the, a, is, và, của, ...) khỏi câu hỏi
      2. Lấy các từ còn lại làm keyword
      3. Duyệt qua tất cả tài liệu trong database
      4. Nếu tài liệu chứa bất kỳ từ nào trong keyword -> là "liên quan"

    Ví dụ:
      Câu hỏi: "làm thế nào để cài đặt python trên windows"
      Stop words: ["làm", "thế", "nào", "để", "trên"] (loại bỏ)
      Keywords: {"cài", "đặt", "python", "windows"}

      Tài liệu 1: "Cài đặt Python nhanh" -> chứa "cài", "đặt", "python" -> LIÊN QUAN
      Tài liệu 2: "Hướng dẫn lập trình" -> chỉ chứa "python" -> LIÊN QUAN (nửa)
      Tài liệu 3: "Trò chơi Windows" -> chỉ chứa "windows" -> LIÊN QUAN (ít)
      Tài liệu 4: "Hướng dẫn Linux" -> không chứa từ nào -> KHÔNG LIÊN QUAN

    === SỬ DỤNG ===

    Từ Settings -> Retrieval Strategy -> Run Retrieval Benchmark

    Nhập câu hỏi -> Bấm Run -> Hệ thống tự động so sánh 3 chiến lược
    """

    def __init__(self, vector_service: "FAISSVectorStoreService"):
        self.vector_service = vector_service

    def _compute_recall_at_k(
        self,
        query: str,
        k: int,
        strategy: str,
    ) -> Tuple[float, List[Document], Dict[str, float]]:
        """
        TÍNH RECALL@K CHO MỘT CHIẾN LƯỢC RETRIEVAL.

        QUY TRÌNH CHI TIẾT 6 BƯỚC:

          BƯỚC 1: XÁC ĐỊNH KEYWORDS TỪ CÂU HỎI
            - Loại bỏ stop words (the, a, is, và, của, ...) khỏi câu hỏi
            - Lấy các từ còn lại làm "keyword" để đánh giá liên quan
            - Ví dụ: "làm thế nào để cài đặt python" -> {"cài", "đặt", "python"}

          BƯỚC 2: TÌM TẤT CẢ TÀI LIỆU LIÊN QUAN (GROUND TRUTH)
            - Duyệt qua TẤT CẢ tài liệu trong database (không phải chỉ top-K)
            - Nếu tài liệu nào chứa bất kỳ keyword nào -> là "liên quan"
            - Ví dụ: tài liệu chứa "python" -> liên quan
                     tài liệu chứa "cài đặt" -> liên quan

          BƯỚC 3: CHẠY RETRIEVAL THEO CHIẾN LƯỢC
            - strategy="vector": chỉ sử dụng FAISS vector search
            - strategy="bm25": chỉ sử dụng BM25 keyword search
            - strategy="hybrid": kết hợp cả 2 (Vector + BM25)

          BƯỚC 4: ĐẾM TÀI LIỆU LIÊN QUAN TRONG TOP-K
            - Sau khi có kết quả top-K, đếm xem có bao nhiêu là "liên quan"

          BƯỚC 5: TÍNH RECALL@K
            - Công thức: Recall@K = (tài liệu liên quan trong top-K) / (tổng tài liệu liên quan)
            - Ví dụ:
              Có 20 tài liệu liên quan trong 100 tài liệu.
              Top-5 trả về 2 tài liệu liên quan.
              Recall@5 = 2/20 = 0.1 = 10%

          BƯỚC 6: TRẢ VỀ KẾT QUẢ
            - recall_score: điểm Recall@K (0.0 -> 1.0)
            - docs: danh sách tài liệu được trả về
            - timing: thống kê thời gian

        STOP WORDS:
          Là những từ thường gặp nhưng không mang ý nghĩa quan trọng.
          VD: "the", "a", "is", "và", "của", "là", "có", "trong"...
          Đóng vai trò như "rổ lọc": loại bỏ nhiễu, giữ lại ý nghĩa.

        Args:
            query: Câu hỏi truy vấn (VD: "làm thế nào để cài đặt python")
            k: Số document muốn lấy về (VD: k=5 -> lấy top-5)
            strategy: Chọn 1 trong 3:
              - "vector": Pure vector search (semantic)
              - "bm25": Pure BM25 search (keyword)
              - "hybrid": Vector + BM25 ensemble

        Returns:
            Tuple: (recall_score, retrieved_docs, timing_stats)
              - recall_score: điểm Recall@K (float 0.0-1.0)
              - retrieved_docs: danh sách Document trả về
              - timing_stats: thống kê thời gian (time_ms)
        """
        all_docs = self.vector_service._all_lc_documents()
        if not all_docs:
            return 0.0, [], {"time_ms": 0.0}

        # === BƯỚC 1: TRÍCH XUẤT KEYWORDS TỪ CÂU HỎI ===
        # Loại bỏ stop words (như "the", "a", "is", "và", "của"...)
        # Chỉ giữ lại những từ dài hơn 1 ký tự và không phải stop word
        # Ví dụ: "làm thế nào để cài đặt python"
        #   -> loại bỏ: "làm", "thế", "nào", "để"
        #   -> giữ lại: "cài", "đặt", "python"
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "of", "at", "by", "for", "with",
            "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "just", "don", "now", "và", "của", "là", "có", "trong", "được",
            "cho", "với", "này", "đó", "những", "một", "các", "tôi", "bạn",
            "anh", "chị", "họ", "nó", "đi", "về", "từ", "ra", "vào", "lên",
        }
        # Trích xuất từ (tokenize): tách câu hỏi thành danh sách từ
        # VD: "làm thế nào" -> ["làm", "thế", "nào"]
        # Lọc: bỏ stop words, chỉ giữ từ dài hơn 1 ký tự
        query_tokens = {
            t.lower() for t in re.findall(r"\w+", query)
            if t.lower() not in stop_words and len(t) > 1
        }

        # === BƯỚC 2: TÌM GROUND TRUTH (TÀI LIỆU LIÊN QUAN) ===
        # Duyệt qua TẤT CẢ tài liệu trong database
        # Nếu tài liệu nào chứa bất kỳ keyword nào -> là "liên quan"
        # Sử dụng MD5 hash để tạo key duy nhất cho mỗi tài liệu
        ground_truth_keys = set()
        for idx, doc in enumerate(all_docs):
            content_lower = doc.page_content.lower()
            # any(): True nếu CÓ ÍT NHẤT 1 keyword xuất hiện trong tài liệu
            if any(token in content_lower for token in query_tokens):
                key = hashlib.md5(doc.page_content.encode()).hexdigest()
                ground_truth_keys.add(key)

        total_relevant = len(ground_truth_keys)  # Tổng số tài liệu liên quan

        # === BƯỚC 3: CHẠY RETRIEVAL THEO CHIẾN LƯỢC ===
        start = time.perf_counter()
        if strategy == "vector":
            # PURE VECTOR: chỉ sử dụng FAISS semantic search
            docs, stats = self.vector_service.search(
                query=query, k=k, use_hybrid=False, use_bm25_only=False, rerank=False
            )
        elif strategy == "bm25":
            # PURE BM25: chỉ sử dụng BM25 keyword search
            docs, stats = self.vector_service.search(
                query=query, k=k, use_hybrid=False, use_bm25_only=True, rerank=False
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            stats = {**stats, "time_ms": elapsed_ms}
        elif strategy == "hybrid":
            # HYBRID: kết hợp vector + BM25
            docs, stats = self.vector_service.search(
                query=query, k=k, use_hybrid=True, use_bm25_only=False, rerank=False
            )
        else:
            docs, stats = [], {}
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not docs:
            return 0.0, [], {"time_ms": elapsed_ms}

        # === BƯỚC 4 & 5: TÍNH RECALL@K ===
        retrieved_relevant = 0
        for doc in docs:
            # Tạo MD5 hash của tài liệu trả về
            key = hashlib.md5(doc.content.encode()).hexdigest()
            # Nếu tài liệu này có trong ground truth -> đếm thêm 1
            if key in ground_truth_keys:
                retrieved_relevant += 1

        # Recall@K = tài liệu liên quan trong top-K / tổng tài liệu liên quan
        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

        timing = dict(stats)
        timing["time_ms"] = round(elapsed_ms, 2)
        # === BƯỚC 6: TRẢ VỀ KẾT QUẢ ===
        return recall, docs, timing

    def run(
        self,
        query: str,
        k: int = 5,
        show_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        CHẠY BENCHMARK ĐẦY ĐỦ CHO 3 CHIẾN LƯỢC.

        QUY TRÌNH:
          1. Kiểm tra vector store có rỗng không
          2. Chạy _compute_recall_at_k cho từng chiến lược (vector, bm25, hybrid)
          3. Tổng hợp kết quả vào dictionary
          4. Tìm chiến lược tốt nhất theo từng metric

        KẾT QUẢ TRẢ VỀ BAO GỒM:
          - strategies: chi tiết kết quả của từng chiến lược
            - recall_at_k: điểm Recall@K
            - time_ms: thời gian phản hồi
            - unique_sources: số nguồn tài liệu duy nhất
            - top_docs: 3 tài liệu đầu tiên (preview)
          - best: chiến lược tốt nhất theo từng metric
            - recall: Vector | BM25 | Hybrid
            - speed: Vector | BM25 | Hybrid
            - coverage: Vector | BM25 | Hybrid

        VÍ DỤ KẾT QUẢ:
          {
            "strategies": {
              "vector": {"recall_at_k": 0.45, "time_ms": 45.2, "unique_sources": 3, ...},
              "bm25":    {"recall_at_k": 0.38, "time_ms": 12.1, "unique_sources": 2, ...},
              "hybrid":  {"recall_at_k": 0.61, "time_ms": 58.3, "unique_sources": 4, ...},
            },
            "best": {
              "recall": "hybrid",
              "speed": "bm25",
              "coverage": "hybrid",
            }
          }

        Args:
            query: Câu hỏi dùng để so sánh
            k: Số document lấy về (VD: k=5 -> lấy top-5)
            show_progress: Callback hiển thị tiến trình (Streamlit spinner)

        Returns:
            Dict chứa kết quả của cả 3 chiến lược và bảng so sánh
        """
        if self.vector_service.vector_store is None:
            return {"error": "Vector store is empty. Please upload documents first."}

        # 3 chiến lược cần so sánh
        strategies = ["vector", "bm25", "hybrid"]
        results: Dict[str, Dict[str, Any]] = {}

        # Nhãn cho hiển thị (hiển thị trong bảng so sánh)
        summary_labels = {
            "vector": "Vector (Semantic)",
            "bm25": "BM25 (Keyword)",
            "hybrid": "Hybrid (Ensemble)",
        }

        # === CHẠY TỪNG CHIẾN LƯỢC ===
        for strat in strategies:
            if show_progress:
                show_progress(f"Testing: {summary_labels.get(strat, strat)}...")

            # Tính Recall@K cho chiến lược này
            recall, docs, timing = self._compute_recall_at_k(query, k, strat)

            # Tổng hợp kết quả vào dictionary
            results[strat] = {
                "strategy": strat,
                "label": summary_labels.get(strat, strat),
                "recall_at_k": round(recall, 4),          # Điểm Recall@K
                "docs_retrieved": len(docs),               # Số lượng tài liệu trả về
                "unique_sources": len({d.metadata.get("source_file", "") for d in docs}),  # Số nguồn khác nhau
                "time_ms": timing.get("time_ms", 0.0),    # Tổng thời gian (ms)
                "vector_time_ms": timing.get("vector_time_ms", 0.0),
                "bm25_time_ms": timing.get("bm25_time_ms", 0.0),
                # Preview 3 tài liệu đầu tiên (hiển thị trong UI)
                "top_docs": [
                    {
                        "source": d.metadata.get("source_file", "unknown"),
                        "page": d.metadata.get("page"),
                        "preview": d.content[:150].replace("\n", " ") + "...",
                    }
                    for d in docs[:3]
                ],
            }

        # === TÌM CHIẾN LƯỢC TỐT NHẤT THEO TỪNG METRIC ===
        # max() / min() tìm key có giá trị lớn nhất / nhỏ nhất
        best_by_recall = max(results, key=lambda s: results[s]["recall_at_k"])
        best_by_speed = min(results, key=lambda s: results[s]["time_ms"])
        best_by_coverage = max(results, key=lambda s: results[s]["unique_sources"])

        # === ĐÁNH DẤU CHIẾN LƯỢC TỐT NHẤT ===
        for strat, res in results.items():
            res["is_best_recall"] = strat == best_by_recall
            res["is_best_speed"] = strat == best_by_speed
            res["is_best_coverage"] = strat == best_by_coverage

        # === TRẢ VỀ KẾT QUẢ CUỐI CÙNG ===
        return {
            "query": query,
            "k": k,
            "query_tokens": list({
                t for t in re.findall(r"\w+", query.lower())
                if len(t) > 1
            }),
            "strategies": results,
            "best": {
                "recall": best_by_recall,
                "speed": best_by_speed,
                "coverage": best_by_coverage,
            },
        }
