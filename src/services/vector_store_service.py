"""Vector store service for SmartDoc AI using FAISS."""

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
    """Abstract interface for vector store operations."""
    
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
    
    Uses HuggingFace embeddings with multilingual support.
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
        Add documents to vector store.

        Prepends a ``[File: <filename>]`` prefix to each chunk's
        ``page_content`` before embedding so that vectors from different
        files are separated in embedding space.  The original content is
        preserved in metadata field ``original_content``.

        Args:
            documents: List of Document objects to add

        Raises:
            VectorStoreError: If adding documents fails
        """
        if not documents:
            logger.warning("No documents to add")
            return

        if self.embeddings is None:
            error_message = self._init_error or "Embedding model is not initialized"
            raise VectorStoreError(error_message)

        try:
            logger.info(f"Adding {len(documents)} documents to vector store")

            # Convert to LangChain document format with filename prefix
            # for better vector separation between different files.
            lc_docs: List[LCDocument] = []
            for doc in documents:
                source_name = Path(doc.metadata.get("source", "")).name or "unknown"
                prefixed_content = f"[File: {source_name}]\n{doc.content}"

                enriched_metadata = {
                    **doc.metadata,
                    "original_content": doc.content,
                }
                lc_docs.append(
                    LCDocument(page_content=prefixed_content, metadata=enriched_metadata)
                )

            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(lc_docs, self.embeddings)
                logger.info("Created new FAISS index with filename-prefixed chunks")
            else:
                self.vector_store.add_documents(lc_docs)
                logger.info("Added documents to existing FAISS index")

            # BM25 needs to be rebuilt from all current docs.
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
        """Advanced retrieval supporting hybrid search, filtering and reranking.

        Retrieval strategies (mutually exclusive):
        - Pure Vector: use_hybrid=False, use_bm25_only=False
        - Pure BM25 (keyword only): use_bm25_only=True
        - Hybrid (Vector + BM25 ensemble): use_hybrid=True

        Args:
            query: Search query string
            k: Number of final documents to return
            metadata_filters: Optional filters by source_file or file_type
            use_hybrid: If True, combine vector and BM25 results
            use_bm25_only: If True, return only BM25 results (for benchmark comparison)
            rerank: If True, apply cross-encoder reranking
            fetch_k: Number of candidates to fetch before filtering/reranking

        Returns:
            Tuple of (retrieved_documents, retrieval_statistics_dict)
        """
        if self.vector_store is None:
            if self.embeddings is None:
                error_message = self._init_error or "Embedding model is not initialized"
                raise VectorStoreError(error_message)
            logger.warning("Vector store not initialized, returning empty results")
            return [], self._build_empty_stats(use_hybrid=use_hybrid, rerank=rerank)

        stats: Dict[str, Any] = {
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

        try:
            start = time.perf_counter()
            vector_candidates = self._vector_search(
                query=query, k=max(k, fetch_k), metadata_filters=metadata_filters,
            )
            stats["vector_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
            stats["vector_candidates"] = len(vector_candidates)

            merged_candidates = vector_candidates
            bm25_candidates: List[Document] = []

            if use_bm25_only:
                # Pure BM25 strategy: skip vector search entirely
                bm25_start = time.perf_counter()
                bm25_candidates = self._bm25_search(query=query, k=max(k, fetch_k))
                stats["bm25_time_ms"] = round((time.perf_counter() - bm25_start) * 1000, 2)
                stats["bm25_candidates"] = len(bm25_candidates)
                merged_candidates = bm25_candidates
                stats["overlap_count"] = 0
            elif use_hybrid:
                bm25_start = time.perf_counter()
                bm25_candidates = self._bm25_search(query=query, k=max(k, fetch_k))
                stats["bm25_time_ms"] = round((time.perf_counter() - bm25_start) * 1000, 2)
                stats["bm25_candidates"] = len(bm25_candidates)

                merged_candidates = self._merge_results(vector_candidates, bm25_candidates)
                stats["overlap_count"] = self._count_overlap(vector_candidates, bm25_candidates)

            if metadata_filters:
                merged_candidates = self._apply_metadata_filters(merged_candidates, metadata_filters)

            stats["merged_candidates"] = len(merged_candidates)

            if rerank:
                rerank_start = time.perf_counter()
                merged_candidates = self._rerank_documents(query, merged_candidates)
                stats["rerank_time_ms"] = round((time.perf_counter() - rerank_start) * 1000, 2)

            final_docs = merged_candidates[:k]
            stats["results"] = len(final_docs)
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

    @staticmethod
    def _build_faiss_filter(
        metadata_filters: Optional[Dict[str, Any]],
    ) -> Optional[Callable[[Dict[str, Any]], bool]]:
        """Build a FAISS-compatible filter function from metadata_filters.

        FAISS ``similarity_search`` accepts a ``filter`` callable that
        receives a doc's metadata dict and returns True if the doc should
        be included.  This is **pre-retrieval** filtering — only matching
        vectors are searched, avoiding the problem where non-matching
        docs fill up the k slots.

        Args:
            metadata_filters: Dict with optional ``source_files`` and ``file_types`` lists

        Returns:
            A filter callable or None if no filters are needed
        """
        if not metadata_filters:
            return None

        source_files = set(metadata_filters.get("source_files", []))
        file_types = set(metadata_filters.get("file_types", []))

        if not source_files and not file_types:
            return None

        def _faiss_filter(metadata: Dict[str, Any]) -> bool:
            source = str(metadata.get("source", ""))
            source_name = Path(source).name
            file_type = str(metadata.get("file_type", Path(source).suffix.lower()))

            if source_files and source_name not in source_files:
                return False
            if file_types and file_type not in file_types:
                return False
            return True

        return _faiss_filter

    def _vector_search(
        self,
        query: str,
        k: int,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Run pure vector search with optional pre-retrieval metadata filtering.

        Uses FAISS's ``filter`` parameter to restrict the search space
        BEFORE similarity computation, ensuring that only chunks from
        the requested files are considered.

        Args:
            query: Search query string
            k: Number of results to return
            metadata_filters: Optional filters to apply during search

        Returns:
            List of domain Document objects with original (non-prefixed) content
        """
        faiss_filter = self._build_faiss_filter(metadata_filters)

        if self.using_offline_fallback:
            query_embedding = self.embeddings.embed_query(query)
            if faiss_filter is not None:
                results = self.vector_store.similarity_search_by_vector(
                    query_embedding, k=k, filter=faiss_filter,
                )
            else:
                results = self.vector_store.similarity_search_by_vector(
                    query_embedding, k=k,
                )
        else:
            if faiss_filter is not None:
                results = self.vector_store.similarity_search(
                    query, k=k, filter=faiss_filter,
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)

        # Extract original content (strip the [File: ...] prefix we added)
        docs: List[Document] = []
        for doc in results:
            content = doc.metadata.get("original_content", doc.page_content)
            metadata = {k: v for k, v in doc.metadata.items() if k != "original_content"}
            docs.append(Document(content=content, metadata=metadata))
        return docs

    def _all_lc_documents(self) -> List[LCDocument]:
        """Get all currently indexed documents from FAISS docstore."""
        if self.vector_store is None:
            return []
        docstore_dict = getattr(getattr(self.vector_store, "docstore", None), "_dict", {})
        return list(docstore_dict.values()) if docstore_dict else []

    def _rebuild_bm25_retriever(self) -> None:
        """Rebuild BM25 index from all currently indexed FAISS documents."""
        try:
            all_docs = self._all_lc_documents()
            if not all_docs:
                self._bm25_retriever = None
                return
            retriever = BM25Retriever.from_documents(all_docs)
            retriever.k = min(20, len(all_docs))
            self._bm25_retriever = retriever
            logger.info("BM25 retriever rebuilt with %s docs", len(all_docs))
        except Exception as bm25_error:
            self._bm25_retriever = None
            logger.warning("Could not rebuild BM25 retriever: %s", bm25_error)

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """Run BM25 keyword retrieval and return domain documents."""
        if self._bm25_retriever is None:
            self._rebuild_bm25_retriever()
        if self._bm25_retriever is None:
            return []

        self._bm25_retriever.k = k
        results = self._bm25_retriever.invoke(query)
        docs: List[Document] = []
        for doc in results:
            content = doc.metadata.get("original_content", doc.page_content)
            metadata = {k: v for k, v in doc.metadata.items() if k != "original_content"}
            docs.append(Document(content=content, metadata=metadata))
        return docs

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Create a deterministic key used for deduplication and overlap metrics."""
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        content_hash = hashlib.md5(doc.content.encode("utf-8")).hexdigest()
        return f"{source}|{page}|{chunk_idx}|{content_hash}"

    def _merge_results(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        """Merge vector and BM25 results with de-duplication preserving rank order."""
        merged: List[Document] = []
        seen: set = set()
        for doc in vector_docs + bm25_docs:
            key = self._doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged

    def _count_overlap(self, vector_docs: List[Document], bm25_docs: List[Document]) -> int:
        """Count overlap between vector and BM25 candidate sets."""
        vector_keys = {self._doc_key(doc) for doc in vector_docs}
        bm25_keys = {self._doc_key(doc) for doc in bm25_docs}
        return len(vector_keys.intersection(bm25_keys))

    @staticmethod
    def _apply_metadata_filters(
        documents: List[Document],
        metadata_filters: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Filter docs by metadata fields such as source_files and file_types."""
        if not metadata_filters:
            return documents

        source_files = set(metadata_filters.get("source_files", []))
        file_types = set(metadata_filters.get("file_types", []))

        filtered: List[Document] = []
        for doc in documents:
            source = str(doc.metadata.get("source", ""))
            source_name = Path(source).name
            file_type = str(doc.metadata.get("file_type", Path(source).suffix.lower()))

            if source_files and source_name not in source_files:
                continue
            if file_types and file_type not in file_types:
                continue
            filtered.append(doc)
        return filtered

    def _load_cross_encoder(self) -> Optional[Any]:
        """Lazy-load cross encoder model only when reranking is requested."""
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self._cross_encoder_name)
            logger.info("Cross-encoder loaded: %s", self._cross_encoder_name)
        except Exception as error:
            logger.warning("Cross-encoder unavailable, skipping rerank: %s", error)
            self._cross_encoder = None
        return self._cross_encoder

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank candidates using cross-encoder relevance scores."""
        if not documents:
            return []

        cross_encoder = self._load_cross_encoder()
        if cross_encoder is None:
            return documents

        pairs = [[query, doc.content] for doc in documents]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)

        reranked_docs: List[Document] = []
        for rank, (doc, score) in enumerate(ranked, start=1):
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = float(score)
            metadata["rerank_rank"] = rank
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
    So sánh 3 chiến lược retrieval:
    1. Pure Vector (semantic search)
    2. Pure BM25 (keyword search)
    3. Hybrid (Vector + BM25 ensemble)

    Dùng 3 proxy metrics:
    - Recall@K: Tỷ lệ chunk chứa từ khóa query được recall về
    - Speed (ms): Thời gian phản hồi trung bình
    - Coverage: Số chunk duy nhất được trả về
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
        Tính Recall@K cho một chiến lược retrieval.

        Recall@K = (số chunk trong top-K chứa từ khóa query) / (tổng số chunk chứa từ khóa)

        Args:
            query: Câu hỏi truy vấn
            k: Số document lấy về
            strategy: 'vector' | 'bm25' | 'hybrid'

        Returns:
            Tuple: (recall_score, retrieved_docs, timing_stats)
        """
        all_docs = self.vector_service._all_lc_documents()
        if not all_docs:
            return 0.0, [], {"time_ms": 0.0}

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
        query_tokens = {
            t.lower() for t in re.findall(r"\w+", query)
            if t.lower() not in stop_words and len(t) > 1
        }

        ground_truth_keys = set()
        for idx, doc in enumerate(all_docs):
            content_lower = doc.page_content.lower()
            if any(token in content_lower for token in query_tokens):
                key = hashlib.md5(doc.page_content.encode()).hexdigest()
                ground_truth_keys.add(key)

        total_relevant = len(ground_truth_keys)

        start = time.perf_counter()
        if strategy == "vector":
            docs, stats = self.vector_service.search(
                query=query, k=k, use_hybrid=False, use_bm25_only=False, rerank=False
            )
        elif strategy == "bm25":
            docs, stats = self.vector_service.search(
                query=query, k=k, use_hybrid=False, use_bm25_only=True, rerank=False
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            stats = {**stats, "time_ms": elapsed_ms}
        elif strategy == "hybrid":
            docs, stats = self.vector_service.search(
                query=query, k=k, use_hybrid=True, use_bm25_only=False, rerank=False
            )
        else:
            docs, stats = [], {}
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not docs:
            return 0.0, [], {"time_ms": elapsed_ms}

        retrieved_relevant = 0
        for doc in docs:
            key = hashlib.md5(doc.content.encode()).hexdigest()
            if key in ground_truth_keys:
                retrieved_relevant += 1

        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
        timing = dict(stats)
        timing["time_ms"] = round(elapsed_ms, 2)
        return recall, docs, timing

    def run(
        self,
        query: str,
        k: int = 5,
        show_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy benchmark đầy đủ cho 3 chiến lược.

        Args:
            query: Câu hỏi test
            k: Số document lấy về
            show_progress: Callback hiển thị tiến trình (tuỳ chọn)

        Returns:
            Dict chứa kết quả của cả 3 chiến lược và bảng so sánh
        """
        if self.vector_service.vector_store is None:
            return {"error": "Vector store is empty. Please upload documents first."}

        strategies = ["vector", "bm25", "hybrid"]
        results: Dict[str, Dict[str, Any]] = {}
        summary_labels = {
            "vector": "Vector (Semantic)",
            "bm25": "BM25 (Keyword)",
            "hybrid": "Hybrid (Ensemble)",
        }

        for strat in strategies:
            if show_progress:
                show_progress(f"Testing: {summary_labels.get(strat, strat)}...")
            recall, docs, timing = self._compute_recall_at_k(query, k, strat)
            results[strat] = {
                "strategy": strat,
                "label": summary_labels.get(strat, strat),
                "recall_at_k": round(recall, 4),
                "docs_retrieved": len(docs),
                "unique_sources": len({d.metadata.get("source_file", "") for d in docs}),
                "time_ms": timing.get("time_ms", 0.0),
                "vector_time_ms": timing.get("vector_time_ms", 0.0),
                "bm25_time_ms": timing.get("bm25_time_ms", 0.0),
                "top_docs": [
                    {
                        "source": d.metadata.get("source_file", "unknown"),
                        "page": d.metadata.get("page"),
                        "preview": d.content[:150].replace("\n", " ") + "...",
                    }
                    for d in docs[:3]
                ],
            }

        best_by_recall = max(results, key=lambda s: results[s]["recall_at_k"])
        best_by_speed = min(results, key=lambda s: results[s]["time_ms"])
        best_by_coverage = max(results, key=lambda s: results[s]["unique_sources"])

        for strat, res in results.items():
            res["is_best_recall"] = strat == best_by_recall
            res["is_best_speed"] = strat == best_by_speed
            res["is_best_coverage"] = strat == best_by_coverage

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
