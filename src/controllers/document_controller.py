"""Document controller for handling document operations."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
from src.services.document_service import DocumentService
from src.services.vector_store_service import AbstractVectorStoreService
from src.services.persistence_service import save_faiss_index, save_loaded_docs, clear_all_state
from src.utils.logger import setup_logger
from src.utils.exceptions import DocumentLoadError
from src.utils.constants import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

logger = setup_logger(__name__)


class DocumentController:
    """
    Controller for document operations.
    
    Handles document upload, validation, processing, and storage.
    """
    
    def __init__(
        self,
        document_service: Optional[DocumentService] = None,
        vector_service: Optional[AbstractVectorStoreService] = None
    ):
        """
        Initialize document controller.
        
        Args:
            document_service: Document service instance
            vector_service: Vector store service instance
        """
        self.document_service = document_service or DocumentService()
        self.vector_service = vector_service
        
        logger.info("DocumentController initialized")
    
    def upload_and_process(self, uploaded_file, use_ocr: bool = False) -> bool:
        """
        Upload and process document.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            use_ocr: Bật chế độ quét ảnh (OCR)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate file
            if not self._validate_file(uploaded_file, use_ocr=use_ocr):
                return False
            
            result = self.upload_and_process_many([uploaded_file], use_ocr=use_ocr)
            return result["success_count"] == 1
                
        except Exception as e:
            logger.exception("Unexpected error in upload_and_process")
            st.error(f"Unexpected error: {str(e)}")
            return False

    def upload_and_process_many(self, uploaded_files: List[Any], use_ocr: bool = False) -> Dict[str, Any]:
        """Upload and process multiple documents in one action."""
        if not uploaded_files:
            return {"success_count": 0, "failed": []}

        if self.vector_service is None:
            st.error("Vector store not initialized")
            return {"success_count": 0, "failed": ["vector_service_unavailable"]}

        success_count = 0
        failed: List[str] = []
        loaded_docs = st.session_state.get("loaded_documents", [])

        for uploaded_file in uploaded_files:
            try:
                if not self._validate_file(uploaded_file, use_ocr=use_ocr):
                    failed.append(uploaded_file.name)
                    continue

                file_path = self._save_uploaded_file(uploaded_file)
                logger.info("Saved file to %s", file_path)

                try:
                    # Truyền cờ use_ocr xuống Service
                    documents = self.document_service.load_document(file_path, use_ocr=use_ocr)
                    logger.info("Loaded %s chunks from %s", len(documents), uploaded_file.name)
                except DocumentLoadError as load_error:
                    failed.append(uploaded_file.name)
                    st.error(f"Cannot load {uploaded_file.name}: {str(load_error)}")
                    continue

                self.vector_service.add_documents(documents)

                # Lưu trạng thái is_ocr vào session để hỗ trợ reload khi benchmark
                # Enrich loaded_documents metadata for display and filtering
                first_chunk_meta = documents[0].metadata
                last_chunk_meta = documents[-1].metadata
                loaded_docs.append(
                    {
                        "name": uploaded_file.name,
                        "path": file_path,
                        "file_type": Path(uploaded_file.name).suffix.lower(),
                        "chunks": len(documents),
                        "is_ocr": use_ocr,
                        "file_size_bytes": first_chunk_meta.get("file_size_bytes", 0),
                        "file_size_mb": first_chunk_meta.get("file_size_mb", 0.0),
                        "uploaded_at": first_chunk_meta.get("uploaded_at", ""),
                        "page_count": first_chunk_meta.get("page_count", 0),
                        "title": first_chunk_meta.get("title", uploaded_file.name),
                    }
                )
                success_count += 1
            except Exception as file_error:
                failed.append(getattr(uploaded_file, "name", "unknown_file"))
                logger.error("Failed processing file %s: %s", getattr(uploaded_file, "name", "unknown"), file_error)

        st.session_state.loaded_documents = loaded_docs
        st.session_state.vector_store_initialized = success_count > 0 or bool(st.session_state.get("vector_store_initialized", False))

        if success_count > 0:
            # Persist FAISS index and document metadata to disk
            save_faiss_index(self.vector_service)
            save_loaded_docs(loaded_docs)
            logger.info("Persisted FAISS index and document metadata to disk")
            st.success(f"Successfully processed {success_count} document(s)")
        if failed:
            st.warning("Could not process: " + ", ".join(failed))

        return {
            "success_count": success_count,
            "failed": failed,
            "total": len(uploaded_files),
        }
    
    def _validate_file(self, uploaded_file, use_ocr: bool = False) -> bool:
        """
        Validate uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            use_ocr: Mở rộng các đuôi file ảnh nếu bật OCR
            
        Returns:
            True if valid, False otherwise
        """
        if uploaded_file is None:
            st.error("No file uploaded")
            return False
        
        # Check file extension
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        # Mở rộng list extension khi bật OCR
        valid_extensions = list(ALLOWED_EXTENSIONS)
        if use_ocr:
            valid_extensions.extend(['.png', '.jpg', '.jpeg'])
            
        if file_ext not in valid_extensions:
            unique_exts = list(set(valid_extensions))
            st.error(f"Invalid file type. Allowed: {', '.join(unique_exts)}")
            return False
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
            return False
        
        logger.info(f"File validation passed: {uploaded_file.name} ({file_size_mb:.2f}MB)")
        return True
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        """
        Save uploaded file to disk.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Path to saved file
        """
        file_path = UPLOAD_DIR / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(file_path)
    
    def update_chunk_config(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Update chunking configuration.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        self.document_service.update_chunk_config(chunk_size, chunk_overlap)
        st.success(f"Updated chunk config: size={chunk_size}, overlap={chunk_overlap}")
        logger.info(f"Chunk config updated via controller")
    
    def clear_vector_store(self) -> None:
        """Clear vector store and all persisted state."""
        if self.vector_service:
            try:
                self.vector_service.clear_store()
                st.session_state.vector_store_initialized = False
                st.session_state.loaded_documents = []
                # Clear all persisted state from disk
                clear_all_state()
                st.success("Vector store cleared successfully")
                logger.warning("Vector store and persisted state cleared by user")
            except Exception as e:
                logger.error(f"Error clearing vector store: {e}")
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Vector store not initialized")

    def benchmark_chunk_configs(
        self,
        query: str,
        configs: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """Compare chunk settings using a lightweight keyword-hit proxy metric."""
        loaded_documents = st.session_state.get("loaded_documents", [])
        if not loaded_documents:
            return []

        normalized_query = (query or "").strip().lower()
        tokens = [token for token in normalized_query.split() if len(token) > 1]
        if not tokens:
            return []

        original_size = self.document_service.chunk_size
        original_overlap = self.document_service.chunk_overlap
        results: List[Dict[str, Any]] = []

        try:
            for chunk_size, chunk_overlap in configs:
                self.document_service.update_chunk_config(chunk_size, chunk_overlap)
                total_chunks = 0
                hit_chunks = 0

                for doc_meta in loaded_documents:
                    doc_path = doc_meta.get("path")
                    # Lấy cờ OCR từ session để đọc lại file chính xác
                    is_ocr = doc_meta.get("is_ocr", False)
                    
                    if not doc_path:
                        continue
                        
                    # Truyền is_ocr vào hàm load
                    chunks = self.document_service.load_document(doc_path, use_ocr=is_ocr)
                    total_chunks += len(chunks)
                    for chunk in chunks:
                        content = chunk.content.lower()
                        if any(token in content for token in tokens):
                            hit_chunks += 1

                accuracy_proxy = (hit_chunks / total_chunks) if total_chunks else 0.0
                results.append(
                    {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "total_chunks": total_chunks,
                        "hit_chunks": hit_chunks,
                        "accuracy_proxy": round(accuracy_proxy, 4),
                    }
                )
        finally:
            self.document_service.update_chunk_config(original_size, original_overlap)

        results.sort(key=lambda row: row["accuracy_proxy"], reverse=True)
        return results