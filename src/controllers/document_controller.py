"""Document controller for handling document operations."""

import os
from pathlib import Path
from typing import Optional
import streamlit as st
from src.services.document_service import DocumentService
from src.services.vector_store_service import AbstractVectorStoreService
from src.utils.logger import setup_logger
from src.utils.exceptions import DocumentLoadError, ValidationError
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
    
    def upload_and_process(self, uploaded_file) -> bool:
        """
        Upload and process document.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate file
            if not self._validate_file(uploaded_file):
                return False
            
            # Save file
            file_path = self._save_uploaded_file(uploaded_file)
            logger.info(f"Saved file to {file_path}")
            
            # Load and chunk document
            try:
                documents = self.document_service.load_document(file_path)
                logger.info(f"Loaded {len(documents)} chunks from document")
            except DocumentLoadError as e:
                st.error(f"❌ Cannot load document: {str(e)}")
                return False
            
            # Store in vector database
            if self.vector_service:
                try:
                    self.vector_service.add_documents(documents)
                    st.success(f"✅ Successfully processed {len(documents)} chunks from {uploaded_file.name}")
                    
                    # Update session state
                    st.session_state.vector_store_initialized = True
                    
                    return True
                except Exception as e:
                    logger.error(f"Vector store error: {e}")
                    st.error(f"❌ Vector store error: {str(e)}")
                    return False
            else:
                st.error("❌ Vector store not initialized")
                return False
                
        except Exception as e:
            logger.exception("Unexpected error in upload_and_process")
            st.error(f"❌ Unexpected error: {str(e)}")
            return False
    
    def _validate_file(self, uploaded_file) -> bool:
        """
        Validate uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if valid, False otherwise
        """
        if uploaded_file is None:
            st.error("❌ No file uploaded")
            return False
        
        # Check file extension
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            st.error(f"❌ Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
            return False
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
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
        st.success(f"✅ Updated chunk config: size={chunk_size}, overlap={chunk_overlap}")
        logger.info(f"Chunk config updated via controller")
    
    def clear_vector_store(self) -> None:
        """Clear vector store."""
        if self.vector_service:
            try:
                self.vector_service.clear_store()
                st.session_state.vector_store_initialized = False
                st.success("✅ Vector store cleared successfully")
                logger.warning("Vector store cleared by user")
            except Exception as e:
                logger.error(f"Error clearing vector store: {e}")
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Vector store not initialized")
