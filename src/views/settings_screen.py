"""Settings screen for SmartDoc AI."""

import streamlit as st
from src.controllers.document_controller import DocumentController
from src.views.components import UIComponents
from src.utils.logger import setup_logger
from src.utils.constants import *

logger = setup_logger(__name__)


class SettingsScreen:
    """
    Settings configuration screen.
    
    Allows users to configure RAG parameters.
    """
    
    def __init__(self, document_controller: DocumentController):
        """
        Initialize settings screen.
        
        Args:
            document_controller: Document controller instance
        """
        self.document_controller = document_controller
        self.components = UIComponents()
    
    def render(self):
        """Render the settings screen."""
        st.title("⚙️ Settings")
        
        st.markdown("""
        Configure the RAG (Retrieval-Augmented Generation) pipeline parameters.
        
        **Note:** Changes to chunk settings will only affect newly uploaded documents.
        """)
        
        st.markdown("---")
        
        # Chunk Configuration
        self._render_chunk_settings()
        
        st.markdown("---")
        
        # LLM Configuration
        self._render_llm_settings()
        
        st.markdown("---")
        
        # System Info
        self._render_system_info()
    
    def _render_chunk_settings(self):
        """Render chunk configuration settings."""
        st.subheader("📝 Text Chunking Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE),
                step=100,
                help="Size of text chunks for processing. Larger chunks = more context but less precision."
            )
        
        with col2:
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=50,
                max_value=300,
                value=st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP),
                step=10,
                help="Overlap between consecutive chunks. Higher overlap = better continuity."
            )
        
        if st.button("💾 Apply Chunk Settings", type="primary"):
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            self.document_controller.update_chunk_config(chunk_size, chunk_overlap)
    
    def _render_llm_settings(self):
        """Render LLM configuration settings."""
        st.subheader("🤖 LLM Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model", DEFAULT_MODEL)
        
        with col2:
            st.metric("Temperature", DEFAULT_TEMPERATURE)
        
        with col3:
            st.metric("Top-P", DEFAULT_TOP_P)
        
        st.info("💡 LLM parameters are currently fixed. Future versions will allow customization.")
    
    def _render_system_info(self):
        """Render system information."""
        st.subheader("🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Embedding Model**")
            st.code(EMBEDDING_MODEL, language="text")
        
        with col2:
            st.markdown("**Vector Database**")
            st.code("FAISS (Local)", language="text")
        
        st.markdown("**Ollama Server**")
        st.code(OLLAMA_BASE_URL, language="text")
        
        # Connection test
        if st.button("🔌 Test Ollama Connection"):
            with self.components.loading_spinner("Testing connection..."):
                try:
                    from src.services.llm_service import OllamaLLMService
                    llm = OllamaLLMService()
                    self.components.success_alert("Connected to Ollama successfully!")
                except Exception as e:
                    self.components.error_alert(
                        "Cannot connect to Ollama",
                        details=str(e)
                    )
