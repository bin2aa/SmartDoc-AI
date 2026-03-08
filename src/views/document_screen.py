"""Document upload screen for SmartDoc AI."""

import streamlit as st
from src.controllers.document_controller import DocumentController
from src.views.components import UIComponents
from src.utils.logger import setup_logger
from src.utils.constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

logger = setup_logger(__name__)


class DocumentScreen:
    """
    Document upload and management screen.
    
    Handles file uploads and document processing.
    """
    
    def __init__(self, controller: DocumentController):
        """
        Initialize document screen.
        
        Args:
            controller: Document controller instance
        """
        self.controller = controller
        self.components = UIComponents()
    
    def render(self):
        """Render the document screen."""
        st.title("📄 Document Management")
        
        st.markdown("""
        Upload your documents here. Supported formats: **PDF, DOCX, TXT**
        
        Once uploaded, your documents will be processed and made available for questions in the Chat tab.
        """)
        
        # File uploader
        uploaded_file = self.components.file_uploader(
            label="Choose a document",
            accepted_types=[ext.replace('.', '') for ext in ALLOWED_EXTENSIONS]
        )
        
        # Upload button
        if uploaded_file is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("📤 Process Document", use_container_width=True, type="primary"):
                    with self.components.loading_spinner("Processing document..."):
                        success = self.controller.upload_and_process(uploaded_file)
                        
                        if success:
                            st.balloons()
        
        # Divider
        st.markdown("---")
        
        # Document info
        self._render_document_info()
        
        # Advanced actions
        st.markdown("---")
        self._render_advanced_actions()
    
    def _render_document_info(self):
        """Display document processing information."""
        st.subheader("ℹ️ Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Max File Size", 
                f"{MAX_FILE_SIZE_MB} MB"
            )
        
        with col2:
            st.metric(
                "Formats", 
                len(ALLOWED_EXTENSIONS)
            )
        
        with col3:
            vs_status = "Ready" if st.session_state.get('vector_store_initialized', False) else "Empty"
            st.metric(
                "Vector Store",
                vs_status
            )
    
    def _render_advanced_actions(self):
        """Render advanced document actions."""
        st.subheader("⚙️ Advanced Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Vector Store", use_container_width=True):
                if st.checkbox("⚠️ Confirm clear all documents?"):
                    self.controller.clear_vector_store()
                    st.rerun()
        
        with col2:
            if st.button("ℹ️ View Upload Folder", use_container_width=True):
                from src.utils.constants import UPLOAD_DIR
                st.code(str(UPLOAD_DIR))
