"""Document upload screen for SmartDoc AI."""

import streamlit as st
from src.controllers.document_controller import DocumentController
from src.views.components import UIComponents, icon
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
        st.markdown(f"## {icon('description')} Document Management", unsafe_allow_html=True)
        
        st.markdown("""
        Upload your documents here. Supported formats: **PDF, DOCX, TXT**
        
        Once uploaded, your documents will be processed and made available for questions in the Chat tab.
        """)
        
        # File uploader
        uploaded_files = self.components.file_uploader(
            label="Choose one or multiple documents",
            accepted_types=[ext.replace('.', '') for ext in ALLOWED_EXTENSIONS],
            accept_multiple_files=True,
        )

        # Upload button
        if uploaded_files:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("Process Documents", use_container_width=True, type="primary"):
                    with self.components.loading_spinner("Processing documents..."):
                        result = self.controller.upload_and_process_many(uploaded_files)
                        if result.get("success_count", 0) > 0:
                            st.balloons()

        self._render_uploaded_document_table()
        
        # Divider
        st.markdown("---")
        
        # Document info
        self._render_document_info()
        
        # Advanced actions
        st.markdown("---")
        self._render_advanced_actions()
    
    def _render_document_info(self):
        """Display document processing information."""
        st.subheader("Information")
        
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
        st.subheader("Advanced Actions")

        col1, col2 = st.columns(2)

        with col1:
            confirm_clear = st.checkbox("Confirm clear all documents and index")
            if st.button("Clear Vector Store", use_container_width=True, disabled=not confirm_clear):
                if confirm_clear:
                    self.controller.clear_vector_store()
                    st.rerun()
        
        with col2:
            if st.button("View Upload Folder", use_container_width=True):
                from src.utils.constants import UPLOAD_DIR
                st.code(str(UPLOAD_DIR))

    def _render_uploaded_document_table(self):
        """Render uploaded documents and metadata filters for retrieval."""
        loaded_documents = st.session_state.get("loaded_documents", [])
        if not loaded_documents:
            return

        st.markdown("---")
        st.subheader("Uploaded Documents")

        source_names = sorted({item.get("name", "") for item in loaded_documents if item.get("name")})
        file_types = sorted({item.get("file_type", "") for item in loaded_documents if item.get("file_type")})

        selected_sources = st.multiselect(
            "Filter retrieval by document",
            options=source_names,
            default=st.session_state.get("active_source_filters", []),
            help="Only selected documents will be used when answering questions.",
        )
        st.session_state.active_source_filters = selected_sources

        selected_types = st.multiselect(
            "Filter retrieval by file type",
            options=file_types,
            default=st.session_state.get("active_file_type_filters", []),
        )
        st.session_state.active_file_type_filters = selected_types

        st.dataframe(loaded_documents, use_container_width=True)
