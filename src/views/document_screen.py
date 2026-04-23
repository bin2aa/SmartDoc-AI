"""Document upload screen for SmartDoc AI — main orchestrator."""

import streamlit as st

from src.controllers.document_controller import DocumentController
from src.views.components import UIComponents, icon
from src.views.document_table import render_uploaded_document_table
from src.utils.logger import setup_logger
from src.utils.constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB
from src.utils.ocr_utils import OCR_AVAILABLE, get_availability_info

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

        # --- THÊM TÙY CHỌN BẬT/TẮT OCR Ở ĐÂY ---
        st.markdown("### Processing Options")

        if not OCR_AVAILABLE:
            ocr_info = get_availability_info()
            missing = ", ".join(ocr_info["missing_deps"])
            st.warning(
                f"⚠️ **OCR unavailable** — missing dependencies: `{missing}`.  \n"
                f"Install with: `pip install {' '.join(ocr_info['missing_deps'])}`"
            )
            enable_ocr = False
        else:
            enable_ocr = st.checkbox(
                "Enable OCR (Read text from Images / Scanned PDFs)",
                help="Check this to process .png, .jpg, or scanned .pdf files. Note: Processing will take longer."
            )

        # Mở rộng danh sách đuôi file được phép trên UI nếu bật OCR
        display_extensions = [ext.replace('.', '') for ext in ALLOWED_EXTENSIONS]
        if enable_ocr:
            display_extensions.extend(['png', 'jpg', 'jpeg'])
        # ----------------------------------------

        # File uploader
        uploaded_files = self.components.file_uploader(
            label="Choose one or multiple documents",
            accepted_types=display_extensions,
            accept_multiple_files=True,
        )

        # Upload button
        if uploaded_files:
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button("Process Documents", use_container_width=True, type="primary"):
                    with self.components.loading_spinner("Processing documents..."):
                        # --- TRUYỀN BIẾN OCR XUỐNG CONTROLLER ---
                        result = self.controller.upload_and_process_many(uploaded_files, use_ocr=enable_ocr)
                        if result.get("success_count", 0) > 0:
                            st.balloons()

        # Render uploaded document table (delegated to document_table module)
        render_uploaded_document_table(st.session_state.get("loaded_documents", []))

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