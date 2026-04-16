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

        # --- THÊM TÙY CHỌN BẬT/TẮT OCR Ở ĐÂY ---
        st.markdown("### Processing Options")
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
        """Render uploaded documents with rich metadata and filter controls."""
        loaded_documents = st.session_state.get("loaded_documents", [])
        if not loaded_documents:
            return

        st.markdown("---")
        st.subheader("Uploaded Documents")

        source_names = sorted({item.get("name", "") for item in loaded_documents if item.get("name")})
        file_types = sorted({item.get("file_type", "") for item in loaded_documents if item.get("file_type")})

        selected_sources = st.multiselect(
            "🔍 Filter by Document",
            options=source_names,
            default=st.session_state.get("active_source_filters", []),
            help="Chỉ các document được chọn mới được dùng khi trả lời câu hỏi.",
        )
        st.session_state.active_source_filters = selected_sources

        selected_types = st.multiselect(
            "📎 Filter by File Type",
            options=file_types,
            default=st.session_state.get("active_file_type_filters", []),
        )
        st.session_state.active_file_type_filters = selected_types

        # ── Document info cards ──────────────────────────────────────────
        st.markdown("##### 📄 Document Details")
        doc_cards = st.columns(min(3, len(loaded_documents)))
        for idx, doc_meta in enumerate(loaded_documents):
            col = doc_cards[idx % len(doc_cards)]
            with col:
                name = doc_meta.get("name", "unknown")
                file_type = doc_meta.get("file_type", "").upper().replace(".", "")
                size_mb = doc_meta.get("file_size_mb", 0)
                chunks = doc_meta.get("chunks", 0)
                pages = doc_meta.get("page_count", "N/A")
                uploaded = doc_meta.get("uploaded_at", "")
                title = doc_meta.get("title", name)
                is_selected = name in selected_sources

                # Icon by file type
                icon_map = {".PDF": "📕", ".DOCX": "📘", ".TXT": "📄"}
                icon = icon_map.get(file_type, "📄")

                # Color strip by file type
                color_map = {
                    ".PDF": "#e53935",
                    ".DOCX": "#1565c0",
                    ".TXT": "#558b2f",
                }
                color = color_map.get(file_type, "#9e9e9e")

                st.markdown(f"""
                <div style="
                    border: 1px solid #ddd;
                    border-left: 4px solid {color};
                    border-radius: 6px;
                    padding: 10px 14px;
                    margin-bottom: 8px;
                    background: {'#f1f8ff' if is_selected else '#fafafa'};
                ">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                        <span style="font-size:1.3em;">{icon}</span>
                        <strong style="font-size:0.95em;">{title}</strong>
                    </div>
                    <div style="font-size:0.8em; color:#555;">
                        <span>📎 {file_type} &nbsp;|&nbsp;
                        💾 {size_mb:.2f} MB &nbsp;|&nbsp;
                        📑 {pages} trang &nbsp;|&nbsp;
                        🧩 {chunks} chunks</span>
                    </div>
                    <div style="font-size:0.78em; color:#888; margin-top:3px;">
                        ⏱ Uploaded: {uploaded}
                    </div>
                    {"<div style='margin-top:4px; font-size:0.8em; color:#1976d2; font-weight:600;'>✅ Active filter</div>" if is_selected else ""}
                </div>
                """, unsafe_allow_html=True)

        # ── Summary stats ────────────────────────────────────────────────
        total_files = len(loaded_documents)
        total_size = sum(item.get("file_size_bytes", 0) for item in loaded_documents)
        total_chunks = sum(item.get("chunks", 0) for item in loaded_documents)
        total_pages = sum(
            item.get("page_count", 0) for item in loaded_documents
            if isinstance(item.get("page_count"), (int, float))
        )
        active_filters = len(selected_sources)

        stat_cols = st.columns(5)
        stat_data = [
            ("📁 Files", f"{total_files}"),
            ("💾 Size", f"{total_size / (1024*1024):.1f} MB"),
            ("📑 Pages", f"{total_pages}"),
            ("🧩 Chunks", f"{total_chunks}"),
            ("✅ Active", f"{active_filters}/{total_files}"),
        ]
        for c_idx, (label, value) in enumerate(stat_data):
            with stat_cols[c_idx]:
                st.metric(label, value)

        # ── Detailed table ───────────────────────────────────────────────
        with st.expander("📋 Chi tiet tat ca documents"):
            display_rows = []
            for item in loaded_documents:
                display_rows.append({
                    "Name": item.get("name", ""),
                    "Type": item.get("file_type", "").upper().replace(".", ""),
                    "Size (MB)": f"{item.get('file_size_mb', 0):.2f}",
                    "Pages": item.get("page_count", "N/A"),
                    "Chunks": item.get("chunks", 0),
                    "Uploaded": item.get("uploaded_at", "")[:19],
                })
            st.dataframe(display_rows, use_container_width=True, hide_index=True)
