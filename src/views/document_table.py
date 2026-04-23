"""Uploaded document table and card rendering for SmartDoc AI."""

import streamlit as st
from typing import List, Dict, Any


def render_uploaded_document_table(loaded_documents: List[Dict[str, Any]]):
    """Render uploaded documents with rich metadata and filter controls.

    Includes document filter multiselects, info cards, summary stats,
    and a detailed expandable table.

    Args:
        loaded_documents: List of document metadata dicts from session state
    """
    if not loaded_documents:
        return

    st.markdown("---")
    st.subheader("Uploaded Documents")

    source_names = sorted({item.get("name", "") for item in loaded_documents if item.get("name")})
    file_types = sorted({item.get("file_type", "") for item in loaded_documents if item.get("file_type")})

    # Apply pending filter changes before creating the widget
    if "_pending_source_filters" in st.session_state:
        st.session_state.active_source_filters = st.session_state.pop("_pending_source_filters")

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
            _render_document_card(doc_meta, selected_sources)

    # ── Summary stats ────────────────────────────────────────────────
    _render_summary_stats(loaded_documents, selected_sources)

    # ── Detailed table ───────────────────────────────────────────────
    _render_detail_table(loaded_documents)


def _render_document_card(doc_meta: Dict[str, Any], selected_sources: List[str]):
    """Render a single document info card.

    Args:
        doc_meta: Document metadata dictionary
        selected_sources: List of currently selected source names
    """
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
    file_icon = icon_map.get(file_type, "📄")

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
            <span style="font-size:1.3em;">{file_icon}</span>
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


def _render_summary_stats(loaded_documents: List[Dict[str, Any]], selected_sources: List[str]):
    """Render summary statistics row.

    Args:
        loaded_documents: List of document metadata dicts
        selected_sources: List of currently selected source names
    """
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


def _render_detail_table(loaded_documents: List[Dict[str, Any]]):
    """Render detailed expandable table of all documents.

    Args:
        loaded_documents: List of document metadata dicts
    """
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