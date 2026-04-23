"""Source citation and detail rendering for SmartDoc AI chat."""

import streamlit as st
from pathlib import Path
from typing import List, Optional

from src.models.document_model import Document
from src.utils.constants import UPLOAD_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def convert_sources_to_details(sources: List[Document]) -> List[dict]:
    """Convert Document objects to serializable source detail dicts.

    Args:
        sources: List of Document objects from retrieval

    Returns:
        List of dicts with content, citation, source_file, page, etc.
    """
    source_details: List[dict] = []
    if not sources:
        return source_details

    for src in sources:
        try:
            source_details.append({
                "content": src.content,
                "citation": src.get_citation(),
                "source_file": src.metadata.get("source_file") or src.source_file,
                "page": src.page_number,
                "used_in_answer": src.metadata.get("used_in_answer", False),
                "used_term_overlap": src.metadata.get("used_term_overlap", 0),
            })
        except Exception as e:
            logger.warning(f"Failed to get details for source: {e}")

    return source_details


def render_source_citations(citations: List[str], msg_idx: int):
    """Render source citations from serialized citation strings.

    Args:
        citations: List of citation strings
        msg_idx: Message index for unique key generation
    """
    with st.expander("Xem nguồn tham khảo"):
        for src_idx, citation in enumerate(citations, 1):
            st.markdown(f"**Nguồn {src_idx}:** {citation}")


def render_source_details(
    source_details: List[dict],
    msg_idx: int,
    rewritten_query: Optional[str] = None,
):
    """Render source citations with detailed info from dictionary.

    Args:
        source_details: List of source detail dictionaries
        msg_idx: Message index for unique key generation
        rewritten_query: Optional rewritten query to display
    """
    with st.expander("Xem chi tiết nguồn tham khảo"):
        if rewritten_query:
            st.info(f"**Câu hỏi đã được tối ưu:** {rewritten_query}")
            st.divider()

        for src_idx, src in enumerate(source_details, 1):
            citation = src.get("citation", "[Unknown source]")
            source_file = src.get("source_file")
            content = src.get("content", "")
            is_used = src.get("used_in_answer", False)
            overlap = src.get("used_term_overlap", 0)

            st.markdown(f"**Nguồn {src_idx}:** {citation}")

            if source_file:
                file_path = Path(UPLOAD_DIR) / source_file
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label="📁 Tải file nguồn",
                            data=f,
                            file_name=source_file,
                            mime="application/octet-stream",
                            key=f"download_source_{msg_idx}_{src_idx}",
                        )
                else:
                    st.caption(f"📄 {source_file}")

            if is_used:
                st.caption(f"Được sử dụng trong câu trả lời (term overlap: {overlap})")
                # Highlight preview
                preview = content[:300] + "..." if len(content) > 300 else content
                st.markdown(
                    f"<mark style='background-color: #ffff0033;'>{preview}</mark>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Ngữ cảnh đã truy xuất (không được dùng trực tiếp)")

            st.text_area(
                f"Nội dung đầy đủ (Nguồn {src_idx})",
                value=content,
                height=150,
                key=f"source_msg{msg_idx}_src{src_idx}",
                disabled=True,
            )


def render_sources(sources: List[Document], msg_idx: int):
    """Render source citations with rich document metadata.

    Hiển thị thông tin chi tiết về document nguồn:
    - Tên file + icon theo loại file
    - Số trang
    - Chunk index
    - Kích thước file
    - Ngày upload
    - Đoạn preview có highlight từ khóa

    Args:
        sources: List of source documents
        msg_idx: Message index for unique key generation
    """
    with st.expander("📚 Nguồn tài liệu"):
        for src_idx, source in enumerate(sources, 1):
            file_type = source.metadata.get("file_type", "").upper().replace(".", "")
            source_file = source.metadata.get("source_file") or source.source_file
            page = source.metadata.get("page")
            chunk_idx = source.metadata.get("chunk_index", 0)
            file_size_mb = source.metadata.get("file_size_mb", 0)
            uploaded_at = source.metadata.get("uploaded_at", "")
            title = source.metadata.get("title", source_file)
            is_used = bool(source.metadata.get("used_in_answer", False))
            rerank_score = source.metadata.get("rerank_score")

            # Icon and color by file type
            icon_map = {".PDF": "📕", ".DOCX": "📘", ".TXT": "📄"}
            color_map = {".PDF": "#e53935", ".DOCX": "#1565c0", ".TXT": "#558b2f"}
            file_icon = icon_map.get(file_type, "📄")
            color = color_map.get(file_type, "#9e9e9e")

            # Page info
            page_info = f", trang {page}" if page is not None else ""
            chunk_info = f" (chunk {chunk_idx + 1})"

            # Rerank badge
            rerank_badge = ""
            if rerank_score is not None:
                rerank_badge = f" | 🎯 relevance = {rerank_score:.3f}"

            st.markdown(
                f"""
            <div style="
                border-left: 4px solid {color};
                padding: 8px 12px;
                margin-bottom: 8px;
                border-radius: 4px;
                background: {'#fff8e1' if is_used else '#f5f5f5'};
            ">
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                    <span style="font-size:1.1em;">{file_icon}</span>
                    <strong>{title}</strong>
                    <span style="font-size:0.8em; color:#666;">{file_type}</span>
                    {"<span style='background:#4caf50;color:white;font-size:0.7em;padding:1px 5px;border-radius:3px;margin-left:4px;'>✅ Dùng trong câu trả lời</span>" if is_used else ""}
                </div>
                <div style="font-size:0.8em; color:#555;">
                    <span>📑 Nguồn: `{source_file}`{page_info}{chunk_info}{rerank_badge}</span>
                </div>
                <div style="font-size:0.78em; color:#888; margin-top:2px;">
                    💾 {file_size_mb:.2f} MB | ⏱ {uploaded_at[:10]}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            preview = source.content[:300].replace("\n", " ") + "..."
            if is_used:
                preview = f"<mark>{preview}</mark>"

            st.text_area(
                f"Noi dung {src_idx}",
                value=source.content,
                height=80,
                key=f"source_msg{msg_idx}_src{src_idx}",
                disabled=True,
            )
            st.markdown(preview, unsafe_allow_html=True)