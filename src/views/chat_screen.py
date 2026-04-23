"""Chat screen for SmartDoc AI — main orchestrator."""

from typing import List

import streamlit as st

from src.controllers.chat_controller import ChatController
from src.views.components import UIComponents, icon
from src.views.chat_history_renderer import render_chat_history
from src.views.chat_input_handler import render_chat_input
from src.views.rag_comparison import render_comparison_display, render_retrieval_metrics
from src.models.chat_model import ChatHistory
from src.services.persistence_service import load_chat_history
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChatScreen:
    """
    Chat interface screen following MVC pattern.

    Displays chat history and handles user interactions.
    Delegates rendering to specialized modules.
    """

    def __init__(self, controller: ChatController):
        """
        Initialize chat screen.

        Args:
            controller: Chat controller instance
        """
        self.controller = controller
        self.components = UIComponents()

    def render(self):
        """Render the chat screen."""
        st.markdown(f"## {icon('chat')} Chat với tài liệu", unsafe_allow_html=True)

        # Check if vector store is ready
        if not st.session_state.get('vector_store_initialized', False):
            self._render_empty_state()
            return

        # Ensure chat_history exists (recovery mechanism)
        self._ensure_history()

        # Render file filter selector above chat input
        self._render_file_filter()

        # Render chat interface
        render_chat_history(st.session_state.get('chat_history'), self.components)
        render_chat_input(self.controller, self.components)

        # Render comparison (persists across reruns via session_state)
        render_comparison_display()

        render_retrieval_metrics()

    def _ensure_history(self):
        """Ensure chat_history exists in session_state, recover from disk if needed."""
        history = st.session_state.get('chat_history')
        if history is not None and len(history) > 0:
            logger.debug(f"Chat history OK: {len(history)} messages")
            return

        # History is empty or missing — try to recover from disk
        if history is None:
            logger.warning("chat_history is None in session_state — recovering from disk")
            saved = load_chat_history()
            if saved and len(saved) > 0:
                st.session_state.chat_history = saved
                logger.info(f"Recovered chat history from disk ({len(saved)} messages)")
            else:
                st.session_state.chat_history = ChatHistory()
                logger.info("Created new empty ChatHistory")
        elif len(history) == 0:
            # Empty history in memory — check disk as well
            saved = load_chat_history()
            if saved and len(saved) > 0:
                logger.warning(f"Memory history is empty but disk has {len(saved)} messages — recovering")
                st.session_state.chat_history = saved

    def _render_file_filter(self):
        """Render compact file selector to filter search by specific documents.

        Displays a multiselect with all uploaded documents, allowing users to
        restrict retrieval to selected files only.  Also renders clickable
        file chips for quick toggle and @mention hints.
        """
        loaded_docs: List[dict] = st.session_state.get("loaded_documents", [])
        if not loaded_docs:
            return

        # Build sorted unique file name list
        source_names = sorted({
            item.get("name", "")
            for item in loaded_docs
            if item.get("name")
        })
        if not source_names:
            return

        # Apply any pending filter changes BEFORE the multiselect widget is
        # instantiated.  Buttons below write to a temporary key
        # (_pending_source_filters) and call st.rerun().  On the next run we
        # apply that pending value here so the widget picks it up cleanly.
        if "_pending_source_filters" in st.session_state:
            st.session_state.active_source_filters = (
                st.session_state.pop("_pending_source_filters")
            )

        # ── Multiselect: use active_source_filters as widget key directly ──
        # This avoids the sync conflict that caused auto-deselect on rerun.
        selected_sources = st.multiselect(
            "🎯 Chọn file để tìm kiếm (để trống = tất cả)",
            options=source_names,
            default=st.session_state.get("active_source_filters", []),
            key="active_source_filters",
            help=(
                "Chọn file cụ thể để giới hạn tìm kiếm. "
                "Hoặc dùng @filename trong câu hỏi (VD: @report.pdf nội dung là gì?)."
            ),
        )

        # Visual indicator + clear button
        indicator_cols = st.columns([5, 1])
        with indicator_cols[0]:
            if selected_sources:
                badges = " ".join(f"`{name}`" for name in selected_sources)
                st.caption(f"📚 Đang tìm trong: {badges} ({len(selected_sources)}/{len(source_names)} files)")
            else:
                st.caption(f"📚 Tìm trong tất cả {len(source_names)} files")
        with indicator_cols[1]:
            if selected_sources:
                if st.button(
                    "✕ Xóa",
                    key="clear_file_filter_btn",
                    help="Xóa bộ lọc file",
                ):
                    # Write to a temporary key and rerun; the value will be
                    # applied *before* the multiselect widget on the next run.
                    st.session_state._pending_source_filters = []
                    st.rerun()

        # ── Quick-select file chips ──────────────────────────────────────
        st.markdown(
            '<span style="font-size:0.8em;color:#888;">💡 Click file để chọn nhanh · Gõ <b>@filename</b> trong chat để focus</span>',
            unsafe_allow_html=True,
        )
        chips = st.columns(min(len(source_names), 5))
        for idx, name in enumerate(source_names):
            col = chips[idx % len(chips)]
            is_selected = name in selected_sources
            label = f"{'✅' if is_selected else '📄'} {name}"
            with col:
                if st.button(
                    label,
                    key=f"chip_{name}",
                    help=f"{'Bỏ chọn' if is_selected else 'Chọn'} {name}",
                    use_container_width=True,
                ):
                    current = list(st.session_state.get("active_source_filters", []))
                    if is_selected:
                        current = [f for f in current if f != name]
                    else:
                        current.append(name)
                    # Write to a temporary key and rerun; the value will be
                    # applied *before* the multiselect widget on the next run.
                    st.session_state._pending_source_filters = current
                    st.rerun()

    def _render_empty_state(self):
        """Show empty state when no documents loaded."""
        self.components.info_alert("Vui lòng tải tài liệu lên trước trong tab Documents")

        st.markdown(f"""
        ### {icon('waving_hand')} Chào mừng đến SmartDoc AI!

        Để bắt đầu:
        1. Nhấn **Documents** ở sidebar bên trái
        2. Tải lên file PDF, DOCX, hoặc TXT
        3. Quay lại đây để đặt câu hỏi về tài liệu

        SmartDoc AI sử dụng RAG (Retrieval-Augmented Generation) để trả lời câu hỏi dựa trên tài liệu của bạn.
        """, unsafe_allow_html=True)