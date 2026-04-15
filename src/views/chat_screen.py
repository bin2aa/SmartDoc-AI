"""Chat screen for SmartDoc AI."""

import streamlit as st
from typing import List
from src.controllers.chat_controller import ChatController
from src.views.components import UIComponents
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import LLMConnectionError

logger = setup_logger(__name__)


class ChatScreen:
    """
    Chat interface screen following MVC pattern.
    
    Displays chat history and handles user interactions.
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
        st.title("💬 Chat with Your Documents")

        # Sidebar is always shown so users can review history.
        self._render_sidebar_actions()
        
        # Check if vector store is ready
        if not st.session_state.get('vector_store_initialized', False):
            self._render_empty_state()
            return
        
        # Render chat interface
        self._render_chat_history()
        self._render_chat_input()
        
        self._render_retrieval_metrics()
    
    def _render_empty_state(self):
        """Show empty state when no documents loaded."""
        self.components.info_alert("Please upload documents first in the Documents tab")
        
        st.markdown("""
        ### 👋 Welcome to SmartDoc AI!
        
        To get started:
        1. Go to the **📄 Documents** tab
        2. Upload a PDF, DOCX, or TXT file
        3. Come back here to ask questions about your documents
        
        SmartDoc AI uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on your documents.
        """)
    
    def _render_chat_history(self):
        """Display chat history."""
        chat_history = st.session_state.get('chat_history')
        
        if not chat_history or len(chat_history) == 0:
            self.components.info_alert("Start a conversation by asking a question below")
            return
        
        # Display messages
        for msg_idx, message in enumerate(chat_history.messages):
            avatar = "🧑" if message.role == "user" else "🤖"
            self.components.chat_message(
                role=message.role,
                content=message.content,
                avatar=avatar
            )
            
            # Display sources if available
            if message.role == "assistant" and message.metadata and message.metadata.get('sources'):
                self._render_sources(message.metadata['sources'], msg_idx)
    
    def _render_sources(self, sources: List[Document], msg_idx: int):
        """
        Render source citations with rich document metadata.

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
                icon = icon_map.get(file_type, "📄")
                color = color_map.get(file_type, "#9e9e9e")

                # Page info
                page_info = f", trang {page}" if page is not None else ""
                chunk_info = f" (chunk {chunk_idx + 1})"

                # Rerank badge
                rerank_badge = ""
                if rerank_score is not None:
                    rerank_badge = f" | 🎯 relevance = {rerank_score:.3f}"

                st.markdown(f"""
                <div style="
                    border-left: 4px solid {color};
                    padding: 8px 12px;
                    margin-bottom: 8px;
                    border-radius: 4px;
                    background: {'#fff8e1' if is_used else '#f5f5f5'};
                ">
                    <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                        <span style="font-size:1.1em;">{icon}</span>
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
                """, unsafe_allow_html=True)

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
    
    def _render_chat_input(self):
        """Render chat input box."""
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            self._add_user_message(prompt)
            
            # Get AI response
            with self.components.loading_spinner("🤔 Thinking..."):
                try:
                    answer, sources = self.controller.process_query(prompt)
                    formatted_answer = self.controller.format_reply_for_streamlit(answer, sources)
                    self.controller.notify_n8n_chat_event(
                        question=prompt,
                        formatted_answer=formatted_answer,
                        raw_answer=answer,
                        sources=sources,
                    )
                    self._add_assistant_message(formatted_answer, sources)
                    st.rerun()
                except LLMConnectionError as e:
                    logger.error(f"LLM error while processing query: {e}")
                    self.components.error_alert(
                        "Khong du RAM de chay model hien tai",
                        details=(
                            f"{str(e)}\n\n"
                            "Goi y giam RAM:\n"
                            "1) Vao tab Settings -> LLM Configuration\n"
                            "2) Chon model nhe da duoc cai trong may\n"
                            "3) Giam num_ctx xuong 256 va num_predict xuong 64\n"
                            "4) Dat keep_alive = 0m de giai phong RAM sau moi cau hoi\n"
                            "5) Neu model chua co, pull truoc: ollama pull <model_name>"
                        ),
                    )
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    self.components.error_alert(
                        "Failed to process your question",
                        details=str(e)
                    )
    
    def _add_user_message(self, content: str):
        """
        Add user message to chat.
        
        Args:
            content: Message content
        """
        st.session_state.chat_history.add_message("user", content)
        logger.info(f"User message added: {content[:50]}...")
    
    def _add_assistant_message(self, content: str, sources: List[Document] = None):
        """
        Add assistant message to chat.
        
        Args:
            content: Message content
            sources: Optional source documents
        """
        metadata = {'sources': sources} if sources else None
        st.session_state.chat_history.add_message("assistant", content, metadata=metadata)
        logger.info(f"Assistant message added: {content[:50]}...")
    
    def _render_sidebar_actions(self):
        """Render sidebar action buttons."""
        st.sidebar.markdown("---")
        self.components.sidebar_section("Chat Actions", "🔧")

        confirm_clear = st.sidebar.checkbox("Confirm clear history")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("🗑️ Clear History", use_container_width=True, disabled=not confirm_clear):
                self.controller.clear_history()
                st.rerun()

        with col2:
            num_messages = len(st.session_state.get('chat_history', []))
            st.metric("Messages", num_messages)

        st.sidebar.markdown("---")
        self.components.sidebar_section("Conversation History", "🕘")
        self._render_sidebar_history()

    def _render_sidebar_history(self):
        """Render compact conversation history in sidebar."""
        chat_history = st.session_state.get("chat_history")
        if not chat_history or len(chat_history) == 0:
            st.sidebar.caption("No messages yet")
            return

        for idx, message in enumerate(chat_history.get_recent(20), start=1):
            if message.role != "user":
                continue
            st.sidebar.markdown(f"{idx}. {message.content}")

    def _render_retrieval_metrics(self):
        """Show retrieval strategy metrics for hybrid/pure-vector comparison."""
        stats = st.session_state.get("last_retrieval_stats", {})
        if not stats:
            return

        with st.expander("📈 Retrieval Metrics"):
            st.json(stats)
            comparison = st.session_state.get("retrieval_comparison")
            if comparison:
                st.markdown("**Hybrid vs Vector**")
                st.write(comparison)
