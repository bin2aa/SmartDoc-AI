"""Chat screen for SmartDoc AI."""

import streamlit as st
from typing import List, Optional
from src.controllers.chat_controller import ChatController
from src.views.components import UIComponents
from src.models.document_model import Document
from src.models.chat_model import ChatHistory
from src.utils.logger import setup_logger
from src.utils.exceptions import LLMConnectionError
from src.services.persistence_service import save_chat_history

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
        st.title("💬 Chat với tài liệu")
        
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
        self.components.info_alert("Vui lòng tải tài liệu lên trước trong tab Documents")
        
        st.markdown("""
        ### 👋 Chào mừng đến SmartDoc AI!
        
        Để bắt đầu:
        1. Nhấn **📄 Documents** ở sidebar bên trái
        2. Tải lên file PDF, DOCX, hoặc TXT
        3. Quay lại đây để đặt câu hỏi về tài liệu
        
        SmartDoc AI sử dụng RAG (Retrieval-Augmented Generation) để trả lời câu hỏi dựa trên tài liệu của bạn.
        """)
    
    def _render_chat_history(self):
        """Display chat history."""
        history = st.session_state.get('chat_history')
        
        if not history or len(history) == 0:
            self.components.info_alert("Bắt đầu cuộc trò chuyện bằng cách đặt câu hỏi bên dưới")
            return
        
        # Display messages
        for msg_idx, message in enumerate(history.messages):
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
        Render source citations.
        
        Args:
            sources: List of source documents
            msg_idx: Message index for unique key generation
        """
        with st.expander("📚 Xem nguồn tham khảo"):
            for src_idx, source in enumerate(sources, 1):
                source_file = source.metadata.get("source_file") or source.source_file
                open_link = f"data/uploads/{source_file}" if source_file else ""
                is_used = bool(source.metadata.get("used_in_answer", False))
                st.markdown(f"**Nguồn {src_idx}:** {source.get_citation()}")
                if open_link:
                    st.markdown(f"[Mở file nguồn]({open_link})")

                overlap = source.metadata.get("used_term_overlap", 0)
                if is_used:
                    st.caption(f"✅ Được sử dụng trong câu trả lời (term overlap: {overlap})")
                else:
                    st.caption("ℹ️ Ngữ cảnh đã truy xuất")

                preview = source.content[:300] + "..." if len(source.content) > 300 else source.content
                if is_used:
                    preview = f"<mark>{preview}</mark>"

                st.text_area(
                    f"Ngữ cảnh {src_idx}",
                    value=source.content,
                    height=100,
                    key=f"source_msg{msg_idx}_src{src_idx}",
                    disabled=True
                )
                st.markdown(preview, unsafe_allow_html=True)
    
    def _render_chat_input(self):
        """Render chat input box with streaming response and step-by-step status."""
        if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
            # Add user message to history and display it
            self._add_user_message(prompt)
            self.components.chat_message("user", prompt, avatar="🧑")

            # Process query with streaming
            try:
                # Step 1: Retrieval with status indicator
                with st.status("🔄 Đang xử lý câu hỏi...", expanded=True) as status:
                    stream_gen, sources = self.controller.process_query_stream(
                        prompt,
                        status_container=status,
                    )
                    # Mark retrieval complete
                    status.update(
                        label="✅ Đã tìm thấy tài liệu liên quan!",
                        state="complete",
                        expanded=False,
                    )

                # Step 2: Stream LLM response into a dedicated chat message bubble
                with st.chat_message("assistant", avatar="🤖"):
                    response_text = st.write_stream(stream_gen)

                # Step 3: Post-processing
                self.controller._mark_used_chunks(answer=response_text, sources=sources)
                formatted_answer = self.controller.format_reply_for_streamlit(response_text, sources)
                self.controller.notify_n8n_chat_event(
                    question=prompt,
                    formatted_answer=formatted_answer,
                    raw_answer=response_text,
                    sources=sources,
                )

                # Step 4: Save to history (use formatted_answer for consistent display)
                self._add_assistant_message(formatted_answer, sources)

                # Step 5: Persist chat history to disk
                save_chat_history(st.session_state.chat_history)

            except LLMConnectionError as e:
                logger.error(f"LLM error while processing query: {e}")
                self.components.error_alert(
                    "Không thể kết nối đến Ollama",
                    details=(
                        f"{str(e)}\n\n"
                        "Kiểm tra:\n"
                        "1) Đảm bảo Ollama đang chạy: ollama serve\n"
                        "2) Kiểm tra model đã tải: ollama list\n"
                        "3) Tải model nếu chưa có: ollama pull <model_name>\n"
                        "4) Vào Settings để đổi model hoặc giảm num_ctx nếu cần"
                    ),
                )
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                self.components.error_alert(
                    "Không thể xử lý câu hỏi của bạn",
                    details=str(e)
                )
    
    def _add_user_message(self, content: str):
        """Add user message to chat history."""
        history = st.session_state.get('chat_history')
        if history:
            history.add_message("user", content)
        logger.info(f"User message added: {content[:50]}...")
    
    def _add_assistant_message(self, content: str, sources: List[Document] = None):
        """Add assistant message to chat history."""
        history = st.session_state.get('chat_history')
        if history:
            metadata = {'sources': sources} if sources else None
            history.add_message("assistant", content, metadata=metadata)
        logger.info(f"Assistant message added: {content[:50]}...")
    
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