"""Chat screen for SmartDoc AI."""

import streamlit as st
from typing import List, Optional
from src.controllers.chat_controller import ChatController
from src.views.components import UIComponents
from src.models.document_model import Document
from src.models.chat_model import ChatHistory
from src.utils.logger import setup_logger
from src.utils.exceptions import LLMConnectionError
from src.services.persistence_service import save_chat_history, load_chat_history

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
        
        # Ensure chat_history exists (recovery mechanism)
        self._ensure_history()
        
        # Render chat interface
        self._render_chat_history()
        self._render_chat_input()
        
        self._render_retrieval_metrics()
    
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
        """Display all chat history messages."""
        history = st.session_state.get('chat_history')
        
        logger.debug(f"_render_chat_history: history={type(history).__name__}, len={len(history) if history else 'None'}")
        
        if not history or len(history) == 0:
            self.components.info_alert("Bắt đầu cuộc trò chuyện bằng cách đặt câu hỏi bên dưới")
            return
        
        # Display ALL messages from history
        for msg_idx, message in enumerate(history.messages):
            avatar = "🧑" if message.role == "user" else "🤖"
            self.components.chat_message(
                role=message.role,
                content=message.content,
                avatar=avatar
            )
            
            # Display sources if available
            if message.role == "assistant" and message.metadata and message.metadata.get('source_citations'):
                self._render_source_citations(message.metadata['source_citations'], msg_idx)
    
    def _render_source_citations(self, citations: List[str], msg_idx: int):
        """
        Render source citations from serialized citation strings.
        
        Args:
            citations: List of citation strings
            msg_idx: Message index for unique key generation
        """
        with st.expander("📚 Xem nguồn tham khảo"):
            for src_idx, citation in enumerate(citations, 1):
                st.markdown(f"**Nguồn {src_idx}:** {citation}")
    
    def _render_sources(self, sources: List[Document], msg_idx: int):
        """
        Render source citations with full document info.
        
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
        """Render chat input box with streaming response and step-by-step status.
        
        Key design: Messages are ONLY added to chat_history AFTER the full
        response is complete. Sources are stored as citation strings (not 
        Document objects) to avoid serialization issues.
        """
        if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
            logger.info(f"Chat input received: '{prompt[:50]}...'")
            
            # Display user message bubble directly (not from history)
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

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

                # Step 4: Convert sources to serializable citation strings
                source_citations = []
                if sources:
                    for src in sources:
                        try:
                            source_citations.append(src.get_citation())
                        except Exception as e:
                            logger.warning(f"Failed to get citation for source: {e}")
                            source_citations.append("[Unknown source]")

                # Step 5: NOW add both messages to history (after display is complete).
                # Store source_citations (list of strings) instead of Document objects.
                history = st.session_state.get('chat_history')
                if history is None:
                    logger.error("chat_history is None before adding messages — creating new one")
                    history = ChatHistory()
                    st.session_state.chat_history = history
                
                history.add_message("user", prompt)
                assistant_metadata = {'source_citations': source_citations} if source_citations else None
                history.add_message("assistant", formatted_answer, metadata=assistant_metadata)
                logger.info(f"Added user + assistant messages to history (total: {len(history)})")

                # Step 6: Persist chat history to disk with verification
                save_ok = save_chat_history(history)
                if save_ok:
                    logger.info("Chat history saved to disk successfully")
                else:
                    logger.error("FAILED to save chat history to disk!")

            except LLMConnectionError as e:
                logger.error(f"LLM error while processing query: {e}")
                # Still save the user message to history even if LLM fails
                self._save_user_message(prompt)
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
                import traceback
                logger.error(traceback.format_exc())
                # Still save the user message to history even if processing fails
                self._save_user_message(prompt)
                self.components.error_alert(
                    "Không thể xử lý câu hỏi của bạn",
                    details=str(e)
                )
    
    def _save_user_message(self, prompt: str):
        """Save user message to history even when processing fails."""
        try:
            history = st.session_state.get('chat_history')
            if history is None:
                history = ChatHistory()
                st.session_state.chat_history = history
            history.add_message("user", prompt)
            save_chat_history(history)
            logger.info(f"Saved user message to history (total: {len(history)})")
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")
    
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