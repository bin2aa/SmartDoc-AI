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
        Render source citations.
        
        Args:
            sources: List of source documents
            msg_idx: Message index for unique key generation
        """
        with st.expander("📚 View Sources"):
            for src_idx, source in enumerate(sources, 1):
                source_file = source.metadata.get("source_file") or source.source_file
                open_link = f"data/uploads/{source_file}" if source_file else ""
                is_used = bool(source.metadata.get("used_in_answer", False))
                st.markdown(f"**Source {src_idx}:** {source.get_citation()}")
                if open_link:
                    st.markdown(f"[Open source file]({open_link})")

                overlap = source.metadata.get("used_term_overlap", 0)
                if is_used:
                    st.caption(f"✅ Highlighted as used context (term overlap: {overlap})")
                else:
                    st.caption("ℹ️ Retrieved context")

                preview = source.content[:300] + "..." if len(source.content) > 300 else source.content
                if is_used:
                    preview = f"<mark>{preview}</mark>"

                st.text_area(
                    f"Context {src_idx}",
                    value=source.content,
                    height=100,
                    key=f"source_msg{msg_idx}_src{src_idx}",
                    disabled=True
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
