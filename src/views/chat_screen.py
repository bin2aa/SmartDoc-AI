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
        
        # Check if vector store is ready
        if not st.session_state.get('vector_store_initialized', False):
            self._render_empty_state()
            return
        
        # Render chat interface
        self._render_chat_history()
        self._render_chat_input()
        
        # Sidebar actions
        self._render_sidebar_actions()
    
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
                st.markdown(f"**Source {src_idx}:** {source.get_citation()}")
                st.text_area(
                    f"Context {src_idx}",
                    value=source.content[:300] + "..." if len(source.content) > 300 else source.content,
                    height=100,
                    key=f"source_msg{msg_idx}_src{src_idx}",
                    disabled=True
                )
    
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
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                self.controller.clear_history()
                st.rerun()
        
        with col2:
            num_messages = len(st.session_state.get('chat_history', []))
            st.metric("Messages", num_messages)
