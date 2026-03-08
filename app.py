"""
SmartDoc AI - Intelligent Document Q&A System
Main application entry point using Streamlit.

This application provides a RAG (Retrieval-Augmented Generation) system
for querying documents using local LLM (Ollama) and vector search (FAISS).
"""

import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.controllers.chat_controller import ChatController
from src.controllers.document_controller import DocumentController
from src.services.vector_store_service import FAISSVectorStoreService
from src.services.document_service import DocumentService
from src.views.chat_screen import ChatScreen
from src.views.document_screen import DocumentScreen
from src.views.settings_screen import SettingsScreen
from src.models.chat_model import ChatHistory
from src.utils.logger import setup_logger
from src.utils.constants import PAGE_TITLE, PAGE_ICON

logger = setup_logger(__name__)


class SessionStateManager:
    """Centralized session state management."""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables."""
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = ChatHistory()
            logger.info("Chat history initialized")
        
        # Vector store initialization flag
        if 'vector_store_initialized' not in st.session_state:
            st.session_state.vector_store_initialized = False
        
        # Vector store service (singleton)
        if 'vector_service' not in st.session_state:
            st.session_state.vector_service = FAISSVectorStoreService()
            logger.info("Vector store service initialized")
        
        # Document service (singleton)
        if 'document_service' not in st.session_state:
            st.session_state.document_service = DocumentService()
            logger.info("Document service initialized")
        
        # Chunk configuration
        if 'chunk_size' not in st.session_state:
            from src.utils.constants import DEFAULT_CHUNK_SIZE
            st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
        
        if 'chunk_overlap' not in st.session_state:
            from src.utils.constants import DEFAULT_CHUNK_OVERLAP
            st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    SessionStateManager.initialize()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize controllers
    chat_controller = ChatController(
        vector_service=st.session_state.vector_service
    )
    
    document_controller = DocumentController(
        document_service=st.session_state.document_service,
        vector_service=st.session_state.vector_service
    )
    
    # Sidebar navigation
    with st.sidebar:
        st.title(f"{PAGE_ICON} {PAGE_TITLE}")
        st.markdown("**Intelligent Document Q&A System**")
        st.markdown("---")
        
        # Navigation menu
        page = st.radio(
            "Navigation",
            ["💬 Chat", "📄 Documents", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Status indicators
        st.markdown("### 📊 Status")
        
        # Vector store status
        if st.session_state.vector_store_initialized:
            st.success("🟢 Documents Loaded")
        else:
            st.warning("🟡 No Documents")
        
        # Chat history status
        num_messages = len(st.session_state.chat_history)
        st.info(f"💬 {num_messages} messages")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            <p>SmartDoc AI v1.0</p>
            <p>OSSD Course - Spring 2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to appropriate screen
    if page == "💬 Chat":
        chat_screen = ChatScreen(chat_controller)
        chat_screen.render()
    
    elif page == "📄 Documents":
        doc_screen = DocumentScreen(document_controller)
        doc_screen.render()
    
    elif page == "⚙️ Settings":
        settings_screen = SettingsScreen(document_controller)
        settings_screen.render()


if __name__ == "__main__":
    logger.info("Starting SmartDoc AI application")
    main()
