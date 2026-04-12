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
from src.services.llm_service import OllamaLLMService
from src.services.n8n_service import N8NWebhookService
from src.views.chat_screen import ChatScreen
from src.views.document_screen import DocumentScreen
from src.views.settings_screen import SettingsScreen
from src.models.chat_model import ChatHistory
from src.utils.logger import setup_logger
from src.utils import ui_icons as icons
from src.utils.constants import (
    PAGE_TITLE,
    PAGE_ICON,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_STREAMLIT_REPLY_TEMPLATES,
    N8N_DEFAULT_ENABLED,
    N8N_DEFAULT_WEBHOOK_URL,
    N8N_TIMEOUT_SECONDS,
)

logger = setup_logger(__name__)

_NAV_LABELS = {
    "chat": f"{icons.CHAT} Chat",
    "documents": f"{icons.DESCRIPTION} Documents",
    "settings": f"{icons.SETTINGS} Settings",
}


@st.cache_resource(show_spinner=False)
def get_vector_service():
    """Get cached vector store service (initialized only once)."""
    logger.info("Creating vector store service (cached)")
    return FAISSVectorStoreService()


@st.cache_resource(show_spinner=False)
def get_document_service():
    """Get cached document service (initialized only once)."""
    logger.info("Creating document service (cached)")
    return DocumentService()


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
        
        # Vector store service (singleton via cache_resource)
        if 'vector_service' not in st.session_state:
            st.session_state.vector_service = get_vector_service()
            logger.info("Vector store service initialized")
        else:
            # Recover from stale cached instance that failed during a previous run.
            cached_service = st.session_state.vector_service
            if getattr(cached_service, "embeddings", None) is None:
                logger.warning("Detected unhealthy cached vector service, rebuilding it")
                get_vector_service.clear()
                st.session_state.vector_service = get_vector_service()
                logger.info("Vector store service reinitialized after health check")
        
        # Document service (singleton via cache_resource)
        if 'document_service' not in st.session_state:
            st.session_state.document_service = get_document_service()
            logger.info("Document service initialized")
        
        # Chunk configuration
        if 'chunk_size' not in st.session_state:
            from src.utils.constants import DEFAULT_CHUNK_SIZE
            st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
        
        if 'chunk_overlap' not in st.session_state:
            from src.utils.constants import DEFAULT_CHUNK_OVERLAP
            st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP

        # LLM runtime tuning
        if 'llm_model' not in st.session_state:
            st.session_state.llm_model = DEFAULT_MODEL

        if 'llm_num_ctx' not in st.session_state:
            st.session_state.llm_num_ctx = DEFAULT_NUM_CTX

        if 'llm_num_predict' not in st.session_state:
            st.session_state.llm_num_predict = DEFAULT_NUM_PREDICT

        if 'llm_keep_alive' not in st.session_state:
            st.session_state.llm_keep_alive = DEFAULT_KEEP_ALIVE

        # Streamlit reply templates (intro/body/footer)
        if 'reply_templates' not in st.session_state:
            st.session_state.reply_templates = DEFAULT_STREAMLIT_REPLY_TEMPLATES

        if 'loaded_documents' not in st.session_state:
            st.session_state.loaded_documents = []

        if 'active_source_filters' not in st.session_state:
            st.session_state.active_source_filters = []

        if 'active_file_type_filters' not in st.session_state:
            st.session_state.active_file_type_filters = []

        if 'use_hybrid_search' not in st.session_state:
            st.session_state.use_hybrid_search = False

        if 'use_rerank' not in st.session_state:
            st.session_state.use_rerank = False

        if 'retrieval_k' not in st.session_state:
            st.session_state.retrieval_k = 3

        if 'last_retrieval_stats' not in st.session_state:
            st.session_state.last_retrieval_stats = {}

        if 'retrieval_comparison' not in st.session_state:
            st.session_state.retrieval_comparison = {}

        # n8n integration settings
        if 'n8n_enabled' not in st.session_state:
            st.session_state.n8n_enabled = N8N_DEFAULT_ENABLED

        if 'n8n_webhook_url' not in st.session_state:
            st.session_state.n8n_webhook_url = N8N_DEFAULT_WEBHOOK_URL


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
    n8n_service = N8NWebhookService(
        webhook_url=st.session_state.get('n8n_webhook_url', N8N_DEFAULT_WEBHOOK_URL),
        enabled=st.session_state.get('n8n_enabled', N8N_DEFAULT_ENABLED),
        timeout_seconds=N8N_TIMEOUT_SECONDS,
    )

    llm_service = OllamaLLMService(
        model=st.session_state.get('llm_model', DEFAULT_MODEL),
        num_ctx=int(st.session_state.get('llm_num_ctx', DEFAULT_NUM_CTX)),
        num_predict=int(st.session_state.get('llm_num_predict', DEFAULT_NUM_PREDICT)),
        keep_alive=st.session_state.get('llm_keep_alive', DEFAULT_KEEP_ALIVE),
    )

    chat_controller = ChatController(
        llm_service=llm_service,
        vector_service=st.session_state.vector_service,
        n8n_service=n8n_service,
    )
    
    document_controller = DocumentController(
        document_service=st.session_state.document_service,
        vector_service=st.session_state.vector_service
    )
    
    # Sidebar navigation
    with st.sidebar:
        st.title(f"{icons.MENU_BOOK} {PAGE_TITLE}")
        st.markdown("**Intelligent Document Q&A System**")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=list(_NAV_LABELS.keys()),
            format_func=lambda k: _NAV_LABELS[k],
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.markdown(f"### {icons.BAR_CHART} Status")

        if st.session_state.vector_store_initialized:
            st.success(f"{icons.CHECK_CIRCLE} Documents loaded")
        else:
            st.warning(f"{icons.FOLDER_OFF} No documents yet")

        num_messages = len(st.session_state.chat_history)
        st.info(f"{icons.FORUM} {num_messages} messages")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            <p>SmartDoc AI v1.0</p>
            <p>OSSD Course - Spring 2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to appropriate screen
    if page == "chat":
        chat_screen = ChatScreen(chat_controller)
        chat_screen.render()

    elif page == "documents":
        doc_screen = DocumentScreen(document_controller)
        doc_screen.render()

    elif page == "settings":
        settings_screen = SettingsScreen(document_controller)
        settings_screen.render()


if __name__ == "__main__":
    logger.info("Starting SmartDoc AI application")
    main()
