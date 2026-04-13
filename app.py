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
from src.views.chat_screen import ChatScreen
from src.views.document_screen import DocumentScreen
from src.views.settings_screen import SettingsScreen
from src.models.chat_model import ChatHistory
from src.services.persistence_service import (
    load_chat_history,
    save_chat_history,
    load_settings,
    save_settings,
    load_loaded_docs,
    save_loaded_docs,
    load_faiss_index,
    save_faiss_index,
)
from src.views.components import icon
from src.utils.logger import setup_logger
from src.utils.constants import (
    PAGE_TITLE,
    PAGE_ICON,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_STREAMLIT_REPLY_TEMPLATES,
    AVAILABLE_MODELS,
)

logger = setup_logger(__name__)


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
        """Initialize all session state variables, loading persisted data if available."""
        # ── Load persisted settings first ───────────────────────
        saved_settings = load_settings()

        # ── Chat history (single) ───────────────────────────────
        if 'chat_history' not in st.session_state:
            saved_history = load_chat_history()
            if saved_history:
                st.session_state.chat_history = saved_history
                logger.info(f"Restored chat history ({len(saved_history)} messages) from disk")
            else:
                st.session_state.chat_history = ChatHistory()
                logger.info("New chat history created")
        else:
            existing = st.session_state.chat_history
            if existing is not None:
                logger.info(f"Chat history already in session_state: {len(existing)} messages")
            else:
                logger.warning("Chat history in session_state is None — will be recovered by ChatScreen")

        # Current page: "chat", "documents", "settings"
        if 'nav_page' not in st.session_state:
            st.session_state.nav_page = "chat"
        
        # ── Vector store ────────────────────────────────────────
        if 'vector_store_initialized' not in st.session_state:
            st.session_state.vector_store_initialized = False
        
        if 'vector_service' not in st.session_state:
            st.session_state.vector_service = get_vector_service()
            logger.info("Vector store service initialized")
        else:
            cached_service = st.session_state.vector_service
            if getattr(cached_service, "embeddings", None) is None:
                logger.warning("Detected unhealthy cached vector service, rebuilding it")
                get_vector_service.clear()
                st.session_state.vector_service = get_vector_service()
                logger.info("Vector store service reinitialized after health check")

        # Always try to restore FAISS index from disk if not currently initialized.
        # This handles page refresh where the in-memory index is lost but disk copy exists.
        vs = st.session_state.vector_service
        if not st.session_state.vector_store_initialized or not vs.is_initialized:
            if load_faiss_index(vs):
                st.session_state.vector_store_initialized = True
                logger.info("FAISS index restored from disk")
        
        # Document service (singleton via cache_resource)
        if 'document_service' not in st.session_state:
            st.session_state.document_service = get_document_service()
            logger.info("Document service initialized")
        
        # ── Settings (with persistence) ─────────────────────────
        if 'chunk_size' not in st.session_state:
            st.session_state.chunk_size = saved_settings.get('chunk_size', DEFAULT_CHUNK_SIZE)
        
        if 'chunk_overlap' not in st.session_state:
            st.session_state.chunk_overlap = saved_settings.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)

        if 'llm_model' not in st.session_state:
            st.session_state.llm_model = saved_settings.get('llm_model', DEFAULT_MODEL)

        if 'llm_num_ctx' not in st.session_state:
            st.session_state.llm_num_ctx = saved_settings.get('llm_num_ctx', DEFAULT_NUM_CTX)

        if 'llm_num_predict' not in st.session_state:
            st.session_state.llm_num_predict = saved_settings.get('llm_num_predict', DEFAULT_NUM_PREDICT)

        if 'llm_keep_alive' not in st.session_state:
            st.session_state.llm_keep_alive = saved_settings.get('llm_keep_alive', DEFAULT_KEEP_ALIVE)

        # Streamlit reply templates (intro/body/footer)
        if 'reply_templates' not in st.session_state:
            st.session_state.reply_templates = DEFAULT_STREAMLIT_REPLY_TEMPLATES

        # ── Loaded documents (with persistence) ─────────────────
        if 'loaded_documents' not in st.session_state:
            saved_docs = load_loaded_docs()
            st.session_state.loaded_documents = saved_docs
            if saved_docs:
                logger.info(f"Restored {len(saved_docs)} loaded document metadata from disk")

        if 'active_source_filters' not in st.session_state:
            st.session_state.active_source_filters = []

        if 'active_file_type_filters' not in st.session_state:
            st.session_state.active_file_type_filters = []

        if 'use_hybrid_search' not in st.session_state:
            st.session_state.use_hybrid_search = saved_settings.get('use_hybrid_search', False)

        if 'use_rerank' not in st.session_state:
            st.session_state.use_rerank = saved_settings.get('use_rerank', False)

        if 'retrieval_k' not in st.session_state:
            st.session_state.retrieval_k = saved_settings.get('retrieval_k', 3)

        if 'last_retrieval_stats' not in st.session_state:
            st.session_state.last_retrieval_stats = {}

        if 'retrieval_comparison' not in st.session_state:
            st.session_state.retrieval_comparison = {}


def _clear_chat_history():
    """Clear chat history and persist."""
    st.session_state.chat_history.clear()
    save_chat_history(st.session_state.chat_history)
    st.rerun()


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
    
    # Custom CSS — Material Symbols font + base styles
    st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
    <style>
        .main { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize controllers
    llm_service = OllamaLLMService(
        model=st.session_state.get('llm_model', DEFAULT_MODEL),
        num_ctx=int(st.session_state.get('llm_num_ctx', DEFAULT_NUM_CTX)),
        num_predict=int(st.session_state.get('llm_num_predict', DEFAULT_NUM_PREDICT)),
        keep_alive=st.session_state.get('llm_keep_alive', DEFAULT_KEEP_ALIVE),
    )

    chat_controller = ChatController(
        llm_service=llm_service,
        vector_service=st.session_state.vector_service,
    )
    
    document_controller = DocumentController(
        document_service=st.session_state.document_service,
        vector_service=st.session_state.vector_service
    )
    
    # ═══════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════
    current_page = st.session_state.get('nav_page', 'chat')

    with st.sidebar:
        # ── Header ─────────────────────────────────────────────
        st.title(f"{PAGE_ICON} {PAGE_TITLE}")

        # ── Status ─────────────────────────────────────────────
        if st.session_state.vector_store_initialized:
            docs_count = len(st.session_state.get("loaded_documents", []))
            st.markdown(
                f'<span class="material-symbols-outlined" style="vertical-align:middle;font-size:1em;color:#4caf50;">check_circle</span> {docs_count} doc(s) loaded',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="material-symbols-outlined" style="vertical-align:middle;font-size:1em;color:#ff9800;">radio_button_unchecked</span> No documents',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Navigation ─────────────────────────────────────────
        btn_type_chat = "primary" if current_page == "chat" else "secondary"
        if st.button("Chat", use_container_width=True, type=btn_type_chat):
            st.session_state.nav_page = "chat"
            st.rerun()

        btn_type_docs = "primary" if current_page == "documents" else "secondary"
        if st.button("Documents", use_container_width=True, type=btn_type_docs):
            st.session_state.nav_page = "documents"
            st.rerun()

        btn_type_set = "primary" if current_page == "settings" else "secondary"
        if st.button("Cài đặt", use_container_width=True, type=btn_type_set):
            st.session_state.nav_page = "settings"
            st.rerun()

        st.markdown("---")

        # ── Clear Chat (only on chat page) ─────────────────────
        if current_page == "chat":
            if st.button("Xóa lịch sử chat", use_container_width=True):
                _clear_chat_history()

            st.markdown("---")

        # ── Model Switch (bottom) ──────────────────────────────
        st.markdown(
            f'**{icon("smart_toy")} Model**',
            unsafe_allow_html=True,
        )
        current_model = st.session_state.get('llm_model', DEFAULT_MODEL)
        model_options = list(AVAILABLE_MODELS)
        if current_model not in model_options:
            model_options.append(current_model)

        new_model = st.selectbox(
            "Switch model",
            options=model_options,
            index=model_options.index(current_model) if current_model in model_options else 0,
            label_visibility="collapsed",
        )
        if new_model != current_model:
            st.session_state.llm_model = new_model
            # Persist model change to disk
            from src.services.persistence_service import save_settings
            save_settings({
                "chunk_size": st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
                "chunk_overlap": st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
                "llm_model": new_model,
                "llm_num_ctx": st.session_state.get("llm_num_ctx", DEFAULT_NUM_CTX),
                "llm_num_predict": st.session_state.get("llm_num_predict", DEFAULT_NUM_PREDICT),
                "llm_keep_alive": st.session_state.get("llm_keep_alive", DEFAULT_KEEP_ALIVE),
                "use_hybrid_search": st.session_state.get("use_hybrid_search", False),
                "use_rerank": st.session_state.get("use_rerank", False),
                "retrieval_k": st.session_state.get("retrieval_k", 3),
            })
            st.rerun()

        st.markdown("---")

        # Footer
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.75em;'>
            <p>SmartDoc AI v1.0 · OSSD Spring 2026</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════
    # MAIN CONTENT — Route based on nav_page
    # ═══════════════════════════════════════════════════════════
    page = st.session_state.get('nav_page', 'chat')

    if page == "documents":
        doc_screen = DocumentScreen(document_controller)
        doc_screen.render()

    elif page == "settings":
        settings_screen = SettingsScreen(document_controller)
        settings_screen.render()

    else:
        # Default: Chat screen
        chat_screen = ChatScreen(chat_controller)
        chat_screen.render()


if __name__ == "__main__":
    logger.info("Starting SmartDoc AI application")
    main()