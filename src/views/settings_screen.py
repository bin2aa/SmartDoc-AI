"""Settings screen for SmartDoc AI — main orchestrator."""

import streamlit as st

from src.controllers.document_controller import DocumentController
from src.controllers.chat_controller import ChatController
from src.views.components import UIComponents, icon
from src.views.chunk_settings import render_chunk_settings
from src.views.retrieval_settings import render_retrieval_settings
from src.views.llm_settings import render_llm_settings, render_system_info, render_rerank_benchmark
from src.services.persistence_service import save_settings
from src.utils.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_KEEP_ALIVE,
    RAG_TYPE_STANDARD,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SettingsScreen:
    """
    Settings configuration screen.

    Allows users to configure RAG parameters.
    Delegates rendering to specialized settings modules.
    """

    def __init__(
        self,
        document_controller: DocumentController,
        chat_controller: ChatController = None,
    ):
        """
        Initialize settings screen.

        Args:
            document_controller: Document controller instance
            chat_controller: Chat controller instance (for retrieval benchmark)
        """
        self.document_controller = document_controller
        self.chat_controller = chat_controller
        self.components = UIComponents()

    def render(self):
        """Render the settings screen."""
        st.markdown(f"## {icon('settings')} Settings", unsafe_allow_html=True)

        st.markdown("""
        Configure the RAG (Retrieval-Augmented Generation) pipeline parameters.

        **Note:** Changes to chunk settings will only affect newly uploaded documents.
        """)

        st.markdown("---")

        # Chunk Configuration
        render_chunk_settings(
            self.document_controller,
            self.components,
            self._persist_current_settings,
        )

        st.markdown("---")

        # Retrieval Strategy
        render_retrieval_settings(
            self.chat_controller,
            self.components,
            self._persist_current_settings,
        )

        # Re-ranking benchmark
        render_rerank_benchmark(self.document_controller, self.components)

        st.markdown("---")

        # LLM Configuration
        render_llm_settings(
            self.components,
            self._persist_current_settings,
        )

        st.markdown("---")

        # System Info
        render_system_info(self.components)

    def _persist_current_settings(self) -> None:
        """Save all current settings from session state to disk."""
        settings = {
            "chunk_size": st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
            "chunk_overlap": st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            "llm_model": st.session_state.get("llm_model", DEFAULT_MODEL),
            "llm_num_ctx": st.session_state.get("llm_num_ctx", DEFAULT_NUM_CTX),
            "llm_num_predict": st.session_state.get("llm_num_predict", DEFAULT_NUM_PREDICT),
            "llm_keep_alive": st.session_state.get("llm_keep_alive", DEFAULT_KEEP_ALIVE),
            "use_hybrid_search": st.session_state.get("use_hybrid_search", False),
            "use_rerank": st.session_state.get("use_rerank", False),
            "use_self_rag": st.session_state.get("use_self_rag", False),
            "retrieval_k": st.session_state.get("retrieval_k", 3),
            "rag_type": st.session_state.get("rag_type", RAG_TYPE_STANDARD),
            "compare_rag": st.session_state.get("compare_rag", False),
        }
        save_settings(settings)
        logger.info("Settings persisted to disk")