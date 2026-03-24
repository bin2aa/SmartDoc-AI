"""Settings screen for SmartDoc AI."""

import streamlit as st
from src.controllers.document_controller import DocumentController
from src.views.components import UIComponents
from src.utils.logger import setup_logger
from src.utils.constants import *

logger = setup_logger(__name__)


class SettingsScreen:
    """
    Settings configuration screen.
    
    Allows users to configure RAG parameters.
    """
    
    def __init__(self, document_controller: DocumentController):
        """
        Initialize settings screen.
        
        Args:
            document_controller: Document controller instance
        """
        self.document_controller = document_controller
        self.components = UIComponents()
    
    def render(self):
        """Render the settings screen."""
        st.title("⚙️ Settings")
        
        st.markdown("""
        Configure the RAG (Retrieval-Augmented Generation) pipeline parameters.
        
        **Note:** Changes to chunk settings will only affect newly uploaded documents.
        """)
        
        st.markdown("---")
        
        # Chunk Configuration
        self._render_chunk_settings()
        
        st.markdown("---")
        
        # LLM Configuration
        self._render_llm_settings()
        
        st.markdown("---")

        # n8n Integration
        self._render_n8n_settings()

        st.markdown("---")
        
        # System Info
        self._render_system_info()
    
    def _render_chunk_settings(self):
        """Render chunk configuration settings."""
        st.subheader("📝 Text Chunking Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE),
                step=100,
                help="Size of text chunks for processing. Larger chunks = more context but less precision."
            )
        
        with col2:
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=50,
                max_value=300,
                value=st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP),
                step=10,
                help="Overlap between consecutive chunks. Higher overlap = better continuity."
            )
        
        if st.button("💾 Apply Chunk Settings", type="primary"):
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            self.document_controller.update_chunk_config(chunk_size, chunk_overlap)
    
    def _render_llm_settings(self):
        """Render LLM configuration settings."""
        st.subheader("🤖 LLM Configuration")

        col1, col2 = st.columns(2)

        with col1:
            llm_model = st.text_input(
                "Ollama Model",
                value=st.session_state.get('llm_model', DEFAULT_MODEL),
                help="Example: qwen2.5:1.5b or qwen2.5:0.5b (lighter RAM)",
            )

            llm_num_ctx = st.slider(
                "Context Window (num_ctx)",
                min_value=256,
                max_value=4096,
                value=int(st.session_state.get('llm_num_ctx', DEFAULT_NUM_CTX)),
                step=256,
                help="Lower value uses less RAM. Recommended 512-1024 on low-memory machines.",
            )

        with col2:
            llm_num_predict = st.slider(
                "Max Output Tokens (num_predict)",
                min_value=64,
                max_value=1024,
                value=int(st.session_state.get('llm_num_predict', DEFAULT_NUM_PREDICT)),
                step=64,
                help="Lower value reduces RAM and response time.",
            )

            llm_keep_alive = st.selectbox(
                "Model Keep Alive",
                options=["0m", "1m", "5m", "30m"],
                index=["0m", "1m", "5m", "30m"].index(
                    st.session_state.get('llm_keep_alive', DEFAULT_KEEP_ALIVE)
                ) if st.session_state.get('llm_keep_alive', DEFAULT_KEEP_ALIVE) in ["0m", "1m", "5m", "30m"] else 0,
                help="0m unloads model after each request to free RAM.",
            )

        col_apply, col_preset = st.columns(2)
        with col_apply:
            if st.button("💾 Apply LLM Settings", type="primary"):
                st.session_state.llm_model = llm_model.strip()
                st.session_state.llm_num_ctx = int(llm_num_ctx)
                st.session_state.llm_num_predict = int(llm_num_predict)
                st.session_state.llm_keep_alive = llm_keep_alive
                self.components.success_alert("LLM settings updated")

        with col_preset:
            if st.button("🪶 Apply Low-RAM Preset"):
                st.session_state.llm_model = LOW_MEMORY_FALLBACK_MODEL
                st.session_state.llm_num_ctx = 256
                st.session_state.llm_num_predict = 64
                st.session_state.llm_keep_alive = "0m"
                self.components.success_alert(
                    f"Applied low-RAM preset: model={LOW_MEMORY_FALLBACK_MODEL}, num_ctx=256, num_predict=64"
                )

        st.info(
            "💡 Tip: If you see memory errors, use an installed lightweight model, num_ctx=256, num_predict=64, keep_alive=0m."
        )
    
    def _render_system_info(self):
        """Render system information."""
        st.subheader("🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Embedding Model**")
            st.code(EMBEDDING_MODEL, language="text")
        
        with col2:
            st.markdown("**Vector Database**")
            st.code("FAISS (Local)", language="text")
        
        st.markdown("**Ollama Server**")
        st.code(OLLAMA_BASE_URL, language="text")
        
        # Connection test
        if st.button("🔌 Test Ollama Connection"):
            with self.components.loading_spinner("Testing connection..."):
                try:
                    from src.services.llm_service import OllamaLLMService
                    llm = OllamaLLMService()
                    self.components.success_alert("Connected to Ollama successfully!")
                except Exception as e:
                    self.components.error_alert(
                        "Cannot connect to Ollama",
                        details=str(e)
                    )

    def _render_n8n_settings(self):
        """Render n8n webhook integration settings."""
        st.subheader("🔄 n8n Integration")

        n8n_enabled = st.checkbox(
            "Enable n8n webhook on each chat",
            value=st.session_state.get("n8n_enabled", False),
            help="When enabled, every Streamlit chat Q&A is sent to n8n webhook.",
        )

        n8n_webhook_url = st.text_input(
            "n8n Webhook URL",
            value=st.session_state.get("n8n_webhook_url", "http://localhost:5678/webhook/smartdoc-chat"),
            help="Example: http://localhost:5678/webhook/smartdoc-chat",
        )

        if st.button("💾 Apply n8n Settings"):
            st.session_state.n8n_enabled = n8n_enabled
            st.session_state.n8n_webhook_url = n8n_webhook_url.strip()
            self.components.success_alert("n8n settings updated")

        if st.session_state.get("n8n_enabled", False):
            st.info("n8n is enabled. Streamlit chat events will be posted to the configured webhook.")
        else:
            st.warning("n8n is currently disabled.")
