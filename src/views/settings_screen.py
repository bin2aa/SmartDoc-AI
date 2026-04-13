"""Settings screen for SmartDoc AI."""

import streamlit as st
from src.controllers.document_controller import DocumentController
from src.views.components import UIComponents
from src.utils.logger import setup_logger
from src.utils.constants import *
from src.services.persistence_service import save_settings

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

        # Retrieval Strategy
        self._render_retrieval_settings()
        
        st.markdown("---")
        
        # LLM Configuration
        self._render_llm_settings()
        
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
            self._persist_current_settings()

        st.markdown("#### 🧪 Chunk Strategy Benchmark")
        benchmark_query = st.text_input(
            "Benchmark query",
            value=st.session_state.get("chunk_benchmark_query", ""),
            help="Use a representative question to compare chunk settings.",
        )
        st.session_state.chunk_benchmark_query = benchmark_query

        if st.button("Run Chunk Benchmark"):
            chunk_sizes = [500, 1000, 1500, 2000]
            chunk_overlaps = [50, 100, 200]
            configs = [(size, overlap) for size in chunk_sizes for overlap in chunk_overlaps]
            results = self.document_controller.benchmark_chunk_configs(
                query=benchmark_query,
                configs=configs,
            )
            if results:
                st.dataframe(results, use_container_width=True)
                best = results[0]
                st.success(
                    "Best proxy accuracy: "
                    f"size={best['chunk_size']}, overlap={best['chunk_overlap']}, "
                    f"score={best['accuracy_proxy']}"
                )
            else:
                st.warning("No benchmark result. Upload documents and provide a query first.")

    def _render_retrieval_settings(self):
        """Render retrieval strategy options including hybrid and rerank."""
        st.subheader("🔎 Retrieval Strategy")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.use_hybrid_search = st.toggle(
                "Hybrid Search (Vector + BM25)",
                value=st.session_state.get("use_hybrid_search", False),
            )

        with col2:
            st.session_state.use_rerank = st.toggle(
                "Cross-Encoder Re-ranking",
                value=st.session_state.get("use_rerank", False),
            )

        with col3:
            st.session_state.retrieval_k = st.selectbox(
                "Top-K Retrieved Chunks",
                options=[3, 5, 8, 10],
                index=[3, 5, 8, 10].index(st.session_state.get("retrieval_k", 3))
                if st.session_state.get("retrieval_k", 3) in [3, 5, 8, 10]
                else 0,
            )

        st.caption(
            "Enable hybrid search to combine semantic and keyword retrieval. "
            "Enable re-ranking to improve relevance at the cost of higher latency."
        )

        # Persist retrieval settings whenever they change
        self._persist_current_settings()
    
    def _render_llm_settings(self):
        """Render LLM configuration settings."""
        st.subheader("🤖 LLM Configuration")

        col1, col2 = st.columns(2)

        with col1:
            current_model = st.session_state.get('llm_model', DEFAULT_MODEL)
            model_options = list(AVAILABLE_MODELS)
            if current_model not in model_options:
                model_options.append(current_model)
            model_options.append("Other (custom)...")

            selected_model = st.selectbox(
                "Ollama Model",
                options=model_options,
                index=model_options.index(current_model) if current_model in model_options else 0,
                help="Select a model or choose 'Other' to enter a custom model name.",
            )

            if selected_model == "Other (custom)...":
                llm_model = st.text_input(
                    "Custom Model Name",
                    value=current_model if current_model not in AVAILABLE_MODELS else "",
                    placeholder="e.g. qwen2.5:7b",
                )
            else:
                llm_model = selected_model

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
                self._persist_current_settings()
                self.components.success_alert("LLM settings updated and saved")

        with col_preset:
            if st.button("🪶 Apply Low-RAM Preset"):
                st.session_state.llm_model = LOW_MEMORY_FALLBACK_MODEL
                st.session_state.llm_num_ctx = 256
                st.session_state.llm_num_predict = 64
                st.session_state.llm_keep_alive = "0m"
                self._persist_current_settings()
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
            "retrieval_k": st.session_state.get("retrieval_k", 3),
        }
        save_settings(settings)
        logger.info("Settings persisted to disk")

