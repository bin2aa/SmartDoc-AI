"""Settings screen for SmartDoc AI."""

import streamlit as st
from src.controllers.document_controller import DocumentController
from src.views.components import UIComponents, icon
from src.controllers.chat_controller import ChatController
from src.utils.logger import setup_logger
from src.utils.constants import *
from src.services.persistence_service import save_settings
from src.utils.constants import RAG_TYPE_STANDARD, RAG_TYPE_CORAG, AVAILABLE_RAG_TYPES

logger = setup_logger(__name__)


class SettingsScreen:
    """
    Settings configuration screen.

    Allows users to configure RAG parameters.
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
        self._render_chunk_settings()

        st.markdown("---")

        # Retrieval Strategy
        self._render_retrieval_settings()
        # Re-ranking benchmark moved here from Chat screen
        self._render_rerank_benchmark()
        
        st.markdown("---")
        
        # LLM Configuration
        self._render_llm_settings()
        
        st.markdown("---")
        
        # System Info
        self._render_system_info()
    
    def _render_chunk_settings(self):
        """Render chunk configuration settings."""
        st.subheader("Text Chunking Configuration")
        
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
        
        if st.button("Apply Chunk Settings", type="primary"):
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            self.document_controller.update_chunk_config(chunk_size, chunk_overlap)
            self._persist_current_settings()

        st.markdown(f"#### {icon('science')} Chunk Strategy Benchmark", unsafe_allow_html=True)
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
        """Render retrieval strategy options including RAG type, hybrid, and rerank."""
        st.subheader("Retrieval Strategy")

        # ── RAG Pipeline Type ─────────────────────────────────────
        rag_type_labels = {
            RAG_TYPE_STANDARD: "Standard RAG",
            RAG_TYPE_CORAG: "Chain-of-RAG (CoRAG)",
        }
        current_rag_type = st.session_state.get("rag_type", RAG_TYPE_STANDARD)
        rag_type_options = list(rag_type_labels.keys())
        rag_display_options = list(rag_type_labels.values())

        selected_rag = st.selectbox(
            "RAG Pipeline Type",
            options=rag_type_options,
            format_func=lambda x: rag_type_labels.get(x, x),
            index=rag_type_options.index(current_rag_type) if current_rag_type in rag_type_options else 0,
            help=(
                "Standard RAG: single retrieval then generate. "
                "Chain-of-RAG: decompose query into sub-questions, retrieve sequentially with refinement, then synthesize."
            ),
        )
        st.session_state.rag_type = selected_rag

        # ── Comparison Toggle ─────────────────────────────────────
        st.session_state.compare_rag = st.toggle(
            "Compare both RAG types side-by-side",
            value=st.session_state.get("compare_rag", False),
            help="Runs both Standard and Chain-of-RAG for each query so you can compare results and timing.",
        )

        # ── Description of selected mode ──────────────────────────
        if selected_rag == RAG_TYPE_CORAG:
            st.info(
                "**Chain-of-RAG** will decompose complex queries into 2-3 sub-questions, "
                "retrieve documents for each sequentially (refining based on previous context), "
                "then synthesize a final answer. This improves quality for multi-faceted questions "
                "but takes longer."
            )
        else:
            st.info(
                "**Standard RAG** retrieves documents in a single step, then generates an answer. "
                "Fast and effective for straightforward questions."
            )

        if st.session_state.get("compare_rag", False):
            st.warning(
                "Comparison mode is ON. Both strategies will run for each query. "
                "Expect ~2x latency."
            )

        st.markdown("---")

        # ── Retrieval Options ─────────────────────────────────────
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

        with col4:
            st.session_state.use_self_rag = st.toggle(
                "Self-RAG (Eval + Confidence)",
                value=st.session_state.get("use_self_rag", False),
            )

        st.caption(
            "Enable hybrid search to combine semantic and keyword retrieval. "
            "Enable re-ranking to improve relevance at the cost of higher latency. "
            "Enable Self-RAG for self-evaluation, multi-hop reasoning, and confidence scoring."
        )

        # Persist retrieval settings whenever they change
        self._persist_current_settings()

        # === Retrieval Benchmark UI ===
        if self.chat_controller is not None:
            st.markdown("##### 📊 Retrieval Strategy Benchmark")
            st.markdown(
                "So sanh 3 chien luong: **Vector (Semantic)**, **BM25 (Keyword)**, "
                "**Hybrid (Ensemble)**. Dùng 3 proxy metrics: "
                "**Recall@K** (do chính xác), **Speed** (thời gian ms), "
                "**Coverage** (số document duy nhất trả về)."
            )

            bench_col1, bench_col2 = st.columns([3, 1])
            with bench_col1:
                bench_query = st.text_input(
                    "Benchmark query",
                    value=st.session_state.get("retrieval_bench_query", ""),
                    placeholder="Nhap cau hoi de so sanh cac chien luong retrieval...",
                    help="Dùng một câu hỏi đại diện để so sánh 3 chiến lược.",
                )
            with bench_col2:
                bench_k = st.selectbox(
                    "Top-K",
                    options=[3, 5, 8, 10],
                    index=[3, 5, 8, 10].index(st.session_state.get("retrieval_bench_k", 5))
                    if st.session_state.get("retrieval_bench_k", 5) in [3, 5, 8, 10]
                    else 1,
                )

            st.session_state.retrieval_bench_query = bench_query
            st.session_state.retrieval_bench_k = bench_k

            if st.button("🚀 Run Retrieval Benchmark", type="primary"):
                if not bench_query.strip():
                    st.warning("Vui long nhap cau hoi benchmark.")
                elif not st.session_state.get("vector_store_initialized", False):
                    st.warning("Vector store trong. Vui long upload tai lieu truoc.")
                else:
                    with st.spinner("Dang chay benchmark retrieval..."):
                        results = self.chat_controller.benchmark_retrieval(
                            query=bench_query,
                            k=bench_k,
                        )

                    if "error" in results:
                        st.error(results["error"])
                    else:
                        strategies = results.get("strategies", {})

                        # Bảng so sánh
                        comparison_data = []
                        for strat, data in strategies.items():
                            row = {
                                "Strategy": data["label"],
                                "Recall@K": f"{data['recall_at_k']:.2%}",
                                "Speed (ms)": f"{data['time_ms']:.1f}",
                                "Sources found": data["unique_sources"],
                                "Docs retrieved": data["docs_retrieved"],
                            }
                            comparison_data.append(row)

                        st.markdown("**📋 Comparison Table**")
                        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

                        # Highlight chiến lược tốt nhất
                        best = results.get("best", {})
                        st.markdown("**🏆 Best Strategy by Metric:**")
                        best_cols = st.columns(3)
                        metric_labels = [
                            ("recall", "Recall@K", "📈"),
                            ("speed", "Speed", "⚡"),
                            ("coverage", "Coverage", "📚"),
                        ]
                        for idx, (key, label, icon) in enumerate(metric_labels):
                            with best_cols[idx]:
                                winner = best.get(key, "N/A")
                                winner_label = strategies.get(winner, {}).get("label", winner)
                                st.metric(
                                    f"{icon} {label}",
                                    winner_label,
                                    delta=(
                                        f"Recall: {strategies[winner]['recall_at_k']:.2%}"
                                        if key == "recall" else
                                        f"{strategies[winner]['time_ms']:.1f}ms"
                                        if key == "speed" else
                                        f"{strategies[winner]['unique_sources']} sources"
                                    )
                                )

                        # Chi tiết top-3 documents của mỗi chiến lược
                        with st.expander("📄 Chi tiet top documents moi chien luong"):
                            for strat, data in strategies.items():
                                st.markdown(f"**{data['label']}** (Recall@K = {data['recall_at_k']:.2%})")
                                if data.get("top_docs"):
                                    for i, doc in enumerate(data["top_docs"], 1):
                                        src = doc["source"]
                                        pg = doc["page"]
                                        pg_label = f", page {pg}" if pg else ""
                                        st.markdown(
                                            f"  {i}. `[{src}{pg_label}]` — "
                                            f"{doc['preview']}"
                                        )
                                else:
                                    st.caption("  Khong co document nao.")
                                st.markdown("")
    
    def _render_llm_settings(self):
        """Render LLM configuration settings."""
        st.subheader("LLM Configuration")

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
            if st.button("Apply LLM Settings", type="primary"):
                st.session_state.llm_model = llm_model.strip()
                st.session_state.llm_num_ctx = int(llm_num_ctx)
                st.session_state.llm_num_predict = int(llm_num_predict)
                st.session_state.llm_keep_alive = llm_keep_alive
                self._persist_current_settings()
                self.components.success_alert("LLM settings updated and saved")

        with col_preset:
            if st.button("Apply Low-RAM Preset"):
                st.session_state.llm_model = LOW_MEMORY_FALLBACK_MODEL
                st.session_state.llm_num_ctx = 256
                st.session_state.llm_num_predict = 64
                st.session_state.llm_keep_alive = "0m"
                self._persist_current_settings()
                self.components.success_alert(
                    f"Applied low-RAM preset: model={LOW_MEMORY_FALLBACK_MODEL}, num_ctx=256, num_predict=64"
                )

        st.info(
            "Tip: If you see memory errors, use an installed lightweight model, num_ctx=256, num_predict=64, keep_alive=0m."
        )
    
    def _render_system_info(self):
        """Render system information."""
        st.subheader("System Information")
        
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
        if st.button("Test Ollama Connection"):
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
            "use_self_rag": st.session_state.get("use_self_rag", False),
            "retrieval_k": st.session_state.get("retrieval_k", 3),
            "rag_type": st.session_state.get("rag_type", RAG_TYPE_STANDARD),
            "compare_rag": st.session_state.get("compare_rag", False),
        }
        save_settings(settings)
        logger.info("Settings persisted to disk")

    def _render_rerank_benchmark(self):
        """Run multi-query re-ranking benchmark for bi-encoder vs cross-encoder."""
        st.markdown("---")
        st.subheader("Re-ranking Benchmark")
        st.caption(
            "Nhập nhiều câu hỏi (mỗi dòng 1 câu) để đo độ trễ và mức thay đổi thứ hạng "
            "giữa bi-encoder và cross-encoder re-ranking."
        )

        default_queries = st.session_state.get("rerank_benchmark_queries", "")
        query_blob = st.text_area(
            "Benchmark queries",
            value=default_queries,
            height=120,
            placeholder="Ví dụ:\nMục tiêu chính của tài liệu là gì?\nĐiểm khác nhau giữa A và B?",
        )
        st.session_state.rerank_benchmark_queries = query_blob

        col1, col2 = st.columns(2)
        with col1:
            benchmark_k = st.selectbox(
                "Top-K benchmark",
                options=[3, 5, 8, 10],
                index=[3, 5, 8, 10].index(st.session_state.get("retrieval_k", 3))
                if st.session_state.get("retrieval_k", 3) in [3, 5, 8, 10]
                else 0,
            )
        with col2:
            run_benchmark = st.button("Run Re-ranking Benchmark")

        if run_benchmark:
            queries = [line.strip() for line in query_blob.splitlines() if line.strip()]
            if not queries:
                self.components.error_alert("Vui lòng nhập ít nhất một câu hỏi để benchmark")
                return

            # Get vector service from document controller or session
            vector_service = None
            if hasattr(self.document_controller, 'vector_service') and self.document_controller.vector_service:
                vector_service = self.document_controller.vector_service
            else:
                vector_service = st.session_state.get('vector_service')

            if vector_service is None or not getattr(vector_service, 'is_initialized', True):
                self.components.error_alert("Vector store chưa được khởi tạo. Vui lòng tải tài liệu lên trước.")
                return

            use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
            selected_sources = st.session_state.get("active_source_filters", [])
            selected_file_types = st.session_state.get("active_file_type_filters", [])
            metadata_filters = {}
            if selected_sources:
                metadata_filters["source_files"] = selected_sources
            if selected_file_types:
                metadata_filters["file_types"] = selected_file_types

            rows = []
            try:
                for q in queries:
                    bi_docs, bi_stats = vector_service.search(
                        query=q,
                        k=benchmark_k,
                        metadata_filters=metadata_filters,
                        use_hybrid=use_hybrid,
                        rerank=False,
                        fetch_k=max(benchmark_k * 4, 20),
                    )
                    rerank_docs, rerank_stats = vector_service.search(
                        query=q,
                        k=benchmark_k,
                        metadata_filters=metadata_filters,
                        use_hybrid=use_hybrid,
                        rerank=True,
                        fetch_k=max(benchmark_k * 4, 20),
                    )

                    bi_ids = [f"{d.metadata.get('source','')}_{d.metadata.get('chunk_index','')}" for d in bi_docs]
                    rerank_ids = [f"{d.metadata.get('source','')}_{d.metadata.get('chunk_index','')}" for d in rerank_docs]
                    top_k = min(len(bi_ids), len(rerank_ids), benchmark_k)
                    overlap = len(set(bi_ids).intersection(set(rerank_ids)))
                    rank_changes = sum(1 for idx in range(top_k) if bi_ids[idx] != rerank_ids[idx])

                    rows.append({
                        'query': q,
                        'bi_encoder_ms': bi_stats.get('total_time_ms', bi_stats.get('vector_time_ms', 0.0)),
                        'rerank_ms': rerank_stats.get('total_time_ms', 0.0),
                        'rerank_only_ms': rerank_stats.get('rerank_time_ms', 0.0),
                        'topk_overlap': overlap,
                        'topk_rank_changes': rank_changes,
                    })

                bi_latencies = [r['bi_encoder_ms'] for r in rows]
                rerank_latencies = [r['rerank_ms'] for r in rows]
                rerank_only_latencies = [r['rerank_only_ms'] for r in rows]
                rank_changes = [r['topk_rank_changes'] for r in rows]

                summary = {
                    'queries': len(rows),
                    'avg_bi_encoder_ms': round(sum(bi_latencies) / len(bi_latencies), 2) if bi_latencies else 0.0,
                    'avg_rerank_ms': round(sum(rerank_latencies) / len(rerank_latencies), 2) if rerank_latencies else 0.0,
                    'avg_rerank_only_ms': round(sum(rerank_only_latencies) / len(rerank_only_latencies), 2) if rerank_only_latencies else 0.0,
                    'avg_rank_changes': round(sum(rank_changes) / len(rank_changes), 2) if rank_changes else 0.0,
                    'mode': 'hybrid' if use_hybrid else 'vector',
                }

                st.session_state.rerank_benchmark_result = {'rows': rows, 'summary': summary}
            except Exception as e:
                self.components.error_alert('Không thể chạy benchmark re-ranking', details=str(e))
                return

        result = st.session_state.get('rerank_benchmark_result')
        if result:
            st.markdown('**Benchmark Summary**')
            st.write(result.get('summary', {}))
            rows = result.get('rows', [])
            if rows:
                st.dataframe(rows, use_container_width=True)

