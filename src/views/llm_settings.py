"""LLM configuration, system info, and re-ranking benchmark settings for SmartDoc AI."""

import streamlit as st

from src.views.components import UIComponents
from src.utils.constants import (
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_KEEP_ALIVE,
    AVAILABLE_MODELS,
    LOW_MEMORY_FALLBACK_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def render_llm_settings(
    components: UIComponents,
    persist_fn,
):
    """Render LLM configuration settings.

    Args:
        components: UIComponents instance for UI rendering
        persist_fn: Callback function to persist settings to disk
    """
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
            persist_fn()
            components.success_alert("LLM settings updated and saved")

    with col_preset:
        if st.button("Apply Low-RAM Preset"):
            st.session_state.llm_model = LOW_MEMORY_FALLBACK_MODEL
            st.session_state.llm_num_ctx = 256
            st.session_state.llm_num_predict = 64
            st.session_state.llm_keep_alive = "0m"
            persist_fn()
            components.success_alert(
                f"Applied low-RAM preset: model={LOW_MEMORY_FALLBACK_MODEL}, num_ctx=256, num_predict=64"
            )

    st.info(
        "Tip: If you see memory errors, use an installed lightweight model, num_ctx=256, num_predict=64, keep_alive=0m."
    )


def render_system_info(components: UIComponents):
    """Render system information panel.

    Args:
        components: UIComponents instance for UI rendering
    """
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
        with components.loading_spinner("Testing connection..."):
            try:
                from src.services.llm_service import OllamaLLMService
                OllamaLLMService()
                components.success_alert("Connected to Ollama successfully!")
            except Exception as e:
                components.error_alert(
                    "Cannot connect to Ollama",
                    details=str(e)
                )


def render_rerank_benchmark(document_controller, components: UIComponents):
    """Run multi-query re-ranking benchmark for bi-encoder vs cross-encoder.

    Args:
        document_controller: DocumentController for accessing vector service
        components: UIComponents instance for UI rendering
    """
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
            components.error_alert("Vui lòng nhập ít nhất một câu hỏi để benchmark")
            return

        # Get vector service from document controller or session
        vector_service = None
        if hasattr(document_controller, 'vector_service') and document_controller.vector_service:
            vector_service = document_controller.vector_service
        else:
            vector_service = st.session_state.get('vector_service')

        if vector_service is None or not getattr(vector_service, 'is_initialized', True):
            components.error_alert("Vector store chưa được khởi tạo. Vui lòng tải tài liệu lên trước.")
            return

        use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
        selected_sources = st.session_state.get("active_source_filters", [])
        selected_file_types = st.session_state.get("active_file_type_filters", [])
        metadata_filters: dict = {}
        if selected_sources:
            metadata_filters["source_files"] = selected_sources
        if selected_file_types:
            metadata_filters["file_types"] = selected_file_types

        rows: list = []
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
            components.error_alert('Không thể chạy benchmark re-ranking', details=str(e))
            return

    result = st.session_state.get('rerank_benchmark_result')
    if result:
        st.markdown('**Benchmark Summary**')
        st.write(result.get('summary', {}))
        result_rows = result.get('rows', [])
        if result_rows:
            st.dataframe(result_rows, use_container_width=True)