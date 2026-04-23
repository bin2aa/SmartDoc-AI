"""Retrieval strategy settings rendering for SmartDoc AI."""

import streamlit as st
from typing import Optional

from src.views.components import UIComponents
from src.utils.constants import RAG_TYPE_STANDARD, RAG_TYPE_CORAG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def render_retrieval_settings(
    chat_controller,
    components: UIComponents,
    persist_fn,
):
    """Render retrieval strategy options including RAG type, hybrid, and rerank.

    Args:
        chat_controller: ChatController instance (for retrieval benchmark)
        components: UIComponents instance for UI rendering
        persist_fn: Callback function to persist settings to disk
    """
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
    col1, col2, col3, col4 = st.columns(4)
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
    persist_fn()

    # === Retrieval Benchmark UI ===
    if chat_controller is not None:
        _render_retrieval_benchmark(chat_controller, components)


def _render_retrieval_benchmark(chat_controller, components: UIComponents):
    """Render retrieval strategy benchmark UI.

    Args:
        chat_controller: ChatController instance for running benchmarks
        components: UIComponents instance for UI rendering
    """
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
                results = chat_controller.benchmark_retrieval(
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
                for idx, (key, label, metric_icon) in enumerate(metric_labels):
                    with best_cols[idx]:
                        winner = best.get(key, "N/A")
                        winner_label = strategies.get(winner, {}).get("label", winner)
                        st.metric(
                            f"{metric_icon} {label}",
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