"""RAG strategy comparison and retrieval metrics rendering for SmartDoc AI."""

import streamlit as st
from typing import Any, Dict, List, Optional

from src.views.source_renderer import convert_sources_to_details
from src.utils.constants import RAG_TYPE_STANDARD, RAG_TYPE_CORAG
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def render_rag_comparison(
    controller,
    primary_answer: str,
    primary_source_details: Optional[List[dict]] = None,
) -> None:
    """Render side-by-side comparison of Standard RAG vs Chain-of-RAG.

    Shows metrics table, sub-questions, answers, and source details
    for both strategies.

    Args:
        controller: ChatController instance (unused but kept for API compat)
        primary_answer: The answer from the primary (selected) strategy
        primary_source_details: Source details from the primary strategy
    """
    comparison = st.session_state.get("rag_comparison_result")
    if not comparison:
        return

    primary_type = comparison["primary_type"]
    other_type = comparison["other_type"]
    other_answer = comparison.get("other_answer", "")
    primary_metrics: Dict[str, Any] = comparison.get("primary_metrics", {})
    other_metrics: Dict[str, Any] = comparison.get("other_metrics", {})
    other_docs = comparison.get("other_docs", [])

    rag_labels = {
        RAG_TYPE_STANDARD: "Standard RAG",
        RAG_TYPE_CORAG: "Chain-of-RAG",
    }

    primary_label = rag_labels.get(primary_type, primary_type)
    other_label = rag_labels.get(other_type, other_type)

    with st.expander(f"⚖️ So sánh {primary_label} vs {other_label}", expanded=True):
        _render_comparison_inner(
            primary_label=primary_label,
            other_label=other_label,
            primary_answer=primary_answer,
            other_answer=other_answer,
            primary_source_details=primary_source_details or [],
            other_source_details=convert_sources_to_details(other_docs),
            primary_metrics=primary_metrics,
            other_metrics=other_metrics,
        )

    # Store display data for survival across reruns (no Document objects)
    other_source_details = convert_sources_to_details(other_docs)
    st.session_state.comparison_display_data = {
        "primary_label": primary_label,
        "other_label": other_label,
        "primary_answer": primary_answer,
        "other_answer": other_answer,
        "primary_source_details": primary_source_details or [],
        "other_source_details": other_source_details,
        "primary_metrics": primary_metrics,
        "other_metrics": other_metrics,
    }


def render_comparison_display() -> None:
    """Render RAG comparison that persists across Streamlit reruns.

    Reads from ``st.session_state.comparison_display_data`` which is set
    by :func:`render_rag_comparison` and survives ``st.rerun()``.
    """
    data = st.session_state.get("comparison_display_data")
    if not data:
        return

    primary_label = data["primary_label"]
    other_label = data["other_label"]
    primary_answer = data["primary_answer"]
    other_answer = data["other_answer"]
    primary_source_details = data.get("primary_source_details", [])
    other_source_details = data.get("other_source_details", [])
    primary_metrics = data.get("primary_metrics", {})
    other_metrics = data.get("other_metrics", {})

    with st.expander(f"⚖️ So sánh {primary_label} vs {other_label}", expanded=True):
        _render_comparison_inner(
            primary_label=primary_label,
            other_label=other_label,
            primary_answer=primary_answer,
            other_answer=other_answer,
            primary_source_details=primary_source_details,
            other_source_details=other_source_details,
            primary_metrics=primary_metrics,
            other_metrics=other_metrics,
        )


def render_retrieval_metrics():
    """Show retrieval strategy metrics for hybrid/pure-vector comparison."""
    if st.session_state.get("is_processing_query", False):
        return

    stats = st.session_state.get("last_retrieval_stats", {})
    # Do not show retrieval metrics when there is no chat history
    history = st.session_state.get("chat_history")
    if not history or (hasattr(history, "messages") and len(history.messages) == 0) or (isinstance(history, list) and len(history) == 0):
        return

    if not stats:
        return

    with st.expander("Retrieval Metrics"):
        st.json(stats)
        comparison = st.session_state.get("retrieval_comparison")
        if comparison:
            hybrid_section = comparison.get("hybrid_vs_vector")
            rerank_section = comparison.get("rerank_vs_biencoder")

            if hybrid_section:
                st.markdown("**Hybrid vs Vector**")
                st.write(hybrid_section)

            if rerank_section:
                st.markdown("**Cross-Encoder Re-rank vs Bi-Encoder**")
                st.write(rerank_section)


def _render_comparison_inner(
    primary_label: str,
    other_label: str,
    primary_answer: str,
    other_answer: str,
    primary_source_details: List[dict],
    other_source_details: List[dict],
    primary_metrics: Dict[str, Any],
    other_metrics: Dict[str, Any],
):
    """Render the inner content shared between live and persisted comparison.

    Args:
        primary_label: Display name for primary strategy
        other_label: Display name for other strategy
        primary_answer: Primary strategy's answer text
        other_answer: Other strategy's answer text
        primary_source_details: Primary strategy's source details
        other_source_details: Other strategy's source details
        primary_metrics: Primary strategy's metrics dict
        other_metrics: Other strategy's metrics dict
    """
    # ── Metrics Table ────────────────────────────────────
    col_metrics = {
        "Metric": [
            "Chiến lược",
            "Retrieval Steps",
            "Docs Retrieved",
            "Retrieval Time",
            "Generation Time",
            "Total Time",
            "Answer Length",
        ],
        primary_label: [
            "🏆 (đang dùng)",
            str(primary_metrics.get("retrieval_steps", "-")),
            str(primary_metrics.get("total_docs_retrieved", "-")),
            f"{primary_metrics.get('retrieval_time_ms', 0):.0f}ms",
            f"{primary_metrics.get('generation_time_ms', 0):.0f}ms",
            f"{primary_metrics.get('total_time_ms', 0):.0f}ms",
            f"{len(primary_answer)} chars",
        ],
        other_label: [
            "🔄 (so sánh)",
            str(other_metrics.get("retrieval_steps", "-")),
            str(other_metrics.get("total_docs_retrieved", "-")),
            f"{other_metrics.get('retrieval_time_ms', 0):.0f}ms",
            f"{other_metrics.get('generation_time_ms', 0):.0f}ms",
            f"{other_metrics.get('total_time_ms', 0):.0f}ms",
            f"{other_metrics.get('answer_length', len(other_answer))} chars",
        ],
    }

    import pandas as pd
    df = pd.DataFrame(col_metrics)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Sub-questions (if CoRAG) ─────────────────────────
    for label, m in [(primary_label, primary_metrics), (other_label, other_metrics)]:
        subs = m.get("sub_questions", [])
        if subs and len(subs) > 1:
            st.markdown(f"**{label} — Sub-questions:**")
            for i, sq in enumerate(subs, 1):
                st.markdown(f"  {i}. `{sq}`")

    st.markdown("---")

    # ── Answer Comparison (side by side) ──────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"### 🏆 {primary_label}")
        st.markdown(primary_answer)

    with col_b:
        st.markdown(f"### 🔄 {other_label}")
        st.markdown(other_answer)

    # ── Source Comparison ─────────────────────────────────
    st.markdown("---")
    col_src_a, col_src_b = st.columns(2)

    with col_src_a:
        st.markdown(f"**📚 Nguồn ({primary_label}):**")
        if primary_source_details:
            for i, src in enumerate(primary_source_details, 1):
                citation = src.get("citation", "[Unknown]")
                badge = " ✅" if src.get("used_in_answer") else ""
                st.markdown(f"  {i}. {citation}{badge}")
        else:
            st.caption("Không có nguồn")

    with col_src_b:
        st.markdown(f"**📚 Nguồn ({other_label}):**")
        if other_source_details:
            for i, src in enumerate(other_source_details, 1):
                citation = src.get("citation", "[Unknown]")
                badge = " ✅" if src.get("used_in_answer") else ""
                st.markdown(f"  {i}. {citation}{badge}")
        else:
            st.caption("Không có nguồn")

    # ── Overlap Analysis ──────────────────────────────────
    if primary_source_details and other_source_details:
        primary_citations = {s.get("citation") for s in primary_source_details}
        other_citations = {s.get("citation") for s in other_source_details}
        overlap = primary_citations.intersection(other_citations)
        st.markdown("---")
        st.markdown(
            f"**🔍 Phân tích chồng lấp:** "
            f"{len(overlap)}/{len(primary_citations)} nguồn trùng nhau "
            f"giữa hai chiến lược"
        )