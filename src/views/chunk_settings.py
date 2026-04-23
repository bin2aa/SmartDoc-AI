"""Chunk configuration settings rendering for SmartDoc AI."""

import streamlit as st

from src.views.components import UIComponents, icon
from src.utils.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def render_chunk_settings(
    document_controller,
    components: UIComponents,
    persist_fn,
):
    """Render chunk configuration settings including sliders and benchmark.

    Args:
        document_controller: DocumentController for chunk config updates
        components: UIComponents instance for UI rendering
        persist_fn: Callback function to persist settings to disk
    """
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
        document_controller.update_chunk_config(chunk_size, chunk_overlap)
        persist_fn()

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
        results = document_controller.benchmark_chunk_configs(
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