"""
RAG strategy implementations for SmartDoc AI.

Provides the Strategy Pattern for different RAG pipeline types:
- StandardRAGStrategy: Single retrieval → generate
- ChainOfRAGStrategy: Decompose → chain retrieve-refine → synthesize
"""

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple

import streamlit as st

from src.models.document_model import Document
from src.services.llm_service import AbstractLLMService
from src.services.vector_store_service import AbstractVectorStoreService
from src.utils.constants import CORAG_MAX_CHAIN_STEPS, CORAG_MIN_QUERY_WORDS
from src.utils.exceptions import VectorStoreError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Abstract Strategy
# ═══════════════════════════════════════════════════════════════


class AbstractRAGStrategy(ABC):
    """Abstract interface for RAG pipeline strategies."""

    @abstractmethod
    def process_query_stream(
        self,
        query: str,
        vector_service: AbstractVectorStoreService,
        llm_service: AbstractLLMService,
        k: int = 3,
        status_container=None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        conversation_context: str = "No previous conversation.",
    ) -> Tuple[Generator[str, None, None], List[Document], Dict[str, Any]]:
        """
        Process a query and return a streaming response.

        Args:
            query: User's question
            vector_service: Vector store service for retrieval
            llm_service: LLM service for generation
            k: Number of documents to retrieve per step
            status_container: Optional ``st.status`` for real-time UI updates
            metadata_filters: Optional metadata filters for retrieval
            conversation_context: Recent conversation history string

        Returns:
            Tuple of (stream_generator, source_documents, metrics_dict)
        """
        pass


# ═══════════════════════════════════════════════════════════════
# Standard RAG Strategy
# ═══════════════════════════════════════════════════════════════


class StandardRAGStrategy(AbstractRAGStrategy):
    """
    Standard RAG: single retrieval → build prompt → stream response.

    This is the existing RAG behavior extracted into a strategy.
    """

    def process_query_stream(
        self,
        query: str,
        vector_service: AbstractVectorStoreService,
        llm_service: AbstractLLMService,
        k: int = 3,
        status_container=None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        conversation_context: str = "No previous conversation.",
    ) -> Tuple[Generator[str, None, None], List[Document], Dict[str, Any]]:
        """Standard single-retrieval RAG pipeline."""
        t_start = time.time()
        metrics: Dict[str, Any] = {
            "strategy": "standard",
            "retrieval_steps": 0,
            "total_docs_retrieved": 0,
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "total_time_ms": 0.0,
        }

        def _step(icon: str, msg: str) -> None:
            if status_container is not None:
                status_container.write(f"{icon} {msg}")

        # ── Validate ───────────────────────────────────────────
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        _step("[search]", f"**Analyzing query:** `{query}`")

        # ── Retrieve ───────────────────────────────────────────
        _step("[docs]", f"Searching documents (k={k})...")
        t_retrieval = time.time()

        if hasattr(vector_service, "search"):
            relevant_docs, retrieval_stats = vector_service.search(
                query=query,
                k=k,
                metadata_filters=metadata_filters,
                use_hybrid=False,
                rerank=False,
                fetch_k=max(k * 4, 20),
            )
        else:
            relevant_docs = vector_service.similarity_search(query, k=k)
            retrieval_stats = {"results": len(relevant_docs)}

        metrics["retrieval_steps"] = 1
        metrics["total_docs_retrieved"] = len(relevant_docs)
        metrics["retrieval_time_ms"] = round((time.time() - t_retrieval) * 1000, 2)

        if relevant_docs:
            total_ms = retrieval_stats.get("total_time_ms", metrics["retrieval_time_ms"])
            _step("[ok]", f"Found **{len(relevant_docs)} documents** ({total_ms:.0f}ms)")
        else:
            _step("[warn]", "No relevant documents found")

        st.session_state.last_retrieval_stats = retrieval_stats

        # ── No docs fallback ───────────────────────────────────
        if not relevant_docs:

            def _no_info():
                yield "I don't have enough information to answer this question."

            metrics["total_time_ms"] = round((time.time() - t_start) * 1000, 2)
            return _no_info(), [], metrics

        # ── Build prompt ───────────────────────────────────────
        context = "\n\n".join(
            [
                f"[Source: {doc.get_citation()} | chunk={doc.metadata.get('chunk_index', 'n/a')}]\n{doc.content}"
                for doc in relevant_docs
            ]
        )

        prompt = _build_standard_prompt(context, query, conversation_context)
        _step("[build]", f"Built prompt ({len(prompt)} chars, {len(context)} context)")

        # ── Stream ─────────────────────────────────────────────
        _step("[llm]", "Calling LLM...")
        t_gen = time.time()
        stream_gen = llm_service.generate_stream(prompt)

        # We wrap the generator to capture generation time
        def _timed_stream(gen):
            try:
                for chunk in gen:
                    yield chunk
            finally:
                metrics["generation_time_ms"] = round((time.time() - t_gen) * 1000, 2)
                metrics["total_time_ms"] = round((time.time() - t_start) * 1000, 2)

        return _timed_stream(stream_gen), relevant_docs, metrics


# ═══════════════════════════════════════════════════════════════
# Chain-of-RAG Strategy
# ═══════════════════════════════════════════════════════════════


class ChainOfRAGStrategy(AbstractRAGStrategy):
    """
    Chain-of-RAG: decompose → sequential retrieve-refine → synthesize.

    Breaks a complex query into a chain of sub-questions, retrieves
    documents for each while refining subsequent questions based on
    accumulated context, then synthesizes a final answer.
    """

    def process_query_stream(
        self,
        query: str,
        vector_service: AbstractVectorStoreService,
        llm_service: AbstractLLMService,
        k: int = 3,
        status_container=None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        conversation_context: str = "No previous conversation.",
    ) -> Tuple[Generator[str, None, None], List[Document], Dict[str, Any]]:
        """Chain-of-RAG pipeline with sequential retrieve-refine loop."""
        t_start = time.time()
        metrics: Dict[str, Any] = {
            "strategy": "chain_of_rag",
            "sub_questions": [],
            "retrieval_steps": 0,
            "total_docs_retrieved": 0,
            "decomposition_time_ms": 0.0,
            "retrieval_time_ms": 0.0,
            "refinement_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "total_time_ms": 0.0,
        }

        def _step(icon: str, msg: str) -> None:
            if status_container is not None:
                status_container.write(f"{icon} {msg}")

        # ── Validate ───────────────────────────────────────────
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        _step("[search]", f"**Analyzing query:** `{query}`")

        # ── Step 1: Decompose query ────────────────────────────
        word_count = len(query.strip().split())
        sub_questions: List[str]

        if word_count < CORAG_MIN_QUERY_WORDS:
            _step("[info]", "Query is simple — treating as single-step chain")
            sub_questions = [query.strip()]
        else:
            _step("[decompose]", "Decomposing query into sub-questions...")
            t_decompose = time.time()
            sub_questions = self._decompose_query(query, llm_service)
            metrics["decomposition_time_ms"] = round((time.time() - t_decompose) * 1000, 2)

            if not sub_questions or len(sub_questions) <= 1:
                _step("[info]", "No decomposition needed — treating as single-step chain")
                sub_questions = [query.strip()]
            else:
                sub_questions = sub_questions[:CORAG_MAX_CHAIN_STEPS]
                _step(
                    "[ok]",
                    f"Decomposed into **{len(sub_questions)} sub-questions**:\n"
                    + "\n".join(f"  {i+1}. `{sq}`" for i, sq in enumerate(sub_questions)),
                )

        metrics["sub_questions"] = sub_questions

        # ── Step 2: Chain retrieve-refine loop ─────────────────
        all_docs: List[Document] = []
        accumulated_context: str = ""
        total_retrieval_ms = 0.0
        total_refinement_ms = 0.0

        for step_idx, sub_q in enumerate(sub_questions):
            chain_label = f"[chain {step_idx + 1}/{len(sub_questions)}]"

            # Refine sub-question using accumulated context (skip for first)
            if step_idx > 0 and accumulated_context:
                _step(chain_label, f"Refining sub-question based on previous context...")
                t_refine = time.time()
                refined_q = self._refine_subquestion(
                    sub_q, accumulated_context, llm_service
                )
                total_refinement_ms += (time.time() - t_refine) * 1000
                if refined_q and refined_q.strip():
                    _step(chain_label, f"Refined: `{refined_q}`")
                else:
                    refined_q = sub_q
            else:
                refined_q = sub_q

            # Retrieve for this sub-question
            _step(chain_label, f"Retrieving for: `{refined_q}`")
            t_retrieval = time.time()

            if hasattr(vector_service, "search"):
                step_docs, _ = vector_service.search(
                    query=refined_q,
                    k=k,
                    metadata_filters=metadata_filters,
                    use_hybrid=False,
                    rerank=False,
                    fetch_k=max(k * 2, 10),
                )
            else:
                step_docs = vector_service.similarity_search(refined_q, k=k)

            step_ms = (time.time() - t_retrieval) * 1000
            total_retrieval_ms += step_ms
            metrics["retrieval_steps"] += 1

            if step_docs:
                _step(
                    chain_label,
                    f"Found **{len(step_docs)} documents** ({step_ms:.0f}ms)",
                )
                all_docs.extend(step_docs)

                # Update accumulated context for next refinement
                new_context = "\n".join(doc.content[:500] for doc in step_docs)
                accumulated_context += f"\n\n[Step {step_idx + 1}: {refined_q}]\n{new_context}"
            else:
                _step(chain_label, "No documents found for this sub-question")

        # Deduplicate documents
        seen_keys: set = set()
        unique_docs: List[Document] = []
        for doc in all_docs:
            key = f"{doc.metadata.get('source', '')}|{doc.metadata.get('chunk_index', '')}|{doc.content[:100]}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_docs.append(doc)

        metrics["total_docs_retrieved"] = len(unique_docs)
        metrics["retrieval_time_ms"] = round(total_retrieval_ms, 2)
        metrics["refinement_time_ms"] = round(total_refinement_ms, 2)

        st.session_state.last_retrieval_stats = {
            "strategy": "chain_of_rag",
            "sub_questions": sub_questions,
            "retrieval_steps": metrics["retrieval_steps"],
            "total_docs": len(unique_docs),
            "total_retrieval_ms": metrics["retrieval_time_ms"],
        }

        # ── No docs fallback ───────────────────────────────────
        if not unique_docs:
            _step("[warn]", "No relevant documents found across all chain steps")

            def _no_info():
                yield "I don't have enough information to answer this question."

            metrics["total_time_ms"] = round((time.time() - t_start) * 1000, 2)
            return _no_info(), [], metrics

        # ── Step 3: Synthesize ─────────────────────────────────
        _step("[synthesize]", "Synthesizing answer from chain evidence...")

        chain_context = "\n\n".join(
            [
                f"[Source: {doc.get_citation()} | chunk={doc.metadata.get('chunk_index', 'n/a')}]\n{doc.content}"
                for doc in unique_docs
            ]
        )

        prompt = _build_corag_prompt(
            query, chain_context, sub_questions, accumulated_context, conversation_context
        )
        _step("[build]", f"Built synthesis prompt ({len(prompt)} chars)")

        # ── Stream ─────────────────────────────────────────────
        _step("[llm]", "Calling LLM for synthesis...")
        t_gen = time.time()
        stream_gen = llm_service.generate_stream(prompt)

        def _timed_stream(gen):
            try:
                for chunk in gen:
                    yield chunk
            finally:
                metrics["generation_time_ms"] = round((time.time() - t_gen) * 1000, 2)
                metrics["total_time_ms"] = round((time.time() - t_start) * 1000, 2)
                logger.info(
                    "Chain-of-RAG complete: steps=%d, docs=%d, total=%.0fms",
                    metrics["retrieval_steps"],
                    metrics["total_docs_retrieved"],
                    metrics["total_time_ms"],
                )

        return _timed_stream(stream_gen), unique_docs, metrics

    # ── CoRAG Helper Methods ────────────────────────────────────

    @staticmethod
    def _decompose_query(query: str, llm_service: AbstractLLMService) -> List[str]:
        """
        Break a complex question into a sequential chain of sub-questions.

        Args:
            query: The user's original question
            llm_service: LLM service for decomposition

        Returns:
            Ordered list of sub-question strings
        """
        decompose_prompt = f"""You are an expert assistant. Your task is to break a question into 2-3 simpler sub-questions.

        RULES:
        1. LANGUAGE MATCHING: You MUST respond in the SAME LANGUAGE as the question provided. 
        - If the question is in Vietnamese, respond in Vietnamese.
        - If the question is in English, respond in English.
        2. If the question is simple, return exactly: "SINGLE"
        3. Output format: Numbered list (1., 2., 3.). No preamble.

        EXAMPLES:
        Question: "Làm thế nào để học code?"
        1. Lộ trình cơ bản để bắt đầu học lập trình là gì?
        2. Có những nguồn tài liệu miễn phí nào để thực hành code?

        Question: "How to bake a cake?"
        1. What are the basic ingredients needed for a standard cake?
        2. What are the step-by-step instructions for mixing and baking?

        QUESTION: {query}
        ANSWER:"""

        try:
            response = llm_service.generate(decompose_prompt)
            sub_questions: List[str] = []

            for line in response.strip().split("\n"):
                line = line.strip()
                cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
                if cleaned and len(cleaned) > 5:
                    sub_questions.append(cleaned)

            # Check if LLM indicated single-step is sufficient
            if len(sub_questions) == 1 and "single" in sub_questions[0].lower():
                return [query.strip()]

            if not sub_questions:
                return [query.strip()]

            logger.info(f"Decomposed into {len(sub_questions)} sub-questions: {sub_questions}")
            return sub_questions

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query.strip()]

    @staticmethod
    def _refine_subquestion(
        sub_question: str, accumulated_context: str, llm_service: AbstractLLMService
    ) -> str:
        """
        Refine a sub-question using context accumulated from previous chain steps.

        Args:
            sub_question: The original sub-question to refine
            accumulated_context: Context gathered from previous retrieval steps
            llm_service: LLM service for refinement

        Returns:
            Refined sub-question string
        """
        refine_prompt = f"""Given the context already retrieved from previous search steps, refine the next sub-question to be more specific and targeted.

PREVIOUSLY RETRIEVED CONTEXT:
{accumulated_context[:2000]}

NEXT SUB-QUESTION TO REFINE:
{sub_question}

Instructions:
- Make the sub-question more specific based on what was already found
- Focus on information gaps in the existing context
- Keep it concise (one sentence)
- If the sub-question doesn't need refinement, return it as-is

Refined sub-question:"""

        try:
            response = llm_service.generate(refine_prompt).strip()
            # Clean up any quotes or extra formatting
            refined = response.strip("\"'").strip()
            if refined and len(refined) > 5:
                logger.info(f"Refined sub-question: '{sub_question}' → '{refined}'")
                return refined
            return sub_question
        except Exception as e:
            logger.warning(f"Sub-question refinement failed: {e}")
            return sub_question


# ═══════════════════════════════════════════════════════════════
# Prompt Templates
# ═══════════════════════════════════════════════════════════════


def _build_standard_prompt(context: str, question: str, chat_history_context: str) -> str:
    """Build prompt for standard RAG generation."""
    return f"""You are SmartDoc AI, an intelligent document assistant.

Answer the QUESTION based ONLY on the CONTEXT below.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."
Detect the question language and respond in the SAME language.
Keep your answer concise (3-4 sentences maximum).
Be factual and precise.

    RECENT CONVERSATION:
    {chat_history_context}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def _build_corag_prompt(
    question: str,
    chain_context: str,
    sub_questions: List[str],
    accumulated_evidence: str,
    chat_history_context: str,
) -> str:
    """Build prompt for Chain-of-RAG synthesis."""
    chain_summary = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(sub_questions))

    return f"""You are SmartDoc AI, an intelligent document assistant using Chain-of-RAG reasoning.

The original question was broken into a chain of sub-questions, and documents were retrieved for each step.
Now synthesize a comprehensive answer using ALL the gathered evidence.

ORIGINAL QUESTION:
{question}

CHAIN OF SUB-QUESTIONS:
{chain_summary}

GATHERED EVIDENCE (from sequential retrieval steps):
{chain_context}

ACCUMULATED REASONING CONTEXT:
{accumulated_evidence[:3000]}

    RECENT CONVERSATION:
    {chat_history_context}

Instructions:
1. Synthesize an answer using evidence from ALL chain steps
2. If the evidence is insufficient, say "I don't have enough information to answer this question."
3. Detect the question language and respond in the SAME language
4. Keep your answer concise (3-4 sentences maximum)
5. Be factual and precise — only use information from the gathered evidence

ANSWER:"""