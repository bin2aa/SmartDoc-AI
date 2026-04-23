"""
CoRAG Debug Tracer — console-only structured logging for Chain-of-RAG pipeline.

Provides a lightweight tracer that captures every CoRAG step and logs
structured debug information to the console via the standard logger.
"""

import time
from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("corag_debug")


class CoRAGDebugger:
    """
    Structured debug tracer for Chain-of-RAG pipeline steps.

    Usage::

        dbg = CoRAGDebugger()
        dbg.log_input(query)
        dbg.log_decompose(raw_output, cleaned, deduplicated)
        dbg.log_chain_step(step_idx, total, sub_q, refined_q, docs, drift_info)
        dbg.log_synthesis(unique_docs, context_len, prompt_len, truncated)
        dbg.log_answer(answer_preview)
        dbg.summary()
    """

    def __init__(self):
        self._start_time: float = time.time()
        self._steps: List[Dict[str, Any]] = []
        self._query: str = ""
        self._sub_questions_raw: List[str] = []
        self._sub_questions_cleaned: List[str] = []
        self._sub_questions_final: List[str] = []
        self._chain_steps: List[Dict[str, Any]] = []
        self._synthesis_info: Dict[str, Any] = {}
        self._answer_preview: str = ""
        self._warnings: List[str] = []

    # ── Input ──────────────────────────────────────────────────

    def log_input(self, query: str) -> None:
        """Log the original user query."""
        self._query = query
        logger.info("=" * 60)
        logger.info("[CoRAG] ═══ NEW QUERY ═══")
        logger.info("[CoRAG] Input query: '%s'", query)
        logger.info("[CoRAG] Word count: %d", len(query.strip().split()))
        logger.info("=" * 60)

    # ── Decompose ──────────────────────────────────────────────

    def log_decompose_raw(self, raw_llm_output: str) -> None:
        """Log the raw LLM output from decomposition."""
        self._sub_questions_raw = raw_llm_output.strip().split("\n")
        logger.info("[CoRAG] ── DECOMPOSE ──")
        logger.info("[CoRAG] Raw LLM output:\n%s", raw_llm_output.strip())

    def log_decompose_cleaned(self, cleaned: List[str]) -> None:
        """Log the cleaned sub-questions (after removing numbering/short lines)."""
        self._sub_questions_cleaned = cleaned
        logger.info("[CoRAG] Cleaned sub-questions (%d):", len(cleaned))
        for i, sq in enumerate(cleaned):
            logger.info("[CoRAG]   %d. '%s'", i + 1, sq)

    def log_decompose_deduplicated(self, final: List[str], removed: List[str]) -> None:
        """Log the final deduplicated sub-questions and what was removed."""
        self._sub_questions_final = final
        logger.info("[CoRAG] After deduplication: %d sub-questions:", len(final))
        for i, sq in enumerate(final):
            logger.info("[CoRAG]   ✅ %d. '%s'", i + 1, sq)
        for sq in removed:
            logger.info("[CoRAG]   ❌ Removed (too similar): '%s'", sq)

    def log_decompose_single(self, reason: str) -> None:
        """Log when decomposition is skipped."""
        self._sub_questions_final = [self._query]
        logger.info("[CoRAG] Decomposition skipped: %s", reason)
        logger.info("[CoRAG] Using single-step chain with original query")

    # ── Sequential Chain v2 ──────────────────────────────────────

    def log_stepback(self, original: str, stepback: str) -> None:
        """Log step-back prompting result."""
        logger.info("[CoRAG] ── STEP-BACK ──")
        logger.info("[CoRAG] Original: '%s'", original)
        logger.info("[CoRAG] Step-back: '%s'", stepback)

    def log_sequential_generate(
        self, step_idx: int, context_preview: str, generated: List[str]
    ) -> None:
        """Log dynamically generated sub-questions from sequential chain."""
        logger.info(
            "[CoRAG] ── SEQUENTIAL GENERATE (step %d) ──", step_idx
        )
        logger.info("[CoRAG] Context preview: '%s...'", context_preview[:120])
        for i, sq in enumerate(generated):
            logger.info("[CoRAG]   Generated sub-question %d: '%s'", i + 1, sq)

    def log_fallback(self, reason: str) -> None:
        """Log when CoRAG falls back to standard RAG retrieval."""
        logger.warning("[CoRAG] ⚠️ FALLBACK TRIGGERED: %s", reason)
        self._warnings.append(f"Fallback to Standard RAG: {reason}")

    def log_early_exit(self, step: int, reason: str) -> None:
        """Log when CoRAG exits the chain early due to sufficient coverage."""
        logger.info("[CoRAG] ✅ EARLY EXIT at step %d: %s", step, reason)
        self._warnings.append(f"Early exit at step {step}: {reason}")

    # ── Chain Steps ────────────────────────────────────────────

    def log_chain_step_start(self, step_idx: int, total: int, sub_question: str) -> None:
        """Log the start of a chain retrieval step."""
        logger.info(
            "[CoRAG] ── CHAIN STEP %d/%d ──", step_idx + 1, total
        )
        logger.info("[CoRAG] Sub-question: '%s'", sub_question)

    def log_refine(
        self,
        original: str,
        refined: str,
        overlap: float,
        reverted: bool,
    ) -> None:
        """Log refinement result and drift check."""
        if reverted:
            logger.warning(
                "[CoRAG] ⚠️ DRIFT DETECTED: overlap=%.2f below threshold", overlap
            )
            logger.warning("[CoRAG]   Original: '%s'", original)
            logger.warning("[CoRAG]   Refined:  '%s'", refined)
            logger.warning("[CoRAG]   → REVERTED to original")
            self._warnings.append(
                f"Step refine drifted (overlap={overlap:.0%}): '{original}' → '{refined}'"
            )
        else:
            logger.info(
                "[CoRAG] Refined: '%s' → '%s' (overlap=%.0f%%)",
                original, refined, overlap * 100,
            )

    def log_retrieval(
        self,
        query_used: str,
        docs_count: int,
        retrieval_ms: float,
        doc_previews: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Log retrieval results for a chain step."""
        logger.info(
            "[CoRAG] Retrieval for: '%s' → %d docs (%.0fms)",
            query_used, docs_count, retrieval_ms,
        )
        if doc_previews:
            for i, preview in enumerate(doc_previews):
                citation = preview.get("citation", "unknown")
                content_preview = preview.get("content_preview", "")[:120]
                page = preview.get("page")
                page_str = f", page={page}" if page is not None else ""
                logger.info(
                    "[CoRAG]   📄 Doc %d: [%s%s] %s...",
                    i + 1, citation, page_str, content_preview,
                )

        if docs_count == 0:
            logger.warning("[CoRAG] ⚠️ No documents retrieved for this step!")
            self._warnings.append(f"No docs retrieved for: '{query_used[:50]}'")

    def log_retrieval_validation(
        self,
        sub_question: str,
        docs_content: List[str],
        has_overlap: bool,
        overlap_ratio: float,
    ) -> None:
        """Log validation of retrieval relevance."""
        if not has_overlap:
            logger.warning(
                "[CoRAG] ⚠️ LOW RELEVANCE: %.0f%% of retrieved docs have "
                "no term overlap with sub-question: '%s'",
                overlap_ratio * 100, sub_question,
            )
            self._warnings.append(
                f"Low retrieval relevance ({overlap_ratio:.0%}): '{sub_question[:50]}'"
            )
        else:
            logger.info(
                "[CoRAG] Retrieval validation OK: %.0f%% docs have term overlap",
                (1 - overlap_ratio) * 100,
            )

    # ── Synthesis ──────────────────────────────────────────────

    def log_synthesis(
        self,
        unique_docs_count: int,
        context_len: int,
        prompt_len: int,
        truncated: bool,
        context_budget: int,
    ) -> None:
        """Log synthesis step details."""
        self._synthesis_info = {
            "unique_docs": unique_docs_count,
            "context_len": context_len,
            "prompt_len": prompt_len,
            "truncated": truncated,
            "context_budget": context_budget,
        }
        logger.info("[CoRAG] ── SYNTHESIS ──")
        logger.info("[CoRAG] Unique docs: %d", unique_docs_count)
        logger.info("[CoRAG] Context length: %d chars", context_len)
        logger.info("[CoRAG] Prompt length: %d chars", prompt_len)
        if truncated:
            logger.warning(
                "[CoRAG] ⚠️ Context TRUNCATED from %d to %d chars! "
                "This may cause hallucination.",
                context_len, context_budget,
            )
            self._warnings.append(
                f"Synthesis context truncated: {context_len} → {context_budget} chars"
            )
        logger.info("[CoRAG] Context budget: %d chars", context_budget)

    # ── Answer ─────────────────────────────────────────────────

    def log_answer(self, answer: str) -> None:
        """Log the final answer preview."""
        self._answer_preview = answer[:200]
        logger.info("[CoRAG] ── ANSWER ──")
        logger.info("[CoRAG] Answer preview: '%s...'", answer[:200])
        logger.info("[CoRAG] Answer length: %d chars", len(answer))

    # ── Summary ────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a structured summary of the entire CoRAG run."""
        elapsed = time.time() - self._start_time
        logger.info("=" * 60)
        logger.info("[CoRAG] ═══ SUMMARY ═══")
        logger.info("[CoRAG] Query: '%s'", self._query[:80])
        logger.info("[CoRAG] Sub-questions: %d", len(self._sub_questions_final))
        for i, sq in enumerate(self._sub_questions_final):
            logger.info("[CoRAG]   %d. %s", i + 1, sq)
        logger.info("[CoRAG] Chain steps: %d", len(self._chain_steps))
        logger.info(
            "[CoRAG] Synthesis docs: %d, context: %d chars",
            self._synthesis_info.get("unique_docs", 0),
            self._synthesis_info.get("context_len", 0),
        )
        if self._warnings:
            logger.warning("[CoRAG] ⚠️ Warnings (%d):", len(self._warnings))
            for w in self._warnings:
                logger.warning("[CoRAG]   - %s", w)
        else:
            logger.info("[CoRAG] ✅ No warnings")
        logger.info("[CoRAG] Total debug time: %.0fms", elapsed * 1000)
        logger.info("=" * 60)