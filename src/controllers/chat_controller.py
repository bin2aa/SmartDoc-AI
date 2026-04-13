"""Chat controller for handling chat operations."""

from typing import Any, Dict, Generator, Optional, List, Tuple
import streamlit as st
from src.services.llm_service import AbstractLLMService, OllamaLLMService
from src.services.vector_store_service import AbstractVectorStoreService
from src.models.chat_model import ChatHistory
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import LLMConnectionError, VectorStoreError
from src.utils.constants import DEFAULT_STREAMLIT_REPLY_TEMPLATES, NO_INFO_MARKERS

logger = setup_logger(__name__)


class ChatController:
    """
    Controller for chat operations.
    
    Handles query processing, context retrieval, and response generation.
    """
    
    def __init__(
        self,
        llm_service: Optional[AbstractLLMService] = None,
        vector_service: Optional[AbstractVectorStoreService] = None,
    ):
        """
        Initialize chat controller.
        
        Args:
            llm_service: LLM service instance (default: OllamaLLMService)
            vector_service: Vector store service instance
        """
        # Dependency injection with defaults
        self.llm_service = llm_service or OllamaLLMService()
        self.vector_service = vector_service
        
        logger.info("ChatController initialized")
    
    def process_query(self, query: str, k: int = 3) -> Tuple[str, List[Document]]:
        """
        Process user query and return response.
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (answer, source_documents)
            
        Raises:
            ValueError: If query is empty
            VectorStoreError: If vector store not initialized
            LLMConnectionError: If LLM generation fails
        """
        # Validation
        if not query or not query.strip():
            logger.warning("Empty query received")
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Processing query: {query[:50]}...")

        use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
        use_rerank = bool(st.session_state.get("use_rerank", False))
        retrieval_k = int(st.session_state.get("retrieval_k", k))
        selected_sources = st.session_state.get("active_source_filters", [])
        selected_file_types = st.session_state.get("active_file_type_filters", [])

        metadata_filters: Dict[str, Any] = {}
        if selected_sources:
            metadata_filters["source_files"] = selected_sources
        if selected_file_types:
            metadata_filters["file_types"] = selected_file_types

        rewritten_query = self._rewrite_query(query)
        conversation_context = self._conversation_context(max_turns=4)
        
        # Check if vector store is initialized
        if self.vector_service is None or not self.vector_service.is_initialized:
            logger.error("Vector store not initialized")
            raise VectorStoreError("Please upload documents first")
        
        # Retrieve relevant documents
        try:
            if hasattr(self.vector_service, "search"):
                relevant_docs, retrieval_stats = self.vector_service.search(
                    query=rewritten_query,
                    k=retrieval_k,
                    metadata_filters=metadata_filters,
                    use_hybrid=use_hybrid,
                    rerank=use_rerank,
                    fetch_k=max(retrieval_k * 4, 20),
                )
            else:
                relevant_docs = self.vector_service.similarity_search(rewritten_query, k=retrieval_k)
                retrieval_stats = {
                    "use_hybrid": False,
                    "rerank": False,
                    "results": len(relevant_docs),
                }

            st.session_state.last_retrieval_stats = retrieval_stats

            if use_hybrid and hasattr(self.vector_service, "search"):
                vector_only_docs, vector_stats = self.vector_service.search(
                    query=rewritten_query,
                    k=retrieval_k,
                    metadata_filters=metadata_filters,
                    use_hybrid=False,
                    rerank=False,
                    fetch_k=max(retrieval_k * 4, 20),
                )
                st.session_state.retrieval_comparison = {
                    "vector_results": len(vector_only_docs),
                    "hybrid_results": len(relevant_docs),
                    "vector_ms": vector_stats.get("total_time_ms", vector_stats.get("vector_time_ms", 0.0)),
                    "hybrid_ms": retrieval_stats.get("total_time_ms", 0.0),
                    "overlap": retrieval_stats.get("overlap_count", 0),
                }

            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise
        
        # If no documents found
        if not relevant_docs:
            logger.warning("No relevant documents found")
            return "I don't have enough information to answer this question.", []
        
        # Build context
        context = "\n\n".join(
            [
                f"[Source: {doc.get_citation()} | chunk={doc.metadata.get('chunk_index', 'n/a')}]\n{doc.content}"
                for doc in relevant_docs
            ]
        )
        logger.debug(f"Context length: {len(context)} characters")
        
        # Create prompt
        prompt = self._build_prompt(context, query, conversation_context)
        
        # Generate response
        try:
            response = self.llm_service.generate(prompt)
            self._mark_used_chunks(answer=response, sources=relevant_docs)
            logger.info("Response generated successfully")
            return response, relevant_docs
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise

    def process_query_stream(
        self,
        query: str,
        k: int = 3,
        status_container=None,
    ) -> Tuple[Generator[str, None, None], List[Document]]:
        """
        Process user query and return a streaming response generator.

        Performs retrieval synchronously, then streams LLM tokens so the
        UI can display them in real-time instead of blocking.

        Args:
            query: User's question
            k: Number of documents to retrieve
            status_container: Optional ``st.status`` container for step-by-step UI updates

        Returns:
            Tuple of (stream_generator, source_documents)

        Raises:
            ValueError: If query is empty
            VectorStoreError: If vector store not initialized
            LLMConnectionError: If LLM generation fails
        """
        import time as _time

        t_start = _time.time()

        def _status_step(icon: str, message: str) -> None:
            """Write a step line into the status container if available."""
            if status_container is not None:
                status_container.write(f"{icon} {message}")

        # ── Step 1: Validate & analyse query ──────────────────────────
        if not query or not query.strip():
            logger.warning("Empty query received")
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing query (stream): {query[:50]}...")
        _status_step("[search]", f"**Phân tích câu hỏi:** `{query}`")

        use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
        use_rerank = bool(st.session_state.get("use_rerank", False))
        retrieval_k = int(st.session_state.get("retrieval_k", k))
        selected_sources = st.session_state.get("active_source_filters", [])
        selected_file_types = st.session_state.get("active_file_type_filters", [])

        metadata_filters: Dict[str, Any] = {}
        if selected_sources:
            metadata_filters["source_files"] = selected_sources
        if selected_file_types:
            metadata_filters["file_types"] = selected_file_types

        rewritten_query = self._rewrite_query(query)
        if rewritten_query != query:
            _status_step("[rewrite]", f"**Viết lại câu hỏi:** `{rewritten_query}`")
        else:
            _status_step("[ok]", "Câu hỏi không cần viết lại")

        conversation_context = self._conversation_context(max_turns=4)

        # ── Step 2: Check vector store ────────────────────────────────
        if self.vector_service is None or not self.vector_service.is_initialized:
            logger.error("Vector store not initialized")
            raise VectorStoreError("Please upload documents first")

        # ── Step 3: Retrieve relevant documents ───────────────────────
        _status_step("[docs]", f"Đang tìm kiếm tài liệu (k={retrieval_k}, hybrid={use_hybrid}, rerank={use_rerank})...")
        t_retrieval = _time.time()

        try:
            if hasattr(self.vector_service, "search"):
                relevant_docs, retrieval_stats = self.vector_service.search(
                    query=rewritten_query,
                    k=retrieval_k,
                    metadata_filters=metadata_filters,
                    use_hybrid=use_hybrid,
                    rerank=use_rerank,
                    fetch_k=max(retrieval_k * 4, 20),
                )
            else:
                relevant_docs = self.vector_service.similarity_search(rewritten_query, k=retrieval_k)
                retrieval_stats = {
                    "use_hybrid": False,
                    "rerank": False,
                    "results": len(relevant_docs),
                }

            st.session_state.last_retrieval_stats = retrieval_stats

            if use_hybrid and hasattr(self.vector_service, "search"):
                vector_only_docs, vector_stats = self.vector_service.search(
                    query=rewritten_query,
                    k=retrieval_k,
                    metadata_filters=metadata_filters,
                    use_hybrid=False,
                    rerank=False,
                    fetch_k=max(retrieval_k * 4, 20),
                )
                st.session_state.retrieval_comparison = {
                    "vector_results": len(vector_only_docs),
                    "hybrid_results": len(relevant_docs),
                    "vector_ms": vector_stats.get("total_time_ms", vector_stats.get("vector_time_ms", 0.0)),
                    "hybrid_ms": retrieval_stats.get("total_time_ms", 0.0),
                    "overlap": retrieval_stats.get("overlap_count", 0),
                }

            retrieval_ms = (_time.time() - t_retrieval) * 1000
            total_ms = retrieval_stats.get("total_time_ms", retrieval_ms)
            _status_step(
                "[ok]",
                f"Tìm thấy **{len(relevant_docs)} tài liệu** ({total_ms:.0f}ms)",
            )
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        except Exception as e:
            _status_step("[err]", f"Lỗi tìm kiếm: {e}")
            logger.error(f"Document retrieval failed: {e}")
            raise

        # If no documents found, return a simple non-streaming fallback
        if not relevant_docs:
            logger.warning("No relevant documents found")
            _status_step("[warn]", "Không tìm thấy tài liệu liên quan")

            def _no_info_stream():
                yield "I don't have enough information to answer this question."

            return _no_info_stream(), []

        # ── Step 4: Build context & prompt ────────────────────────────
        context = "\n\n".join(
            [
                f"[Source: {doc.get_citation()} | chunk={doc.metadata.get('chunk_index', 'n/a')}]\n{doc.content}"
                for doc in relevant_docs
            ]
        )
        logger.debug(f"Context length: {len(context)} characters")

        prompt = self._build_prompt(context, query, conversation_context)
        _status_step(
            "[build]",
            f"Đã xây dựng prompt ({len(prompt)} ký tự, {len(context)} ngữ cảnh)",
        )

        # ── Step 5: Stream LLM response ──────────────────────────────
        elapsed = (_time.time() - t_start) * 1000
        _status_step("[llm]", f"Đang gọi LLM (chuẩn bị mất {elapsed:.0f}ms)...")

        stream_gen = self.llm_service.generate_stream(prompt)
        return stream_gen, relevant_docs

    def _build_prompt(self, context: str, question: str, chat_history_context: str) -> str:
        """
        Build prompt for LLM.
        
        Args:
            context: Retrieved context from documents
            question: User's question
            
        Returns:
            Formatted prompt string
        """
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

    def _conversation_context(self, max_turns: int = 4) -> str:
        """Build compact conversational memory from session chat history."""
        chat_history = st.session_state.get("chat_history")
        if not chat_history or len(chat_history) == 0:
            return "No previous conversation."

        recent_messages = chat_history.get_recent(n=max_turns * 2)
        lines: List[str] = []
        for message in recent_messages:
            role = "User" if message.role == "user" else "Assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def _rewrite_query(self, query: str) -> str:
        """Rewrite short follow-up queries using previous user message context."""
        chat_history = self._get_active_history()
        if not chat_history or len(chat_history) == 0:
            return query

        stripped = query.strip()
        lowered = stripped.lower()
        followup_markers = ["nó", "cái đó", "điều đó", "thế còn", "it", "that", "those", "they", "them"]
        is_followup = len(stripped.split()) <= 8 or any(marker in lowered for marker in followup_markers)

        if not is_followup:
            return stripped

        recent_user_questions = [msg.content for msg in chat_history.get_recent(8) if msg.role == "user"]
        previous_question = recent_user_questions[-1] if recent_user_questions else ""

        if not previous_question:
            return stripped

        rewritten = f"{stripped} (follow-up to previous question: {previous_question})"
        logger.info("Rewrote follow-up query for conversational retrieval")
        return rewritten

    @staticmethod
    def _mark_used_chunks(answer: str, sources: List[Document]) -> None:
        """Tag likely used chunks for UI highlighting."""
        answer_terms = {term for term in (answer or "").lower().split() if len(term) > 3}
        for source in sources:
            content = source.content.lower()
            overlap = sum(1 for term in answer_terms if term in content)
            source.metadata["used_in_answer"] = overlap > 0
            source.metadata["used_term_overlap"] = overlap
    
    @staticmethod
    def _get_active_history():
        """Get chat history from the active session (multi-chat support)."""
        sessions = st.session_state.get("chat_sessions", [])
        active_id = st.session_state.get("active_chat_id")
        for session in sessions:
            if session.id == active_id:
                return session.history
        # Fallback to legacy chat_history
        return st.session_state.get("chat_history")

    def clear_history(self) -> None:
        """Clear chat history for the active session."""
        history = self._get_active_history()
        if history:
            history.clear()
            logger.info("Chat history cleared for active session")

    def format_reply_for_streamlit(self, answer: str, sources: List[Document]) -> str:
        """
        Format assistant reply for Streamlit using intro/body/footer templates.

        Args:
            answer: Raw answer from LLM
            sources: Retrieved source documents

        Returns:
            Formatted text for Streamlit UI
        """
        templates = st.session_state.get('reply_templates', DEFAULT_STREAMLIT_REPLY_TEMPLATES)
        found_template = templates.get('found', DEFAULT_STREAMLIT_REPLY_TEMPLATES['found'])
        not_found_template = templates.get('not_found', DEFAULT_STREAMLIT_REPLY_TEMPLATES['not_found'])

        normalized_answer = (answer or '').strip().lower()
        has_no_info_marker = any(marker in normalized_answer for marker in NO_INFO_MARKERS)
        has_answer = bool((answer or '').strip()) and (not has_no_info_marker) and bool(sources)

        selected = found_template if has_answer else not_found_template
        parts: List[str] = [selected.get('intro', '').strip()]

        if has_answer:
            parts.append((answer or '').strip())

        body = selected.get('body', '').strip()
        if body:
            parts.append(body)

        footer = selected.get('footer', '').strip()
        if footer:
            parts.append(footer)

        # Filter empty parts and keep spacing predictable in chat bubbles.
        return "\n\n".join([part for part in parts if part])

    # ── Self-RAG: Self-evaluation, Confidence, Multi-hop ──────────

    def _self_evaluate(self, question: str, answer: str, context: str) -> Tuple[float, str]:
        """
        Ask the LLM to evaluate its own answer (Self-RAG).

        Returns:
            Tuple of (score 1-5, justification)
        """
        eval_prompt = f"""You are an answer quality evaluator. Rate the following answer on a scale of 1-5.

QUESTION: {question}

CONTEXT PROVIDED:
{context[:2000]}

ANSWER:
{answer}

Rate the answer on these criteria (1=poor, 5=excellent):
1. Accuracy: Is the answer supported by the context?
2. Completeness: Does it fully address the question?
3. Relevance: Is it on-topic?

Respond in this EXACT format:
SCORE: <number 1-5>
JUSTIFICATION: <one sentence explanation>"""

        try:
            response = self.llm_service.generate(eval_prompt)
            score = 3.0  # default
            justification = "No justification provided"

            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip().split()[0])
                        score = max(1.0, min(5.0, score))
                    except (ValueError, IndexError):
                        pass
                elif line.upper().startswith("JUSTIFICATION:"):
                    justification = line.split(":", 1)[1].strip()

            logger.info(f"Self-evaluation score: {score}/5 — {justification}")
            return score, justification
        except Exception as e:
            logger.warning(f"Self-evaluation failed: {e}")
            return 3.0, f"Evaluation unavailable: {str(e)}"

    def _compute_confidence(self, sources: List[Document], self_eval_score: float) -> Tuple[float, str]:
        """
        Compute confidence score based on retrieval quality and self-evaluation.

        Returns:
            Tuple of (confidence 0-100%, level description)
        """
        if not sources:
            return 0.0, "Không có tài liệu tham khảo"

        # Retrieval quality: average term overlap
        overlaps = [s.metadata.get("used_term_overlap", 0) for s in sources]
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        retrieval_quality = min(avg_overlap / 10.0, 1.0)  # normalize to 0-1

        # Number of sources factor
        source_factor = min(len(sources) / 3.0, 1.0)

        # Self-evaluation factor (normalize 1-5 to 0-1)
        eval_factor = (self_eval_score - 1.0) / 4.0

        # Weighted combination
        confidence = (retrieval_quality * 0.3 + source_factor * 0.2 + eval_factor * 0.5) * 100
        confidence = round(max(0.0, min(100.0, confidence)), 1)

        if confidence >= 80:
            level = "High confidence"
        elif confidence >= 60:
            level = "Moderate confidence"
        elif confidence >= 40:
            level = "Low confidence"
        else:
            level = "Very low confidence"

        return confidence, level

    def _multi_hop_reasoning(self, query: str, k: int = 3) -> Tuple[str, List[Document]]:
        """
        Break complex questions into sub-queries for multi-hop reasoning.

        Returns:
            Tuple of (synthesized answer, all sources)
        """
        # Generate sub-queries
        decompose_prompt = f"""Break this complex question into 2-3 simpler sub-questions.
Each sub-question should focus on one aspect.

QUESTION: {query}

Respond with one sub-question per line, numbered 1-3. No explanation needed."""

        try:
            response = self.llm_service.generate(decompose_prompt)
            sub_queries = []
            for line in response.strip().split("\n"):
                line = line.strip()
                # Remove numbering like "1. ", "2) ", etc.
                import re
                cleaned = re.sub(r'^[\d]+[.)]\s*', '', line).strip()
                if cleaned and len(cleaned) > 5:
                    sub_queries.append(cleaned)

            if not sub_queries:
                return "", []

            logger.info(f"Multi-hop: decomposed into {len(sub_queries)} sub-queries")

            # Retrieve for each sub-query
            all_sources: List[Document] = []
            sub_answers: List[str] = []

            for sub_q in sub_queries[:3]:  # max 3 hops
                try:
                    if hasattr(self.vector_service, "search"):
                        docs, _ = self.vector_service.search(query=sub_q, k=k, use_hybrid=False, rerank=False)
                    else:
                        docs = self.vector_service.similarity_search(sub_q, k=k)

                    all_sources.extend(docs)
                    if docs:
                        context = "\n".join([d.content for d in docs[:2]])
                        sub_answers.append(f"Sub-question: {sub_q}\nAnswer: {context[:500]}")
                except Exception as e:
                    logger.warning(f"Multi-hop sub-query failed: {e}")

            # Deduplicate sources
            seen = set()
            unique_sources = []
            for s in all_sources:
                key = s.content[:100]
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)

            # Synthesize
            if sub_answers:
                synthesis_prompt = f"""Based on the following sub-answers, provide a comprehensive answer to the original question.

ORIGINAL QUESTION: {query}

SUB-ANSWERS:
{chr(10).join(sub_answers)}

Provide a clear, comprehensive answer:"""
                synthesis = self.llm_service.generate(synthesis_prompt)
                return synthesis, unique_sources[:k]

            return "", unique_sources[:k]

        except Exception as e:
            logger.warning(f"Multi-hop reasoning failed: {e}")
            return "", []

    def process_query_with_self_rag(
        self,
        query: str,
        k: int = 3,
    ) -> Tuple[str, List[Document], float, str, str]:
        """
        Process query with full Self-RAG pipeline.

        Returns:
            Tuple of (answer, sources, confidence_score, confidence_level, self_eval_justification)
        """
        # Step 1: Query rewriting
        rewritten = self._rewrite_query(query)

        # Step 2: Check if complex enough for multi-hop
        word_count = len(query.split())
        has_complex_markers = any(m in query.lower() for m in ["và", "and", "so sánh", "compare", "khác gì", "difference"])
        use_multi_hop = word_count > 12 and has_complex_markers

        # Step 3: Retrieval
        if self.vector_service is None or not self.vector_service.is_initialized:
            raise VectorStoreError("Please upload documents first")

        if use_multi_hop:
            logger.info("Using multi-hop reasoning for complex query")
            answer, sources = self._multi_hop_reasoning(rewritten, k)
            if not answer:
                # Fallback to normal retrieval
                answer, sources = self._normal_retrieval(rewritten, k)
        else:
            answer, sources = self._normal_retrieval(rewritten, k)

        if not answer:
            return "I don't have enough information to answer this question.", [], 0.0, "Very low confidence", ""

        # Step 4: Self-evaluation
        context = "\n".join([s.content[:500] for s in sources[:3]])
        eval_score, eval_justification = self._self_evaluate(query, answer, context)

        # Step 5: Confidence scoring
        self._mark_used_chunks(answer, sources)
        confidence, confidence_level = self._compute_confidence(sources, eval_score)

        logger.info(f"Self-RAG complete: confidence={confidence}%, eval={eval_score}/5")

        return answer, sources, confidence, confidence_level, eval_justification

    def _normal_retrieval(self, query: str, k: int = 3) -> Tuple[str, List[Document]]:
        """Standard single-hop retrieval and generation."""
        use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
        use_rerank = bool(st.session_state.get("use_rerank", False))
        retrieval_k = int(st.session_state.get("retrieval_k", k))

        if hasattr(self.vector_service, "search"):
            relevant_docs, _ = self.vector_service.search(
                query=query, k=retrieval_k, use_hybrid=use_hybrid, rerank=use_rerank,
                fetch_k=max(retrieval_k * 4, 20),
            )
        else:
            relevant_docs = self.vector_service.similarity_search(query, k=retrieval_k)

        if not relevant_docs:
            return "", []

        context = "\n\n".join([
            f"[Source: {doc.get_citation()}]\n{doc.content}"
            for doc in relevant_docs
        ])

        conversation_context = self._conversation_context(max_turns=4)
        prompt = self._build_prompt(context, query, conversation_context)
        response = self.llm_service.generate(prompt)
        return response, relevant_docs

