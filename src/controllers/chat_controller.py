"""Chat controller for handling chat operations."""

from typing import Any, Dict, Generator, Optional, List, Tuple
import re
import time
import streamlit as st
from src.services.llm_service import AbstractLLMService, OllamaLLMService
from src.services.vector_store_service import AbstractVectorStoreService, RetrievalBenchmark
from src.services.rag_service import (
    AbstractRAGStrategy,
    StandardRAGStrategy,
    ChainOfRAGStrategy,
)
from src.models.chat_model import ChatHistory
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import LLMConnectionError, VectorStoreError
from src.utils.constants import (
    DEFAULT_STREAMLIT_REPLY_TEMPLATES,
    LANG_REPLY_TEMPLATES,
    NO_INFO_MARKERS,
    NO_INFO_MESSAGE,
    NO_INFO_INSTRUCTION,
    RAG_TYPE_STANDARD,
    RAG_TYPE_CORAG,
)

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
    
    @staticmethod
    def _get_source_filters() -> List[str]:
        """Read current source filters, respecting temporary overrides.

        The ``active_source_filters`` key is bound to a multiselect widget
        and cannot be modified after the widget is instantiated.  Views
        that need to temporarily override the filter (e.g. @mention focus
        mode) write to ``_source_filter_override`` instead.  This helper
        checks the override first and falls back to the widget value.

        Returns:
            List of selected source file names (may be empty).
        """
        override = st.session_state.get("_source_filter_override")
        if override is not None:
            return override
        return st.session_state.get("active_source_filters", [])

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
        selected_sources = self._get_source_filters()
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
            comparison = self._build_retrieval_comparison(
                query=rewritten_query,
                k=retrieval_k,
                metadata_filters=metadata_filters,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                retrieved_docs=relevant_docs,
                retrieval_stats=retrieval_stats,
            )
            if comparison:
                st.session_state.retrieval_comparison = comparison
            else:
                st.session_state.pop("retrieval_comparison", None)

            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise
        
        # If no documents found
        if not relevant_docs:
            logger.warning("No relevant documents found")
            lang = self._detect_target_language(query)
            no_info = NO_INFO_MESSAGE.get(lang, NO_INFO_MESSAGE["en"])
            return no_info, []
        
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
    ) -> Tuple[Generator[str, None, None], List[Document], str]:
        """
        Process user query and return a streaming response generator.

        Performs retrieval synchronously, then streams LLM tokens so the
        UI can display them in real-time instead of blocking.

        Args:
            query: User's question
            k: Number of documents to retrieve
            status_container: Optional ``st.status`` container for step-by-step UI updates

        Returns:
            Tuple of (stream_generator, source_documents, rewritten_query)

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
        selected_sources = self._get_source_filters()
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
            comparison = self._build_retrieval_comparison(
                query=rewritten_query,
                k=retrieval_k,
                metadata_filters=metadata_filters,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                retrieved_docs=relevant_docs,
                retrieval_stats=retrieval_stats,
            )
            if comparison:
                st.session_state.retrieval_comparison = comparison
            else:
                st.session_state.pop("retrieval_comparison", None)

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

            lang = self._detect_target_language(query)
            _no_info_msg = NO_INFO_MESSAGE.get(lang, NO_INFO_MESSAGE["en"])

            def _no_info_stream():
                yield _no_info_msg

            return _no_info_stream(), [], rewritten_query

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
        return stream_gen, relevant_docs, rewritten_query

    @staticmethod
    def _doc_identity(doc: Document) -> str:
        """Build a deterministic doc identifier for overlap/rank comparisons."""
        return (
            f"{doc.metadata.get('source', '')}|"
            f"{doc.metadata.get('page', '')}|"
            f"{doc.metadata.get('chunk_index', '')}|"
            f"{doc.content[:160]}"
        )

    def _build_retrieval_comparison(
        self,
        query: str,
        k: int,
        metadata_filters: Dict[str, Any],
        use_hybrid: bool,
        use_rerank: bool,
        retrieved_docs: List[Document],
        retrieval_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create comparison metrics for hybrid/vector and rerank/bi-encoder."""
        if not hasattr(self.vector_service, "search"):
            return {}

        fetch_k = max(k * 4, 20)
        comparison: Dict[str, Any] = {}

        if use_hybrid:
            vector_docs, vector_stats = self.vector_service.search(
                query=query,
                k=k,
                metadata_filters=metadata_filters,
                use_hybrid=False,
                rerank=False,
                fetch_k=fetch_k,
            )
            overlap = len(
                {self._doc_identity(doc) for doc in vector_docs}.intersection(
                    {self._doc_identity(doc) for doc in retrieved_docs}
                )
            )
            comparison["hybrid_vs_vector"] = {
                "vector_results": len(vector_docs),
                "hybrid_results": len(retrieved_docs),
                "vector_ms": vector_stats.get("total_time_ms", vector_stats.get("vector_time_ms", 0.0)),
                "hybrid_ms": retrieval_stats.get("total_time_ms", 0.0),
                "overlap": overlap,
            }

        if use_rerank:
            bi_docs, bi_stats = self.vector_service.search(
                query=query,
                k=k,
                metadata_filters=metadata_filters,
                use_hybrid=use_hybrid,
                rerank=False,
                fetch_k=fetch_k,
            )

            rerank_ids = [self._doc_identity(doc) for doc in retrieved_docs]
            bi_ids = [self._doc_identity(doc) for doc in bi_docs]
            top_k = min(len(rerank_ids), len(bi_ids), k)
            rank_changes = sum(1 for idx in range(top_k) if rerank_ids[idx] != bi_ids[idx])
            overlap = len(set(rerank_ids).intersection(set(bi_ids)))

            comparison["rerank_vs_biencoder"] = {
                "mode": "hybrid+rerank" if use_hybrid else "vector+rerank",
                "bi_encoder_results": len(bi_docs),
                "rerank_results": len(retrieved_docs),
                "bi_encoder_ms": bi_stats.get("total_time_ms", bi_stats.get("vector_time_ms", 0.0)),
                "rerank_ms": retrieval_stats.get("total_time_ms", 0.0),
                "rerank_only_ms": retrieval_stats.get("rerank_time_ms", 0.0),
                "topk_overlap": overlap,
                "topk_rank_changes": rank_changes,
            }

        return comparison

    def benchmark_rerank_queries(self, queries: List[str], k: int = 3) -> Dict[str, Any]:
        """Run A/B benchmark for bi-encoder vs cross-encoder rerank on multiple queries."""
        if self.vector_service is None or not self.vector_service.is_initialized:
            raise VectorStoreError("Please upload documents first")
        if not hasattr(self.vector_service, "search"):
            raise VectorStoreError("Advanced search API is unavailable")

        cleaned_queries = [query.strip() for query in queries if query and query.strip()]
        if not cleaned_queries:
            raise ValueError("Please provide at least one benchmark query")

        use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
        metadata_filters: Dict[str, Any] = {}
        selected_sources = self._get_source_filters()
        selected_file_types = st.session_state.get("active_file_type_filters", [])
        if selected_sources:
            metadata_filters["source_files"] = selected_sources
        if selected_file_types:
            metadata_filters["file_types"] = selected_file_types

        rows: List[Dict[str, Any]] = []
        for query in cleaned_queries:
            bi_docs, bi_stats = self.vector_service.search(
                query=query,
                k=k,
                metadata_filters=metadata_filters,
                use_hybrid=use_hybrid,
                rerank=False,
                fetch_k=max(k * 4, 20),
            )
            rerank_docs, rerank_stats = self.vector_service.search(
                query=query,
                k=k,
                metadata_filters=metadata_filters,
                use_hybrid=use_hybrid,
                rerank=True,
                fetch_k=max(k * 4, 20),
            )

            bi_ids = [self._doc_identity(doc) for doc in bi_docs]
            rerank_ids = [self._doc_identity(doc) for doc in rerank_docs]
            top_k = min(len(bi_ids), len(rerank_ids), k)
            overlap = len(set(bi_ids).intersection(set(rerank_ids)))
            rank_changes = sum(1 for idx in range(top_k) if bi_ids[idx] != rerank_ids[idx])

            rows.append(
                {
                    "query": query,
                    "bi_encoder_ms": bi_stats.get("total_time_ms", bi_stats.get("vector_time_ms", 0.0)),
                    "rerank_ms": rerank_stats.get("total_time_ms", 0.0),
                    "rerank_only_ms": rerank_stats.get("rerank_time_ms", 0.0),
                    "topk_overlap": overlap,
                    "topk_rank_changes": rank_changes,
                }
            )

        bi_latencies = [row["bi_encoder_ms"] for row in rows]
        rerank_latencies = [row["rerank_ms"] for row in rows]
        rerank_only_latencies = [row["rerank_only_ms"] for row in rows]
        rank_changes = [row["topk_rank_changes"] for row in rows]

        summary = {
            "queries": len(rows),
            "avg_bi_encoder_ms": round(sum(bi_latencies) / len(bi_latencies), 2),
            "avg_rerank_ms": round(sum(rerank_latencies) / len(rerank_latencies), 2),
            "avg_rerank_only_ms": round(sum(rerank_only_latencies) / len(rerank_only_latencies), 2),
            "avg_rank_changes": round(sum(rank_changes) / len(rank_changes), 2),
            "mode": "hybrid" if use_hybrid else "vector",
        }
        return {"rows": rows, "summary": summary}

    def _build_prompt(self, context: str, question: str, chat_history_context: str) -> str:
        """
        Build prompt for LLM with explicit language enforcement.
        
        Args:
            context: Retrieved context from documents
            question: User's question
            chat_history_context: Recent conversation history string
            
        Returns:
            Formatted prompt string
        """
        # Detect question language for explicit enforcement
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", question))
        has_vn = any(
            c in question.lower()
            for c in ["đ", "ă", "â", "ê", "ô", "ơ", "ư", "á", "à", "ả", "ã", "ạ"]
        )

        if has_cjk:
            lang_instruction = "Respond in Chinese (中文)."
        elif has_vn:
            lang_instruction = (
                "Respond ONLY in Vietnamese (Tiếng Việt).\n"
                "DO NOT use Chinese characters or Chinese language."
            )
        else:
            lang_instruction = (
                "Respond ONLY in English.\n"
                "DO NOT use Chinese characters or Chinese language under any circumstances."
            )

        lang = self._detect_target_language(question)
        no_info_instr = NO_INFO_INSTRUCTION.get(lang, NO_INFO_INSTRUCTION["en"])

        return f"""You are SmartDoc AI, an intelligent document assistant.

CRITICAL LANGUAGE RULE (HIGHEST PRIORITY):
{lang_instruction}

Answer the QUESTION based ONLY on the CONTEXT below.
{no_info_instr}
Keep your answer concise (3-4 sentences maximum).
Be factual and precise.

    RECENT CONVERSATION (for resolving references only — do NOT use as a knowledge source):
    {chat_history_context}

CONTEXT (this is your ONLY source of factual information):
{context}

QUESTION:
{question}

RULE: If the user asks you to add features, code, or logic that are NOT in the documents, explicitly REFUSE that specific part of the request in your text, and DO NOT write the code for it.

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
        """Rewrite follow-up queries using LLM to create a standalone question.
        
        Only rewrites when the query is a genuine follow-up (contains pronouns
        or references to previous conversation). Rejects rewrites that introduce
        unrelated topics from chat history.
        """
        chat_history = self._get_active_history()
        if not chat_history or len(chat_history) == 0:
            return query

        # Stricter follow-up detection: require pronoun/reference markers
        # AND short query length to qualify as a follow-up.
        stripped = query.strip()
        lowered = stripped.lower()
        followup_markers = [
            "nó", "cái đó", "điều đó", "thế còn", "cái kia", "vấn đề đó",
            "phần đó", "chuyện đó", "vậy còn", "thì sao",
            "it", "that", "those", "they", "them", "this", "what about",
        ]
        has_followup_marker = any(marker in lowered for marker in followup_markers)
        word_count = len(stripped.split())

        # Only rewrite if there's clear evidence of a follow-up:
        # - Short query (<=5 words) with a pronoun marker, OR
        # - Has pronoun marker and moderately short (<=8 words)
        is_followup = (word_count <= 5 and has_followup_marker) or (
            has_followup_marker and word_count <= 8
        )

        if not is_followup:
            logger.debug(f"Query not identified as follow-up, keeping original: '{stripped}'")
            return stripped

        # Build conversation context for rewriting
        recent_messages = chat_history.get_recent(n=6)
        history_lines = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        
        history_text = "\n".join(history_lines)

        rewrite_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a STANDALONE question that can be understood without the conversation history.

CRITICAL RULES:
1. Maintain the original language of the follow-up question.
2. If the follow-up question is about a DIFFERENT TOPIC than the conversation, return it AS-IS.
3. Do NOT merge or combine topics from previous conversation into the new question.
4. Only resolve pronouns and references (it, that, they, nó, cái đó, etc.) to their actual meaning.
5. If the question is already clear and self-contained, return it exactly as-is.

CONVERSATION HISTORY:
{history_text}

FOLLOW-UP QUESTION: {query}

STANDALONE QUESTION:"""

        try:
            rewritten = self.llm_service.generate(rewrite_prompt).strip()
            # Basic validation: if LLM returns something too short or fails, fallback
            if len(rewritten) > 5:
                # Topic divergence guard: reject rewrite if it introduces old topics
                if self._is_topic_divergent(stripped, rewritten, history_text):
                    logger.warning(
                        f"Rewrite rejected (topic divergence): "
                        f"'{stripped}' -> '{rewritten}'. Using original."
                    )
                    return stripped
                logger.info(f"LLM rewrote query: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"LLM query rewriting failed: {e}")

        # Fallback to original query — do NOT append previous topic
        return stripped

    @staticmethod
    def _is_topic_divergent(
        original_query: str, rewritten_query: str, history_text: str
    ) -> bool:
        """
        Detect if the rewritten query has drifted to a different topic.

        Compares key terms between original query, rewritten query, and chat
        history. Returns True if the rewrite introduces too many terms from
        history that were NOT present in the original query.

        Args:
            original_query: The user's raw input query
            rewritten_query: The LLM-rewritten standalone query
            history_text: Recent conversation history text

        Returns:
            True if topic divergence is detected (rewrite should be rejected)
        """
        # Common stop words to exclude from term analysis
        stop_words = {
            "và", "của", "là", "có", "không", "được", "cho", "với", "một", "các",
            "những", "này", "đó", "khi", "nếu", "thì", "để", "từ", "về", "đã",
            "sẽ", "đang", "tôi", "bạn", "chúng", "họ", "nó", "cái", "còn", "thế",
            "như", "sau", "trước", "nào", "vậy", "gì", "thêm", "nữa", "đi", "lên",
            "xuống", "ra", "vào", "lại", "mà", "nhưng", "vì", "nên", "rằng", "cũng",
            "đây", "kia", "vẫn", "mới", "phải", "nhiều", "ít", "hơn", "nhất",
            "the", "and", "for", "with", "this", "that", "from", "are", "was",
            "were", "been", "have", "has", "had", "will", "would", "could",
            "should", "does", "what", "which", "when", "where", "who", "how",
            "about", "into", "than", "then", "also", "just", "only", "over",
            "some", "very", "can", "not", "but", "its", "our", "your", "their",
        }

        def extract_terms(text: str) -> set:
            """Extract meaningful terms (>3 chars, not stop words)."""
            words = re.findall(r'\b[a-zà-ỹ]{4,}\b', text.lower())
            return {w for w in words if w not in stop_words}

        original_terms = extract_terms(original_query)
        rewritten_terms = extract_terms(rewritten_query)
        history_terms = extract_terms(history_text)

        if not original_terms:
            return False  # Can't determine topic from very short original

        # Terms in rewrite that came from history but weren't in original
        history_intrusion = (rewritten_terms - original_terms) & history_terms

        # Terms from original that were completely lost in rewrite
        lost_terms = original_terms - rewritten_terms

        # Calculate divergence metrics
        loss_ratio = len(lost_terms) / len(original_terms)
        intrusion_count = len(history_intrusion)

        # Signal 1: High original term loss AND history terms intruding
        if loss_ratio > 0.5 and intrusion_count >= 2:
            logger.info(
                f"Topic divergence: lost {lost_terms}, "
                f"history intrusion {history_intrusion}"
            )
            return True

        # Signal 2: Excessive history term intrusion regardless of loss
        if intrusion_count >= 4:
            logger.info(f"Excessive history intrusion: {history_intrusion}")
            return True

        return False

    @staticmethod
    def _detect_target_language(query: str) -> str:
        """Detect target output language from user query (vi/en/zh)."""
        lowered = (query or "").lower()
        if re.search(r"[\u4e00-\u9fff]", query or ""):
            return "zh"
        vietnamese_markers = [
            "đ", "ă", "â", "ê", "ô", "ơ", "ư", "á", "à", "ả", "ã", "ạ",
            "câu", "tài liệu", "và", "không", "như thế nào", "là gì",
        ]
        if any(marker in lowered for marker in vietnamese_markers):
            return "vi"
        return "en"

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """Check whether text contains CJK characters."""
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    def _enforce_answer_language(self, answer: str, query: str) -> str:
        """Force answer language to follow user query when model drifts."""
        target_lang = self._detect_target_language(query)
        if target_lang == "zh":
            return answer

        if not self._contains_cjk(answer):
            return answer

        language_name = "Vietnamese" if target_lang == "vi" else "English"
        rewrite_prompt = f"""Rewrite the answer below into {language_name}.

Requirements:
1) Keep the meaning unchanged.
2) Do not add new facts.
3) Do not use Chinese.
4) Keep concise (3-4 sentences).

ANSWER:
{answer}

REWRITTEN ANSWER:"""
        try:
            rewritten = self.llm_service.generate(rewrite_prompt).strip()
            if rewritten:
                return rewritten
        except Exception as rewrite_error:
            logger.warning(f"Language enforcement rewrite failed: {rewrite_error}")
        return answer

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

    def format_reply_for_streamlit(
        self,
        answer: str,
        sources: List[Document],
        query: str = "",
    ) -> str:
        """
        Format assistant reply for Streamlit using language-aware templates.

        Detects the query language and selects intro/body/footer templates
        that match. Falls back to English templates if language-specific
        ones are not available.

        Args:
            answer: Raw answer from LLM
            sources: Retrieved source documents
            query: Original user question (used for language detection)

        Returns:
            Formatted text for Streamlit UI
        """
        # Detect language from query (or from answer if query unavailable)
        detected_lang = self._detect_target_language(query or answer or "")
        lang_templates = LANG_REPLY_TEMPLATES.get(detected_lang, LANG_REPLY_TEMPLATES.get("en", {}))

        # Use language-aware templates.  Session-state templates are only
        # used when the user has *explicitly* customised them (non-empty intro).
        user_templates = st.session_state.get('reply_templates', None)
        using_custom = (
            user_templates
            and user_templates.get("found", {}).get("intro", "").strip()
        )
        if using_custom:
            found_template = user_templates.get('found', lang_templates.get('found', {}))
            not_found_template = user_templates.get('not_found', lang_templates.get('not_found', {}))
        else:
            found_template = lang_templates.get('found', {})
            not_found_template = lang_templates.get('not_found', {})

        # Use the same strict refusal check as _compute_confidence
        # to avoid false positives (e.g. long answers mentioning "thông tin")
        is_refusal = self._is_refusal_response(answer) if answer else True
        has_answer = bool((answer or '').strip()) and (not is_refusal) and bool(sources)

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
JUSTIFICATION: <one concise sentence in Vietnamese explaining the rating>"""

        try:
            response = self.llm_service.generate(eval_prompt)
            score = 3.0  # default
            justification = "Chưa có giải thích"

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
            return 3.0, f"Không thể tự đánh giá: {str(e)}"

    def _compute_confidence(
        self,
        sources: List[Document],
        self_eval_score: float,
        answer: str = "",
        entities: Optional[List[str]] = None,
    ) -> Tuple[float, str]:
        """
        Compute confidence score based on retrieval quality, self-evaluation,
        and entity coverage.

        Args:
            sources: Retrieved source documents
            self_eval_score: Self-evaluation score (1-5)
            answer: The actual answer text (used to detect refusals)
            entities: List of detected entities from query (for coverage boost)

        Returns:
            Tuple of (confidence 0-100%, level description)
        """
        if not sources:
            return 0.0, "Không có tài liệu tham khảo"

        # If the answer is a refusal/no-info response, confidence is low
        # regardless of retrieval quality or self-eval score.
        if answer and self._is_refusal_response(answer):
            logger.info(
                "Confidence penalized: answer is a refusal/no-info response"
            )
            # Still compute base score for transparency, but cap it low
            base_eval = max(1.0, min(5.0, self_eval_score))
            eval_factor = (base_eval - 1.0) / 4.0
            confidence = round(eval_factor * 25.0, 1)  # Max ~25% for refusals
            confidence = max(0.0, min(30.0, confidence))
            level = "Độ tin cậy rất thấp"
            return confidence, level

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

        # ── Entity Coverage Boost ───────────────────────────────
        # If all query entities appear in the answer text, boost confidence
        # significantly. This prevents low scores when comparing entities
        # with different names (word overlap is low but coverage is complete).
        if entities and answer:
            answer_lower = answer.lower()
            covered = sum(1 for e in entities if e.lower() in answer_lower)
            entity_coverage = covered / len(entities) if entities else 0.0

            if entity_coverage >= 1.0:
                # All entities found → strong boost
                confidence = max(confidence, 75.0)
                logger.info(
                    "Entity coverage boost: %d/%d entities found in answer → confidence=%.1f%%",
                    covered, len(entities), confidence,
                )
            elif entity_coverage >= 0.5:
                # At least half of entities found → moderate boost
                confidence = max(confidence, 60.0)
                logger.info(
                    "Entity coverage partial boost: %d/%d entities → confidence=%.1f%%",
                    covered, len(entities), confidence,
                )

        confidence = round(max(0.0, min(100.0, confidence)), 1)

        if confidence >= 80:
            level = "Độ tin cậy cao"
        elif confidence >= 60:
            level = "Độ tin cậy trung bình"
        elif confidence >= 40:
            level = "Độ tin cậy thấp"
        else:
            level = "Độ tin cậy rất thấp"

        return confidence, level

    @staticmethod
    def _is_refusal_response(answer: str) -> bool:
        """
        Check if the answer is a genuine refusal/no-info response.

        Uses a strict multi-criteria check to avoid false positives:
        - Short answers (< 150 chars): check directly against markers
        - Longer answers: only refusal if the answer is purely a refusal
          phrase with minimal extra content (< 80 chars beyond marker).

        This prevents false positives where a detailed answer happens to
        contain "thông tin" or starts with "không đủ thông tin" but
        still provides valuable partial information.

        Examples:
        - "Tài liệu không chứa đủ thông tin." → True (pure refusal)
        - "Tài liệu không chứa đủ thông tin. Tài liệu chỉ cung cấp thông
           tin về A và B..." → False (has substantive content beyond refusal)

        Args:
            answer: The answer text to check

        Returns:
            True if this is a genuine refusal response
        """
        if not answer or not answer.strip():
            return True  # Empty answer = de facto refusal

        normalized = answer.strip().lower()

        # Short responses (< 150 chars): check directly against markers.
        # Genuine refusals are typically very short.
        if len(normalized) < 150:
            return any(marker in normalized for marker in NO_INFO_MARKERS)

        # Longer responses: check if there's substantive content beyond
        # the refusal marker. If the LLM says "không đủ thông tin" but
        # then adds partial info (e.g. "Tài liệu chỉ cung cấp thông tin
        # về A và B, nhưng không so sánh trực tiếp"), that's valuable
        # content that should be shown to the user.
        for marker in NO_INFO_MARKERS:
            if marker in normalized:
                marker_idx = normalized.find(marker)
                # Content excluding the marker phrase itself
                before = normalized[:marker_idx].strip()
                after = normalized[marker_idx + len(marker):].strip()
                substantive_content = len(before) + len(after)
                # If there's meaningful content (> 80 chars) beyond
                # the marker, this is NOT a pure refusal — it's an
                # answer with partial info + disclaimer.
                if substantive_content > 80:
                    continue
                # Very little content beyond marker → pure refusal
                return True

        return False

    def _multi_hop_reasoning(
        self,
        query: str,
        k: int = 3,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Document]]:
        """
        Break complex questions into sub-queries for multi-hop reasoning.

        Returns:
            Tuple of (synthesized answer, all sources)
        """
        # Generate sub-queries
        target_lang = self._detect_target_language(query)
        language_name = "Vietnamese" if target_lang == "vi" else ("Chinese" if target_lang == "zh" else "English")

        decompose_prompt = f"""Break this complex question into 2-3 simpler sub-questions.
Each sub-question should focus on one aspect.
    Write sub-questions in {language_name}.

QUESTION: {query}

Respond with one sub-question per line, numbered 1-3. No explanation needed."""

        try:
            total_start = time.perf_counter()
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
            hop_details: List[Dict[str, Any]] = []

            for sub_q in sub_queries[:3]:  # max 3 hops
                try:
                    hop_start = time.perf_counter()
                    if hasattr(self.vector_service, "search"):
                        docs, _ = self.vector_service.search(
                            query=sub_q,
                            k=k,
                            metadata_filters=metadata_filters,
                            use_hybrid=False,
                            rerank=False,
                            fetch_k=max(k * 4, 20),
                        )
                    else:
                        docs = self.vector_service.similarity_search(sub_q, k=k)

                    hop_details.append(
                        {
                            "sub_query": sub_q,
                            "results": len(docs),
                            "time_ms": round((time.perf_counter() - hop_start) * 1000, 2),
                        }
                    )

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
                st.session_state.last_retrieval_stats = {
                    "mode": "self_rag_multi_hop",
                    "hops": len(hop_details),
                    "hop_details": hop_details,
                    "total_time_ms": round((time.perf_counter() - total_start) * 1000, 2),
                    "results": len(unique_sources[:k]),
                }
                st.session_state.pop("retrieval_comparison", None)

                synthesis_prompt = f"""Based on the following sub-answers, provide a comprehensive answer to the original question.

Language rule:
- Respond in {language_name}.
- If the question is not Chinese, do not use Chinese.

ORIGINAL QUESTION: {query}

SUB-ANSWERS:
{chr(10).join(sub_answers)}

Provide a clear, comprehensive answer:"""
                synthesis = self.llm_service.generate(synthesis_prompt)
                synthesis = self._enforce_answer_language(synthesis, query)
                return synthesis, unique_sources[:k]

            return "", unique_sources[:k]

        except Exception as e:
            logger.warning(f"Multi-hop reasoning failed: {e}")
            return "", []

    def process_query_with_self_rag(
        self,
        query: str,
        k: int = 3,
    ) -> Tuple[str, List[Document], float, str, str, str]:
        """
        Process query with full Self-RAG pipeline.

        Returns:
            Tuple of (answer, sources, confidence_score, confidence_level, self_eval_justification, rewritten_query)
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

        selected_sources = self._get_source_filters()
        selected_file_types = st.session_state.get("active_file_type_filters", [])
        metadata_filters: Dict[str, Any] = {}
        if selected_sources:
            metadata_filters["source_files"] = selected_sources
        if selected_file_types:
            metadata_filters["file_types"] = selected_file_types

        if use_multi_hop:
            logger.info("Using multi-hop reasoning for complex query")
            answer, sources = self._multi_hop_reasoning(rewritten, k, metadata_filters)
            if not answer:
                # Fallback to normal retrieval
                answer, sources = self._normal_retrieval(rewritten, k, metadata_filters)
        else:
            answer, sources = self._normal_retrieval(rewritten, k, metadata_filters)

        if not answer:
            lang = self._detect_target_language(query)
            no_info = NO_INFO_MESSAGE.get(lang, NO_INFO_MESSAGE["en"])
            return no_info, [], 0.0, "Độ tin cậy rất thấp", "", rewritten

        answer = self._enforce_answer_language(answer, query)

        # Step 4: Self-evaluation
        context = "\n".join([s.content[:500] for s in sources[:3]])
        eval_score, eval_justification = self._self_evaluate(query, answer, context)

        # Step 5: Confidence scoring
        self._mark_used_chunks(answer, sources)
        confidence, confidence_level = self._compute_confidence(sources, eval_score)

        logger.info(f"Self-RAG complete: confidence={confidence}%, eval={eval_score}/5")

        return answer, sources, confidence, confidence_level, eval_justification, rewritten

    # ── RAG Strategy Delegation ────────────────────────────────────

    @staticmethod
    def _get_strategy(rag_type: str) -> AbstractRAGStrategy:
        """Return the appropriate RAG strategy instance.

        Args:
            rag_type: One of RAG_TYPE_STANDARD or RAG_TYPE_CORAG

        Returns:
            An instance of AbstractRAGStrategy
        """
        if rag_type == RAG_TYPE_CORAG:
            return ChainOfRAGStrategy()
        return StandardRAGStrategy()

    def process_query_with_strategy(
        self,
        query: str,
        k: int = 3,
        status_container=None,
    ) -> Tuple[Generator[str, None, None], List[Document], Dict[str, Any]]:
        """
        Process a query using the selected RAG strategy.

        Reads ``rag_type`` and ``compare_rag`` from session state. When
        comparison mode is enabled, both Standard and Chain-of-RAG are
        executed sequentially and the comparison results are stored in
        ``st.session_state.rag_comparison_result``.

        Args:
            query: User's question
            k: Number of documents to retrieve per step
            status_container: Optional ``st.status`` container

        Returns:
            Tuple of (stream_generator, source_documents, metrics)
        """
        rag_type = st.session_state.get("rag_type", RAG_TYPE_STANDARD)
        compare = bool(st.session_state.get("compare_rag", False))
        retrieval_k = int(st.session_state.get("retrieval_k", k))
        use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
        use_rerank = bool(st.session_state.get("use_rerank", False))

        selected_sources = self._get_source_filters()
        selected_file_types = st.session_state.get("active_file_type_filters", [])
        metadata_filters: Dict[str, Any] = {}
        if selected_sources:
            metadata_filters["source_files"] = selected_sources
        if selected_file_types:
            metadata_filters["file_types"] = selected_file_types

        conversation_context = self._conversation_context(max_turns=4)

        # ── Rewrite query for better retrieval ──────────────────
        rewritten_query = self._rewrite_query(query)

        # ── Check vector store ──────────────────────────────────
        if self.vector_service is None or not self.vector_service.is_initialized:
            raise VectorStoreError("Please upload documents first")

        # ── Run primary strategy ────────────────────────────────
        strategy = self._get_strategy(rag_type)
        logger.info(f"Using RAG strategy: {rag_type}")

        stream_gen, docs, metrics = strategy.process_query_stream(
            query=rewritten_query,
            vector_service=self.vector_service,
            llm_service=self.llm_service,
            k=retrieval_k,
            status_container=status_container,
            metadata_filters=metadata_filters,
            conversation_context=conversation_context,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
        )
        metrics["rewritten_query"] = rewritten_query if rewritten_query != query else None

        # ── Comparison mode: run the other strategy ─────────────
        if compare:
            import time as _time
            import copy

            other_type = RAG_TYPE_STANDARD if rag_type == RAG_TYPE_CORAG else RAG_TYPE_CORAG
            other_strategy = self._get_strategy(other_type)

            logger.info(f"Comparison mode: also running {other_type}")

            # Run the other strategy (non-streaming — collect full answer)
            try:
                other_stream, other_docs, other_metrics = other_strategy.process_query_stream(
                    query=rewritten_query,
                    vector_service=self.vector_service,
                    llm_service=self.llm_service,
                    k=retrieval_k,
                    status_container=None,  # Don't clutter UI
                    metadata_filters=metadata_filters,
                    conversation_context=conversation_context,
                    use_hybrid=use_hybrid,
                    use_rerank=use_rerank,
                )

                # Collect the full answer from the other strategy
                other_answer = "".join(other_stream)
                other_metrics["answer_length"] = len(other_answer)

                # Mark used chunks for comparison answer
                self._mark_used_chunks(answer=other_answer, sources=other_docs)

                st.session_state.rag_comparison_result = {
                    "primary_type": rag_type,
                    "primary_metrics": metrics,
                    "primary_answer_pending": True,
                    "other_type": other_type,
                    "other_metrics": other_metrics,
                    "other_answer": other_answer,
                    "other_docs": other_docs,
                }

            except Exception as e:
                logger.warning(f"Comparison strategy ({other_type}) failed: {e}")
                st.session_state.rag_comparison_result = None

        return stream_gen, docs, metrics

    def _normal_retrieval(self, query: str, k: int = 3) -> Tuple[str, List[Document]]:
        """Standard single-hop retrieval and generation."""
        use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
        use_rerank = bool(st.session_state.get("use_rerank", False))
        retrieval_k = int(st.session_state.get("retrieval_k", k))

        selected_sources = self._get_source_filters()
        selected_file_types = st.session_state.get("active_file_type_filters", [])
        metadata_filters: Dict[str, Any] = {}
        if selected_sources:
            metadata_filters["source_files"] = selected_sources
        if selected_file_types:
            metadata_filters["file_types"] = selected_file_types

        if hasattr(self.vector_service, "search"):
            relevant_docs, retrieval_stats = self.vector_service.search(
                query=query,
                k=retrieval_k,
                metadata_filters=metadata_filters,
                use_hybrid=use_hybrid,
                rerank=use_rerank,
                fetch_k=max(retrieval_k * 4, 20),
            )

            st.session_state.last_retrieval_stats = retrieval_stats
            comparison = self._build_retrieval_comparison(
                query=query,
                k=retrieval_k,
                metadata_filters=metadata_filters or {},
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                retrieved_docs=relevant_docs,
                retrieval_stats=retrieval_stats,
            )
            if comparison:
                st.session_state.retrieval_comparison = comparison
            else:
                st.session_state.pop("retrieval_comparison", None)
        else:
            relevant_docs = self.vector_service.similarity_search(query, k=retrieval_k)
            st.session_state.last_retrieval_stats = {
                "use_hybrid": False,
                "rerank": False,
                "results": len(relevant_docs),
            }
            st.session_state.pop("retrieval_comparison", None)

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

    def benchmark_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        So sánh performance giữa pure vector, pure BM25 và hybrid search.

        Benchmark chạy trên vector_service hiện tại và trả về:
        - Recall@K cho mỗi chiến lược
        - Thời gian phản hồi (ms)
        - Số document duy nhất được trả về (coverage)
        - Chiến lược tốt nhất theo từng metric

        Args:
            query: Câu hỏi dùng để test retrieval
            k: Số document lấy về (mặc định 5)

        Returns:
            Dict chứa kết quả benchmark của cả 3 chiến lược

        Raises:
            VectorStoreError: Nếu vector store chưa được khởi tạo
        """
        if self.vector_service is None or not self.vector_service.is_initialized:
            raise VectorStoreError("Vector store chua duoc khoi tao. Vui long upload tai lieu truoc.")

        logger.info(f"Running retrieval benchmark for query: {query[:50]}...")
        benchmark = RetrievalBenchmark(self.vector_service)
        results = benchmark.run(query=query, k=k)
        logger.info(
            f"Benchmark complete | best recall={results.get('best', {}).get('recall')} "
            f"best speed={results.get('best', {}).get('speed')}"
        )
        return results

