"""Chat controller for handling chat operations."""

from typing import Any, Dict, Optional, List, Tuple
import streamlit as st
from src.services.llm_service import AbstractLLMService, OllamaLLMService
from src.services.vector_store_service import AbstractVectorStoreService, RetrievalBenchmark
from src.services.n8n_service import N8NWebhookService
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
        n8n_service: Optional[N8NWebhookService] = None,
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
        self.n8n_service = n8n_service
        
        logger.info("ChatController initialized")
    
    def process_query(self, query: str, k: int = 3) -> Tuple[str, List[Document]]:
        """
        XỬ LÝ CÂU HỎI NGƯỜI DÙNG - ENTRY POINT CHÍNH CỦA CHAT.

        ĐÂY LÀ HÀM ĐƯỢC GỌI KHI NGƯỜI DÙNG NHẬP CÂU HỎI TRONG CHAT.

        QUY TRÌNH 6 BƯỚC:

          BƯỚC 1: VALIDATE (Kiểm tra)
            - Kiểm tra câu hỏi không được rỗng

          BƯỚC 2: ĐỌC CÁC TUỲ CHỌN (TỪ SESSION STATE)
            - use_hybrid_search: có bật hybrid search không?
            - use_rerank: có bật cross-encoder rerank không?
            - retrieval_k: số tài liệu muốn lấy về
            - active_source_filters: lọc theo tên file nào?
            - active_file_type_filters: lọc theo loại file nào?

          BƯỚC 3: RETRIEVAL (TÌM TÀI LIỆU)
            - Gọi vector_service.search() để tìm tài liệu liên quan
            - Sử dụng tuỳ chọn: hybrid, rerank, metadata_filters
            - Lưu retrieval_stats để hiển thị trong UI

          BƯỚC 4: SO SÁNH HYBRID vs VECTOR (nếu hybrid bật)
            - Nếu hybrid được bật, chạy thêm vector-only để so sánh
            - Lưu kết quả so sánh vào session_state

          BƯỚC 5: BUILD CONTEXT (Xây dựng ngữ cảnh)
            - Hợp nhất nội dung các tài liệu thành một chuỗi context
            - Định dạng: [Source: filename, page X, chunk Y]\nnoi_dung

          BƯỚC 6: GENERATE RESPONSE (Tạo câu trả lời)
            - Gọi LLM để tạo câu trả lời từ context
            - Trả về câu trả lời + danh sách tài liệu nguồn

        Args:
            query: Câu hỏi người dùng (VD: "làm thế nào để cài đặt python")
            k: Số lượng tài liệu mặc định muốn lấy về

        Returns:
            Tuple: (câu trả lời từ AI, danh sách tài liệu nguồn)

        Raises:
            ValueError: Nếu câu hỏi rỗng
            VectorStoreError: Nếu chưa upload tài liệu
            LLMConnectionError: Nếu không kết nối được LLM
        """
        # === BƯỚC 1: VALIDATE ===
        if not query or not query.strip():
            logger.warning("Empty query received")
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing query: {query[:50]}...")

        # === BƯỚC 2: ĐỌC CÁC TUỲ CHỌN TỪ SESSION STATE ===
        # Session state là "biến toàn cục" trong Streamlit
        # Các toggle trong Settings -> Retrieval Strategy sẽ lưu tại đây
        use_hybrid = bool(st.session_state.get("use_hybrid_search", False))
        use_rerank = bool(st.session_state.get("use_rerank", False))
        retrieval_k = int(st.session_state.get("retrieval_k", k))

        # Lấy filter từ Document screen (multiselect)
        selected_sources = st.session_state.get("active_source_filters", [])
        selected_file_types = st.session_state.get("active_file_type_filters", [])

        # Đóng gói metadata filters
        metadata_filters: Dict[str, Any] = {}
        if selected_sources:
            metadata_filters["source_files"] = selected_sources
        if selected_file_types:
            metadata_filters["file_types"] = selected_file_types

        # Rewrite câu hỏi ngắn (follow-up) để cải thiện retrieval
        rewritten_query = self._rewrite_query(query)
        # Lấy ngữ cảnh cuộc hội thoại (4 lượt gần nhất)
        conversation_context = self._conversation_context(max_turns=4)

        # === BƯỚC 3: RETRIEVAL ===
        if self.vector_service is None or not self.vector_service.is_initialized:
            logger.error("Vector store not initialized")
            raise VectorStoreError("Please upload documents first")

        try:
            if hasattr(self.vector_service, "search"):
                # GỌI HYBRID SEARCH
                # search() nhận các tham số:
                #   - query: câu hỏi (đã được rewrite nếu là follow-up)
                #   - k: số tài liệu muốn lấy
                #   - metadata_filters: lọc theo file/loại file
                #   - use_hybrid: True = kết hợp vector + BM25
                #   - rerank: True = sử dụng cross-encoder
                #   - fetch_k: số candidate lấy trước khi lọc (mặc định 20)
                relevant_docs, retrieval_stats = self.vector_service.search(
                    query=rewritten_query,
                    k=retrieval_k,
                    metadata_filters=metadata_filters,
                    use_hybrid=use_hybrid,
                    rerank=use_rerank,
                    fetch_k=max(retrieval_k * 4, 20),
                )
            else:
                # Vector service cũ, không có search() -> fallback về similarity_search()
                relevant_docs = self.vector_service.similarity_search(rewritten_query, k=retrieval_k)
                retrieval_stats = {
                    "use_hybrid": False,
                    "rerank": False,
                    "results": len(relevant_docs),
                }

            # Lưu stats để hiển thị trong UI (Chat -> Retrieval Metrics)
            st.session_state.last_retrieval_stats = retrieval_stats

            # === BƯỚC 4: SO SÁNH HYBRID vs PURE VECTOR ===
            # Nếu hybrid được bật, chạy thêm vector-only để so sánh
            # Kết quả so sánh sẽ hiển thị trong UI
            if use_hybrid and hasattr(self.vector_service, "search"):
                # Chạy lại retrieval nhưng không sử dụng hybrid
                vector_only_docs, vector_stats = self.vector_service.search(
                    query=rewritten_query,
                    k=retrieval_k,
                    metadata_filters=metadata_filters,
                    use_hybrid=False,    # Chỉ vector, không BM25
                    use_bm25_only=False,
                    rerank=False,
                    fetch_k=max(retrieval_k * 4, 20),
                )
                # Lưu kết quả so sánh để hiển thị trong UI
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

        # === BƯỚC 5: BUILD CONTEXT ===
        if not relevant_docs:
            logger.warning("No relevant documents found")
            return "I don't have enough information to answer this question.", []

        # Hợp nhất các tài liệu thành một chuỗi context
        # Định dạng: [Source: filename, page X, chunk Y]\nnoi_dung
        context = "\n\n".join(
            [
                f"[Source: {doc.get_citation()} | chunk={doc.metadata.get('chunk_index', 'n/a')}]\n{doc.content}"
                for doc in relevant_docs
            ]
        )
        logger.debug(f"Context length: {len(context)} characters")

        # === BƯỚC 6: GENERATE RESPONSE ===
        # Xây dựng prompt cho LLM
        prompt = self._build_prompt(context, query, conversation_context)

        try:
            # Gọi LLM (Ollama) để tạo câu trả lời
            response = self.llm_service.generate(prompt)
            # Đánh dấu những tài liệu được sử dụng trong câu trả lời (cho UI hightlight)
            self._mark_used_chunks(answer=response, sources=relevant_docs)
            logger.info("Response generated successfully")
            return response, relevant_docs
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise

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
        chat_history = st.session_state.get("chat_history")
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
    
    def clear_history(self) -> None:
        """Clear chat history from session state."""
        if 'chat_history' in st.session_state:
            st.session_state.chat_history.clear()
            logger.info("Chat history cleared")

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

    def notify_n8n_chat_event(
        self,
        question: str,
        formatted_answer: str,
        raw_answer: str,
        sources: List[Document],
    ) -> bool:
        """Send current chat interaction to n8n webhook when integration is enabled."""
        if self.n8n_service is None:
            return False

        return self.n8n_service.send_chat_event(
            question=question,
            formatted_answer=formatted_answer,
            raw_answer=raw_answer,
            sources=sources,
        )

    def benchmark_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        SO SÁNH HIỆU SUẤT GIỮA 3 CHIẾN LƯỢC RETRIEVAL.

        HÀM NÀY ĐƯỢC GỌI TỪ Settings -> Retrieval Strategy -> Run Retrieval Benchmark.

        NÓ GỌI RetrievalBenchmark.run() ĐỂ:
          1. Chạy 3chiến lược: Vector, BM25, Hybrid trên cùng câu hỏi
          2. Tính 3 metrics: Recall@K, Speed, Coverage
          3. Tìm chiến lược tốt nhất theo từng metric
          4. Trả về kết quả để hiển thị trong bảng so sánh UI

        CƠ CHẾ HOẠT ĐỘNG:
          - Lấy vector_service hiện tại (đã có tài liệu)
          - Tạo RetrievalBenchmark với vector_service đó
          - Gọi benchmark.run(query, k) để chạy so sánh
          - Tra ve ket qua

        VI DU KET QUA TRA VE:
          {
            "strategies": {
              "vector": {"recall_at_k": 0.45, "time_ms": 45, "unique_sources": 3},
              "bm25":    {"recall_at_k": 0.38, "time_ms": 12, "unique_sources": 2},
              "hybrid":  {"recall_at_k": 0.61, "time_ms": 58, "unique_sources": 4},
            },
            "best": {
              "recall": "hybrid",
              "speed": "bm25",
              "coverage": "hybrid",
            }
          }

        Args:
            query: Cau hoi dung de so sanh
            k: So document lay ve (mac dinh 5)

        Returns:
            Dict chua ket qua benchmark cua ca 3 chien luong

        Raises:
            VectorStoreError: Neu vector store chua duoc khoi tao
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
