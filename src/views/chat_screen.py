"""Chat screen for SmartDoc AI."""

import streamlit as st
from typing import List, Optional
from src.controllers.chat_controller import ChatController
from src.views.components import UIComponents, icon
from src.models.document_model import Document
from src.models.chat_model import ChatHistory
from src.utils.logger import setup_logger
from src.utils.exceptions import LLMConnectionError
from src.services.persistence_service import save_chat_history, load_chat_history

logger = setup_logger(__name__)


class ChatScreen:
    """
    Chat interface screen following MVC pattern.

    Displays chat history and handles user interactions.
    """

    def __init__(self, controller: ChatController):
        """
        Initialize chat screen.

        Args:
            controller: Chat controller instance
        """
        self.controller = controller
        self.components = UIComponents()

    def render(self):
        """Render the chat screen."""
        st.markdown(f"## {icon('chat')} Chat với tài liệu", unsafe_allow_html=True)

        # Check if vector store is ready
        if not st.session_state.get('vector_store_initialized', False):
            self._render_empty_state()
            return

        # Ensure chat_history exists (recovery mechanism)
        self._ensure_history()

        # Render chat interface
        self._render_chat_history()
        self._render_chat_input()

        self._render_retrieval_metrics()

    def _ensure_history(self):
        """Ensure chat_history exists in session_state, recover from disk if needed."""
        history = st.session_state.get('chat_history')
        if history is not None and len(history) > 0:
            logger.debug(f"Chat history OK: {len(history)} messages")
            return

        # History is empty or missing — try to recover from disk
        if history is None:
            logger.warning("chat_history is None in session_state — recovering from disk")
            saved = load_chat_history()
            if saved and len(saved) > 0:
                st.session_state.chat_history = saved
                logger.info(f"Recovered chat history from disk ({len(saved)} messages)")
            else:
                st.session_state.chat_history = ChatHistory()
                logger.info("Created new empty ChatHistory")
        elif len(history) == 0:
            # Empty history in memory — check disk as well
            saved = load_chat_history()
            if saved and len(saved) > 0:
                logger.warning(f"Memory history is empty but disk has {len(saved)} messages — recovering")
                st.session_state.chat_history = saved

    def _render_empty_state(self):
        """Show empty state when no documents loaded."""
        self.components.info_alert("Vui lòng tải tài liệu lên trước trong tab Documents")

        st.markdown(f"""
        ### {icon('waving_hand')} Chào mừng đến SmartDoc AI!

        Để bắt đầu:
        1. Nhấn **Documents** ở sidebar bên trái
        2. Tải lên file PDF, DOCX, hoặc TXT
        3. Quay lại đây để đặt câu hỏi về tài liệu

        SmartDoc AI sử dụng RAG (Retrieval-Augmented Generation) để trả lời câu hỏi dựa trên tài liệu của bạn.
        """, unsafe_allow_html=True)

    def _render_chat_history(self):
        """Display all chat history messages."""
        history = st.session_state.get('chat_history')

        logger.debug(f"_render_chat_history: history={type(history).__name__}, len={len(history) if history else 'None'}")

        if not history or len(history) == 0:
            self.components.info_alert("Bắt đầu cuộc trò chuyện bằng cách đặt câu hỏi bên dưới")
            return

        # Display ALL messages from history
        for msg_idx, message in enumerate(history.messages):
            avatar = "user" if message.role == "user" else "assistant"
            self.components.chat_message(
                role=message.role,
                content=message.content,
                avatar=avatar
            )

            if message.role == "assistant" and message.metadata and message.metadata.get('used_self_rag'):
                self._render_self_rag_metadata(message.metadata)

            if message.role == "assistant" and message.metadata:
                rewritten = message.metadata.get('rewritten_query')
                if message.metadata.get('source_details'):
                    self._render_source_details(message.metadata['source_details'], msg_idx, rewritten_query=rewritten)
                elif message.metadata.get('source_citations'):
                    self._render_source_citations(message.metadata['source_citations'], msg_idx)

    @staticmethod
    def _render_self_rag_metadata(metadata: dict):
        """Render confidence and self-evaluation details for Self-RAG responses."""
        confidence_score = metadata.get("confidence_score")
        confidence_level = metadata.get("confidence_level")
        self_eval = metadata.get("self_eval_justification")

        if confidence_score is not None and confidence_level:
            st.caption(f"Do tin cay Self-RAG: {confidence_score}% ({confidence_level})")
        if self_eval:
            st.caption(f"Tu danh gia: {self_eval}")

    def _render_source_citations(self, citations: List[str], msg_idx: int):
        """
        Render source citations from serialized citation strings.

        Args:
            citations: List of citation strings
            msg_idx: Message index for unique key generation
        """
        with st.expander("Xem nguồn tham khảo"):
            for src_idx, citation in enumerate(citations, 1):
                st.markdown(f"**Nguồn {src_idx}:** {citation}")

    def _render_source_details(self, source_details: List[dict], msg_idx: int, rewritten_query: Optional[str] = None):
        """
        Render source citations with detailed info from dictionary.

        Args:
            source_details: List of source detail dictionaries
            msg_idx: Message index for unique key generation
            rewritten_query: Optional rewritten query to display
        """
        with st.expander("Xem chi tiết nguồn tham khảo"):
            if rewritten_query:
                st.info(f"**Câu hỏi đã được tối ưu:** {rewritten_query}")
                st.divider()

            for src_idx, src in enumerate(source_details, 1):
                citation = src.get("citation", "[Unknown source]")
                source_file = src.get("source_file")
                content = src.get("content", "")
                is_used = src.get("used_in_answer", False)
                overlap = src.get("used_term_overlap", 0)

                st.markdown(f"**Nguồn {src_idx}:** {citation}")

                if source_file:
                    open_link = f"data/uploads/{source_file}"
                    st.markdown(f"[Mở file nguồn]({open_link})")

                if is_used:
                    st.caption(f"Được sử dụng trong câu trả lời (term overlap: {overlap})")
                    # Highlight preview
                    preview = content[:300] + "..." if len(content) > 300 else content
                    st.markdown(f"<mark style='background-color: #ffff0033;'>{preview}</mark>", unsafe_allow_html=True)
                else:
                    st.caption("Ngữ cảnh đã truy xuất (không được dùng trực tiếp)")

                st.text_area(
                    f"Nội dung đầy đủ (Nguồn {src_idx})",
                    value=content,
                    height=150,
                    key=f"source_msg{msg_idx}_src{src_idx}",
                    disabled=True
                )

    def _render_sources(self, sources: List[Document], msg_idx: int):
        """
        Render source citations with rich document metadata.

        Hiển thị thông tin chi tiết về document nguồn:
        - Tên file + icon theo loại file
        - Số trang
        - Chunk index
        - Kích thước file
        - Ngày upload
        - Đoạn preview có highlight từ khóa

        Args:
            sources: List of source documents
            msg_idx: Message index for unique key generation
        """
        with st.expander("📚 Nguồn tài liệu"):
            for src_idx, source in enumerate(sources, 1):
                file_type = source.metadata.get("file_type", "").upper().replace(".", "")
                source_file = source.metadata.get("source_file") or source.source_file
                page = source.metadata.get("page")
                chunk_idx = source.metadata.get("chunk_index", 0)
                file_size_mb = source.metadata.get("file_size_mb", 0)
                uploaded_at = source.metadata.get("uploaded_at", "")
                title = source.metadata.get("title", source_file)
                is_used = bool(source.metadata.get("used_in_answer", False))
                rerank_score = source.metadata.get("rerank_score")

                # Icon and color by file type
                icon_map = {".PDF": "📕", ".DOCX": "📘", ".TXT": "📄"}
                color_map = {".PDF": "#e53935", ".DOCX": "#1565c0", ".TXT": "#558b2f"}
                icon = icon_map.get(file_type, "📄")
                color = color_map.get(file_type, "#9e9e9e")

                # Page info
                page_info = f", trang {page}" if page is not None else ""
                chunk_info = f" (chunk {chunk_idx + 1})"

                # Rerank badge
                rerank_badge = ""
                if rerank_score is not None:
                    rerank_badge = f" | 🎯 relevance = {rerank_score:.3f}"

                st.markdown(f"""
                <div style="
                    border-left: 4px solid {color};
                    padding: 8px 12px;
                    margin-bottom: 8px;
                    border-radius: 4px;
                    background: {'#fff8e1' if is_used else '#f5f5f5'};
                ">
                    <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                        <span style="font-size:1.1em;">{icon}</span>
                        <strong>{title}</strong>
                        <span style="font-size:0.8em; color:#666;">{file_type}</span>
                        {"<span style='background:#4caf50;color:white;font-size:0.7em;padding:1px 5px;border-radius:3px;margin-left:4px;'>✅ Dùng trong câu trả lời</span>" if is_used else ""}
                    </div>
                    <div style="font-size:0.8em; color:#555;">
                        <span>📑 Nguồn: `{source_file}`{page_info}{chunk_info}{rerank_badge}</span>
                    </div>
                    <div style="font-size:0.78em; color:#888; margin-top:2px;">
                        💾 {file_size_mb:.2f} MB | ⏱ {uploaded_at[:10]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                preview = source.content[:300].replace("\n", " ") + "..."
                if is_used:
                    preview = f"<mark>{preview}</mark>"

                st.text_area(
                    f"Noi dung {src_idx}",
                    value=source.content,
                    height=80,
                    key=f"source_msg{msg_idx}_src{src_idx}",
                    disabled=True,
                )
                st.markdown(preview, unsafe_allow_html=True)

    def _render_chat_input(self):
        """Render chat input box with streaming response and step-by-step status.

        Key design: Messages are ONLY added to chat_history AFTER the full
        response is complete. Sources are stored as citation strings (not
        Document objects) to avoid serialization issues.
        """
        if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
            logger.info(f"Chat input received: '{prompt[:50]}...'")
            use_self_rag = bool(st.session_state.get("use_self_rag", False))
            retrieval_k = int(st.session_state.get("retrieval_k", 3))
            st.session_state.is_processing_query = True
            # Avoid rendering retrieval metrics in the same render cycle as loading/status.
            st.session_state.skip_retrieval_metrics_once = True

            # Display user message bubble directly (not from history)
            with st.chat_message("user", avatar="user"):
                st.markdown(prompt)

            # Process query with streaming
            try:
                confidence_score = None
                confidence_level = None
                self_eval_justification = None

                if use_self_rag:
                    with st.status("Đang xử lý câu hỏi...", expanded=True) as status:
                        status.write(f"[search] **Phân tích câu hỏi:** `{prompt}`")
                        status.write("[rewrite] Query rewriting + Self-RAG")
                        status.write("[retrieve] Retrieval / multi-hop reasoning")
                        response_text, sources, confidence_score, confidence_level, self_eval_justification, rewritten_query = (
                            self.controller.process_query_with_self_rag(prompt, k=retrieval_k)
                        )
                        status.write("[eval] Self-evaluation + confidence scoring")
                        status.update(
                            label="Đã tìm thấy tài liệu liên quan!",
                            state="complete",
                            expanded=False,
                        )

                    with st.chat_message("assistant", avatar="assistant"):
                        response_text = st.write_stream(self._stream_text(response_text))

                    if confidence_score is not None and confidence_level:
                        st.caption(f"Do tin cay Self-RAG: {confidence_score}% ({confidence_level})")
                    if self_eval_justification:
                        st.caption(f"Tu danh gia: {self_eval_justification}")
                else:
                    # Step 1: Retrieval with status indicator
                    with st.status("Đang xử lý câu hỏi...", expanded=True) as status:
                        stream_gen, sources, rewritten_query = self.controller.process_query_stream(
                            prompt,
                            status_container=status,
                        )
                        # Mark retrieval complete
                        status.update(
                            label="Đã tìm thấy tài liệu liên quan!",
                            state="complete",
                            expanded=False,
                        )

                    # Step 2: Stream LLM response into a dedicated chat message bubble
                    with st.chat_message("assistant", avatar="assistant"):
                        response_text = st.write_stream(stream_gen)

                    # Step 3: Post-processing
                    self.controller._mark_used_chunks(answer=response_text, sources=sources)

                formatted_answer = self.controller.format_reply_for_streamlit(response_text, sources)

                # Step 4: Convert sources to serializable detailed info
                source_details = []
                if sources:
                    for src in sources:
                        try:
                            source_details.append({
                                "content": src.content,
                                "citation": src.get_citation(),
                                "source_file": src.metadata.get("source_file") or src.source_file,
                                "page": src.page_number,
                                "used_in_answer": src.metadata.get("used_in_answer", False),
                                "used_term_overlap": src.metadata.get("used_term_overlap", 0)
                            })
                        except Exception as e:
                            logger.warning(f"Failed to get details for source: {e}")

                # Show sources immediately after answer for both modes
                if source_details:
                    self._render_source_details(source_details, msg_idx=-1, rewritten_query=rewritten_query)

                # Step 5: NOW add both messages to history (after display is complete).
                history = st.session_state.get('chat_history')
                if history is None:
                    logger.error("chat_history is None before adding messages — creating new one")
                    history = ChatHistory()
                    st.session_state.chat_history = history

                history.add_message("user", prompt)
                assistant_metadata = {
                    'source_details': source_details,
                    'used_self_rag': use_self_rag,
                    'rewritten_query': rewritten_query if rewritten_query != prompt else None
                }
                if use_self_rag:
                    assistant_metadata['confidence_score'] = confidence_score
                    assistant_metadata['confidence_level'] = confidence_level
                    assistant_metadata['self_eval_justification'] = self_eval_justification

                history.add_message("assistant", formatted_answer, metadata=assistant_metadata)
                logger.info(f"Added user + assistant messages to history (total: {len(history)})")

                # Step 6: Persist chat history to disk with verification
                save_ok = save_chat_history(history)
                if save_ok:
                    logger.info("Chat history saved to disk successfully")
                else:
                    logger.error("FAILED to save chat history to disk!")

            except LLMConnectionError as e:
                logger.error(f"LLM error while processing query: {e}")
                # Still save the user message to history even if LLM fails
                self._save_user_message(prompt)
                self.components.error_alert(
                    "Không thể kết nối đến Ollama",
                    details=(
                        f"{str(e)}\n\n"
                        "Kiểm tra:\n"
                        "1) Đảm bảo Ollama đang chạy: ollama serve\n"
                        "2) Kiểm tra model đã tải: ollama list\n"
                        "3) Tải model nếu chưa có: ollama pull <model_name>\n"
                        "4) Vào Settings để đổi model hoặc giảm num_ctx nếu cần"
                    ),
                )
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Still save the user message to history even if processing fails
                self._save_user_message(prompt)
                self.components.error_alert(
                    "Không thể xử lý câu hỏi của bạn",
                    details=str(e)
                )
            finally:
                st.session_state.is_processing_query = False
                if st.session_state.get("skip_retrieval_metrics_once", False):
                    st.session_state.skip_retrieval_metrics_once = False
                    st.rerun()

    @staticmethod
    def _stream_text(text: str):
        """Yield text chunks so non-streaming answers can use stream-like UI."""
        if not text:
            return
        for token in text.split(" "):
            yield token + " "

    def _save_user_message(self, prompt: str):
        """Save user message to history even when processing fails."""
        try:
            history = st.session_state.get('chat_history')
            if history is None:
                history = ChatHistory()
                st.session_state.chat_history = history
            history.add_message("user", prompt)
            save_chat_history(history)
            logger.info(f"Saved user message to history (total: {len(history)})")
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")

    def _render_retrieval_metrics(self):
        """Show retrieval strategy metrics for hybrid/pure-vector comparison."""
        if st.session_state.get("is_processing_query", False):
            return

        stats = st.session_state.get("last_retrieval_stats", {})
        # Do not show retrieval metrics when there is no chat history
        history = st.session_state.get('chat_history')
        if not history or (hasattr(history, 'messages') and len(history.messages) == 0) or (isinstance(history, list) and len(history) == 0):
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

    def _render_rerank_benchmark(self):
        """Run multi-query benchmark for bi-encoder vs cross-encoder reranking."""
        with st.expander("Re-ranking Benchmark"):
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
                try:
                    result = self.controller.benchmark_rerank_queries(queries=queries, k=benchmark_k)
                    st.session_state.rerank_benchmark_result = result
                    logger.info("Ran rerank benchmark with %s queries", len(queries))
                except Exception as benchmark_error:
                    self.components.error_alert(
                        "Không thể chạy benchmark re-ranking",
                        details=str(benchmark_error),
                    )

            result = st.session_state.get("rerank_benchmark_result")
            if result:
                st.markdown("**Benchmark Summary**")
                st.write(result.get("summary", {}))
                rows = result.get("rows", [])
                if rows:
                    st.dataframe(rows, use_container_width=True)
