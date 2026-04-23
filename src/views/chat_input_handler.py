"""Chat input handling and query processing for SmartDoc AI."""

import streamlit as st
from typing import Optional

from src.controllers.chat_controller import ChatController
from src.views.components import UIComponents
from src.views.source_renderer import convert_sources_to_details, render_source_details
from src.views.rag_comparison import render_rag_comparison
from src.models.chat_model import ChatHistory
from src.services.persistence_service import save_chat_history
from src.utils.constants import RAG_TYPE_STANDARD
from src.utils.exceptions import LLMConnectionError
from src.utils.helpers import parse_file_mentions
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def render_chat_input(controller: ChatController, components: UIComponents):
    """Render chat input box with streaming response and step-by-step status.

    Unified flow for ALL RAG strategies (Standard, Chain-of-RAG):
    1. Call ``process_query_with_strategy`` → streaming + sources + metrics
    2. Stream LLM response
    3. Post-processing (mark used chunks, optional Self-RAG evaluation)
    4. Convert sources to serializable details and display
    5. Save to chat history and disk

    Messages are ONLY added to chat_history AFTER the full response is
    complete. Sources are stored as serializable dicts (not Document
    objects) to avoid serialization issues.

    Args:
        controller: ChatController instance for query processing
        components: UIComponents instance for UI rendering
    """
    if prompt := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
        logger.info(f"Chat input received: '{prompt[:50]}...'")
        st.session_state.is_processing_query = True
        st.session_state.skip_retrieval_metrics_once = True

        # Clear previous comparison data before processing new query
        st.session_state.pop("comparison_display_data", None)
        st.session_state.pop("rag_comparison_result", None)

        # ── Parse @filename mentions for metadata filtering ──
        clean_prompt, mentioned_files = parse_file_mentions(prompt)
        _prev_source_filters = st.session_state.get("active_source_filters", [])
        _mention_filters_active = False

        if mentioned_files:
            logger.info(f"@mention Focus Mode activated: {mentioned_files}")
            # Use override key — cannot modify active_source_filters
            # after the multiselect widget with that key is instantiated.
            st.session_state._source_filter_override = mentioned_files
            _mention_filters_active = True

        # Display user message bubble directly (not from history)
        with st.chat_message("user", avatar="user"):
            st.markdown(prompt)
            if mentioned_files:
                st.caption(f"🎯 Focus Mode (@mention): {', '.join(mentioned_files)}")
            elif _prev_source_filters:
                st.caption(f"🎯 Filter: {', '.join(_prev_source_filters)}")

        try:
            rag_type = st.session_state.get("rag_type", RAG_TYPE_STANDARD)
            compare = bool(st.session_state.get("compare_rag", False))
            use_self_rag = bool(st.session_state.get("use_self_rag", False))

            # ── Step 1: Call RAG strategy (Standard or Chain-of-RAG) ──
            rag_label = "Standard RAG" if rag_type == RAG_TYPE_STANDARD else "Chain-of-RAG"
            with st.status(f"Đang xử lý câu hỏi ({rag_label})...", expanded=True) as status:
                stream_gen, sources, metrics = controller.process_query_with_strategy(
                    clean_prompt,
                    status_container=status,
                )
                status.update(
                    label=f"Đã tìm thấy tài liệu liên quan! ({rag_label})",
                    state="complete",
                    expanded=False,
                )

            # ── Step 2: Stream LLM response ──────────────────────────
            with st.chat_message("assistant", avatar="assistant"):
                response_text = st.write_stream(stream_gen)

            # ── Step 3: Post-processing ──────────────────────────────
            # Enforce language: if LLM responded in Chinese despite prompt
            # instructions, rewrite it in the user's language.
            response_text = controller._enforce_answer_language(response_text, prompt)

            controller._mark_used_chunks(answer=response_text, sources=sources)

            # ── Step 4: Self-RAG post-evaluation (optional) ──────────
            confidence_score = None
            confidence_level = None
            self_eval_justification = None

            if use_self_rag and sources:
                with st.status("Đang đánh giá Self-RAG...", expanded=True) as eval_status:
                    eval_status.write("[eval] Self-evaluation + confidence scoring")
                    context = "\n".join([s.content[:500] for s in sources[:3]])
                    eval_score, self_eval_justification = controller._self_evaluate(
                        prompt, response_text, context,
                    )
                    # Pass entities from dispatcher for entity coverage boost
                    query_entities = metrics.get("entities", [])
                    confidence_score, confidence_level = controller._compute_confidence(
                        sources, eval_score, response_text, entities=query_entities,
                    )
                    eval_status.update(
                        label="Đánh giá hoàn tất!",
                        state="complete",
                        expanded=False,
                    )

                if confidence_score is not None and confidence_level:
                    st.caption(f"Độ tin cậy Self-RAG: {confidence_score}% ({confidence_level})")
                if self_eval_justification:
                    st.caption(f"Tự đánh giá: {self_eval_justification}")

            # Get rewritten query from strategy metrics
            rewritten_query = metrics.get("rewritten_query")

            # ── Step 5: Format and convert sources ───────────────────
            formatted_answer = controller.format_reply_for_streamlit(response_text, sources, query=prompt)

            source_details = convert_sources_to_details(sources)

            # Show sources immediately after answer
            if source_details:
                render_source_details(source_details, msg_idx=-1, rewritten_query=rewritten_query)

            # ── Step 6: Save to chat history ─────────────────────────
            history = st.session_state.get("chat_history")
            if history is None:
                logger.error("chat_history is None before adding messages — creating new one")
                history = ChatHistory()
                st.session_state.chat_history = history

            history.add_message("user", prompt)
            assistant_metadata = {
                "source_details": source_details,
                "used_self_rag": use_self_rag,
                "rewritten_query": rewritten_query,
                "rag_type": rag_type,
                "metrics": {
                    "strategy": metrics.get("strategy"),
                    "retrieval_steps": metrics.get("retrieval_steps"),
                    "total_docs_retrieved": metrics.get("total_docs_retrieved"),
                    "total_time_ms": metrics.get("total_time_ms"),
                },
            }
            if use_self_rag:
                assistant_metadata["confidence_score"] = confidence_score
                assistant_metadata["confidence_level"] = confidence_level
                assistant_metadata["self_eval_justification"] = self_eval_justification

            history.add_message("assistant", formatted_answer, metadata=assistant_metadata)
            logger.info(f"Added user + assistant messages to history (total: {len(history)})")

            # ── Step 7: Persist to disk ──────────────────────────────
            save_ok = save_chat_history(history)
            if save_ok:
                logger.info("Chat history saved to disk successfully")
            else:
                logger.error("FAILED to save chat history to disk!")

            # ── Step 8: Show RAG comparison if enabled ───────────────
            if compare:
                render_rag_comparison(
                    controller, response_text,
                    primary_source_details=source_details,
                )

        except LLMConnectionError as e:
            logger.error(f"LLM error while processing query: {e}")
            _save_user_message(prompt)
            components.error_alert(
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
            _save_user_message(prompt)
            components.error_alert(
                "Không thể xử lý câu hỏi của bạn",
                details=str(e),
            )
        finally:
            st.session_state.is_processing_query = False
            # Clean up the override so the next rerun reads from the
            # multiselect widget state directly.
            if _mention_filters_active:
                st.session_state.pop("_source_filter_override", None)
                logger.info("Cleaned up source filter override after @mention processing")
            if st.session_state.get("skip_retrieval_metrics_once", False):
                st.session_state.skip_retrieval_metrics_once = False
                st.rerun()


def stream_text(text: str):
    """Yield text chunks so non-streaming answers can use stream-like UI.

    Args:
        text: Full text to stream word by word

    Yields:
        Individual word tokens with spaces
    """
    if not text:
        return
    for token in text.split(" "):
        yield token + " "


def _save_user_message(prompt: str):
    """Save user message to history even when processing fails.

    Args:
        prompt: The user's input text
    """
    try:
        history = st.session_state.get("chat_history")
        if history is None:
            history = ChatHistory()
            st.session_state.chat_history = history
        history.add_message("user", prompt)
        save_chat_history(history)
        logger.info(f"Saved user message to history (total: {len(history)})")
    except Exception as e:
        logger.error(f"Failed to save user message: {e}")