"""Chat history rendering for SmartDoc AI."""

import streamlit as st
from typing import Optional

from src.views.components import UIComponents
from src.views.source_renderer import render_source_citations, render_source_details
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def render_chat_history(history, components: UIComponents):
    """Display all chat history messages.

    Args:
        history: ChatHistory object from session state
        components: UIComponents instance for rendering
    """
    logger.debug(
        f"render_chat_history: history={type(history).__name__}, "
        f"len={len(history) if history else 'None'}"
    )

    if not history or len(history) == 0:
        components.info_alert("Bắt đầu cuộc trò chuyện bằng cách đặt câu hỏi bên dưới")
        return

    # Display ALL messages from history
    for msg_idx, message in enumerate(history.messages):
        avatar = "user" if message.role == "user" else "assistant"
        components.chat_message(
            role=message.role,
            content=message.content,
            avatar=avatar,
        )

        if message.role == "assistant" and message.metadata and message.metadata.get("used_self_rag"):
            render_self_rag_metadata(message.metadata)

        if message.role == "assistant" and message.metadata:
            rewritten = message.metadata.get("rewritten_query")
            if message.metadata.get("source_details"):
                render_source_details(
                    message.metadata["source_details"], msg_idx, rewritten_query=rewritten
                )
            elif message.metadata.get("source_citations"):
                render_source_citations(message.metadata["source_citations"], msg_idx)


def render_self_rag_metadata(metadata: dict):
    """Render confidence and self-evaluation details for Self-RAG responses.

    Args:
        metadata: Metadata dictionary containing self-rag evaluation info
    """
    confidence_score = metadata.get("confidence_score")
    confidence_level = metadata.get("confidence_level")
    self_eval = metadata.get("self_eval_justification")

    if confidence_score is not None and confidence_level:
        st.caption(f"Độ tin cậy Self-RAG: {confidence_score}% ({confidence_level})")
    if self_eval:
        st.caption(f"Tự đánh giá: {self_eval}")
