"""n8n webhook integration service."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib import request, error

from src.models.document_model import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class N8NWebhookService:
    """Service for sending chat events from Streamlit to n8n webhook."""

    def __init__(self, webhook_url: str, enabled: bool = False, timeout_seconds: int = 5):
        self.webhook_url = webhook_url
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds

    def send_chat_event(
        self,
        question: str,
        formatted_answer: str,
        raw_answer: str,
        sources: Optional[List[Document]] = None,
    ) -> bool:
        """
        Send a chat event payload to n8n webhook.

        Returns:
            True when request succeeds with 2xx response, otherwise False.
        """
        if not self.enabled:
            logger.debug("n8n webhook is disabled")
            return False

        if not self.webhook_url:
            logger.warning("n8n webhook is enabled but URL is empty")
            return False

        payload = self._build_payload(question, formatted_answer, raw_answer, sources or [])

        try:
            req = request.Request(
                url=self.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                status_code = getattr(response, "status", 0)
                if 200 <= status_code < 300:
                    logger.info("Sent chat event to n8n successfully")
                    return True

                logger.warning(f"n8n webhook responded with non-success status: {status_code}")
                return False

        except error.URLError as exc:
            logger.warning(f"Could not send event to n8n webhook: {exc}")
            return False
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"Unexpected error when sending event to n8n: {exc}")
            return False

    def _build_payload(
        self,
        question: str,
        formatted_answer: str,
        raw_answer: str,
        sources: List[Document],
    ) -> Dict[str, Any]:
        """Build payload shape to be consumed by n8n workflow."""
        source_items = []
        for source in sources:
            source_items.append(
                {
                    "citation": source.get_citation(),
                    "content": source.content,
                    "metadata": source.metadata,
                }
            )

        return {
            "event": "streamlit_chat",
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "formatted_answer": formatted_answer,
            "raw_answer": raw_answer,
            "has_sources": len(source_items) > 0,
            "sources": source_items,
        }
