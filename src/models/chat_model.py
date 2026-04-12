"""Chat domain models for SmartDoc AI."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional, Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ChatMessage:
    """
    Value object for chat messages.
    
    Attributes:
        role: Message role (user, assistant, system)
        content: Message content
        timestamp: When the message was created
        metadata: Optional metadata (sources, citations, etc.)
    """
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict] = None


@dataclass
class ChatHistory:
    """
    Conversation history management.
    
    Attributes:
        messages: List of chat messages
        max_history: Maximum number of messages to keep
    """
    messages: List[ChatMessage] = field(default_factory=list)
    max_history: int = 50
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a new message to history.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
        """
        self.messages.append(ChatMessage(role=role, content=content, metadata=metadata))
        
        # Trim if exceeds max
        if len(self.messages) > self.max_history:
            removed = self.messages.pop(0)
            logger.debug(f"Removed old message from history: {removed.timestamp}")
    
    def clear(self) -> None:
        """Clear all messages."""
        logger.info(f"Clearing {len(self.messages)} messages from history")
        self.messages.clear()
    
    def get_recent(self, n: int = 10) -> List[ChatMessage]:
        """
        Get n most recent messages.
        
        Args:
            n: Number of recent messages to retrieve
            
        Returns:
            List of recent ChatMessage objects
        """
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def to_dict(self) -> dict:
        """Serialize history to dict for JSON persistence."""
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": self._serialize_metadata(m.metadata),
                }
                for m in self.messages
            ],
            "max_history": self.max_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatHistory":
        """Deserialize history from dict."""
        history = cls(max_history=data.get("max_history", 50))
        for msg_data in data.get("messages", []):
            msg = ChatMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]) if "timestamp" in msg_data else datetime.now(),
                metadata=msg_data.get("metadata"),
            )
            history.messages.append(msg)
        return history

    @staticmethod
    def _serialize_metadata(metadata) -> dict:
        """Serialize metadata, handling non-serializable objects like Document sources."""
        if metadata is None:
            return None
        result = {}
        for key, value in metadata.items():
            if key == "sources":
                # Sources contain Document objects — skip them for persistence
                continue
            try:
                import json
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                result[key] = str(value)
        return result if result else None

    def __len__(self) -> int:
        """Return number of messages."""
        return len(self.messages)