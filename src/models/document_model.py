"""Document domain models for SmartDoc AI."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class Document:
    """
    Domain entity representing a document.
    
    Attributes:
        content: Text content of the document
        metadata: Additional metadata (source, page, etc.)
        id: Unique identifier
        created_at: Creation timestamp
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validation after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Document content cannot be empty")
    
    @property
    def page_number(self) -> Optional[int]:
        """Get page number from metadata if available."""
        return self.metadata.get('page')
    
    @property
    def source_file(self) -> str:
        """Get source file from metadata."""
        return self.metadata.get('source', 'Unknown')
    
    def get_citation(self) -> str:
        """
        Format citation string.
        
        Returns:
            Formatted citation like "[filename.pdf, page 5]"
        """
        from pathlib import Path
        page = self.page_number
        source = Path(self.source_file).name if self.source_file else 'Unknown'
        
        if page:
            return f"[{source}, page {page}]"
        return f"[{source}]"
