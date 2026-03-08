"""Configuration models for SmartDoc AI."""

from dataclasses import dataclass
from src.utils.constants import *


@dataclass
class RAGConfig:
    """
    Configuration for RAG pipeline.
    
    Attributes:
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between consecutive chunks
        retrieval_k: Number of documents to retrieve
        temperature: LLM temperature parameter
        top_p: LLM top_p parameter
        repeat_penalty: LLM repeat penalty parameter
        max_tokens: Maximum tokens in LLM response
        llm_model: LLM model name
        embedding_model: Embedding model name
    """
    # Text Splitting
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    
    # Retrieval
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    
    # LLM Parameters
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY
    max_tokens: int = DEFAULT_MAX_TOKENS
    
    # Models
    llm_model: str = DEFAULT_MODEL
    embedding_model: str = EMBEDDING_MODEL


@dataclass
class AppConfig:
    """
    Application-wide configuration.
    
    Attributes:
        rag: RAG pipeline configuration
        max_chat_history: Maximum chat history size
        allowed_extensions: Allowed file extensions
        max_file_size_mb: Maximum file size in MB
    """
    rag: RAGConfig = None
    max_chat_history: int = MAX_CHAT_HISTORY
    allowed_extensions: list = None
    max_file_size_mb: int = MAX_FILE_SIZE_MB
    
    def __post_init__(self):
        """Initialize default values."""
        if self.rag is None:
            self.rag = RAGConfig()
        if self.allowed_extensions is None:
            self.allowed_extensions = ALLOWED_EXTENSIONS
