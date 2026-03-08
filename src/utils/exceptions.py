"""Custom exceptions for SmartDoc AI application."""


class SmartDocError(Exception):
    """Base exception for SmartDoc application."""
    pass


class DocumentLoadError(SmartDocError):
    """Raised when document loading fails."""
    pass


class VectorStoreError(SmartDocError):
    """Raised when vector store operations fail."""
    pass


class LLMConnectionError(SmartDocError):
    """Raised when cannot connect to Ollama LLM."""
    pass


class ValidationError(SmartDocError):
    """Raised when input validation fails."""
    pass
