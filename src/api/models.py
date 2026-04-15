"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for document query."""
    query: str = Field(..., description="User's question")
    k: int = Field(3, description="Number of documents to retrieve", ge=1, le=10)
    temperature: float = Field(0.7, description="LLM temperature", ge=0.0, le=1.0)
    use_self_rag: bool = Field(False, description="Enable Self-RAG pipeline")
    
    class Config:
        example = {
            "query": "What is the main topic?",
            "k": 3,
            "temperature": 0.7,
            "use_self_rag": False,
        }


class BatchQueryRequest(BaseModel):
    """Request model for batch queries."""
    queries: List[str] = Field(..., description="List of questions")
    k: int = Field(3, description="Number of documents to retrieve", ge=1, le=10)
    use_self_rag: bool = Field(False, description="Enable Self-RAG pipeline for all queries")
    
    class Config:
        example = {
            "queries": ["What is AI?", "How does it work?"],
            "k": 3,
            "use_self_rag": False,
        }


class SourceDocument(BaseModel):
    """Source document in response."""
    content: str = Field(..., description="Document content snippet")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for query results."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(..., description="Source documents used")
    timestamp: str = Field(..., description="Response timestamp")
    confidence: float = Field(0.8, description="Confidence score", ge=0.0, le=1.0)
    confidence_level: Optional[str] = Field(None, description="Confidence level label")
    self_evaluation: Optional[str] = Field(None, description="Self-evaluation justification")
    
    class Config:
        example = {
            "query": "What is AI?",
            "answer": "AI is artificial intelligence...",
            "sources": [],
            "timestamp": "2024-01-15T10:30:00",
            "confidence": 0.85,
            "confidence_level": "Moderate confidence",
            "self_evaluation": "Answer is grounded in context.",
        }


class DocumentInfo(BaseModel):
    """Document metadata."""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    chunks_count: int = Field(..., description="Number of chunks created")
    upload_date: str = Field(..., description="Upload timestamp")


class DocumentsListResponse(BaseModel):
    """Response for listing documents."""
    total_documents: int = Field(..., description="Total documents loaded")
    documents: List[DocumentInfo] = Field(..., description="Document list")


class StatusResponse(BaseModel):
    """System status response."""
    is_vector_store_ready: bool = Field(..., description="Vector store initialized")
    documents_loaded: int = Field(..., description="Number of loaded documents")
    ollama_status: str = Field(..., description="Ollama connection status")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    
    class Config:
        example = {
            "is_vector_store_ready": True,
            "documents_loaded": 3,
            "ollama_status": "connected",
            "uptime_seconds": 3600
        }


class UploadResponse(BaseModel):
    """Document upload response."""
    status: str = Field(..., description="Upload status")
    doc_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Uploaded filename")
    chunks_created: int = Field(..., description="Number of chunks created")
    total_characters: int = Field(..., description="Total characters processed")
    
    class Config:
        example = {
            "status": "success",
            "doc_id": "doc_0",
            "filename": "report.pdf",
            "chunks_created": 45,
            "total_characters": 25000
        }


class ClearResponse(BaseModel):
    """Clear storage response."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[str] = Field(None, description="Additional details")
