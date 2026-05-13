"""
FastAPI server for SmartDocAI
Provides REST API endpoints for document processing and querying.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import asyncio

# Import your services
from src.controllers.chat_controller import ChatController
from src.controllers.document_controller import DocumentController
from src.services.vector_store_service import FAISSVectorStoreService
from src.services.document_service import DocumentService
from src.services.llm_service import OllamaLLMService
from src.utils.logger import setup_logger
from src.utils.constants import UPLOAD_DIR
from src.utils.exceptions import (
    LLMConnectionError,
    VectorStoreError,
    DocumentLoadError
)

# Import API models
from src.api.models import (
    QueryRequest,
    QueryResponse,
    BatchQueryRequest,
    SourceDocument,
    DocumentInfo,
    DocumentsListResponse,
    StatusResponse,
    UploadResponse,
    ClearResponse,
    ErrorResponse
)

logger = setup_logger(__name__)

# ============ Initialize FastAPI app ============

app = FastAPI(
    title="SmartDocAI API",
    description="REST API - Document Q&A System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Enable CORS for external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Global Services (Singleton) ============

try:
    vector_service = FAISSVectorStoreService()
    logger.info("Vector Store Service initialized")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_service = None

try:
    document_service = DocumentService()
    logger.info("Document Service initialized")
except Exception as e:
    logger.error(f"Failed to initialize document service: {e}")
    document_service = None

try:
    llm_service = OllamaLLMService()
    logger.info("LLM Service initialized")
except Exception as e:
    logger.error(f"Failed to initialize LLM service: {e}")
    llm_service = None

try:
    chat_controller = ChatController(llm_service, vector_service)
    document_controller = DocumentController()
    logger.info("Controllers initialized")
except Exception as e:
    logger.error(f"Failed to initialize controllers: {e}")
    chat_controller = None
    document_controller = None

# Track loaded documents
loaded_documents: Dict[str, DocumentInfo] = {}
server_start_time = datetime.now()

# ============ Root Endpoints ============

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - Health check."""
    return {
        "service": "SmartDocAI",
        "version": "1.0.0",
        "status": "running",
        "description": "Intelligent Document Q&A System with LLM"
    }


@app.get("/api/status", response_model=StatusResponse, tags=["Status"])
async def get_status() -> StatusResponse:
    """
    Get system status and service information.
    
    Returns:
        StatusResponse with system health indicators
    """
    logger.info("Status check requested")
    
    try:
        uptime = (datetime.now() - server_start_time).total_seconds()
        
        is_ready = False
        if vector_service and vector_service.vector_store is not None:
            is_ready = True
        
        ollama_status = "connected"
        if llm_service is None:
            ollama_status = "disconnected"
        
        return StatusResponse(
            is_vector_store_ready=is_ready,
            documents_loaded=len(loaded_documents),
            ollama_status=ollama_status,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Document Management Endpoints ============

@app.post("/api/document/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload and process a document.
    
    Supported formats: PDF, DOCX, TXT
    
    Args:
        file: Document file to upload
        
    Returns:
        UploadResponse with processing details
        
    Raises:
        HTTPException: If file type not supported or processing fails
    """
    logger.info(f"Upload requested for file: {file.filename}")
    
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is empty")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.pdf', '.docx', '.txt']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: .pdf, .docx, .txt"
            )
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"File saved to {file_path} ({len(content)} bytes)")
        
        # Load and process document
        if not document_service:
            raise HTTPException(status_code=500, detail="Document service not initialized")
        
        documents = document_service.load_document(file_path)
        if not documents:
            raise HTTPException(status_code=400, detail="No content extracted from document")
        
        logger.info(f"Created {len(documents)} chunks")
        
        # Add to vector store (documents already chunked from load_document)
        if not vector_service:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        vector_service.add_documents(documents)
        logger.info("Documents added to vector store")
        
        # Track document
        doc_id = f"doc_{len(loaded_documents)}"
        loaded_documents[doc_id] = DocumentInfo(
            id=doc_id,
            filename=file.filename,
            chunks_count=len(documents),
            upload_date=datetime.now().isoformat()
        )
        
        logger.info(f"Document processing completed: {doc_id}")
        
        return UploadResponse(
            status="success",
            doc_id=doc_id,
            filename=file.filename,
            chunks_created=len(documents),
            total_characters=sum(len(chunk.content) for chunk in documents)
        )
        
    except HTTPException:
        raise
    except DocumentLoadError as e:
        logger.error(f"Document load error: {e}")
        raise HTTPException(status_code=400, detail=f"Document load failed: {str(e)}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/documents", response_model=DocumentsListResponse, tags=["Documents"])
async def list_documents() -> DocumentsListResponse:
    """
    List all loaded documents.
    
    Returns:
        DocumentsListResponse with list of loaded documents
    """
    logger.info("Document list requested")
    
    return DocumentsListResponse(
        total_documents=len(loaded_documents),
        documents=list(loaded_documents.values())
    )


@app.delete("/api/clear", response_model=ClearResponse, tags=["Documents"])
async def clear_storage() -> ClearResponse:
    """
    Clear vector store and all loaded documents.
    
    Returns:
        ClearResponse with confirmation
    """
    logger.warning("Clear storage requested")
    
    try:
        if vector_service:
            vector_service.clear_store()
        
        loaded_documents.clear()
        
        logger.info("Vector store and documents cleared")
        
        return ClearResponse(
            status="success",
            message="Vector store and all documents cleared successfully"
        )
    except Exception as e:
        logger.error(f"Clear storage failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


# ============ Query Endpoints ============

@app.post("/api/query", response_model=QueryResponse, tags=["Queries"])
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query loaded documents and get AI-generated response.
    
    Args:
        request: QueryRequest with question and parameters
        
    Returns:
        QueryResponse with answer and source documents
        
    Raises:
        HTTPException: If query processing fails
    """
    logger.info(f"Query received: {request.query[:50]}...")
    
    try:
        # Check if vector store is ready
        if not vector_service or vector_service.vector_store is None:
            raise HTTPException(
                status_code=400,
                detail="No documents loaded. Please upload documents first."
            )
        
        confidence = 0.85
        confidence_level = None
        self_evaluation = None

        # Process query
        if request.use_self_rag:
            answer, source_docs, confidence_score, confidence_level, self_evaluation = (
                chat_controller.process_query_with_self_rag(
                    request.query,
                    k=request.k,
                )
            )
            confidence = max(0.0, min(1.0, round(confidence_score / 100.0, 3)))
        else:
            answer, source_docs = chat_controller.process_query(
                request.query,
                k=request.k
            )
        
        # Format sources
        sources = [
            SourceDocument(
                content=doc.content[:200] if len(doc.content) > 200 else doc.content,
                metadata=doc.metadata
            )
            for doc in source_docs
        ]
        
        logger.info("Query processed successfully")
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            confidence_level=confidence_level,
            self_evaluation=self_evaluation,
        )
        
    except HTTPException:
        raise
    except VectorStoreError as e:
        logger.error(f"Vector store error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except LLMConnectionError as e:
        logger.error(f"LLM connection error: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/batch-query", response_model=List[QueryResponse], tags=["Queries"])
async def batch_query(request: BatchQueryRequest) -> List[QueryResponse]:
    """
    Process multiple queries in batch.
    
    Args:
        request: BatchQueryRequest with list of questions
        
    Returns:
        List of QueryResponse objects
    """
    logger.info(f"Batch query received with {len(request.queries)} questions")
    
    results = []
    errors = []
    
    for idx, query in enumerate(request.queries, 1):
        try:
            query_request = QueryRequest(
                query=query,
                k=request.k,
                use_self_rag=request.use_self_rag,
            )
            result = await query_documents(query_request)
            results.append(result)
            logger.info(f"  Query {idx}/{len(request.queries)} processed")
        except HTTPException as e:
            logger.error(f"  Query {idx}/{len(request.queries)} failed: {e.detail}")
            errors.append({"query": query, "error": str(e.detail)})
            continue
        except Exception as e:
            logger.error(f"  Query {idx}/{len(request.queries)} failed: {e}")
            errors.append({"query": query, "error": str(e)})
            continue
    
    if errors:
        logger.warning(f"Batch completed with {len(errors)} errors")
    else:
        logger.info(f"All {len(results)} queries processed successfully")
    
    return results


# ============ Error Handlers ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            status_code=exc.status_code,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            timestamp=datetime.now().isoformat(),
            details=str(exc)
        ).dict()
    )


# ============ Startup/Shutdown ============

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 50)
    logger.info("SmartDocAI API Server starting...")
    logger.info("=" * 50)
    
    logger.info(f"Vector Store Status: {'Ready' if vector_service and vector_service.vector_store else 'Not initialized'}")
    logger.info(f"LLM Service Status: {'Ready' if llm_service else 'Not connected'}")
    logger.info(f"Document Service Status: {'Ready' if document_service else 'Not initialized'}")
    logger.info(f"Upload Directory: {UPLOAD_DIR}")
    logger.info(f"API Docs: http://localhost:8001/api/docs")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("=" * 50)
    logger.info("SmartDocAI API Server shutting down...")
    logger.info("=" * 50)


# ============ Run the Server ============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
