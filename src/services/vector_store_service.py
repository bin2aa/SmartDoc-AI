"""Vector store service for SmartDoc AI using FAISS."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import os
import re
import hashlib
from pathlib import Path
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import VectorStoreError
from src.utils.constants import EMBEDDING_MODEL

logger = setup_logger(__name__)

# Configure model cache directory and ensure it is writable.
def _resolve_cache_dir() -> Path:
    """Return a writable cache directory for HuggingFace models."""
    project_cache = Path(__file__).resolve().parents[2] / "models_cache"

    try:
        project_cache.mkdir(parents=True, exist_ok=True)
        test_file = project_cache / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return project_cache
    except Exception as permission_error:
        fallback = Path.home() / ".cache" / "smartdocai" / "models_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Project cache directory is not writable (%s). Falling back to %s",
            permission_error,
            fallback,
        )
        return fallback


_MODEL_CACHE_DIR = _resolve_cache_dir()
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_MODEL_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_MODEL_CACHE_DIR))


class AbstractVectorStoreService(ABC):
    """Abstract interface for vector store operations."""
    
    @abstractmethod
    def add_documents(self, documents: List[LCDocument]) -> None:
        """Add documents to vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> List[LCDocument]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def clear_store(self) -> None:
        """Clear the vector store."""
        pass


class LocalHashEmbeddings:
    """Deterministic local embeddings fallback that works fully offline."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, byteorder="big") % self.dimension
            vec[bucket] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class FAISSVectorStoreService(AbstractVectorStoreService):
    """
    FAISS vector store implementation.
    
    Uses HuggingFace embeddings with multilingual support.
    """
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model: HuggingFace embedding model name
            
        Raises:
            VectorStoreError: If embedding model cannot be loaded
        """
        logger.info(f"Initializing FAISS with embedding model: {embedding_model}")
        self.embedding_model = embedding_model
        self.embeddings: Optional[Any] = None
        self.vector_store: Optional[FAISS] = None
        self._init_error: Optional[str] = None
        self.using_offline_fallback: bool = False
        
        try:
            self.embeddings = self._create_embeddings(local_files_only=False)
            logger.info("FAISS vector store initialized successfully")
            
        except Exception as online_error:
            logger.warning(
                "Online embedding initialization failed for '%s': %s",
                embedding_model,
                online_error,
            )

            if "client has been closed" in str(online_error).lower():
                self._reset_hf_client()
                try:
                    self.embeddings = self._create_embeddings(local_files_only=False)
                    self.vector_store = None
                    logger.info("FAISS vector store initialized after resetting HuggingFace client")
                    return
                except Exception as retry_error:
                    logger.warning("Embedding init retry failed: %s", retry_error)

            try:
                self.embeddings = self._create_embeddings(local_files_only=True)
                logger.info("FAISS vector store initialized from local cache only")
                return
            except Exception as local_error:
                local_error_msg = str(local_error)
                logger.error(
                    "Failed to initialize FAISS vector store (online and local): %s | %s",
                    online_error,
                    local_error,
                )
                self._init_error = (
                    f"Failed to initialize embedding model '{embedding_model}'.\n"
                    f"Online init error: {online_error}\n"
                    f"Local-cache init error: {local_error_msg}\n"
                    f"Model cache path: {_MODEL_CACHE_DIR}\n"
                    "Please ensure internet is available for first download, or pre-download the model into the local cache."
                )
                logger.error(self._init_error)
                self.embeddings = LocalHashEmbeddings(dimension=768)
                self.using_offline_fallback = True
                logger.warning(
                    "Using LocalHashEmbeddings fallback (offline mode). Retrieval quality may be lower until HuggingFace model is available."
                )

    def _create_embeddings(self, local_files_only: bool) -> HuggingFaceEmbeddings:
        """Create embedding client with optional offline-only loading."""
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            cache_folder=str(_MODEL_CACHE_DIR),
            model_kwargs={
                'device': 'cpu',
                'local_files_only': local_files_only,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32,
            },
        )

    @staticmethod
    def _reset_hf_client() -> None:
        """Reset HuggingFace HTTP sessions when a stale closed client is detected."""
        try:
            from huggingface_hub.utils._http import reset_sessions

            reset_sessions()
            logger.info("Reset HuggingFace Hub HTTP sessions")
        except Exception as reset_error:
            logger.debug("Unable to reset HuggingFace Hub sessions: %s", reset_error)
    
    def add_documents(self, documents: List[LCDocument]) -> None:
        """
        Add documents to vector store.
        
        Args:
            documents: List of Document objects to add
            
        Raises:
            VectorStoreError: If adding documents fails
        """
        if not documents:
            logger.warning("No documents to add")
            return

        if self.embeddings is None:
            error_message = self._init_error or "Embedding model is not initialized"
            raise VectorStoreError(error_message)
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Convert to LangChain document format
            lc_docs = [
                LCDocument(page_content=doc.content, metadata=doc.metadata)
                for doc in documents
            ]
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(lc_docs, self.embeddings)
                logger.info("Created new FAISS index")
            else:
                self.vector_store.add_documents(lc_docs)
                logger.info("Added documents to existing FAISS index")
                
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[LCDocument]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar Document objects
        """
        if self.vector_store is None:
            if self.embeddings is None:
                error_message = self._init_error or "Embedding model is not initialized"
                raise VectorStoreError(error_message)
            logger.warning("Vector store not initialized, returning empty results")
            return []
        
        try:
            logger.info(f"Searching for {k} similar documents")
            results = self.vector_store.similarity_search(query, k=k)
            
            # Convert back to domain Document
            docs = [
                Document(content=doc.page_content, metadata=doc.metadata)
                for doc in results
            ]
            
            logger.info(f"Found {len(docs)} similar documents")
            return docs
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def clear_store(self) -> None:
        """Clear the vector store."""
        logger.warning("Clearing vector store")
        self.vector_store = None
        logger.info("Vector store cleared")
    
    @property
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self.vector_store is not None
