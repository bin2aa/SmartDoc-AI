"""Vector store service for SmartDoc AI using FAISS."""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import VectorStoreError
from src.utils.constants import EMBEDDING_MODEL

logger = setup_logger(__name__)


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
        """
        logger.info(f"Initializing FAISS with embedding model: {embedding_model}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store: Optional[FAISS] = None
        logger.info("FAISS vector store initialized")
    
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
