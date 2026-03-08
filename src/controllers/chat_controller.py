"""Chat controller for handling chat operations."""

from typing import Optional, List, Tuple
import streamlit as st
from src.services.llm_service import AbstractLLMService, OllamaLLMService
from src.services.vector_store_service import AbstractVectorStoreService
from src.models.chat_model import ChatHistory
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import LLMConnectionError, VectorStoreError

logger = setup_logger(__name__)


class ChatController:
    """
    Controller for chat operations.
    
    Handles query processing, context retrieval, and response generation.
    """
    
    def __init__(
        self,
        llm_service: Optional[AbstractLLMService] = None,
        vector_service: Optional[AbstractVectorStoreService] = None
    ):
        """
        Initialize chat controller.
        
        Args:
            llm_service: LLM service instance (default: OllamaLLMService)
            vector_service: Vector store service instance
        """
        # Dependency injection with defaults
        self.llm_service = llm_service or OllamaLLMService()
        self.vector_service = vector_service
        
        logger.info("ChatController initialized")
    
    def process_query(self, query: str, k: int = 3) -> Tuple[str, List[Document]]:
        """
        Process user query and return response.
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (answer, source_documents)
            
        Raises:
            ValueError: If query is empty
            VectorStoreError: If vector store not initialized
            LLMConnectionError: If LLM generation fails
        """
        # Validation
        if not query or not query.strip():
            logger.warning("Empty query received")
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # Check if vector store is initialized
        if self.vector_service is None or not self.vector_service.is_initialized:
            logger.error("Vector store not initialized")
            raise VectorStoreError("Please upload documents first")
        
        # Retrieve relevant documents
        try:
            relevant_docs = self.vector_service.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise
        
        # If no documents found
        if not relevant_docs:
            logger.warning("No relevant documents found")
            return "I don't have enough information to answer this question.", []
        
        # Build context
        context = "\n\n".join([doc.content for doc in relevant_docs])
        logger.debug(f"Context length: {len(context)} characters")
        
        # Create prompt
        prompt = self._build_prompt(context, query)
        
        # Generate response
        try:
            response = self.llm_service.generate(prompt)
            logger.info("Response generated successfully")
            return response, relevant_docs
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise
    
    def _build_prompt(self, context: str, question: str) -> str:
        """
        Build prompt for LLM.
        
        Args:
            context: Retrieved context from documents
            question: User's question
            
        Returns:
            Formatted prompt string
        """
        return f"""You are SmartDoc AI, an intelligent document assistant.

Answer the QUESTION based ONLY on the CONTEXT below.
If the context doesn't contain enough information, say "I don't have enough information to answer this question."
Detect the question language and respond in the SAME language.
Keep your answer concise (3-4 sentences maximum).
Be factual and precise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    
    def clear_history(self) -> None:
        """Clear chat history from session state."""
        if 'chat_history' in st.session_state:
            st.session_state.chat_history.clear()
            logger.info("Chat history cleared")
