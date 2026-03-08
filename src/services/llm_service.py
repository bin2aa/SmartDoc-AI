"""LLM service for SmartDoc AI using Ollama."""

from abc import ABC, abstractmethod
from typing import Optional
from langchain_community.llms import Ollama
from src.utils.exceptions import LLMConnectionError
from src.utils.logger import setup_logger
from src.utils.constants import *

logger = setup_logger(__name__)


class AbstractLLMService(ABC):
    """Abstract interface for LLM services."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        pass


class OllamaLLMService(AbstractLLMService):
    """
    Ollama LLM service implementation.
    
    Uses local Ollama instance with qwen2.5:7b model.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY
    ):
        """
        Initialize Ollama LLM service.
        
        Args:
            model: Model name (default: qwen2.5:7b)
            base_url: Ollama server URL
            temperature: Temperature parameter
            top_p: Top-p parameter
            repeat_penalty: Repeat penalty parameter
            
        Raises:
            LLMConnectionError: If cannot connect to Ollama
        """
        logger.info(f"Initializing Ollama with model: {model}")
        
        try:
            self.llm = Ollama(
                model=model,
                base_url=base_url,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty
            )
            logger.info("Ollama LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise LLMConnectionError(
                f"Cannot connect to Ollama. Ensure Ollama is running on {base_url}"
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
            
        Raises:
            LLMConnectionError: If generation fails
        """
        try:
            logger.info(f"Generating response for prompt length: {len(prompt)}")
            response = self.llm.invoke(prompt, **kwargs)
            logger.info(f"Generated response length: {len(response)}")
            return response
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise LLMConnectionError(f"Failed to generate response: {str(e)}")
