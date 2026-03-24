"""LLM service for SmartDoc AI using Ollama."""

from abc import ABC, abstractmethod
from typing import Optional
import subprocess
from langchain_ollama import OllamaLLM
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
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        num_ctx: int = DEFAULT_NUM_CTX,
        num_predict: int = DEFAULT_NUM_PREDICT,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ):
        """
        Initialize Ollama LLM service.
        
        Args:
            model: Model name (default: qwen2.5:7b)
            base_url: Ollama server URL
            temperature: Temperature parameter
            top_p: Top-p parameter
            repeat_penalty: Repeat penalty parameter
            num_ctx: Context window token limit
            num_predict: Maximum generated tokens
            keep_alive: Ollama keep_alive setting to control model residency
            
        Raises:
            LLMConnectionError: If cannot connect to Ollama
        """
        logger.info(f"Initializing Ollama with model: {model}")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.num_ctx = int(num_ctx)
        self.num_predict = int(num_predict)
        self.keep_alive = keep_alive
        
        try:
            self.llm = self._build_client(model=model, num_ctx=self.num_ctx)
            logger.info("Ollama LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise LLMConnectionError(
                f"Cannot connect to Ollama. Ensure Ollama is running on {base_url}"
            )

    def _build_client(self, model: str, num_ctx: int) -> OllamaLLM:
        """Build an Ollama LLM client with current runtime parameters."""
        return OllamaLLM(
            model=model,
            base_url=self.base_url,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            num_ctx=max(256, int(num_ctx)),
            num_predict=max(64, int(self.num_predict)),
            keep_alive=self.keep_alive,
        )

    @staticmethod
    def _is_memory_error(message: str) -> bool:
        """Detect common Ollama insufficient memory errors."""
        lowered = (message or "").lower()
        return (
            "requires more system memory" in lowered
            or "insufficient memory" in lowered
            or "out of memory" in lowered
        )

    @staticmethod
    def _is_model_not_found_error(message: str) -> bool:
        """Detect Ollama model-not-found errors."""
        lowered = (message or "").lower()
        return "not found" in lowered and "model" in lowered

    @staticmethod
    def _list_installed_models() -> list:
        """Read local Ollama model names from `ollama list`."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode != 0:
                return []

            models = []
            for line in result.stdout.splitlines()[1:]:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if parts:
                    models.append(parts[0])
            return models
        except Exception:
            return []

    def _candidate_fallback_models(self) -> list:
        """Return fallback candidates ordered by expected memory usage."""
        installed = self._list_installed_models()
        candidates = [
            LOW_MEMORY_FALLBACK_MODEL,
            "qwen2.5:1.5b",
            "qwen2.5:0.5b",
            self.model,
        ]

        # Keep unique order and prefer installed models only.
        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in unique_candidates:
                unique_candidates.append(candidate)

        if installed:
            return [candidate for candidate in unique_candidates if candidate in installed]
        return unique_candidates
    
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
            error_message = str(e)
            logger.error(f"Generation failed: {error_message}")

            if self._is_model_not_found_error(error_message):
                installed_models = self._list_installed_models()
                installed_text = ", ".join(installed_models) if installed_models else "(none detected)"
                raise LLMConnectionError(
                    f"Model '{self.model}' is not installed in Ollama. "
                    f"Installed models: {installed_text}. "
                    f"Run: ollama pull {self.model}"
                )

            if self._is_memory_error(error_message):
                reduced_ctx = max(256, self.num_ctx // 2)
                logger.warning(
                    "Memory pressure detected. Retrying with reduced context: "
                    f"model={self.model}, num_ctx={reduced_ctx}"
                )

                try:
                    self.llm = self._build_client(model=self.model, num_ctx=reduced_ctx)
                    self.num_ctx = reduced_ctx
                    response = self.llm.invoke(prompt, **kwargs)
                    logger.info("Generation succeeded after reducing num_ctx")
                    return response
                except Exception as reduced_error:
                    reduced_message = str(reduced_error)
                    logger.error(f"Reduced-context retry failed: {reduced_message}")

                    for fallback_model in self._candidate_fallback_models():
                        if fallback_model == self.model:
                            continue

                        logger.warning(
                            "Retrying with fallback model: "
                            f"{fallback_model}, num_ctx={reduced_ctx}"
                        )
                        try:
                            self.model = fallback_model
                            self.llm = self._build_client(model=self.model, num_ctx=reduced_ctx)
                            response = self.llm.invoke(prompt, **kwargs)
                            logger.info("Generation succeeded with fallback model")
                            return response
                        except Exception as fallback_error:
                            fallback_message = str(fallback_error)
                            logger.error(
                                f"Fallback-model retry failed ({fallback_model}): {fallback_message}"
                            )

                    installed_models = self._list_installed_models()
                    installed_text = ", ".join(installed_models) if installed_models else "(none detected)"
                    raise LLMConnectionError(
                        "Failed to generate response due to low system memory. "
                        "Try reducing num_ctx to 256, num_predict to 64, keep_alive to 0m, "
                        "or pull and use a smaller model. "
                        f"Installed models: {installed_text}. "
                        f"Original error: {reduced_message}"
                    )

            raise LLMConnectionError(f"Failed to generate response: {error_message}")
