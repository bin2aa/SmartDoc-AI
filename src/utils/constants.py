"""Application-wide constants for SmartDoc AI."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# LLM Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# RAG Parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_RETRIEVAL_K = 3
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_MAX_TOKENS = 512

# Chat Configuration
MAX_CHAT_HISTORY = 50

# File Upload Configuration
ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt']
MAX_FILE_SIZE_MB = 10

# UI Configuration
PAGE_TITLE = "SmartDoc AI"
PAGE_ICON = "📚"
