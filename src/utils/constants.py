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
LOW_MEMORY_FALLBACK_MODEL = "qwen2.5:1.5b"
AVAILABLE_MODELS = ["qwen2.5:1.5b", "llama3.2:3b", "qwen2.5:7b"]
DEFAULT_NUM_CTX = 1024
DEFAULT_NUM_PREDICT = 256
DEFAULT_KEEP_ALIVE = "0m"
DEFAULT_LLM_TIMEOUT = 120  # seconds
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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

# Streamlit chat reply templates (Google Sheet style: intro/body/footer)
DEFAULT_STREAMLIT_REPLY_TEMPLATES = {
	"found": {
		"intro": "Chào bạn, mình đã tìm được câu trả lời:",
		"body": "",
		"footer": "Nếu cần mình có thể giải thích thêm bằng tiếng Việt."
	},
	"not_found": {
		"intro": "Chào bạn, mình chưa tìm được câu trả lời:",
		"body": "Vui lòng hỏi rõ hơn hoặc upload thêm tài liệu liên quan trong tab Documents.",
		"footer": "Bạn hãy nói rõ hơn được không?"
	}
}

NO_INFO_MARKERS = [
	"i don't have enough information to answer this question.",
	"i dont have enough information to answer this question.",
	"không có đủ thông tin",
	"không đủ thông tin",
	"không tìm thấy thông tin"
]

# File Upload Configuration
ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt']
MAX_FILE_SIZE_MB = 10

# UI Configuration
PAGE_TITLE = "SmartDoc AI"
PAGE_ICON = "📚"

