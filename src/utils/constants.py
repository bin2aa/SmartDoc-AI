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
AVAILABLE_MODELS = ["qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:3b", "qwen2.5:7b"]
DEFAULT_NUM_CTX = 4096
DEFAULT_NUM_PREDICT = 512
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

# Streamlit chat reply templates (language-aware)
# Each template has variants for different query languages.
DEFAULT_STREAMLIT_REPLY_TEMPLATES = {
	"found": {
		"intro": "",  # Empty — let LLM answer speak for itself
		"body": "",
		"footer": ""
	},
	"not_found": {
		"intro": "",
		"body": "",
		"footer": ""
	}
}

# Legacy Vietnamese templates (kept for backward compatibility)
LEGACY_VI_REPLY_TEMPLATES = {
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

# Language-aware intro/footer templates keyed by detected language
LANG_REPLY_TEMPLATES = {
	"vi": {
		"found": {
			"intro": "Chào bạn, mình đã tìm được câu trả lời:",
			"footer": "Nếu cần mình có thể giải thích thêm."
		},
		"not_found": {
			"intro": "Chào bạn, mình chưa tìm được câu trả lời:",
			"body": "Vui lòng hỏi rõ hơn hoặc upload thêm tài liệu liên quan.",
			"footer": ""
		},
	},
	"en": {
		"found": {
			"intro": "Here's what I found:",
			"footer": "Let me know if you need more details."
		},
		"not_found": {
			"intro": "I couldn't find an answer:",
			"body": "Please try rephrasing your question or upload more documents.",
			"footer": ""
		},
	},
	"zh": {
		"found": {
			"intro": "找到了以下答案：",
			"footer": ""
		},
		"not_found": {
			"intro": "未找到答案：",
			"body": "请尝试更清楚地表述您的问题。",
			"footer": ""
		},
	},
}

NO_INFO_MARKERS = [
	"i don't have enough information to answer this question.",
	"i dont have enough information to answer this question.",
	"không có đủ thông tin",
	"không đủ thông tin",
	"không tìm thấy thông tin",
	"the provided documents do not contain enough information",
	"tài liệu không chứa đủ thông tin",
	"not enough information",
]

# Language-aware "no info" fallback messages
NO_INFO_MESSAGE = {
	"vi": "Tài liệu không chứa đủ thông tin để trả lời câu hỏi này.",
	"en": "The provided documents do not contain enough information to fully answer this question.",
	"zh": "提供的文档没有足够的信息来完全回答这个问题。",
}

# Language-aware "no info" instruction for LLM prompts
NO_INFO_INSTRUCTION = {
	"vi": 'Nếu ngữ cảnh không chứa đủ thông tin, hãy nói: "Tài liệu không chứa đủ thông tin để trả lời câu hỏi này."',
	"en": 'If the context doesn\'t contain enough information, say "I don\'t have enough information to answer this question."',
	"zh": '如果上下文没有包含足够的信息，请说："提供的文档没有足够的信息来完全回答这个问题。"',
}

# File Upload Configuration
ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt']
MAX_FILE_SIZE_MB = 10

# RAG Pipeline Types
RAG_TYPE_STANDARD = "standard"
RAG_TYPE_CORAG = "chain_of_rag"
AVAILABLE_RAG_TYPES = [RAG_TYPE_STANDARD, RAG_TYPE_CORAG]
DEFAULT_RAG_TYPE = RAG_TYPE_STANDARD

# Chain-of-RAG Configuration
CORAG_MAX_CHAIN_STEPS = 3
CORAG_MIN_QUERY_WORDS = 6  # Below this, skip decomposition
CORAG_ACCUMULATED_CONTEXT_MAX = 2500  # Max chars for accumulated context per step (v3: was 1500)
CORAG_SYNTHESIS_CONTEXT_MAX = 6000  # Max chars for final synthesis context (v3: was 4000)
CORAG_DOC_PREVIEW_CHARS = 300  # Chars to include per doc in accumulated context
CORAG_SUB_QUESTION_MIN_CHARS = 8  # Min chars for a valid sub-question
CORAG_CHAIN_RETRIEVAL_K = 2  # Docs per chain step (lower = less noise)
CORAG_REFINE_DRIFT_THRESHOLD = 0.65  # Min word overlap; below this → revert refine

# Chain-of-RAG v2: Sequential Chain & Fallback
CORAG_LOW_RELEVANCE_THRESHOLD = 0.05  # Overlap below this = low relevance doc
CORAG_LOW_RELEVANCE_FALLBACK_RATIO = 1.0  # If 100% steps low-relevance → fallback
CORAG_STEPBACK_ENABLED = True  # Enable step-back prompting before decomposition
CORAG_SEQUENTIAL_CHAIN_ENABLED = True  # Enable sequential (dependent) sub-questions
CORAG_ENTITY_DISCOVERY_K = 5  # More docs for entity discovery step 1

# Chain-of-RAG v3: Semantic Relevance & Pipeline Optimization
CORAG_SEMANTIC_WEIGHT = 0.7  # Weight for cosine similarity in relevance blend
CORAG_WORD_OVERLAP_WEIGHT = 0.3  # Weight for word overlap in relevance blend
CORAG_COSINE_LOW_RELEVANCE = 0.15  # Cosine similarity below this = low relevance
CORAG_EARLY_EXIT_ENTITY_COVERAGE = 1.0  # Exit when this fraction of entities found
CORAG_EARLY_EXIT_CONTEXT_RATIO = 0.6  # Exit when context fills this ratio of budget

# Hallucination Guard v2: Confidence-based (not binary)
CORAG_HALLUCINATION_GUARD_MIN_USEFUL = 50  # Chars of useful content to not strip (relaxed)
CORAG_HALLUCINATION_GUARD_SHORT_LIMIT = 100  # Below this length = true refusal (relaxed)

# UI Configuration
PAGE_TITLE = "SmartDoc AI"
PAGE_ICON = "📚"

