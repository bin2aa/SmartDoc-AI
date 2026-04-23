"""Persistence service for saving/loading app state to disk."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.models.chat_model import ChatHistory
from src.utils.logger import setup_logger
from src.utils.constants import DATA_DIR

logger = setup_logger(__name__)

# Persistence file paths
PERSIST_DIR = DATA_DIR / "persistence"
CHAT_HISTORY_FILE = PERSIST_DIR / "chat_history.json"
SETTINGS_FILE = PERSIST_DIR / "settings.json"
LOADED_DOCS_FILE = PERSIST_DIR / "loaded_docs.json"
FAISS_INDEX_DIR = PERSIST_DIR / "faiss_index"


def _ensure_persist_dir():
    """Ensure persistence directory exists."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# ── Chat History ──────────────────────────────────────────────

def save_chat_history(history: ChatHistory) -> bool:
    """
    Save chat history to JSON file.
    
    Args:
        history: ChatHistory object
        
    Returns:
        True if save successful
    """
    try:
        _ensure_persist_dir()
        data = history.to_dict()
        msg_count = len(data.get("messages", []))
        
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Verify the file was actually written
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                verify_data = json.load(f)
            verify_count = len(verify_data.get("messages", []))
            if verify_count == msg_count:
                logger.info(f"Saved & verified chat history ({msg_count} messages) to {CHAT_HISTORY_FILE}")
                return True
            else:
                logger.error(f"Verification FAILED: wrote {msg_count} messages but read back {verify_count}")
                return False
        else:
            logger.error(f"File {CHAT_HISTORY_FILE} does not exist after write!")
            return False
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def load_chat_history() -> Optional[ChatHistory]:
    """
    Load chat history from JSON file.
    
    Returns:
        ChatHistory object, or None if file not found
    """
    try:
        if not CHAT_HISTORY_FILE.exists():
            logger.info("No saved chat history found")
            return None

        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        history = ChatHistory.from_dict(data)
        logger.info(f"Loaded chat history ({len(history)} messages) from {CHAT_HISTORY_FILE}")
        return history
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
        return None


# ── Settings ───────────────────────────────────────────────────

def save_settings(settings: Dict[str, Any]) -> bool:
    """
    Save settings to JSON file.
    
    Args:
        settings: Dictionary of settings values
    """
    try:
        _ensure_persist_dir()
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
        logger.info(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return False


def load_settings() -> Dict[str, Any]:
    """
    Load settings from JSON file.
    
    Returns:
        Dictionary of settings, or empty dict if not found
    """
    try:
        if not SETTINGS_FILE.exists():
            return {}

        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Settings loaded from {SETTINGS_FILE}")
        return data
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        return {}


# ── Loaded Documents ───────────────────────────────────────────

def save_loaded_docs(docs: List[Dict[str, Any]]) -> bool:
    """Save loaded document metadata."""
    try:
        _ensure_persist_dir()
        with open(LOADED_DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        logger.info(f"Loaded docs saved ({len(docs)} items)")
        return True
    except Exception as e:
        logger.error(f"Failed to save loaded docs: {e}")
        return False


def load_loaded_docs() -> List[Dict[str, Any]]:
    """Load saved document metadata."""
    try:
        if not LOADED_DOCS_FILE.exists():
            return []
        with open(LOADED_DOCS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load loaded docs: {e}")
        return []


# ── FAISS Index ────────────────────────────────────────────────

def save_faiss_index(vector_service) -> bool:
    """
    Save FAISS index to disk.
    
    Args:
        vector_service: FAISSVectorStoreService instance
    """
    try:
        if vector_service is None or vector_service.vector_store is None:
            return False

        _ensure_persist_dir()
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vector_service.vector_store.save_local(str(FAISS_INDEX_DIR))
        logger.info(f"FAISS index saved to {FAISS_INDEX_DIR}")
        return True
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
        return False


def load_faiss_index(vector_service) -> bool:
    """
    Load FAISS index from disk into vector_service.
    
    Args:
        vector_service: FAISSVectorStoreService instance
        
    Returns:
        True if index loaded successfully
    """
    try:
        if not FAISS_INDEX_DIR.exists():
            logger.info("No saved FAISS index found")
            return False

        if vector_service.embeddings is None:
            logger.warning("Embeddings not initialized, cannot load FAISS index")
            return False

        from langchain_community.vectorstores import FAISS
        vector_service.vector_store = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            vector_service.embeddings,
            allow_dangerous_deserialization=True,
        )
        # Rebuild BM25 from loaded docs
        vector_service._rebuild_bm25_retriever()
        logger.info(f"FAISS index loaded from {FAISS_INDEX_DIR}")
        return True
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return False


# ── Full State Save/Load ──────────────────────────────────────

def save_all_state(
    chat_history: ChatHistory,
    settings: Dict[str, Any],
    loaded_docs: List[Dict[str, Any]],
    vector_service=None,
) -> bool:
    """Save all application state."""
    results = [
        save_chat_history(chat_history),
        save_settings(settings),
        save_loaded_docs(loaded_docs),
    ]
    if vector_service is not None:
        results.append(save_faiss_index(vector_service))
    return all(results)


def clear_all_state() -> bool:
    """Clear all persisted state (for reset)."""
    try:
        for f in [CHAT_HISTORY_FILE, SETTINGS_FILE, LOADED_DOCS_FILE]:
            if f.exists():
                f.unlink()
        if FAISS_INDEX_DIR.exists():
            import shutil
            shutil.rmtree(FAISS_INDEX_DIR)
        logger.info("All persisted state cleared")
        return True
    except Exception as e:
        logger.error(f"Failed to clear state: {e}")
        return False
