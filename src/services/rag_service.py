"""
RAG strategy implementations for SmartDoc AI.

Provides the Strategy Pattern for different RAG pipeline types:
- StandardRAGStrategy: Single retrieval → generate
- ChainOfRAGStrategy: Sequential chain with step-back prompting,
  entity discovery, and low-relevance fallback.

v3 improvements:
- Confidence-based Hallucination Guard (not binary)
- Cosine Similarity + Max-segment Relevance (replaces Jaccard word overlap)
- LLM-based Query Complexity Detection (replaces simple word-count heuristic)
- Rank-based Context Reordering (mitigates "Lost in the Middle")
- Early Exit for Sequential Chain (saves latency when entities found)
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import streamlit as st

from src.models.document_model import Document
from src.services.llm_service import AbstractLLMService
from src.services.vector_store_service import AbstractVectorStoreService
from src.utils.constants import (
    CORAG_MAX_CHAIN_STEPS,
    CORAG_MIN_QUERY_WORDS,
    CORAG_ACCUMULATED_CONTEXT_MAX,
    CORAG_SYNTHESIS_CONTEXT_MAX,
    CORAG_DOC_PREVIEW_CHARS,
    CORAG_SUB_QUESTION_MIN_CHARS,
    CORAG_CHAIN_RETRIEVAL_K,
    CORAG_REFINE_DRIFT_THRESHOLD,
    CORAG_LOW_RELEVANCE_THRESHOLD,
    CORAG_LOW_RELEVANCE_FALLBACK_RATIO,
    CORAG_STEPBACK_ENABLED,
    CORAG_SEQUENTIAL_CHAIN_ENABLED,
    CORAG_ENTITY_DISCOVERY_K,
    NO_INFO_MARKERS,
    NO_INFO_MESSAGE,
    NO_INFO_INSTRUCTION,
    CORAG_SEMANTIC_WEIGHT,
    CORAG_WORD_OVERLAP_WEIGHT,
    CORAG_COSINE_LOW_RELEVANCE,
    CORAG_EARLY_EXIT_ENTITY_COVERAGE,
    CORAG_EARLY_EXIT_CONTEXT_RATIO,
    CORAG_HALLUCINATION_GUARD_MIN_USEFUL,
    CORAG_HALLUCINATION_GUARD_SHORT_LIMIT,
)
from src.utils.corag_debugger import CoRAGDebugger
from src.utils.exceptions import VectorStoreError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Shared Helpers
# ═══════════════════════════════════════════════════════════════


# Stop words to exclude from similarity calculations — these inflate
# Jaccard overlap and cause drift checks to miss semantic topic changes.
_STOP_WORDS = frozenset({
    # Articles, determiners, conjunctions, prepositions
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "into", "over", "after", "before", "up",
    "out", "between",
    # Common verbs / auxiliaries
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "need", "dare",
    # Pronouns
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their",
    # Question words (inflate Jaccard without adding topic signal)
    "what", "which", "who", "whom", "how", "when", "where", "why",
    # Negation / qualifiers
    "not", "no", "nor", "as", "if", "then", "than", "too", "very", "so",
    "just", "about", "also", "some", "any", "all", "each", "every", "both",
    "few", "more", "most", "other", "such", "only", "own", "same",
    # Question-framing verbs (high-frequency, zero topic signal)
    "list", "give", "tell", "explain", "describe", "provide", "show",
    "compare", "discuss", "mention", "include",
    # Deictic / filler
    "there", "here", "please", "thank", "thanks",
    # Vietnamese stop words
    "của", "và", "hoặc", "là", "gì", "từ", "đâu", "được", "có",
    "những", "các", "một", "sự", "cho", "với", "như", "thì", "mà",
    "này", "kia", "đó", "nào", "về", "việc", "nhận", "bài", "tập",
    "yêu", "cầu", "trong", "ngoài", "trên", "dưới", "sau", "trước",
    "để", "làm", "cần", "phải", "nên", "đã", "sẽ", "đang", "khi",
    "nếu", "vì", "nhưng", "mỗi", "tất", "cả", "vẫn", "mới", "thêm",
    "nữa", "rất", "nhất", "hơn", "cũng", "vậy", "thế", "ra", "vào",
    "đầu", "liệu", "dữ", "tập", "chức", "thực", "hiện", "phương",
    "thức", "chương", "trình", "hệ", "thống", "nguồn", "kiểu", "loại",
    "trả", "kết", "quả", "quá", "trình", "số", "tham",
})


def _extract_refusal_sentence(text: str) -> str:
    """
    Extract the refusal sentence from a response that contains a no-info marker.

    Args:
        text: Full LLM response text

    Returns:
        Clean refusal sentence string
    """
    lowered = text.lower()
    for marker in NO_INFO_MARKERS:
        idx = lowered.find(marker)
        if idx >= 0:
            start = idx
            while start > 0 and text[start - 1] not in ".!?\n":
                start -= 1
            end = idx + len(marker)
            while end < len(text) and text[end] not in ".!?\n":
                end += 1
            if end < len(text) and text[end] in ".!?":
                end += 1
            sentence = text[start:end].strip()
            if sentence:
                return sentence
            return marker.strip().capitalize()
    # Language-aware fallback (default to English)
    return NO_INFO_MESSAGE.get("en", "I don't have enough information to answer this question.")


def _has_refusal_marker(text: str) -> bool:
    """Check if text contains any refusal/no-info marker."""
    lowered = text.lower()
    return any(marker in lowered for marker in NO_INFO_MARKERS)


def _hallucination_guard_stream(gen, is_direct: bool = False) -> Generator[str, None, None]:
    """
    Confidence-based hallucination guard (v3).

    Instead of stripping ALL content when a refusal marker is found,
    this guard uses a confidence-based approach:

    - If the response is SHORT (< threshold) AND
      contains a refusal → TRUE refusal, strip everything → yield only refusal.
    - If the response is LONGER and contains a refusal BUT also has substantial
      useful content (>= threshold chars after marker)
      → WARNING only, yield full response.
    - If no refusal marker → pass through as-is.
    - For DIRECT strategy: much more lenient thresholds to avoid stripping
      real data from successful single-entity lookups.

    Args:
        gen: Raw token generator from LLM streaming
        is_direct: Whether this is a DIRECT strategy query (more lenient guard)

    Yields:
        Cleaned token strings
    """
    REFUSAL_BUFFER_LIMIT = 300
    buffer: List[str] = []
    buffer_size = 0

    for chunk in gen:
        buffer.append(chunk)
        buffer_size += len(chunk)

        if buffer_size >= REFUSAL_BUFFER_LIMIT:
            text = "".join(buffer)

            if _has_refusal_marker(text):
                # Confidence-based decision
                refusal_sentence = _extract_refusal_sentence(text)

                # Check if there's substantial useful content beyond the refusal
                refusal_idx = -1
                lowered = text.lower()
                for marker in NO_INFO_MARKERS:
                    idx = lowered.find(marker)
                    if idx >= 0:
                        refusal_idx = idx
                        break

                # Content after the refusal marker
                after_refusal = text[refusal_idx + len(refusal_sentence):].strip() if refusal_idx >= 0 else ""
                # Content before the refusal marker
                before_refusal = text[:refusal_idx].strip() if refusal_idx > 0 else ""

                total_useful = len(after_refusal) + len(before_refusal)

                # For DIRECT strategy: much more lenient thresholds
                # Only strip if response is truly empty or completely unrelated
                short_limit = CORAG_HALLUCINATION_GUARD_SHORT_LIMIT
                min_useful = CORAG_HALLUCINATION_GUARD_MIN_USEFUL
                if is_direct:
                    short_limit = max(short_limit, 200)  # Only very short = refusal
                    min_useful = 20  # Even tiny useful content passes

                if len(text) < short_limit or total_useful < min_useful:
                    # TRUE refusal — short response with no useful content
                    # Consume remaining generator and yield only the refusal
                    for _ in gen:
                        pass
                    logger.warning(
                        "Hallucination guard STRIPPED (true refusal, is_direct=%s): "
                        "len=%d, useful=%d, refusal=%s",
                        is_direct, len(text), total_useful, refusal_sentence[:80],
                    )
                    yield refusal_sentence
                    return
                else:
                    # Has useful content — just warn but pass through
                    logger.warning(
                        "Hallucination guard WARNING (has useful content): "
                        "len=%d, useful=%d chars — keeping full response",
                        len(text), total_useful,
                    )
                    # Fall through to yield buffered content
            else:
                pass  # No refusal marker — fall through

            # No stripping needed — yield buffered content
            for buffered_chunk in buffer:
                yield buffered_chunk
            buffer = []
            break

    if buffer:
        text = "".join(buffer)

        if _has_refusal_marker(text):
            refusal_sentence = _extract_refusal_sentence(text)

            short_limit = CORAG_HALLUCINATION_GUARD_SHORT_LIMIT if not is_direct else 200
            if len(text) < short_limit:
                # Short true refusal
                logger.warning(
                    "Hallucination guard STRIPPED (short response, is_direct=%s): len=%d",
                    is_direct, len(text),
                )
                yield refusal_sentence
                return
            else:
                # Longer response with refusal but potentially useful content
                logger.warning(
                    "Hallucination guard WARNING (long response with marker): "
                    "len=%d — keeping full response",
                    len(text),
                )
        # Yield buffered content
        for buffered_chunk in buffer:
            yield buffered_chunk

    for chunk in gen:
        yield chunk


def _word_overlap(a: str, b: str) -> float:
    """
    Content-word Jaccard similarity between two strings.

    Stop words are filtered out so that topic drift is detected even when
    the refined question reuses common function words from the original.
    """
    words_a = set(a.lower().split()) - _STOP_WORDS
    words_b = set(b.lower().split()) - _STOP_WORDS
    if not words_a or not words_b:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two numpy vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _semantic_relevance(
    query: str,
    doc_content: str,
    vector_service: Optional[AbstractVectorStoreService] = None,
) -> float:
    """
    Semantic relevance between a query and a document chunk.

    Uses Max-segment Similarity: splits the document into sentences,
    computes cosine similarity for each sentence against the query,
    and takes the maximum. This handles long chunks where only one
    sentence may be relevant.

    Falls back to word overlap if embeddings are unavailable.

    Args:
        query: Search query string
        doc_content: Document chunk content
        vector_service: Vector store service (for access to embeddings)

    Returns:
        Blended relevance score in [0, 1]
    """
    word_overlap_score = _word_overlap(query, doc_content)

    # Try cosine similarity if embeddings are available
    if vector_service is not None and hasattr(vector_service, 'embeddings'):
        embeddings = getattr(vector_service, 'embeddings', None)
        if embeddings is not None and not getattr(vector_service, 'using_offline_fallback', False):
            try:
                # Split doc into sentences for max-segment similarity
                sentences = [s.strip() for s in re.split(r'[.!?\n]', doc_content) if len(s.strip()) > 10]

                if not sentences:
                    # Use full content as single segment
                    sentences = [doc_content[:500]]

                # Embed query
                query_vec = np.array(embeddings.embed_query(query))

                # Compute similarity for each sentence, take max
                max_sim = 0.0
                # Batch embed sentences (limit to top 5 longest to save time)
                top_sentences = sorted(sentences, key=len, reverse=True)[:5]
                for sent in top_sentences:
                    sent_vec = np.array(embeddings.embed_query(sent))
                    sim = _cosine_similarity(query_vec, sent_vec)
                    if sim > max_sim:
                        max_sim = sim

                # Blend: semantic weight + word overlap weight
                blended = (
                    CORAG_SEMANTIC_WEIGHT * max_sim
                    + CORAG_WORD_OVERLAP_WEIGHT * word_overlap_score
                )
                logger.debug(
                    "Semantic relevance: cosine=%.3f, word_overlap=%.3f, blended=%.3f",
                    max_sim, word_overlap_score, blended,
                )
                return blended

            except Exception as e:
                logger.warning(
                    "Cosine similarity failed, falling back to word overlap: %s", e
                )

    # Fallback to word overlap only
    return word_overlap_score


def _detect_lang(query: str) -> str:
    """Detect language code from query text (vi/en/zh)."""
    if re.search(r"[\u4e00-\u9fff]", query):
        return "zh"
    if any(c in query.lower() for c in ["đ", "ă", "â", "ê", "ô", "ơ", "ư", "á", "à", "ả", "ã", "ạ"]):
        return "vi"
    return "en"


def _lang_rule(query: str) -> str:
    """Return language enforcement instruction for LLM prompts."""
    lang = _detect_lang(query)
    if lang == "zh":
        return "1. Output in the same language as the question."
    if lang == "vi":
        return (
            "1. You MUST output ONLY in Vietnamese (Tiếng Việt).\n"
            "2. DO NOT use Chinese characters (中文) under any circumstances."
        )
    return (
        "1. You MUST output ONLY in English.\n"
        "2. DO NOT use Chinese characters (中文) under any circumstances."
    )


def _dispatch_query(
    query: str,
    llm_service: AbstractLLMService,
) -> Dict[str, Any]:
    """
    Điều hướng truy vấn dựa trên phân tích cấu trúc logic và thực thể.
    Áp dụng cho mọi domain (General Purpose).
    """
    
    dispatch_prompt = f"""Bạn là Bộ điều phối Truy vấn thông minh. Nhiệm vụ của bạn là phân tích cấu trúc câu hỏi để chọn chiến lược RAG phù hợp.

### QUY TẮC PHÂN LOẠI CHIẾN LƯỢC:

1. DIRECT (Strategy: "DIRECT"):
   - Khi câu hỏi tập trung vào MỘT đối tượng, MỘT chủ đề duy nhất.
   - Khi yêu cầu là tóm tắt, giải thích hoặc trích xuất thông tin từ một nguồn cụ thể.
   - Ví dụ: "Dự án X là gì?", "Tóm tắt tài liệu này", "Cách sử dụng hàm Y".

2. DECOMPOSE (Strategy: "DECOMPOSE"):
   - Khi câu hỏi yêu cầu SO SÁNH giữa 2 hoặc nhiều đối tượng.
   - Khi câu hỏi có tính chất LIỆT KÊ hoặc TỔNG HỢP thông tin từ nhiều nguồn rời rạc.
   - Khi câu hỏi lồng ghép nhiều bước thực hiện (Multi-step reasoning).
   - Ví dụ: "Điểm khác nhau giữa A và B là gì?", "Liệt kê ưu điểm của X, Y và Z".

### NGUYÊN TẮC TRÍCH XUẤT THỰC THỂ (ENTITIES):
- Trích xuất tất cả Danh từ riêng, Mã định danh (ID), Tên công nghệ, Tên người, Tên dự án xuất hiện.
- GIỮ NGUYÊN định dạng gốc (Ví dụ: "vd1", "Nguyễn Thành Nam", "ReactJS").

### YÊU CẦU ĐẦU RA (JSON ONLY):
{{
  "strategy": "DIRECT" | "DECOMPOSE",
  "intent": "summarize" | "compare" | "extract" | "explain" | "analyze",
  "is_complex": true | false,
  "entities": ["danh sách thực thể tìm thấy"],
  "sub_questions": ["Câu hỏi con 1", "Câu hỏi con 2"]
}}
*Lưu ý: "sub_questions" chỉ điền khi strategy là "DECOMPOSE".*

CÂU HỎI CẦN PHÂN TÍCH: "{query}"

JSON:"""

    default_result: Dict[str, Any] = {
        "strategy": "DIRECT",
        "intent": "extract",
        "is_complex": False,
        "entities": _extract_entities(query), 
        "sub_questions": [],
    }

    try:
        response = llm_service.generate(dispatch_prompt).strip()
        # Xử lý lấy JSON (loại bỏ markdown code fences nếu có)
        json_str = response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # --- Logic xử lý kết quả ---
            strategy = str(result.get("strategy", "DIRECT")).upper()
            
            # Kiểm soát chặt chẽ: Nếu strategy là DIRECT thì KHÔNG được phép có sub_questions
            if strategy == "DIRECT":
                sub_questions = []
                is_complex = False
            else:
                sub_questions = result.get("sub_questions", [])
                is_complex = True

            # Merge thực thể từ LLM và Regex để không bỏ sót
            llm_entities = result.get("entities", [])
            regex_entities = _extract_entities(query)
            merged_entities = list(set([e.strip() for e in llm_entities + regex_entities if e]))

            dispatch_result = {
                "strategy": strategy,
                "intent": result.get("intent", "analyze"),
                "is_complex": is_complex,
                "entities": merged_entities,
                "sub_questions": [sq for sq in sub_questions if len(str(sq)) > 5],
            }

            logger.info(f"[Dispatcher] Strategy: {strategy} | Entities: {merged_entities}")
            return dispatch_result
            
        return default_result
    except Exception as e:
        logger.error(f"Query dispatch error: {e}")
        return default_result


def _build_entity_metadata_filters(
    entities: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Build metadata filters for entity-gated retrieval.

    Uses entity names to filter document source files, ensuring
    retrieval only returns chunks from files related to the entity.

    Args:
        entities: List of entity strings from dispatcher

    Returns:
        Dict of metadata filters, or None if no filters needed
    """
    if not entities:
        return None

    # Build filter: match source files containing entity names
    # Use a list of entity keywords for source file matching
    entity_keywords = []
    for ent in entities:
        # Extract last word of multi-word entities (e.g., "Nam" from "Nguyễn Thành Nam")
        words = ent.strip().split()
        if len(words) >= 2:
            # Use the full name and the last name component
            entity_keywords.append(ent)
            entity_keywords.append(words[-1])
        else:
            entity_keywords.append(ent)

    # Deduplicate
    entity_keywords = list(dict.fromkeys(kw.lower() for kw in entity_keywords))

    if not entity_keywords:
        return None

    # Return as a filter dict that vector_store_service.search() can use
    # The vector store will use these to filter chunks by source filename
    return {
        "entity_keywords": entity_keywords,
    }


def _rank_reorder_docs(
    docs: List[Document],
    relevance_scores: Dict[str, float],
) -> List[Document]:
    """
    Rank-based context reordering to mitigate "Lost in the Middle".

    Places the most relevant chunks at the beginning and end of the list
    (where LLMs pay most attention), and less relevant chunks in the middle.

    Args:
        docs: List of Document objects
        relevance_scores: Dict mapping doc key → relevance score

    Returns:
        Reordered list of Document objects
    """
    if len(docs) <= 2:
        return docs

    # Score each doc
    scored: List[Tuple[Document, float]] = []
    for doc in docs:
        key = f"{doc.metadata.get('source', '')}|{doc.metadata.get('chunk_index', '')}|{doc.content[:100]}"
        score = relevance_scores.get(key, 0.5)
        scored.append((doc, score))

    # Sort by relevance descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Interleave: top scores at start and end, low scores in middle
    # Pattern: best, 2nd-best, ..., worst, ..., 3rd-best, 2nd-best-from-end
    result: List[Document] = []
    left = True
    for doc, score in scored:
        if left:
            result.insert(0, doc)  # Prepend (top positions)
        else:
            result.append(doc)  # Append (bottom positions)
        left = not left

    logger.info(
        "Rank-based reordering: %d docs, scores range [%.3f, %.3f]",
        len(docs),
        scored[0][1] if scored else 0,
        scored[-1][1] if scored else 0,
    )
    return result


def _check_entity_coverage(
    accumulated_context: str,
    entities: List[str],
) -> float:
    """
    Check what fraction of original query entities appear in accumulated context.

    Args:
        accumulated_context: Context gathered from chain steps so far
        entities: Entities extracted from original query

    Returns:
        Fraction of entities found (0.0 to 1.0). Returns 1.0 if no entities.
    """
    if not entities:
        return 1.0  # No entities to check → consider "fully covered"

    context_lower = accumulated_context.lower()
    found = sum(1 for e in entities if e.lower() in context_lower)
    return found / len(entities)


# ═══════════════════════════════════════════════════════════════
# Entity Extraction & Validation (CoRAG v3 — Entity Guard)
# ═══════════════════════════════════════════════════════════════


# Vietnamese common words that look capitalized but are NOT entities
_VI_NON_ENTITY_WORDS = frozenset({
    "Hà", "Nội", "Hồ", "Chí", "Minh", "Đà", "Nẵng", "Huế",
    "Việt", "Nam", "Thành", "Phố", "Bà", "Rịa", "Vũng", "Tàu",
    "Đồng", "Nai", "Bình", "Dương", "Long", "An", "Tiền", "Giang",
    "Bắc", "Ninh", "Nam", "Định", "Hải", "Phòng", "Quảng", "Ninh",
})


def _is_uppercase_start(word: str) -> bool:
    """Check if a word starts with an uppercase letter (including Vietnamese)."""
    if not word:
        return False
    return word[0].isupper()


def _extract_context_entities(
    context: str,
    query_entities: List[str],
) -> List[str]:
    """
    Extract actual entity names found IN the document context.

    When the user asks about 'cv' but the context contains 'Thanh-Nam Nguyen',
    this function discovers the real names/entities present in the documents
    so they can be injected into the synthesis prompt as a bridge.

    Args:
        context: Full document context text
        query_entities: Entities extracted from the user query

    Returns:
        List of entity names actually found in the context (deduplicated)
    """
    # Use regex entity extraction on the context itself
    context_entities = _extract_entities(context)

    # Also look for names near query entity mentions in context
    # E.g., if query_entity='cv', find names like "Nguyễn Thành Nam" near "CV" in context
    bridged_entities: List[str] = []
    for qe in query_entities:
        qe_lower = qe.lower()
        # Search for this entity in context and extract surrounding names
        for match in re.finditer(re.escape(qe_lower), context.lower()):
            start = max(0, match.start() - 200)
            end = min(len(context), match.end() + 200)
            window = context[start:end]
            window_entities = _extract_entities(window)
            for we in window_entities:
                if we.lower() not in qe_lower and qe_lower not in we.lower():
                    bridged_entities.append(we)

    # Combine and deduplicate
    all_found = context_entities + bridged_entities
    seen = set()
    unique: List[str] = []
    for e in all_found:
        lower = e.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(e)

    if unique:
        logger.info(
            "Context entity discovery: found %s in documents (query entities: %s)",
            unique[:5], query_entities,
        )

    return unique[:10]  # Limit to top 10


def _extract_entities(query: str) -> List[str]:
    """
    Extract named entities (proper nouns, identifiers) from query.

    Detects:
    - Vietnamese names: 2-3 consecutive capitalized words (e.g. "Nguyễn Thành Nam")
    - Technical identifiers: alphanumeric codes like "vd1", "Django", "Unity"
    - Quoted strings

    Args:
        query: User query string

    Returns:
        List of extracted entity strings (may be empty)
    """
    entities: List[str] = []

    # 1. Quoted strings — highest confidence
    for match in re.finditer(r'["\']([^"\']+)["\']', query):
        entities.append(match.group(1))

    # 2. Vietnamese names: 2-4 consecutive words starting with uppercase
    words = query.split()
    i = 0
    while i < len(words):
        clean = words[i].strip('.,;:!?()[]{}')
        if _is_uppercase_start(clean) and len(clean) > 1:
            name_words = [clean]
            j = i + 1
            while j < len(words):
                next_clean = words[j].strip('.,;:!?()[]{}')
                if _is_uppercase_start(next_clean) and len(next_clean) > 1:
                    name_words.append(next_clean)
                    j += 1
                else:
                    break

            if len(name_words) >= 2:
                name = " ".join(name_words)
                non_location = [w for w in name_words if w not in _VI_NON_ENTITY_WORDS]
                if non_location:
                    entities.append(name)
            i = j
        else:
            i += 1

    # 3. Technical identifiers: short alphanumeric codes (vd1, p1, abc123)
    for match in re.finditer(r'\b([a-zA-Z]{1,6}\d{1,4})\b', query):
        ident = match.group(1)
        if len(ident) >= 2:
            entities.append(ident)

    # 4. Capitalized technical terms (Django, Unity, React, etc.)
    for match in re.finditer(r'\b([A-Z][a-zA-Z]{2,})\b', query):
        term = match.group(1)
        if term.lower() not in {
            "the", "and", "but", "for", "not", "all", "any",
            "what", "which", "who", "how", "when", "where", "why",
        }:
            entities.append(term)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for e in entities:
        lower = e.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(e)

    if unique:
        logger.info(f"Extracted entities from query: {unique}")

    return unique


def _validate_sub_questions_preserve_entities(
    sub_questions: List[str],
    original_query: str,
    entities: List[str],
) -> List[str]:
    """
    Validate that each sub-question preserves at least one entity from
    the original query. Sub-questions that lose ALL entities are rejected.

    If ALL sub-questions lose entities, falls back to the original query.

    Args:
        sub_questions: Decomposed sub-questions to validate
        original_query: The user's original query
        entities: Extracted entities from the original query

    Returns:
        Validated list of sub-questions
    """
    if not entities:
        return sub_questions

    entity_lower = [e.lower() for e in entities]
    valid: List[str] = []
    rejected: List[str] = []

    for sq in sub_questions:
        sq_lower = sq.lower()
        has_any_entity = any(ent in sq_lower for ent in entity_lower)
        if has_any_entity:
            valid.append(sq)
        else:
            rejected.append(sq)
            logger.warning(
                "Sub-question lost all entities — rejected: '%s' "
                "(missing: %s)",
                sq[:80], entities,
            )

    if rejected:
        logger.info(
            "Entity validation: %d/%d sub-questions kept, %d rejected: %s",
            len(valid), len(sub_questions), len(rejected), rejected,
        )

    if not valid and sub_questions:
        logger.warning(
            "ALL sub-questions lost entities — falling back to original query: '%s'",
            original_query[:80],
        )
        return [original_query.strip()]

    return valid


# ═══════════════════════════════════════════════════════════════
# Abstract Strategy
# ═══════════════════════════════════════════════════════════════


class AbstractRAGStrategy(ABC):
    """Abstract interface for RAG pipeline strategies."""

    @abstractmethod
    def process_query_stream(
        self,
        query: str,
        vector_service: AbstractVectorStoreService,
        llm_service: AbstractLLMService,
        k: int = 3,
        status_container=None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        conversation_context: str = "No previous conversation.",
        use_hybrid: bool = False,
        use_rerank: bool = False,
    ) -> Tuple[Generator[str, None, None], List[Document], Dict[str, Any]]:
        pass


# ═══════════════════════════════════════════════════════════════
# Standard RAG Strategy
# ═══════════════════════════════════════════════════════════════


class StandardRAGStrategy(AbstractRAGStrategy):
    """
    Standard RAG: single retrieval → build prompt → stream response.
    """

    def process_query_stream(
        self,
        query: str,
        vector_service: AbstractVectorStoreService,
        llm_service: AbstractLLMService,
        k: int = 3,
        status_container=None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        conversation_context: str = "No previous conversation.",
        use_hybrid: bool = False,
        use_rerank: bool = False,
    ) -> Tuple[Generator[str, None, None], List[Document], Dict[str, Any]]:
        """Standard single-retrieval RAG pipeline."""
        t_start = time.time()
        metrics: Dict[str, Any] = {
            "strategy": "standard",
            "retrieval_steps": 0,
            "total_docs_retrieved": 0,
            "retrieval_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "total_time_ms": 0.0,
        }

        def _step(icon: str, msg: str) -> None:
            if status_container is not None:
                status_container.write(f"{icon} {msg}")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        _step("[search]", f"**Analyzing query:** `{query}`")

        # ── Retrieve ───────────────────────────────────────────
        _step("[docs]", f"Searching documents (k={k})...")
        t_retrieval = time.time()

        if hasattr(vector_service, "search"):
            relevant_docs, retrieval_stats = vector_service.search(
                query=query,
                k=k,
                metadata_filters=metadata_filters,
                use_hybrid=use_hybrid,
                rerank=use_rerank,
                fetch_k=max(k * 4, 20),
            )
        else:
            relevant_docs = vector_service.similarity_search(query, k=k)
            retrieval_stats = {"results": len(relevant_docs)}

        metrics["retrieval_steps"] = 1
        metrics["total_docs_retrieved"] = len(relevant_docs)
        metrics["retrieval_time_ms"] = round((time.time() - t_retrieval) * 1000, 2)

        if relevant_docs:
            total_ms = retrieval_stats.get("total_time_ms", metrics["retrieval_time_ms"])
            _step("[ok]", f"Found **{len(relevant_docs)} documents** ({total_ms:.0f}ms)")
        else:
            _step("[warn]", "No relevant documents found")

        st.session_state.last_retrieval_stats = retrieval_stats

        if not relevant_docs:
            _no_info_msg = NO_INFO_MESSAGE.get(
                _detect_lang(query), NO_INFO_MESSAGE["en"]
            )

            def _no_info():
                yield _no_info_msg

            metrics["total_time_ms"] = round((time.time() - t_start) * 1000, 2)
            return _no_info(), [], metrics

        context = "\n\n".join(
            [
                f"[Source: {doc.get_citation()} | chunk={doc.metadata.get('chunk_index', 'n/a')}]\n{doc.content}"
                for doc in relevant_docs
            ]
        )

        prompt = _build_standard_prompt(context, query, conversation_context)
        _step("[build]", f"Built prompt ({len(prompt)} chars, {len(context)} context)")

        _step("[llm]", "Calling LLM...")
        t_gen = time.time()
        stream_gen = llm_service.generate_stream(prompt)

        def _timed_stream(gen):
            try:
                for chunk in gen:
                    yield chunk
            finally:
                metrics["generation_time_ms"] = round((time.time() - t_gen) * 1000, 2)
                metrics["total_time_ms"] = round((time.time() - t_start) * 1000, 2)

        return _timed_stream(_hallucination_guard_stream(stream_gen)), relevant_docs, metrics


# ═══════════════════════════════════════════════════════════════
# Chain-of-RAG Strategy v3
# ═══════════════════════════════════════════════════════════════


class ChainOfRAGStrategy(AbstractRAGStrategy):
    """
    Chain-of-RAG v3: complexity detection → optional step-back → entity
    discovery → sequential chain with early exit → rank-based reordering
    → synthesize.

    Key improvements over v2:
    - **Confidence-based hallucination guard**: Only strips content when
      response is truly a bare refusal (short + no useful content).
    - **Semantic relevance**: Uses cosine similarity (max-segment) blended
      with word overlap for relevance scoring.
    - **LLM-based complexity detection**: Classifies queries as
      direct/complex to avoid over-processing simple questions.
    - **Rank-based reordering**: Places best chunks at start/end of
      synthesis context to mitigate "Lost in the Middle".
    - **Early exit**: Stops chain early when entity coverage is sufficient.
    """

    def process_query_stream(
        self,
        query: str,
        vector_service: AbstractVectorStoreService,
        llm_service: AbstractLLMService,
        k: int = 3,
        status_container=None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        conversation_context: str = "No previous conversation.",
        use_hybrid: bool = False,
        use_rerank: bool = False,
    ) -> Tuple[Generator[str, None, None], List[Document], Dict[str, Any]]:
        """Chain-of-RAG v3 pipeline with semantic relevance and early exit."""
        t_start = time.time()
        metrics: Dict[str, Any] = {
            "strategy": "chain_of_rag",
            "sub_questions": [],
            "retrieval_steps": 0,
            "total_docs_retrieved": 0,
            "decomposition_time_ms": 0.0,
            "retrieval_time_ms": 0.0,
            "refinement_time_ms": 0.0,
            "generation_time_ms": 0.0,
            "total_time_ms": 0.0,
            "fallback_triggered": False,
            "early_exit_triggered": False,
            "early_exit_step": None,
            "query_complexity": "unknown",
        }

        def _step(icon: str, msg: str) -> None:
            if status_container is not None:
                status_container.write(f"{icon} {msg}")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        dbg = CoRAGDebugger()
        dbg.log_input(query)

        _step("[search]", f"**Analyzing query:** `{query}`")

        # ── Query Dispatch (v4: unified classifier + entity extractor) ──
        word_count = len(query.strip().split())
        is_direct_strategy = False  # Track DIRECT vs DECOMPOSE

        if word_count >= CORAG_MIN_QUERY_WORDS:
            # Use LLM-based dispatcher for queries with enough words
            t_classify = time.time()
            dispatch_result = _dispatch_query(query, llm_service)
            metrics["decomposition_time_ms"] += round((time.time() - t_classify) * 1000, 2)

            strategy = dispatch_result["strategy"]
            intent = dispatch_result["intent"]
            is_complex = dispatch_result["is_complex"]
            entities = dispatch_result["entities"]
            dispatcher_sub_questions = dispatch_result["sub_questions"]

            is_direct_strategy = (strategy == "DIRECT")

            metrics["entities"] = entities
            metrics["query_complexity"] = intent
            metrics["dispatch_strategy"] = strategy

            _step(
                "[dispatch]",
                f"📊 Query dispatched as **{strategy}** "
                f"(intent={intent}, entities={entities})",
            )

            if entities and not use_hybrid:
                logger.info(
                    "Auto-enabling hybrid search for entity query (entities=%s)",
                    entities,
                )
                use_hybrid = True
                _step("[entity]", f"🔑 Entities: {entities} — hybrid search auto-enabled")

            # Build entity-gated metadata filters
            entity_filters = _build_entity_metadata_filters(entities)
            if entity_filters:
                # Merge with existing metadata_filters (entity filters take priority)
                if metadata_filters:
                    metadata_filters.update(entity_filters)
                else:
                    metadata_filters = entity_filters
                _step("[entity]", f"🔒 Entity-gated retrieval: filtering by {list(entity_filters.values())}")
        else:
            # Short query → always direct, use regex entity extraction
            is_complex = False
            is_direct_strategy = True
            intent = "direct_lookup"
            entities = _extract_entities(query)
            dispatcher_sub_questions = []
            metrics["entities"] = entities
            metrics["query_complexity"] = "direct_lookup_short"
            metrics["dispatch_strategy"] = "DIRECT"

            if entities and not use_hybrid:
                use_hybrid = True
                _step("[entity]", f"🔑 Entities: {entities} — hybrid search auto-enabled")

            entity_filters = _build_entity_metadata_filters(entities)
            if entity_filters:
                if metadata_filters:
                    metadata_filters.update(entity_filters)
                else:
                    metadata_filters = entity_filters

        # ── Step 0: Step-back prompting (only for complex queries) ──
        stepback_query = query
        if CORAG_STEPBACK_ENABLED and is_complex:
            if entities:
                logger.info(
                    "Skipping step-back: named entities detected: %s",
                    entities,
                )
                _step("[entity]", f"🔒 Step-back skipped — preserving entities: {entities}")
            else:
                t_stepback = time.time()
                stepback_query = self._step_back_query(query, llm_service)
                metrics["decomposition_time_ms"] += round((time.time() - t_stepback) * 1000, 2)
                if stepback_query != query:
                    dbg.log_stepback(query, stepback_query)
                    _step("[stepback]", f"**Step-back:** `{stepback_query}`")
                else:
                    _step("[info]", "Step-back not needed for this query")
        elif not is_complex:
            _step("[info]", "🔒 Step-back skipped — direct query")

        # ── Step 1: Decompose query (only for complex queries) ──
        initial_sub_questions: List[str]

        if not is_complex:
            # Direct query → single-step chain with original query
            _step("[info]", "Query is direct — treating as single-step chain")
            dbg.log_decompose_single(f"direct query (intent={intent})")
            initial_sub_questions = [query.strip()]
        else:
            _step("[decompose]", "Decomposing query into sub-questions...")
            t_decompose = time.time()
            initial_sub_questions, dbg = self._decompose_query(
                query, stepback_query, llm_service, dbg
            )
            metrics["decomposition_time_ms"] += round((time.time() - t_decompose) * 1000, 2)

            if not initial_sub_questions:
                _step("[info]", "No decomposition produced — treating as single-step chain")
                dbg.log_decompose_single("LLM returned no sub-questions")
                initial_sub_questions = [query.strip()]
            else:
                initial_sub_questions = initial_sub_questions[:CORAG_MAX_CHAIN_STEPS]

                if entities:
                    pre_valid = list(initial_sub_questions)
                    initial_sub_questions = _validate_sub_questions_preserve_entities(
                        initial_sub_questions, query, entities,
                    )
                    if len(initial_sub_questions) != len(pre_valid):
                        _step(
                            "[entity]",
                            f"🔒 Entity validation: {len(initial_sub_questions)}/{len(pre_valid)} "
                            "sub-questions kept (entities preserved)",
                        )

                _step(
                    "[ok]",
                    f"Decomposed into **{len(initial_sub_questions)} initial sub-questions**:\n"
                    + "\n".join(f"  {i+1}. `{sq}`" for i, sq in enumerate(initial_sub_questions)),
                )

        metrics["sub_questions"] = list(initial_sub_questions)

        # ── Step 2: Sequential chain with early exit ───────────
        all_docs: List[Document] = []
        accumulated_context: str = ""
        total_retrieval_ms = 0.0
        total_refinement_ms = 0.0
        step_relevance_scores: List[float] = []
        doc_relevance_map: Dict[str, float] = {}  # For rank-based reordering

        sub_q_queue: List[str] = list(initial_sub_questions)
        processed_steps = 0
        early_exit = False

        while sub_q_queue and processed_steps < CORAG_MAX_CHAIN_STEPS:
            sub_q = sub_q_queue.pop(0)
            chain_label = f"[chain {processed_steps + 1}/{CORAG_MAX_CHAIN_STEPS}]"
            dbg.log_chain_step_start(processed_steps, CORAG_MAX_CHAIN_STEPS, sub_q)

            # Refine sub-question using accumulated context (skip for first)
            refined_q = sub_q
            if processed_steps > 0 and accumulated_context:
                _step(chain_label, "Refining sub-question based on previous context...")
                t_refine = time.time()
                refined_q = self._refine_subquestion(
                    sub_q, accumulated_context, llm_service
                )
                total_refinement_ms += (time.time() - t_refine) * 1000

                if refined_q and refined_q.strip():
                    # Use semantic relevance for drift check
                    similarity = _semantic_relevance(sub_q, refined_q, vector_service)
                    if similarity < CORAG_REFINE_DRIFT_THRESHOLD:
                        logger.warning(
                            "Refine drifted too far (relevance=%.2f): "
                            "'%s' → '%s'. Reverting to original.",
                            similarity, sub_q, refined_q,
                        )
                        _step(
                            chain_label,
                            f"⚠️ Refine drifted (relevance={similarity:.0%}) — keeping original",
                        )
                        dbg.log_refine(sub_q, refined_q, similarity, reverted=True)
                        refined_q = sub_q
                    else:
                        _step(chain_label, f"Refined: `{refined_q}` (relevance={similarity:.0%})")
                        dbg.log_refine(sub_q, refined_q, similarity, reverted=False)
                else:
                    refined_q = sub_q

            # Determine k for this step
            chain_k = CORAG_ENTITY_DISCOVERY_K if processed_steps == 0 else min(k, CORAG_CHAIN_RETRIEVAL_K)

            _step(chain_label, f"Retrieving for: `{refined_q}` (k={chain_k})")
            t_retrieval = time.time()

            if hasattr(vector_service, "search"):
                step_docs, _ = vector_service.search(
                    query=refined_q,
                    k=chain_k,
                    metadata_filters=metadata_filters,
                    use_hybrid=use_hybrid,
                    rerank=use_rerank,
                    fetch_k=max(chain_k * 2, 10),
                )
            else:
                step_docs = vector_service.similarity_search(refined_q, k=chain_k)

            step_ms = (time.time() - t_retrieval) * 1000
            total_retrieval_ms += step_ms
            metrics["retrieval_steps"] += 1
            processed_steps += 1

            # Debug: log retrieval results
            doc_previews = []
            if step_docs:
                for d in step_docs:
                    doc_previews.append({
                        "citation": d.get_citation(),
                        "content_preview": d.content[:120].replace("\n", " "),
                        "page": d.metadata.get("page"),
                    })
            dbg.log_retrieval(refined_q, len(step_docs), step_ms, doc_previews)

            # ── Relevance validation (semantic) ─────────────────
            step_relevance = 1.0
            if step_docs:
                no_relevance_count = 0
                for d in step_docs:
                    rel = _semantic_relevance(refined_q, d.content[:500], vector_service)
                    # Store for rank-based reordering
                    doc_key = f"{d.metadata.get('source', '')}|{d.metadata.get('chunk_index', '')}|{d.content[:100]}"
                    doc_relevance_map[doc_key] = rel
                    if rel < CORAG_COSINE_LOW_RELEVANCE:
                        no_relevance_count += 1
                low_ratio = no_relevance_count / len(step_docs)
                step_relevance = 1.0 - low_ratio

                dbg.log_retrieval_validation(
                    refined_q,
                    [d.content[:500] for d in step_docs],
                    has_overlap=low_ratio < 0.8,
                    overlap_ratio=low_ratio,
                )
            step_relevance_scores.append(step_relevance)

            if step_docs:
                _step(
                    chain_label,
                    f"Found **{len(step_docs)} documents** ({step_ms:.0f}ms, relevance={step_relevance:.0%})",
                )
                all_docs.extend(step_docs)

                new_context = "\n".join(
                    doc.content[:CORAG_DOC_PREVIEW_CHARS] for doc in step_docs
                )
                accumulated_context += (
                    f"\n\n[Step {processed_steps}: {refined_q}]\n{new_context}"
                )
                if len(accumulated_context) > CORAG_ACCUMULATED_CONTEXT_MAX:
                    accumulated_context = accumulated_context[
                        -CORAG_ACCUMULATED_CONTEXT_MAX:
                    ]

                # ── Early exit check (v3) ───────────────────────
                if not early_exit and processed_steps >= 1:
                    entity_coverage = _check_entity_coverage(accumulated_context, entities)
                    context_fill_ratio = len(accumulated_context) / CORAG_ACCUMULATED_CONTEXT_MAX

                    should_exit = False
                    exit_reason = ""

                    if entity_coverage >= CORAG_EARLY_EXIT_ENTITY_COVERAGE and entities:
                        should_exit = True
                        exit_reason = f"entity coverage={entity_coverage:.0%} (all entities found)"
                    elif context_fill_ratio >= CORAG_EARLY_EXIT_CONTEXT_RATIO:
                        should_exit = True
                        exit_reason = f"context fill={context_fill_ratio:.0%} (sufficient context)"

                    if should_exit:
                        early_exit = True
                        metrics["early_exit_triggered"] = True
                        metrics["early_exit_step"] = processed_steps
                        logger.info(
                            "CoRAG EARLY EXIT at step %d: %s",
                            processed_steps, exit_reason,
                        )
                        _step(
                            "[early-exit]",
                            f"✅ Early exit at step {processed_steps}: {exit_reason}",
                        )
                        # Clear remaining queue
                        sub_q_queue.clear()
                        dbg.log_early_exit(processed_steps, exit_reason)

                # ── Sequential generation (BLOCKED for DIRECT strategy) ──
                if (
                    CORAG_SEQUENTIAL_CHAIN_ENABLED
                    and not is_direct_strategy  # ← v4: block sequential for DIRECT
                    and not sub_q_queue
                    and not early_exit
                    and processed_steps < CORAG_MAX_CHAIN_STEPS
                ):
                    next_questions = self._generate_next_sub_questions(
                        original_query=query,
                        accumulated_context=accumulated_context,
                        llm_service=llm_service,
                        max_questions=CORAG_MAX_CHAIN_STEPS - processed_steps,
                    )
                    if next_questions and entities:
                        next_questions = _validate_sub_questions_preserve_entities(
                            next_questions, query, entities,
                        )

                    if next_questions:
                        dbg.log_sequential_generate(
                            processed_steps,
                            accumulated_context[:200],
                            next_questions,
                        )
                        sub_q_queue.extend(next_questions)
                        metrics["sub_questions"].extend(next_questions)
                        _step(
                            chain_label,
                            f"Generated **{len(next_questions)} follow-up sub-questions** from results",
                        )
            else:
                _step(chain_label, "No documents found for this sub-question")

        metrics["retrieval_time_ms"] = round(total_retrieval_ms, 2)
        metrics["refinement_time_ms"] = round(total_refinement_ms, 2)

        # Deduplicate documents
        seen_keys: set = set()
        unique_docs: List[Document] = []
        for doc in all_docs:
            key = f"{doc.metadata.get('source', '')}|{doc.metadata.get('chunk_index', '')}|{doc.content[:100]}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_docs.append(doc)

        metrics["total_docs_retrieved"] = len(unique_docs)

        st.session_state.last_retrieval_stats = {
            "strategy": "chain_of_rag",
            "sub_questions": metrics["sub_questions"],
            "retrieval_steps": metrics["retrieval_steps"],
            "total_docs": len(unique_docs),
            "total_retrieval_ms": metrics["retrieval_time_ms"],
            "early_exit": metrics["early_exit_triggered"],
        }

        # ── Low Relevance Fallback ─────────────────────────────
        all_low_relevance = (
            len(step_relevance_scores) > 0
            and all(s <= CORAG_LOW_RELEVANCE_THRESHOLD for s in step_relevance_scores)
        )

        if all_low_relevance and step_relevance_scores:
            logger.warning(
                "CoRAG LOW RELEVANCE FALLBACK: all %d steps had relevance ≤ %.2f",
                len(step_relevance_scores), CORAG_LOW_RELEVANCE_THRESHOLD,
            )
            dbg.log_fallback(
                f"All {len(step_relevance_scores)} steps returned low relevance"
            )
            metrics["fallback_triggered"] = True
            _step(
                "[fallback]",
                "⚠️ Low relevance across all steps — falling back to standard retrieval",
            )

            t_fb = time.time()
            if hasattr(vector_service, "search"):
                fb_docs, fb_stats = vector_service.search(
                    query=query,
                    k=k,
                    metadata_filters=metadata_filters,
                    use_hybrid=use_hybrid,
                    rerank=use_rerank,
                    fetch_k=max(k * 4, 20),
                )
            else:
                fb_docs = vector_service.similarity_search(query, k=k)
                fb_stats = {"results": len(fb_docs)}

            metrics["retrieval_steps"] += 1
            metrics["retrieval_time_ms"] += round((time.time() - t_fb) * 1000, 2)
            _step("[ok]", f"Fallback found **{len(fb_docs)} documents**")

            unique_docs = fb_docs
            metrics["total_docs_retrieved"] = len(fb_docs)

        # ── No docs fallback ───────────────────────────────────
        if not unique_docs:
            _step("[warn]", "No relevant documents found across all chain steps")
            _no_info_msg = NO_INFO_MESSAGE.get(
                _detect_lang(query), NO_INFO_MESSAGE["en"]
            )

            def _no_info():
                yield _no_info_msg

            metrics["total_time_ms"] = round((time.time() - t_start) * 1000, 2)
            return _no_info(), [], metrics

        # ── Rank-based reordering (v3) ─────────────────────────
        if doc_relevance_map and len(unique_docs) > 2:
            _step("[reorder]", "Reordering documents by relevance (rank-based)...")
            unique_docs = _rank_reorder_docs(unique_docs, doc_relevance_map)

        # ── Step 3: Synthesize ─────────────────────────────────
        _step("[synthesize]", "Synthesizing answer from chain evidence...")

        chain_context = "\n\n".join(
            [
                f"[Source: {doc.get_citation()} | chunk={doc.metadata.get('chunk_index', 'n/a')}]\n{doc.content}"
                for doc in unique_docs
            ]
        )
        original_context_len = len(chain_context)
        truncated = original_context_len > CORAG_SYNTHESIS_CONTEXT_MAX
        if truncated:
            chain_context = chain_context[-CORAG_SYNTHESIS_CONTEXT_MAX:]

        # ── Entity Bridge: discover real names from context (C) ──
        context_entities = _extract_context_entities(chain_context, entities or [])
        # Merge query entities with context-discovered entities for richer prompt
        all_synthesis_entities = list(entities or [])
        for ce in context_entities:
            if ce.lower() not in [e.lower() for e in all_synthesis_entities]:
                all_synthesis_entities.append(ce)

        prompt = _build_corag_prompt(
            query, chain_context, metrics["sub_questions"], conversation_context,
            entities=all_synthesis_entities,
            context_entities=context_entities,
            intent=intent,
        )
        _step("[build]", f"Built synthesis prompt ({len(prompt)} chars, {len(chain_context)} context, "
             f"context_entities={context_entities[:3]})")

        dbg.log_synthesis(
            unique_docs_count=len(unique_docs),
            context_len=original_context_len,
            prompt_len=len(prompt),
            truncated=truncated,
            context_budget=CORAG_SYNTHESIS_CONTEXT_MAX,
        )

        # ── Stream ─────────────────────────────────────────────
        _step("[llm]", "Calling LLM for synthesis...")
        t_gen = time.time()
        stream_gen = llm_service.generate_stream(prompt)

        _answer_chunks: List[str] = []

        def _timed_stream(gen):
            try:
                for chunk in gen:
                    _answer_chunks.append(chunk)
                    yield chunk
            finally:
                full_answer = "".join(_answer_chunks)
                metrics["generation_time_ms"] = round((time.time() - t_gen) * 1000, 2)
                metrics["total_time_ms"] = round((time.time() - t_start) * 1000, 2)
                logger.info(
                    "Chain-of-RAG v3 complete: steps=%d, docs=%d, fallback=%s, "
                    "early_exit=%s, total=%.0fms",
                    metrics["retrieval_steps"],
                    metrics["total_docs_retrieved"],
                    metrics["fallback_triggered"],
                    metrics["early_exit_triggered"],
                    metrics["total_time_ms"],
                )
                dbg.log_answer(full_answer)
                dbg.summary()

        return _timed_stream(_hallucination_guard_stream(stream_gen, is_direct=is_direct_strategy)), unique_docs, metrics

    # ── CoRAG Helper Methods ────────────────────────────────────

    @staticmethod
    def _step_back_query(query: str, llm_service: AbstractLLMService) -> str:
        """
        Generate a more abstract, broader version of the query (Step-back Prompting).

        Args:
            query: Original user query
            llm_service: LLM service for generation

        Returns:
            Step-back query string, or original query if step-back fails
        """
        lang_r = _lang_rule(query)

        stepback_prompt = f"""Create a BROADER, more abstract version of this question.
The abstract version should capture the OVERALL information need, not specific details.

RULES:
{lang_r}
3. Keep the same language as the original question.
4. Output EXACTLY ONE question on a single line. No explanation.
5. If the question already has specific entities, return it as-is.

ORIGINAL: {query}

ABSTRACT QUESTION:"""

        try:
            response = llm_service.generate(stepback_prompt).strip()
            cleaned = response.strip("\"'").strip()
            if re.search(r"[\u4e00-\u9fff]", cleaned):
                logger.warning("Step-back produced CJK — reverting to original")
                return query
            if cleaned and len(cleaned) > 10:
                logger.info(f"Step-back: '{query}' → '{cleaned}'")
                return cleaned
            return query
        except Exception as e:
            logger.warning(f"Step-back prompting failed: {e}")
            return query

    @staticmethod
    def _decompose_query(
        query: str,
        stepback_query: str,
        llm_service: AbstractLLMService,
        dbg: CoRAGDebugger,
    ) -> Tuple[List[str], CoRAGDebugger]:
        """
        Break a complex question into sub-questions with entity-discovery awareness.

        Args:
            query: The user's original question
            stepback_query: The broader step-back version of the query
            llm_service: LLM service for decomposition
            dbg: Debugger instance for tracing

        Returns:
            Tuple of (sub-questions list, updated debugger)
        """
        lang_r = _lang_rule(query)

        context_hint = ""
        if stepback_query != query:
            context_hint = f"\nBROADER CONTEXT: {stepback_query}"

        extracted = _extract_entities(query)
        entity_rule = ""
        if extracted:
            entity_list = ", ".join(f"'{e}'" for e in extracted)
            entity_rule = f"""
MANDATORY ENTITY PRESERVATION:
The following entity names MUST appear in EVERY sub-question: {entity_list}
If you cannot include them, output the ORIGINAL QUESTION as-is.
"""

        decompose_prompt = f"""Break this question into exactly 2 sub-questions for document search.

CRITICAL RULES:
{lang_r}{entity_rule}
3. If the question asks to compare or list items but the specific items are UNKNOWN, your FIRST sub-question MUST be an "entity discovery" question that asks what items/entities exist in the documents.
4. Your second sub-question should then ask about details of those entities.
5. NEVER ask follow-up questions back to the user. All sub-questions must be answerable from documents alone.
6. Keep exact identifiers like 'vd1', 'vd2', 'Django', 'Unity' intact. Do not translate them.
7. Output EXACTLY one question per line, numbered 1 and 2. DO NOT output more than 2 lines.
8. Each sub-question must be self-contained and searchable.
{context_hint}

QUESTION: {query}

SUB-QUESTIONS:"""

        try:
            response = llm_service.generate(decompose_prompt)
            dbg.log_decompose_raw(response)

            sub_questions: List[str] = []

            for line in response.strip().split("\n"):
                line = line.strip()
                cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
                if not cleaned or len(cleaned) < CORAG_SUB_QUESTION_MIN_CHARS:
                    continue
                if re.search(r"[\u4e00-\u9fff]", cleaned):
                    logger.warning("Rejecting CJK hallucinated sub-question: '%s'", cleaned[:80])
                    continue
                sub_questions.append(cleaned)

            dbg.log_decompose_cleaned(sub_questions)

            if len(sub_questions) == 1 and "single" in sub_questions[0].lower():
                dbg.log_decompose_single("LLM responded with SINGLE")
                return [query.strip()], dbg

            if not sub_questions:
                dbg.log_decompose_single("No valid sub-questions extracted")
                return [query.strip()], dbg

            pre_dedup = list(sub_questions)
            sub_questions = ChainOfRAGStrategy._deduplicate_sub_questions(
                sub_questions, query
            )
            removed = [sq for sq in pre_dedup if sq not in sub_questions]
            dbg.log_decompose_deduplicated(sub_questions, removed)

            logger.info("Decomposed into %d sub-questions: %s", len(sub_questions), sub_questions)
            return sub_questions, dbg

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            dbg.log_decompose_single(f"Exception: {e}")
            return [query.strip()], dbg

    @staticmethod
    def _generate_next_sub_questions(
        original_query: str,
        accumulated_context: str,
        llm_service: AbstractLLMService,
        max_questions: int = 2,
    ) -> List[str]:
        """
        Generate follow-up sub-questions based on retrieval results so far.

        Args:
            original_query: The user's original question
            accumulated_context: Context gathered from previous retrieval steps
            llm_service: LLM service for generation
            max_questions: Maximum number of questions to generate

        Returns:
            List of generated sub-questions (may be empty)
        """
        lang_r = _lang_rule(original_query)
        remaining = min(max_questions, 2)

        generate_prompt = f"""Based on the documents found so far and the original question, create {remaining} specific follow-up search questions.

The documents mention specific entities/projects/topics. Create questions that dig deeper into EACH entity found.

RULES:
{lang_r}
3. NEVER ask follow-up questions back to the user.
4. Each question must be answerable from documents alone.
5. Keep exact identifiers and names from the context intact.
6. Output EXACTLY one question per line, numbered. No explanation.
7. If the context does not mention any specific entities, output NOTHING (empty).

ORIGINAL QUESTION: {original_query}

DOCUMENTS FOUND SO FAR:
{accumulated_context[:2000]}

FOLLOW-UP SEARCH QUESTIONS:"""

        try:
            response = llm_service.generate(generate_prompt).strip()
            if not response:
                return []

            generated: List[str] = []
            for line in response.strip().split("\n"):
                line = line.strip()
                cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
                if not cleaned or len(cleaned) < CORAG_SUB_QUESTION_MIN_CHARS:
                    continue
                if re.search(r"[\u4e00-\u9fff]", cleaned):
                    continue
                generated.append(cleaned)

            generated = ChainOfRAGStrategy._deduplicate_sub_questions(generated, original_query)

            logger.info(f"Sequential generation: {len(generated)} follow-up questions: {generated}")
            return generated[:remaining]

        except Exception as e:
            logger.warning(f"Sequential question generation failed: {e}")
            return []

    @staticmethod
    def _deduplicate_sub_questions(
        sub_questions: List[str], original_query: str,
    ) -> List[str]:
        """
        Remove sub-questions that are too similar to each other or to the
        original query, preventing redundant retrieval steps.
        """
        if len(sub_questions) <= 1:
            return sub_questions

        unique: List[str] = []
        for sq in sub_questions:
            if _word_overlap(sq, original_query) > 0.7:
                logger.debug(f"Skipping sub-question too similar to original: '{sq}'")
                continue
            is_dup = False
            for existing in unique:
                if _word_overlap(sq, existing) > 0.65:
                    logger.debug(f"Skipping duplicate sub-question: '{sq}' ≈ '{existing}'")
                    is_dup = True
                    break
            if not is_dup:
                unique.append(sq)

        if not unique and sub_questions:
            unique = [sub_questions[0]]

        return unique

    @staticmethod
    def _refine_subquestion(
        sub_question: str, 
        accumulated_context: str, 
        llm_service: AbstractLLMService
    ) -> str:
        """
        Tinh chỉnh câu hỏi dựa trên bối cảnh đã thu thập để tăng độ chính xác khi tìm kiếm.
        """
        if not accumulated_context:
            return sub_question

        refine_prompt = f"""
### ROLE:
Bạn là một Query Optimizer chuyên nghiệp. Nhiệm vụ của bạn là làm rõ câu hỏi ORIGINAL QUERY dựa trên CONTEXT để bước tìm kiếm tiếp theo đạt kết quả tốt nhất.

### INPUT:
- CONTEXT (Bối cảnh đã thu thập): {accumulated_context[:2000]}
- ORIGINAL QUERY (Câu hỏi gốc): {sub_question}

### RULES (BẮT BUỘC):
1. CHỈ LÀM RÕ (CLARIFY), KHÔNG MỞ RỘNG (EXPAND): Chỉ sử dụng CONTEXT để xác định các đại từ mơ hồ (ví dụ: "nó", "dự án đó", "người này") hoặc các thực thể chưa rõ danh tính.
2. BẢO TỒN Ý ĐỊNH (INTENT PRESERVATION): Không được thêm các chủ đề mới hoặc các chi tiết kỹ thuật mà ORIGINAL QUERY không yêu cầu.
3. TÍNH TRUNG LẬP: Nếu CONTEXT không chứa thông tin giúp làm rõ ORIGINAL QUERY, hoặc CONTEXT thuộc một chủ đề hoàn toàn khác, bạn PHẢI trả về nguyên văn ORIGINAL QUERY.
4. ĐỊNH DẠNG: Chỉ trả về duy nhất 1 câu truy vấn đã tinh chỉnh. Không giải thích, không thêm ký tự đặc biệt.

### VÍ DỤ:
- Context: "Tài liệu đang nói về dự án X." + Query: "Nó dùng công nghệ gì?" -> Refined: "Dự án X dùng công nghệ gì?"
- Context: "Hôm nay trời đẹp." + Query: "Cách cài đặt Docker?" -> Refined: "Cách cài đặt Docker?" (Vì không liên quan)

REFINED QUERY:"""

        try:
            response = llm_service.generate(refine_prompt).strip()
            refined = response.strip("\"'").strip()
            
            # Guard: Nếu model trả về lời giải thích hoặc từ chối
            if len(refined.split('\n')) > 1 or "xin lỗi" in refined.lower():
                return sub_question
                
            return refined if len(refined) > 3 else sub_question
        except Exception as e:
            logger.error(f"Refine logic error: {e}")
            return sub_question


# ═══════════════════════════════════════════════════════════════
# Prompt Templates
# ═══════════════════════════════════════════════════════════════


def _build_entity_priority_instruction(
    entities: List[str],
    lang: str,
    context_entities: Optional[List[str]] = None,
) -> str:
    """
    Build entity-priority instruction block for synthesis prompts.

    This instruction ensures the LLM prioritizes information about the
    detected entities and only says "not enough information" when the
    context is truly empty or contains NO entity-related keywords.

    v3: Added CONTEXT ACKNOWLEDGMENT RULE to force model to acknowledge
    data, and ENTITY BRIDGE to connect query entities with context entities.

    Args:
        entities: List of detected entities from query + context
        lang: Detected language code
        context_entities: Entities actually found in document context

    Returns:
        Instruction string block, or empty string if no entities
    """
    if not entities:
        return ""

    entity_list = ", ".join(f"'{e}'" for e in entities)

    # Build entity bridge instruction if context entities were discovered
    bridge_instruction = ""
    if context_entities:
        context_names = ", ".join(f"'{e}'" for e in context_entities[:5])
        if lang == "vi":
            bridge_instruction = f"""
LIÊN KẾT THỰC THỂ (ENTITY BRIDGE):
- Người dùng hỏi về các thực thể trong câu hỏi, và hệ thống đã tìm thấy thông tin về: {context_names}.
- Đây chính là đối tượng được hỏi. Bạn PHẢI tóm tắt thông tin về những người/thực thể này ngay lập tức.
"""
        else:
            bridge_instruction = f"""
ENTITY BRIDGE:
- The user asked about entities in the question, and the system found information about: {context_names}.
- These ARE the subjects being asked about. You MUST summarize information about them immediately.
"""

    if lang == "vi":
        return f"""
QUY TẮC THỪA NHẬN NGỮ CẢNH (CONTEXT ACKNOWLEDGMENT) — ƯU TIÊN CAO NHẤT:
- Bạn đang thực hiện nhiệm vụ tóm tắt tài liệu. Context dưới đây là nội dung duy nhất và chính xác nhất bạn có.
- KHÔNG ĐƯỢC bắt đầu bằng "Tài liệu không đủ thông tin" hoặc "Không có thông tin".
- Nếu thấy thông tin về một cá nhân hoặc dự án, hãy mặc định đó là đối tượng được hỏi và tóm tắt ngay lập tức.
{bridge_instruction}
THỰC THỂ ƯU TIÊN: [{entity_list}]
- Nếu CONTEXT chứa thông tin về các thực thể này, bạn PHẢI trình bày thông tin đó.
- Chỉ được nói "không đủ thông tin" khi CONTEXT hoàn toàn trống rỗng hoặc KHÔNG chứa BẤT KỲ từ khóa nào liên quan đến: {entity_list}.
- Ưu tiên trình bày thông tin trực tiếp đề cập đến: {entity_list}
"""
    else:
        return f"""
CONTEXT ACKNOWLEDGMENT RULE (HIGHEST PRIORITY):
- You are performing a document summarization task. The context below is the ONLY and MOST ACCURATE content you have.
- DO NOT start with "The document does not contain enough information" or "No information available".
- If you see information about an individual or project, ASSUME that IS the subject being asked about and summarize it immediately.
{bridge_instruction}
ENTITY PRIORITY: [{entity_list}]
- If the CONTEXT contains information about these entities, you MUST present it.
- Only say "not enough information" when the context is COMPLETELY empty or contains NO keywords related to: {entity_list}.
- Prioritize information directly mentioning: {entity_list}
"""


def _build_comparison_instruction(intent: str, lang: str) -> str:
    """
    Build comparison-specific instruction block for synthesis prompts.

    When the dispatcher detects intent="compare", this adds structured
    comparison guidance to the prompt so the LLM generates a table
    instead of saying "not enough information".

    Args:
        intent: Query intent from dispatcher (e.g. "compare", "summarize")
        lang: Detected language code

    Returns:
        Instruction string block, or empty string if not a comparison
    """
    if intent != "compare":
        return ""

    if lang == "vi":
        return """
📊 SO SÁNH (COMPARISON MODE):
Đây là yêu cầu SO SÁNH giữa các thực thể. Hãy tuân thủ các quy tắc sau:

1. TỰ DO TRONG KIỂM SOÁT (Grounded Reasoning):
   - Nếu Context không có sẵn bảng so sánh, bạn PHẢI tự phân tích các thuộc tính
     tương đồng của các thực thể (Kỹ năng, Học vấn, Kinh nghiệm, Dự án) để lập bảng.
   - Tuyệt đối KHÔNG bịa đặt thông tin không có trong Context.

2. XỬ LÝ THIẾU THÔNG TIN:
   - Thay vì nói "Tài liệu không đủ thông tin", hãy nói:
     "Dựa trên dữ liệu có sẵn, tôi tìm thấy các điểm sau để so sánh..."
   - Chỉ từ chối khi Context hoàn toàn không liên quan đến thực thể được hỏi.

3. ĐỊNH DẠNG ĐẦU RA:
   - Ưu tiên sử dụng BẢNG (Markdown Table) cho các tiêu chí so sánh.
   - Các tiêu chí gợi ý: [Tên, Chuyên môn, Kỹ năng chính, Dự án tiêu biểu, Học vấn]
"""

    return """
📊 COMPARISON MODE:
This is a COMPARISON request between entities. Follow these rules:

1. GROUNDED REASONING:
   - If the Context doesn't have a ready-made comparison, you MUST analyze
     matching attributes (Skills, Education, Experience, Projects) to build a table.
   - Do NOT fabricate information not present in the Context.

2. HANDLING MISSING INFO:
   - Instead of saying "not enough information", say:
     "Based on available data, here are the comparison points I found..."
   - Only refuse when Context is completely unrelated to the entities asked.

3. OUTPUT FORMAT:
   - Use a MARKDOWN TABLE for comparison criteria.
   - Suggested criteria: [Name, Specialty, Key Skills, Notable Projects, Education]
"""


def _build_standard_prompt(
    context: str,
    question: str,
    chat_history_context: str,
    entities: Optional[List[str]] = None,
    intent: str = "",
) -> str:
    """Build prompt for standard RAG generation with language-aware instructions."""
    lang = _detect_lang(question)
    lang_instruction = _lang_rule(question)
    no_info_instr = NO_INFO_INSTRUCTION.get(lang, NO_INFO_INSTRUCTION["en"])
    entity_priority = _build_entity_priority_instruction(entities or [], lang)
    comparison_instr = _build_comparison_instruction(intent, lang)

    return f"""You are SmartDoc AI, an intelligent document assistant.

CRITICAL LANGUAGE RULE (HIGHEST PRIORITY):
{lang_instruction}
{entity_priority}
{comparison_instr}
Answer the QUESTION based ONLY on the CONTEXT below.
{no_info_instr}
Keep your answer concise (3-4 sentences maximum).
Be factual and precise.

    RECENT CONVERSATION (for resolving references only — do NOT use as a knowledge source):
    {chat_history_context}

CONTEXT (this is your ONLY source of factual information):
{context}

QUESTION:
{question}

RULE: If the user asks you to add features, code, or logic that are NOT in the documents, explicitly REFUSE that specific part of the request in your text, and DO NOT write the code for it.

ANSWER:"""


def _build_corag_prompt(
    question: str,
    chain_context: str,
    sub_questions: List[str],
    chat_history_context: str,
    entities: Optional[List[str]] = None,
    context_entities: Optional[List[str]] = None,
    intent: str = "",
) -> str:
    """Build prompt for Chain-of-RAG synthesis with language-aware instructions."""
    chain_summary = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(sub_questions))
    lang = _detect_lang(question)
    lang_instruction = _lang_rule(question)
    no_info_msg = NO_INFO_MESSAGE.get(lang, NO_INFO_MESSAGE["en"])
    entity_priority = _build_entity_priority_instruction(
        entities or [], lang, context_entities=context_entities,
    )
    comparison_instr = _build_comparison_instruction(intent, lang)

    return f"""You are SmartDoc AI, an intelligent document assistant using Chain-of-RAG reasoning.

The original question was broken into a chain of sub-questions, and documents were retrieved for each step.
Synthesize a clear, accurate answer using ONLY the gathered evidence.

CRITICAL LANGUAGE RULE (HIGHEST PRIORITY):
{lang_instruction}
{entity_priority}
{comparison_instr}
ORIGINAL QUESTION:
{question}

CHAIN OF SUB-QUESTIONS:
{chain_summary}

GATHERED EVIDENCE (from sequential retrieval steps):
{chain_context}

    RECENT CONVERSATION (for resolving references only — do NOT use as a knowledge source):
    {chat_history_context}

STRICT INSTRUCTIONS:
1. You MUST answer the ORIGINAL QUESTION using the facts provided in the GATHERED EVIDENCE above.
2. The GATHERED EVIDENCE has been verified as relevant — start summarizing immediately.
3. DO NOT use your general knowledge to fill gaps. Do not invent, assume, or deduce code, fields, or relationships that are not explicitly written in the evidence.
4. When citing information, include the [Source: filename] tag exactly as it appears in the evidence block.
5. Only say "{no_info_msg}" if the GATHERED EVIDENCE block above is COMPLETELY empty or contains NO relevant text at all.

RULE: If the user asks you to add features, code, or logic that are NOT in the documents, explicitly REFUSE that specific part of the request in your text, and DO NOT write the code for it.

ANSWER:"""
