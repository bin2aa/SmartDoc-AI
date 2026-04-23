"""
OCR utility for SmartDoc AI.

Extracts text from scanned PDFs and image files using Tesseract OCR.
Falls back gracefully when dependencies are unavailable.
"""

import os
import platform
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Dependency availability flags ───────────────────────────────
OCR_AVAILABLE = False
_MISSING_DEPS: list = []

try:
    import pytesseract  # noqa: F401
    logger.debug("pytesseract imported successfully")
except ImportError:
    _MISSING_DEPS.append("pytesseract")
    logger.warning("pytesseract not installed — OCR unavailable")

try:
    from pdf2image import convert_from_path  # noqa: F401
    logger.debug("pdf2image imported successfully")
except ImportError:
    _MISSING_DEPS.append("pdf2image")
    logger.warning("pdf2image not installed — scanned PDF OCR unavailable")

try:
    from PIL import Image  # noqa: F401
    logger.debug("Pillow imported successfully")
except ImportError:
    _MISSING_DEPS.append("Pillow")
    logger.warning("Pillow not installed — image OCR unavailable")

OCR_AVAILABLE = len(_MISSING_DEPS) == 0

if OCR_AVAILABLE:
    logger.info("OCR dependencies available: pytesseract, pdf2image, Pillow")
else:
    logger.warning(
        "OCR unavailable — missing dependencies: %s. "
        "Install with: pip install %s",
        ", ".join(_MISSING_DEPS),
        " ".join(_MISSING_DEPS),
    )

# ── Platform-specific paths ─────────────────────────────────────
OS_NAME = platform.system()
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

if OS_NAME == "Windows":
    if not TESSERACT_CMD:
        TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not POPPLER_PATH:
        POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

if OCR_AVAILABLE and TESSERACT_CMD:
    import pytesseract as _pytesseract

    _pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    logger.debug("Tesseract cmd set to: %s", TESSERACT_CMD)


# ── Quality helpers ─────────────────────────────────────────────

def _assess_quality(text: str) -> str:
    """
    Quick heuristic assessment of OCR output quality.

    Returns one of: "good", "poor", "empty".
    """
    if not text or not text.strip():
        return "empty"

    stripped = text.strip()
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    total_chars = len(stripped)

    if total_chars == 0:
        return "empty"

    alpha_ratio = alpha_chars / total_chars
    if alpha_ratio < 0.3:
        return "poor"  # Mostly symbols/garbage
    return "good"


def get_availability_info() -> dict:
    """Return OCR availability status for UI display."""
    return {
        "available": OCR_AVAILABLE,
        "missing_deps": _MISSING_DEPS,
        "os": OS_NAME,
        "tesseract_cmd": TESSERACT_CMD,
        "poppler_path": POPPLER_PATH,
    }


# ── Main extraction function ────────────────────────────────────

def extract_text_with_ocr(file_path: str, lang: str = "vie+eng") -> str:
    """
    Extract text from a scanned PDF or image file using Tesseract OCR.

    Args:
        file_path: Path to the PDF or image file
        lang: Tesseract language pack (default: vie+eng for Vietnamese+English)

    Returns:
        Extracted text string (may be empty if OCR reads nothing)

    Raises:
        RuntimeError: If OCR dependencies are not installed
        FileNotFoundError: If the input file does not exist
    """
    if not OCR_AVAILABLE:
        raise RuntimeError(
            f"OCR unavailable — missing dependencies: {', '.join(_MISSING_DEPS)}. "
            f"Install with: pip install {' '.join(_MISSING_DEPS)}"
        )

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    import pytesseract as _pytesseract
    from pdf2image import convert_from_path as _convert_from_path
    from PIL import Image

    extension = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    extracted_text = ""
    total_chars = 0
    page_count = 0

    logger.info("=" * 50)
    logger.info("[OCR] Starting OCR for: %s (lang=%s)", filename, lang)
    logger.info("[OCR] File type: %s | OS: %s", extension, OS_NAME)

    try:
        # ── Case 1: PDF file ──────────────────────────────────
        if extension == ".pdf":
            pdf_kwargs = {"dpi": 300}
            if POPPLER_PATH and OS_NAME == "Windows":
                if not os.path.exists(POPPLER_PATH):
                    raise FileNotFoundError(
                        f"Poppler directory not found: {POPPLER_PATH}"
                    )
                pdf_kwargs["poppler_path"] = POPPLER_PATH

            logger.info("[OCR] Converting PDF to images (dpi=%s)...", pdf_kwargs.get("dpi"))
            pages = _convert_from_path(file_path, **pdf_kwargs)
            page_count = len(pages)
            logger.info("[OCR] PDF has %d pages", page_count)

            for i, page_image in enumerate(pages):
                page_num = i + 1
                logger.info("[OCR] Processing page %d/%d...", page_num, page_count)

                text = _pytesseract.image_to_string(
                    page_image, lang=lang, config="--psm 6"
                )
                page_text = text.strip() if text else ""
                quality = _assess_quality(page_text)
                page_chars = len(page_text)

                logger.info(
                    "[OCR] Page %d/%d: %d chars, quality=%s",
                    page_num, page_count, page_chars, quality,
                )

                if page_text:
                    # Log a preview of extracted text for debugging
                    preview = page_text[:200].replace("\n", " ")
                    logger.debug("[OCR] Page %d preview: '%s'", page_num, preview)
                else:
                    logger.warning(
                        "[OCR] Page %d/%d: NO TEXT EXTRACTED — page may be blank or unreadable",
                        page_num, page_count,
                    )

                extracted_text += f"\n\n--- Page {page_num} ---\n\n"
                extracted_text += text if text else ""
                total_chars += page_chars

        # ── Case 2: Image file (PNG, JPG, etc.) ───────────────
        else:
            logger.info("[OCR] Processing image file directly...")
            img = Image.open(file_path)
            page_count = 1

            text = _pytesseract.image_to_string(img, lang=lang, config="--psm 6")
            page_text = text.strip() if text else ""
            quality = _assess_quality(page_text)
            page_chars = len(page_text)

            logger.info(
                "[OCR] Image result: %d chars, quality=%s",
                page_chars, quality,
            )

            if page_text:
                preview = page_text[:200].replace("\n", " ")
                logger.debug("[OCR] Image preview: '%s'", preview)
            else:
                logger.warning("[OCR] Image: NO TEXT EXTRACTED — image may be blank or unreadable")

            extracted_text += text if text else ""
            total_chars = page_chars

        # ── Final summary ─────────────────────────────────────
        overall_quality = _assess_quality(extracted_text)

        logger.info("-" * 50)
        logger.info(
            "[OCR] COMPLETE: %s | pages=%d | total_chars=%d | quality=%s",
            filename, page_count, total_chars, overall_quality,
        )

        if overall_quality == "empty":
            logger.warning(
                "[OCR] ⚠️ NO readable text found in '%s'. "
                "The file may be blank, heavily image-based, or require a different language pack.",
                filename,
            )
        elif overall_quality == "poor":
            logger.warning(
                "[OCR] ⚠️ POOR quality text in '%s' (%d chars, low alpha ratio). "
                "OCR may have misread the content. Consider reviewing the original file.",
                filename, total_chars,
            )
        else:
            logger.info("[OCR] ✅ Good quality extraction from '%s'", filename)

        logger.info("=" * 50)

        return extracted_text.strip()

    except Exception as e:
        logger.error("[OCR] ❌ FAILED for '%s': %s", filename, str(e))
        logger.error("[OCR] Error type: %s", type(e).__name__)
        raise RuntimeError(f"OCR processing failed for '{filename}': {str(e)}") from e