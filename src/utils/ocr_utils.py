"""
OCR utility for SmartDoc AI.

Extracts text from scanned PDFs and image files using Tesseract OCR.
Falls back gracefully when dependencies are unavailable.
"""

import os
import platform
import shutil
from typing import Dict, List

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


def _resolve_poppler_from_path() -> str:
    """Resolve Poppler bin directory from PATH if available."""
    candidate = shutil.which("pdftoppm")
    if not candidate:
        return ""
    return os.path.dirname(candidate)

if OS_NAME == "Windows":
    if not TESSERACT_CMD:
        TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if POPPLER_PATH and not os.path.exists(POPPLER_PATH):
        logger.warning(
            "POPPLER_PATH is set but invalid: %s. Falling back to PATH lookup.",
            POPPLER_PATH,
        )
        POPPLER_PATH = ""
    if not POPPLER_PATH:
        POPPLER_PATH = _resolve_poppler_from_path()

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


def summarize_text_quality(text: str, min_alpha_ratio: float = 0.3) -> Dict[str, float]:
    """Return basic quality metrics for a text string."""
    stripped = (text or "").strip()
    total_chars = len(stripped)
    if total_chars == 0:
        return {"total_chars": 0, "alpha_ratio": 0.0}
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    alpha_ratio = alpha_chars / total_chars
    return {"total_chars": total_chars, "alpha_ratio": alpha_ratio}


def is_text_quality_good(
    text: str,
    min_chars: int = 200,
    min_alpha_ratio: float = 0.3,
) -> bool:
    """Heuristic check for usable extracted text (non-OCR)."""
    metrics = summarize_text_quality(text, min_alpha_ratio=min_alpha_ratio)
    if metrics["total_chars"] < min_chars:
        return False
    return metrics["alpha_ratio"] >= min_alpha_ratio


def _preprocess_image(img, enable: bool):
    """Apply lightweight preprocessing to improve OCR on scans."""
    if not enable:
        return img

    from PIL import ImageFilter, ImageOps

    gray = ImageOps.grayscale(img)
    enhanced = ImageOps.autocontrast(gray)
    denoised = enhanced.filter(ImageFilter.MedianFilter(size=3))

    # Simple binarization to sharpen text edges
    threshold = 170
    binary = denoised.point(lambda x: 255 if x > threshold else 0, mode="1")
    return binary


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

def extract_pages_with_ocr(
    file_path: str,
    lang: str = "vie+eng",
    dpi: int = 300,
    psm: int = 6,
    oem: int = 3,
    preprocess: bool = True,
) -> List[Dict[str, object]]:
    """
    Extract text by page from a scanned PDF or image file using Tesseract OCR.

    Returns a list of dicts: {"page": int, "text": str}.
    """
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
    pages_out: List[Dict[str, object]] = []
    total_chars = 0
    page_count = 0

    logger.info("=" * 50)
    logger.info("[OCR] Starting OCR for: %s (lang=%s)", filename, lang)
    logger.info("[OCR] File type: %s | OS: %s", extension, OS_NAME)

    try:
        # ── Case 1: PDF file ──────────────────────────────────
        if extension == ".pdf":
            pdf_kwargs = {"dpi": dpi}
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

                processed = _preprocess_image(page_image, preprocess)
                text = _pytesseract.image_to_string(
                    processed, lang=lang, config=f"--psm {psm} --oem {oem}"
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

                pages_out.append({"page": page_num, "text": text or ""})
                total_chars += page_chars

        # ── Case 2: Image file (PNG, JPG, etc.) ───────────────
        else:
            logger.info("[OCR] Processing image file directly...")
            img = Image.open(file_path)
            page_count = 1

            processed = _preprocess_image(img, preprocess)
            text = _pytesseract.image_to_string(
                processed, lang=lang, config=f"--psm {psm} --oem {oem}"
            )
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

            pages_out.append({"page": 1, "text": text or ""})
            total_chars = page_chars

        # ── Final summary ─────────────────────────────────────
        combined = "\n".join(page.get("text", "") for page in pages_out)
        overall_quality = _assess_quality(combined)

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

        return pages_out

    except Exception as e:
        logger.error("[OCR] ❌ FAILED for '%s': %s", filename, str(e))
        logger.error("[OCR] Error type: %s", type(e).__name__)
        raise RuntimeError(f"OCR processing failed for '{filename}': {str(e)}") from e


def extract_text_with_ocr(
    file_path: str,
    lang: str = "vie+eng",
    dpi: int = 300,
    psm: int = 6,
    oem: int = 3,
    preprocess: bool = True,
) -> str:
    """
    Extract text from a scanned PDF or image file using Tesseract OCR.

    Returns:
        Extracted text string (may be empty if OCR reads nothing)
    """
    pages = extract_pages_with_ocr(
        file_path=file_path,
        lang=lang,
        dpi=dpi,
        psm=psm,
        oem=oem,
        preprocess=preprocess,
    )
    combined = []
    for page in pages:
        page_num = page.get("page")
        text = page.get("text", "")
        combined.append(f"\n\n--- Page {page_num} ---\n\n{text}")
    return "".join(combined).strip()