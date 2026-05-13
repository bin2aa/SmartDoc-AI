"""Document service for loading and processing documents."""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import DocumentLoadError
from src.utils.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Import OCR with availability check
from src.utils.ocr_utils import (
    extract_pages_with_ocr,
    OCR_AVAILABLE,
    get_availability_info,
    is_text_quality_good,
)
from src.utils.constants import (
    DEFAULT_OCR_LANG,
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_PSM,
    DEFAULT_OCR_OEM,
    DEFAULT_OCR_PREPROCESS,
    DEFAULT_OCR_AUTO_PDF,
    DEFAULT_OCR_MIN_TEXT_CHARS,
    DEFAULT_OCR_MIN_ALPHA_RATIO,
)

logger = setup_logger(__name__)


class DocumentLoaderFactory:
    """Factory pattern for document loaders."""

    LOADERS = {
        '.pdf': PDFPlumberLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
    }

    @classmethod
    def create_loader(cls, file_path: str):
        """
        Create appropriate loader based on file type.

        Args:
            file_path: Path to the document file

        Returns:
            Document loader instance

        Raises:
            DocumentLoadError: If file type not supported
        """
        extension = Path(file_path).suffix.lower()
        loader_class = cls.LOADERS.get(extension)

        if not loader_class:
            raise DocumentLoadError(
                f"File type {extension} not supported. "
                f"Supported types: {list(cls.LOADERS.keys())}"
            )

        logger.info(f"Creating {loader_class.__name__} for {file_path}")
        return loader_class(file_path)


# Define a simple text splitter as a replacement for RecursiveCharacterTextSplitter
class SimpleTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        chunks = []
        for doc in documents:
            text = doc.page_content
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                chunk_text = text[i:i + self.chunk_size]
                if not chunk_text.strip():
                    continue
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_start": i,
                    "chunk_end": i + len(chunk_text),
                }
                chunks.append(LCDocument(page_content=chunk_text, metadata=chunk_metadata))
        return chunks


class DocumentService:
    """Service for document loading and processing."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        Initialize document service.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = SimpleTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"DocumentService initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")

    def load_document(
        self,
        file_path: str,
        use_ocr: bool = False,
        ocr_config: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Load a document from file, extract rich metadata, and split into chunks.

        **Enriched metadata fields:**
        - `source`: Absolute path to the file
        - `source_file`: File name only (without path)
        - `file_type`: File extension (e.g., `.pdf`)
        - `uploaded_at`: ISO timestamp when the file was loaded
        - `chunk_index`: Index of the chunk within the document (0-based)
        - `chunk_start`: Character offset where the chunk begins
        - `chunk_end`: Character offset where the chunk ends
        - `file_size_bytes`: Size of the original file in bytes
        - `total_chunks`: Total number of chunks created from this document
        - `page_count`: Number of pages/sections in the document (PDF only)
        - `title`: Document title derived from filename
        - `is_ocr`: Whether OCR was used for this document

        Args:
            file_path: Path to the document file
            use_ocr: Bật chế độ quét ảnh (OCR) cho file PDF và file ảnh trực tiếp

        Returns:
            List of Document objects (one per chunk)

        Raises:
            DocumentLoadError: If loading fails
        """
        try:
            logger.info(f"Loading document: {file_path} | use_ocr: {use_ocr}")

            extension = Path(file_path).suffix.lower()
            lc_docs = []
            ocr_config = ocr_config or {}
            ocr_lang = ocr_config.get("lang", DEFAULT_OCR_LANG)
            ocr_dpi = int(ocr_config.get("dpi", DEFAULT_OCR_DPI))
            ocr_psm = int(ocr_config.get("psm", DEFAULT_OCR_PSM))
            ocr_oem = int(ocr_config.get("oem", DEFAULT_OCR_OEM))
            ocr_preprocess = bool(ocr_config.get("preprocess", DEFAULT_OCR_PREPROCESS))
            ocr_auto_pdf = bool(ocr_config.get("auto_pdf", DEFAULT_OCR_AUTO_PDF))

            # --- KHAI BÁO CÁC ĐUÔI ẢNH ĐƯỢC PHÉP CHẠY OCR ---
            image_extensions = ['.png', '.jpg', '.jpeg']

            # ĐIỀU KIỆN CHẠY OCR:
            # (Là file PDF VÀ người dùng có bật OCR) HOẶC (Bản chất nó là file ảnh)
            is_image_file = extension in image_extensions
            is_pdf_ocr = use_ocr and extension == '.pdf'
            used_ocr = False

            if is_pdf_ocr or is_image_file:
                # Guard: check OCR availability before attempting
                if not OCR_AVAILABLE:
                    ocr_info = get_availability_info()
                    missing = ", ".join(ocr_info["missing_deps"])
                    if is_image_file:
                        raise DocumentLoadError(
                            f"Cannot process image file '{extension}' — "
                            f"OCR dependencies not installed: {missing}. "
                            f"Install with: pip install {' '.join(ocr_info['missing_deps'])}"
                        )
                    else:
                        raise DocumentLoadError(
                            f"Cannot OCR scanned PDF — "
                            f"OCR dependencies not installed: {missing}. "
                            f"Install with: pip install {' '.join(ocr_info['missing_deps'])}"
                        )

                if is_pdf_ocr and ocr_auto_pdf:
                    try:
                        loader = DocumentLoaderFactory.create_loader(file_path)
                        lc_docs = loader.load()
                        combined_text = "".join(doc.page_content or "" for doc in lc_docs)
                        if is_text_quality_good(
                            combined_text,
                            min_chars=DEFAULT_OCR_MIN_TEXT_CHARS,
                            min_alpha_ratio=DEFAULT_OCR_MIN_ALPHA_RATIO,
                        ):
                            logger.info(
                                "[OCR] PDF has usable text; skipping OCR (auto mode)"
                            )
                        else:
                            logger.info(
                                "[OCR] Low PDF text detected; falling back to OCR (auto mode)"
                            )
                            ocr_pages = extract_pages_with_ocr(
                                file_path,
                                lang=ocr_lang,
                                dpi=ocr_dpi,
                                psm=ocr_psm,
                                oem=ocr_oem,
                                preprocess=ocr_preprocess,
                            )
                            lc_docs = [
                                LCDocument(
                                    page_content=page.get("text", ""),
                                    metadata={"source": file_path, "page": page.get("page")},
                                )
                                for page in ocr_pages
                            ]
                            used_ocr = True
                    except Exception as pdf_error:
                        logger.warning(
                            "[OCR] Auto-detect failed; falling back to OCR: %s",
                            pdf_error,
                        )
                        ocr_pages = extract_pages_with_ocr(
                            file_path,
                            lang=ocr_lang,
                            dpi=ocr_dpi,
                            psm=ocr_psm,
                            oem=ocr_oem,
                            preprocess=ocr_preprocess,
                        )
                        lc_docs = [
                            LCDocument(
                                page_content=page.get("text", ""),
                                metadata={"source": file_path, "page": page.get("page")},
                            )
                            for page in ocr_pages
                        ]
                        used_ocr = True
                else:
                    logger.info("[OCR] Starting OCR extraction for: %s (type=%s)", file_path, extension)
                    ocr_pages = extract_pages_with_ocr(
                        file_path,
                        lang=ocr_lang,
                        dpi=ocr_dpi,
                        psm=ocr_psm,
                        oem=ocr_oem,
                        preprocess=ocr_preprocess,
                    )
                    lc_docs = [
                        LCDocument(
                            page_content=page.get("text", ""),
                            metadata={"source": file_path, "page": page.get("page")},
                        )
                        for page in ocr_pages
                    ]
                    used_ocr = True

                combined = "".join(doc.page_content or "" for doc in lc_docs)
                if not combined or not combined.strip():
                    logger.warning(
                        "[OCR] ⚠️ No text extracted from '%s' — skipping file. "
                        "The document may be blank or unreadable.",
                        file_path,
                    )
                    raise DocumentLoadError(
                        f"OCR could not read any text from '{Path(file_path).name}'. "
                        f"The file may be blank, contain only images, or be unreadable. "
                        f"Try a different language pack or higher DPI setting."
                    )

                char_count = len(combined.strip())
                logger.info(
                    "[OCR] ✅ Successfully extracted %d chars from '%s'",
                    char_count, file_path,
                )
            if not lc_docs:
                # Logic cũ bình thường: Create appropriate loader
                loader = DocumentLoaderFactory.create_loader(file_path)
                lc_docs = loader.load()

            logger.info(f"Loaded {len(lc_docs)} pages/sections from document")

            # Split into chunks
            chunks = self.text_splitter.split_documents(lc_docs)
            logger.info(f"Split into {len(chunks)} chunks")

            if not chunks:
                logger.warning(
                    "No text chunks produced for '%s' — document may be image-only or empty",
                    file_path,
                )
                raise DocumentLoadError(
                    f"No readable text found in '{Path(file_path).name}'. "
                    "If this is a scanned document, enable OCR and try again."
                )

            # Extract metadata for enriched fields
            source_path = Path(file_path)
            upload_time = datetime.now().isoformat()
            file_size_bytes = source_path.stat().st_size
            file_name = source_path.name
            file_ext = source_path.suffix.lower()
            num_pages = len(lc_docs)
            total_chunks = len(chunks)
            title = file_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")

            # Convert to domain Document objects
            documents = [
                Document(
                    content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        'source': file_path,
                        'source_file': source_path.name,
                        'file_type': source_path.suffix.lower(),
                        'uploaded_at': upload_time,
                        'chunk_index': idx,
                        'is_ocr': used_ocr,
                        'ocr_lang': ocr_lang,
                        'ocr_dpi': ocr_dpi,
                        'ocr_psm': ocr_psm,
                        'ocr_oem': ocr_oem,
                        'ocr_preprocess': ocr_preprocess,
                        'ocr_auto_pdf': ocr_auto_pdf,
                        "file_size_bytes": file_size_bytes,
                        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                        "title": title,
                        "page_count": num_pages,
                        "total_chunks": total_chunks,
                    }
                )
                for idx, chunk in enumerate(chunks)
            ]

            logger.info(f"Successfully processed {file_path}: {len(documents)} chunks, "
                        f"{num_pages} pages, {file_size_bytes} bytes")
            return documents

        except DocumentLoadError:
            raise
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise DocumentLoadError(f"Cannot find file: {file_path}")
        except Exception as e:
            logger.exception(f"Unexpected error loading {file_path}")
            raise DocumentLoadError(f"Failed to load document: {str(e)}")

    def update_chunk_config(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Update chunking configuration.

        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = SimpleTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"Updated chunk config: size={chunk_size}, overlap={chunk_overlap}")