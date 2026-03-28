"""Document service for loading and processing documents."""

from pathlib import Path
from datetime import datetime
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from src.models.document_model import Document
from src.utils.logger import setup_logger
from src.utils.exceptions import DocumentLoadError
from src.utils.constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

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
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects (already chunked)
            
        Raises:
            DocumentLoadError: If loading fails
        """
        try:
            logger.info(f"Loading document: {file_path}")
            
            # Create appropriate loader
            loader = DocumentLoaderFactory.create_loader(file_path)
            
            # Load document
            lc_docs = loader.load()
            logger.info(f"Loaded {len(lc_docs)} pages/sections from document")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(lc_docs)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Convert to domain Document objects
            source_path = Path(file_path)
            upload_time = datetime.now().isoformat()
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
                    }
                )
                for idx, chunk in enumerate(chunks)
            ]
            
            logger.info(f"Successfully processed {file_path}: {len(documents)} chunks")
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
