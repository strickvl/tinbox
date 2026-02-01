"""Package initialization for document processors."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tinbox.core.types import FileType
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentContent(BaseModel):
    """Represents a document ready for translation."""

    pages: list[Union[str, bytes]]  # Individual pages for translation
    content_type: str = Field(pattern=r"^(text|image)/.+$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @field_validator("pages")
    @classmethod
    def validate_pages_not_empty(
        cls, v: list[Union[str, bytes]]
    ) -> list[Union[str, bytes]]:
        """Validate that pages list is not empty."""
        if not v:
            raise ValueError("Pages cannot be empty")
        return v


class DocumentMetadata(BaseModel):
    """Metadata about a document being processed."""

    file_type: FileType
    total_pages: int = Field(ge=1)
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class DocumentProcessor(Protocol):
    """Protocol for document processors."""

    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor.

        Returns:
            Set of supported file types
        """
        ...

    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from a document.

        Args:
            file_path: Path to the document

        Returns:
            Document metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """
        ...

    async def extract_content(
        self, file_path: Path, *, start_page: int = 1, end_page: int | None = None
    ) -> AsyncIterator[Union[str, bytes]]:
        """Extract content from a document.

        Args:
            file_path: Path to the document
            start_page: First page to extract (1-indexed)
            end_page: Last page to extract (inclusive), or None for all pages

        Returns:
            Iterator of page contents (text or image bytes)

        Raises:
            ProcessingError: If content extraction fails
        """
        ...


class BaseDocumentProcessor(ABC):
    """Base class for document processors."""

    def __init__(self) -> None:
        """Initialize the processor."""
        self._logger = logger

    @property
    @abstractmethod
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor.

        Returns:
            Set of supported file types
        """
        ...

    def supports_file_type(self, file_type: FileType) -> bool:
        """Check if this processor supports a file type.

        Args:
            file_type: File type to check

        Returns:
            True if the file type is supported
        """
        return file_type in self.supported_types

    async def validate_file(self, file_path: Path) -> None:
        """Validate that a file can be processed.

        Args:
            file_path: Path to the file to validate

        Raises:
            ProcessingError: If the file is invalid or cannot be processed
        """
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ProcessingError(f"Not a file: {file_path}")

        try:
            file_type = FileType(file_path.suffix.lstrip(".").lower())
        except ValueError:
            raise ProcessingError(f"Unsupported file type: {file_path.suffix}")

        if not self.supports_file_type(file_type):
            raise ProcessingError(
                f"File type {file_type.value} not supported by {type(self).__name__}"
            )


class ProcessingError(Exception):
    """Error during document processing."""

    pass


def get_processor_for_file_type(
    file_type: FileType,
    *,
    settings: dict[str, Any] | None = None,
) -> DocumentProcessor:
    """Get the appropriate processor for a file type.

    Args:
        file_type: File type to get processor for
        settings: Optional settings dict passed to processor (e.g., {"dpi": 300} for PDF)

    Returns:
        Document processor instance

    Raises:
        ProcessingError: If no processor is available for the file type
    """
    # Import processors lazily to avoid loading unnecessary dependencies
    if file_type == FileType.PDF:
        from tinbox.core.processor.pdf import PdfProcessor

        return PdfProcessor(settings=settings)
    elif file_type == FileType.DOCX:
        from tinbox.core.processor.docx import WordProcessor as DocxProcessor

        return DocxProcessor()
    elif file_type == FileType.TXT:
        from tinbox.core.processor.text import TextProcessor

        return TextProcessor()
    else:
        raise ProcessingError(f"No processor available for file type: {file_type}")


async def load_document(
    file_path: Path,
    *,
    processor_settings: dict[str, Any] | None = None,
) -> DocumentContent:
    """Load a document and prepare it for translation.

    Args:
        file_path: Path to the document to load
        processor_settings: Optional settings passed to the processor (e.g., {"dpi": 300} for PDFs)

    Returns:
        Document content ready for translation

    Raises:
        ProcessingError: If document loading fails
    """
    try:
        file_type = FileType(file_path.suffix.lstrip(".").lower())
        processor = get_processor_for_file_type(file_type, settings=processor_settings)

        # Get metadata first
        metadata = await processor.get_metadata(file_path)

        # Extract all pages
        pages = []
        async for page in processor.extract_content(file_path):
            pages.append(page)

        # Determine content type based on first page
        content_type = "image/png" if isinstance(pages[0], bytes) else "text/plain"

        return DocumentContent(
            pages=pages,
            content_type=content_type,
            metadata={"file_type": file_type, **metadata.model_dump()},
        )

    except Exception as e:
        raise ProcessingError(f"Failed to load document: {e!s}") from e


__all__ = [
    "BaseDocumentProcessor",
    "DocumentContent",
    "DocumentMetadata",
    "DocumentProcessor",
    "FileType",
    "ProcessingError",
    "get_processor_for_file_type",
    "load_document",
]
