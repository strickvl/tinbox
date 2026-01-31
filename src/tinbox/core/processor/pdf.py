"""PDF document processor implementation."""

import io
import shutil
from pathlib import Path
from typing import AsyncIterator, Union

import pypdf

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentMetadata,
    FileType,
    ProcessingError,
)
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


def _get_convert_from_path():
    """Lazily import pdf2image.convert_from_path with helpful error message."""
    try:
        from pdf2image import convert_from_path
        return convert_from_path
    except ImportError:
        raise ProcessingError(
            "pdf2image is not installed. PDF image conversion requires the PDF extras.\n\n"
            "To install with PDF support, run:\n"
            "  pip install tinbox[pdf]\n"
            "or:\n"
            "  uv pip install tinbox[pdf]\n\n"
            "For full installation with all extras:\n"
            "  pip install tinbox[all]"
        )


def _check_poppler_available() -> None:
    """Check if poppler-utils is available on the system."""
    if shutil.which("pdfinfo") is None:
        raise ProcessingError(
            "poppler-utils is not installed. PDF processing requires poppler-utils to be installed on your system.\n\n"
            "Installation instructions:\n"
            "- macOS: brew install poppler\n"
            "- Ubuntu/Debian: sudo apt-get install poppler-utils\n"
            "- CentOS/RHEL: sudo yum install poppler-utils\n"
            "- Fedora: sudo dnf install poppler-utils\n"
            "- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/\n\n"
            "For more details, see the README.md file."
        )


class PdfProcessor(BaseDocumentProcessor):
    """Processor for PDF files."""

    def __init__(self, settings: dict | None = None):
        """Initialize the PDF processor.

        Args:
            settings: Optional dictionary of settings (e.g., {'dpi': 300})
        """
        super().__init__()
        self.settings = settings or {}
        self.dpi = self.settings.get("dpi", 200)  # Default DPI

    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor.

        Returns:
            Set containing only FileType.PDF
        """
        return {FileType.PDF}

    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from a PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            Document metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """
        if not file_path.exists():
            raise ProcessingError("File not found")

        try:
            with open(file_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                info = pdf.metadata

                # Use filename as title if no title in PDF metadata
                title = info.get("/Title")
                if not title:
                    title = file_path.name

                return DocumentMetadata(
                    file_type=FileType.PDF,
                    total_pages=len(pdf.pages),
                    title=title,
                    author=info.get("/Author", None),
                    creation_date=info.get("/CreationDate", None),
                    modification_date=info.get("/ModDate", None),
                    custom_metadata={"pdf_info": dict(info) if info else {}},
                )
        except pypdf.errors.PdfReadError as e:
            raise ProcessingError("File format not supported") from e
        except Exception as e:
            logger.exception("Failed to extract PDF metadata")
            raise ProcessingError(f"Failed to extract PDF metadata: {str(e)}") from e

    async def extract_content(
        self, file_path: Path, *, start_page: int = 1, end_page: int | None = None
    ) -> AsyncIterator[Union[str, bytes]]:
        """Extract content from a PDF document.

        Args:
            file_path: Path to the PDF file
            start_page: First page to extract (1-indexed)
            end_page: Last page to extract (inclusive), or None for all pages

        Yields:
            PNG image bytes for each page

        Raises:
            ProcessingError: If content extraction fails
        """
        if not file_path.exists():
            raise ProcessingError("File not found")

        # Check if poppler is available before proceeding
        _check_poppler_available()

        # Validate page range
        if start_page < 1:
            raise ProcessingError("Invalid page range: start_page must be >= 1")

        try:
            # Get total pages to validate end_page
            with open(file_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                total_pages = len(pdf.pages)

            if end_page is not None:
                if end_page < start_page:
                    raise ProcessingError(
                        "Invalid page range: end_page must be >= start_page"
                    )
                if end_page > total_pages:
                    end_page = total_pages

            # Convert pages to images (lazy import for optional dependency)
            convert_from_path = _get_convert_from_path()
            pages = convert_from_path(
                file_path,
                first_page=start_page,
                last_page=end_page,
                dpi=self.dpi,
            )

            for page in pages:
                with io.BytesIO() as bio:
                    page.save(bio, format="PNG")
                    yield bio.getvalue()

        except ProcessingError:
            raise
        except Exception as e:
            logger.exception("Failed to extract PDF content")
            raise ProcessingError(f"Failed to extract PDF content: {str(e)}") from e
