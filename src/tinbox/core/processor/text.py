"""Text document processor implementation."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Union

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentMetadata,
    FileType,
    ProcessingError,
)
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


def detect_rtl(text: str) -> bool:
    """Detect if text contains RTL characters.

    Args:
        text: Text to check for RTL characters

    Returns:
        True if RTL characters are found, False otherwise
    """
    # RTL Unicode ranges
    rtl_ranges = [
        (0x0590, 0x05FF),  # Hebrew
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]

    return any(
        any(start <= ord(char) <= end for start, end in rtl_ranges) for char in text
    )


def detect_encoding(file_path: Path) -> str:
    """Detect the encoding of a text file.

    Args:
        file_path: Path to the text file

    Returns:
        Detected encoding (defaults to utf-8)

    Raises:
        ProcessingError: If file cannot be read with UTF-8 encoding
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            f.read()
        return "utf-8"
    except UnicodeDecodeError:
        raise ProcessingError("Invalid UTF-8 encoding")


class TextProcessor(BaseDocumentProcessor):
    """Processor for text files."""

    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor.

        Returns:
            Set containing only FileType.TXT
        """
        return {FileType.TXT}

    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from a text document.

        Args:
            file_path: Path to the text file

        Returns:
            Document metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """
        if not file_path.exists():
            raise ProcessingError("File not found")

        if file_path.suffix.lower() != ".txt":
            raise ProcessingError("File format not supported")

        try:
            # Detect encoding and read file
            encoding = detect_encoding(file_path)
            text = file_path.read_text(encoding=encoding)

            # Get file stats
            stats = file_path.stat()

            return DocumentMetadata(
                file_type=FileType.TXT,
                total_pages=1,  # Text files are always single page
                title=file_path.name,
                author=None,
                creation_date=str(stats.st_ctime),
                modification_date=str(stats.st_mtime),
                custom_metadata={
                    "size_bytes": stats.st_size,
                    "encoding": encoding,
                    "contains_rtl": detect_rtl(text),
                },
            )
        except ProcessingError:
            raise
        except Exception as e:
            logger.exception("Failed to extract text metadata")
            raise ProcessingError(f"Failed to extract text metadata: {e!s}") from e

    async def extract_content(
        self, file_path: Path, *, start_page: int = 1, end_page: int | None = None
    ) -> AsyncIterator[Union[str, bytes]]:
        """Extract content from a text document.

        Args:
            file_path: Path to the text file
            start_page: Must be 1 for text files
            end_page: Must be 1 or None for text files

        Yields:
            Text content as a single string

        Raises:
            ProcessingError: If content extraction fails
        """
        if not file_path.exists():
            raise ProcessingError("File not found")

        if start_page != 1 or (end_page is not None and end_page != 1):
            raise ProcessingError("Text files only support page 1")

        try:
            # Detect encoding and read file
            encoding = detect_encoding(file_path)
            text = file_path.read_text(encoding=encoding)
            yield text

        except ProcessingError:
            raise
        except Exception as e:
            logger.exception("Failed to extract text content")
            raise ProcessingError(f"Failed to extract text content: {e!s}") from e
