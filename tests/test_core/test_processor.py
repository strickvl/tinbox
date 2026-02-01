"""Tests for document processing interfaces and base functionality."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentContent,
    DocumentMetadata,
    FileType,
    ProcessingError,
)


class MockProcessor(BaseDocumentProcessor):
    """Mock document processor for testing."""

    @property
    def supported_types(self) -> set[FileType]:
        """Get supported file types."""
        return {FileType.TXT, FileType.PDF}

    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Mock metadata extraction."""
        if not file_path.exists():
            raise ProcessingError("File not found")
        if file_path.suffix.lower() not in {".txt", ".pdf"}:
            raise ProcessingError("File type not supported")
        return DocumentMetadata(
            file_type=FileType.TXT,
            total_pages=1,
            title=file_path.name,
            custom_metadata={},
        )

    async def extract_content(
        self, file_path: Path, *, start_page: int = 1, end_page: int | None = None
    ) -> AsyncIterator[str]:
        """Mock content extraction."""
        if not file_path.exists():
            raise ProcessingError("File not found")
        if file_path.suffix.lower() not in {".txt", ".pdf"}:
            raise ProcessingError("File type not supported")
        if start_page < 1:
            raise ProcessingError("Invalid page range")
        if end_page is not None and end_page < start_page:
            raise ProcessingError("Invalid page range")
        yield "Test content"


@pytest.fixture
def processor() -> MockProcessor:
    """Fixture providing a mock processor instance."""
    return MockProcessor()


def test_document_content_validation():
    """Test document content validation."""
    # Valid content
    content = DocumentContent(
        pages=["Test content"],
        content_type="text/plain",
        metadata={"test": "metadata"},
    )
    assert content.pages == ["Test content"]
    assert content.content_type == "text/plain"
    assert content.metadata == {"test": "metadata"}

    # Invalid content type
    with pytest.raises(ValidationError, match=r"String should match pattern"):
        DocumentContent(
            pages=["Test content"],
            content_type="invalid",
            metadata={},
        )

    # Empty pages
    with pytest.raises(ValueError, match="Pages cannot be empty"):
        DocumentContent(
            pages=[],
            content_type="text/plain",
            metadata={},
        )


def test_document_metadata_validation():
    """Test DocumentMetadata validation."""
    # Valid metadata
    metadata = DocumentMetadata(
        file_type=FileType.TXT,
        total_pages=1,
        title="Test Document",
        custom_metadata={},
    )
    assert metadata.file_type == FileType.TXT
    assert metadata.total_pages == 1
    assert metadata.title == "Test Document"

    # Invalid total pages
    with pytest.raises(ValidationError):
        DocumentMetadata(
            file_type=FileType.TXT,
            total_pages=0,  # Must be >= 1
            custom_metadata={},
        )


def test_processor_supported_types(processor: MockProcessor):
    """Test supported file type checking."""
    assert processor.supports_file_type(FileType.TXT)
    assert processor.supports_file_type(FileType.PDF)
    assert not processor.supports_file_type(FileType.DOCX)


@pytest.mark.asyncio
async def test_processor_validate_file(processor: MockProcessor, tmp_path: Path):
    """Test file validation."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")

    # Valid file
    await processor.validate_file(test_file)

    # Non-existent file
    with pytest.raises(ProcessingError, match="File not found"):
        await processor.validate_file(tmp_path / "nonexistent.txt")

    # Directory instead of file
    with pytest.raises(ProcessingError, match="Not a file"):
        await processor.validate_file(tmp_path)

    # Unsupported file type
    invalid_file = tmp_path / "test.docx"
    invalid_file.write_text("Test content")
    with pytest.raises(ProcessingError, match="not supported"):
        await processor.validate_file(invalid_file)


@pytest.mark.asyncio
async def test_processor_get_metadata(processor: MockProcessor, tmp_path: Path):
    """Test metadata extraction."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")

    metadata = await processor.get_metadata(test_file)
    assert isinstance(metadata, DocumentMetadata)
    assert metadata.file_type == FileType.TXT
    assert metadata.title == test_file.name


@pytest.mark.asyncio
async def test_processor_extract_content(processor: MockProcessor, tmp_path: Path):
    """Test content extraction."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")

    pages = []
    async for page in processor.extract_content(test_file):
        pages.append(page)

    assert len(pages) == 1
    assert pages[0] == "Test content"
