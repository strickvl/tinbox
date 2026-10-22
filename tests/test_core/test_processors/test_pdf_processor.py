"""Tests for the PDF document processor."""

import io
import shutil
from pathlib import Path

import pytest
from PIL import Image

from tinbox.core.processor import DocumentMetadata, ProcessingError
from tinbox.core.processor.pdf import PdfProcessor
from tinbox.core.types import FileType


def _skip_if_no_poppler():
    """Skip test if poppler is not available."""
    if shutil.which("pdfinfo") is None:
        pytest.skip("poppler-utils not installed (required for PDF processing)")


@pytest.fixture
def processor() -> PdfProcessor:
    """Fixture providing a PDF processor instance."""
    return PdfProcessor()


@pytest.fixture
def sample_pdf() -> Path:
    """Fixture providing the path to the sample PDF file."""
    return Path("tests/data/sample_ar.pdf")


@pytest.mark.asyncio
async def test_pdf_processor_metadata(processor: PdfProcessor, sample_pdf: Path):
    """Test PDF metadata extraction."""
    _skip_if_no_poppler()
    metadata = await processor.get_metadata(sample_pdf)

    assert isinstance(metadata, DocumentMetadata)
    assert metadata.file_type == FileType.PDF
    assert metadata.total_pages > 0  # Should have at least one page
    assert metadata.title == sample_pdf.name

    # These might be None, but should exist
    assert hasattr(metadata, "title")
    assert hasattr(metadata, "author")
    assert hasattr(metadata, "creation_date")
    assert hasattr(metadata, "modification_date")

    # Custom metadata should include PDF-specific fields
    assert "pdf_info" in metadata.custom_metadata
    assert isinstance(metadata.custom_metadata["pdf_info"], dict)


@pytest.mark.asyncio
async def test_pdf_processor_content_extraction(
    processor: PdfProcessor, sample_pdf: Path
):
    """Test PDF content extraction."""
    _skip_if_no_poppler()
    content = None
    async for page in processor.extract_content(sample_pdf, start_page=1, end_page=1):
        content = page
        break

    assert isinstance(content, bytes)  # Should be PNG image bytes

    # Should be able to load as PIL Image
    image = Image.open(io.BytesIO(content))
    assert image.format == "PNG"
    assert image.width > 0
    assert image.height > 0


@pytest.mark.asyncio
async def test_pdf_processor_multi_page(processor: PdfProcessor, sample_pdf: Path):
    """Test multi-page PDF processing."""
    _skip_if_no_poppler()
    metadata = await processor.get_metadata(sample_pdf)
    total_pages = metadata.total_pages

    # Process all pages
    pages = []
    async for content in processor.extract_content(sample_pdf):
        pages.append(content)

    assert len(pages) == total_pages
    assert all(isinstance(p, bytes) for p in pages)


@pytest.mark.asyncio
async def test_pdf_processor_page_range(processor: PdfProcessor, sample_pdf: Path):
    """Test PDF processing with specific page ranges."""
    _skip_if_no_poppler()
    # Process pages 2-3
    pages = []
    async for content in processor.extract_content(
        sample_pdf, start_page=2, end_page=3
    ):
        pages.append(content)

    assert len(pages) == 2


@pytest.mark.asyncio
async def test_pdf_processor_invalid_file(processor: PdfProcessor, tmp_path: Path):
    """Test handling of invalid files."""
    # Non-existent file
    with pytest.raises(ProcessingError, match="File not found"):
        await processor.get_metadata(tmp_path / "nonexistent.pdf")

    # Invalid file type
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("Not a PDF")
    with pytest.raises(ProcessingError, match="not supported"):
        await processor.get_metadata(invalid_file)


@pytest.mark.asyncio
async def test_pdf_processor_invalid_page_range(
    processor: PdfProcessor, sample_pdf: Path
):
    """Test handling of invalid page ranges."""
    _skip_if_no_poppler()
    # Start page < 1
    with pytest.raises(ProcessingError, match="Invalid page range"):
        async for _ in processor.extract_content(sample_pdf, start_page=0):
            pass

    # End page < start page
    with pytest.raises(ProcessingError, match="Invalid page range"):
        async for _ in processor.extract_content(sample_pdf, start_page=2, end_page=1):
            pass


@pytest.mark.asyncio
async def test_pdf_processor_dpi_settings(sample_pdf: Path):
    """Test PDF processing with different DPI settings."""
    _skip_if_no_poppler()
    # Create processors with different DPI settings
    low_dpi = PdfProcessor({"dpi": 72})
    high_dpi = PdfProcessor({"dpi": 300})

    # Get first page from each
    low_content = None
    high_content = None

    async for content in low_dpi.extract_content(sample_pdf, end_page=1):
        low_content = content
        break

    async for content in high_dpi.extract_content(sample_pdf, end_page=1):
        high_content = content
        break

    # High DPI should result in larger images
    low_image = Image.open(io.BytesIO(low_content))
    high_image = Image.open(io.BytesIO(high_content))

    assert high_image.width > low_image.width
    assert high_image.height > low_image.height
