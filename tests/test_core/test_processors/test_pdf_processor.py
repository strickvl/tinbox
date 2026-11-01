"""Tests for the PDF document processor."""

import io
import shutil
from pathlib import Path

import pytest
from PIL import Image
from pypdf import PdfWriter

from tinbox.core.processor import DocumentMetadata, ProcessingError, load_document
from tinbox.core.processor.pdf import PdfProcessor
from tinbox.core.types import FileType

requires_poppler = pytest.mark.skipif(
    shutil.which("pdfinfo") is None or shutil.which("pdftoppm") is None,
    reason="poppler-utils not installed (required for PDF processing)",
)


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


async def test_pdf_processor_without_metadata(processor: PdfProcessor, tmp_path: Path):
    file_path = tmp_path / "no_metadata.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.metadata = None
    writer.write(file_path)

    metadata = await processor.get_metadata(file_path)

    assert metadata.title == file_path.name
    assert metadata.total_pages == 1
    assert metadata.author is None
    assert metadata.custom_metadata["pdf_info"] == {}


@pytest.mark.asyncio
@requires_poppler
async def test_pdf_processor_content_extraction(
    processor: PdfProcessor, sample_pdf: Path
):
    """Test PDF content extraction."""
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
@requires_poppler
async def test_pdf_processor_multi_page(processor: PdfProcessor, sample_pdf: Path):
    """Test multi-page PDF processing."""
    metadata = await processor.get_metadata(sample_pdf)
    total_pages = metadata.total_pages

    # Process all pages
    pages = []
    async for content in processor.extract_content(sample_pdf):
        pages.append(content)

    assert len(pages) == total_pages
    assert all(isinstance(p, bytes) for p in pages)


@pytest.mark.asyncio
@requires_poppler
async def test_pdf_processor_page_range(processor: PdfProcessor, sample_pdf: Path):
    """Test PDF processing with specific page ranges."""
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
@requires_poppler
async def test_pdf_processor_invalid_page_range(
    processor: PdfProcessor, sample_pdf: Path
):
    """Test handling of invalid page ranges."""
    # Start page < 1
    with pytest.raises(ProcessingError, match="Invalid page range"):
        async for _ in processor.extract_content(sample_pdf, start_page=0):
            pass

    # End page < start page
    with pytest.raises(ProcessingError, match="Invalid page range"):
        async for _ in processor.extract_content(sample_pdf, start_page=2, end_page=1):
            pass


@pytest.mark.asyncio
@requires_poppler
async def test_pdf_processor_dpi_settings(sample_pdf: Path):
    """Test PDF processing with different DPI settings."""
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

    assert isinstance(low_content, bytes)
    assert isinstance(high_content, bytes)

    # High DPI should result in larger images
    low_image = Image.open(io.BytesIO(low_content))
    high_image = Image.open(io.BytesIO(high_content))

    assert high_image.width > low_image.width
    assert high_image.height > low_image.height


@pytest.mark.asyncio
async def test_pdf_processor_converts_one_page_at_a_time(tmp_path: Path, monkeypatch):
    """Only one decoded page image is held at a time, so long PDFs fit in memory."""
    pdf_path = tmp_path / "three_pages.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    calls: list[tuple[int, int]] = []
    closed: list[int] = []

    def fake_convert(path, *, first_page, last_page, dpi):
        # The previous page's image must be closed before the next is decoded
        assert len(closed) == len(calls)
        calls.append((first_page, last_page))
        image = Image.new("RGB", (10, 10), color=(first_page * 40, 0, 0))
        close = image.close
        monkeypatch.setattr(
            image, "close", lambda: (closed.append(first_page), close())[1]
        )
        return [image]

    monkeypatch.setattr(
        "tinbox.core.processor.pdf._get_convert_from_path", lambda: fake_convert
    )
    monkeypatch.setattr(
        "tinbox.core.processor.pdf._check_poppler_available", lambda: None
    )

    pages = [page async for page in PdfProcessor().extract_content(pdf_path)]

    assert calls == [(1, 1), (2, 2), (3, 3)]
    pngs = [page for page in pages if isinstance(page, bytes)]
    assert [Image.open(io.BytesIO(png)).getpixel((0, 0)) for png in pngs] == [
        (40, 0, 0),
        (80, 0, 0),
        (120, 0, 0),
    ]


@pytest.mark.asyncio
@requires_poppler
async def test_load_document_loads_only_selected_pages(sample_pdf: Path):
    whole = await load_document(sample_pdf)
    selected = await load_document(sample_pdf, pages=[2, 3, 5])

    assert selected.page_numbers == [2, 3, 5]
    assert selected.pages == [whole.pages[1], whole.pages[2], whole.pages[4]]
    assert whole.page_numbers is None


@pytest.mark.asyncio
async def test_load_document_refuses_page_selection_for_text(tmp_path: Path):
    text = tmp_path / "doc.txt"
    text.write_text("Hallo")
    with pytest.raises(ProcessingError, match="only with PDF files"):
        await load_document(text, pages=[1])
