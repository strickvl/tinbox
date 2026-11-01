"""Tests for the Word document processor."""

from pathlib import Path

import pytest
from docx import Document
from docx.opc.constants import CONTENT_TYPE as CT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part

from tinbox.core.processor import DocumentMetadata, ProcessingError
from tinbox.core.processor.docx import WordProcessor
from tinbox.core.types import FileType


@pytest.fixture
def processor() -> WordProcessor:
    """Fixture providing a Word processor instance."""
    return WordProcessor()


@pytest.fixture
def temp_docx(tmp_path: Path) -> Path:
    """Fixture providing a temporary Word document for testing."""
    doc = Document()
    doc.add_heading("Test Document", 0)

    # Add some regular paragraphs
    doc.add_paragraph("This is a test paragraph.")
    doc.add_paragraph("Another test paragraph.")

    # Add some Arabic text
    doc.add_paragraph("هذا نص عربي للاختبار")
    doc.add_paragraph("مزيد من النص العربي")

    # Save the document
    file_path = tmp_path / "test.docx"
    doc.save(str(file_path))
    return file_path


@pytest.mark.asyncio
async def test_word_processor_metadata(processor: WordProcessor, temp_docx: Path):
    """Test Word document metadata extraction."""
    metadata = await processor.get_metadata(temp_docx)

    assert isinstance(metadata, DocumentMetadata)
    assert metadata.file_type == FileType.DOCX
    assert metadata.total_pages == 1  # Word files are treated as single page
    assert metadata.title == temp_docx.name

    # These might be None, but should exist
    assert hasattr(metadata, "title")
    assert hasattr(metadata, "author")
    assert hasattr(metadata, "creation_date")
    assert hasattr(metadata, "modification_date")

    # Custom metadata should include Word-specific fields
    assert "language" in metadata.custom_metadata
    assert "contains_rtl" in metadata.custom_metadata
    assert metadata.custom_metadata["contains_rtl"]  # Should be True for Arabic text


@pytest.mark.asyncio
async def test_word_processor_content_extraction(
    processor: WordProcessor, temp_docx: Path
):
    """Test Word document content extraction."""
    pages = []
    async for page in processor.extract_content(temp_docx):
        pages.append(page)

    assert len(pages) == 1  # Word files are treated as single page
    content = pages[0]
    assert isinstance(content, str)
    assert "Test Document" in content
    assert "This is a test paragraph" in content
    assert "هذا نص عربي" in content  # Arabic text should be present


@pytest.mark.asyncio
async def test_word_processor_rtl_content(processor: WordProcessor, temp_docx: Path):
    """Test handling of right-to-left text."""
    pages = []
    async for page in processor.extract_content(temp_docx):
        pages.append(page)

    assert len(pages) == 1
    content = pages[0]
    assert isinstance(content, str)
    assert "هذا نص عربي" in content


@pytest.mark.asyncio
async def test_word_processor_invalid_file(processor: WordProcessor, tmp_path: Path):
    """Test handling of invalid files."""
    # Non-existent file
    with pytest.raises(ProcessingError, match="File not found"):
        await processor.get_metadata(tmp_path / "nonexistent.docx")

    # Invalid file type
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("Not a Word document")
    with pytest.raises(ProcessingError, match="not supported"):
        await processor.get_metadata(invalid_file)

    # Corrupted Word document
    corrupted_file = tmp_path / "corrupted.docx"
    corrupted_file.write_text("This is not a valid DOCX file")
    with pytest.raises(ProcessingError, match="Invalid or corrupted"):
        await processor.get_metadata(corrupted_file)


# Removed test_word_processor_invalid_page_range since we no longer validate page ranges


FOOTNOTES_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>
  <w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:footnote>
  <w:footnote w:id="1"><w:p><w:r><w:t>Payment is due</w:t></w:r><w:r><w:t xml:space="preserve"> within 30 days.</w:t></w:r></w:p></w:footnote>
</w:footnotes>"""


@pytest.fixture
def structured_docx(tmp_path: Path) -> Path:
    """A document whose text lives outside body paragraphs as well as in them."""
    doc = Document()
    doc.sections[0].header.paragraphs[0].text = "Confidential draft"
    doc.sections[0].footer.paragraphs[0].text = "Page footer text"
    doc.add_paragraph("Intro paragraph.")
    table = doc.add_table(rows=2, cols=3)
    table.cell(0, 0).text = "Item"
    table.cell(0, 1).text = "Price"
    table.cell(0, 2).text = "Due"
    table.cell(1, 0).text = "Licence"
    table.cell(1, 1).merge(table.cell(1, 2)).text = "500 EUR on signing"
    doc.add_paragraph("Closing paragraph.")
    footnotes = Part(
        PackURI("/word/footnotes.xml"),
        CT.WML_FOOTNOTES,
        FOOTNOTES_XML,
        doc.part.package,
    )
    doc.part.relate_to(footnotes, RT.FOOTNOTES)
    file_path = tmp_path / "structured.docx"
    doc.save(str(file_path))
    return file_path


@pytest.mark.asyncio
async def test_word_processor_keeps_tables_headers_footers_and_notes(
    processor: WordProcessor, structured_docx: Path
):
    pages = [page async for page in processor.extract_content(structured_docx)]

    assert pages == [
        "Confidential draft\n\n"
        "Intro paragraph.\n\n"
        "Item | Price | Due\n"
        "Licence | 500 EUR on signing\n\n"
        "Closing paragraph.\n\n"
        "Page footer text\n\n"
        "[1] Payment is due within 30 days."
    ]


@pytest.mark.asyncio
async def test_word_processor_separates_paragraphs_with_blank_lines(
    processor: WordProcessor, temp_docx: Path
):
    """Blank lines let the chunkers split on paragraph breaks."""
    pages = [page async for page in processor.extract_content(temp_docx)]

    assert isinstance(pages[0], str)
    assert "This is a test paragraph.\n\nAnother test paragraph." in pages[0]


@pytest.mark.asyncio
async def test_word_processor_reads_vertically_merged_cells_once(
    processor: WordProcessor, tmp_path: Path
):
    doc = Document()
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).merge(table.cell(1, 0)).text = "North"
    table.cell(0, 1).text = "r1"
    table.cell(1, 1).text = "r2"
    file_path = tmp_path / "vmerge.docx"
    doc.save(str(file_path))

    pages = [page async for page in processor.extract_content(file_path)]

    assert pages == ["North | r1\nr2"]


@pytest.mark.asyncio
async def test_word_processor_skips_header_variants_word_does_not_show(
    processor: WordProcessor, tmp_path: Path
):
    """First-page and even-page headers stay in the file after the option is off."""
    doc = Document()
    section = doc.sections[0]
    section.header.paragraphs[0].text = "Default header"
    section.different_first_page_header_footer = True
    section.first_page_header.paragraphs[0].text = "First page header"
    section.even_page_header.paragraphs[0].text = "Even page header"
    section.different_first_page_header_footer = False
    doc.add_paragraph("Body")
    file_path = tmp_path / "headers.docx"
    doc.save(str(file_path))

    pages = [page async for page in processor.extract_content(file_path)]

    assert pages == ["Default header\n\nBody"]


@pytest.mark.asyncio
async def test_word_processor_reads_first_page_header_inherited_by_later_section(
    processor: WordProcessor, tmp_path: Path
):
    """Section 2 turns on Different First Page and inherits section 1's hidden part."""
    doc = Document()
    first = doc.sections[0]
    first.header.paragraphs[0].text = "Default header"
    first.first_page_header.paragraphs[0].text = "Title page header"
    doc.add_paragraph("Section one")
    second = doc.add_section()
    second.different_first_page_header_footer = True
    doc.add_paragraph("Section two")
    file_path = tmp_path / "inherited.docx"
    doc.save(str(file_path))

    pages = [page async for page in processor.extract_content(file_path)]

    assert pages == [
        "Default header\n\nTitle page header\n\nSection one\n\nSection two"
    ]
