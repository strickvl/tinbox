"""Word document processor implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.blkcntnr import BlockItemContainer
from docx.document import Document as DocxDocument
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.parser import parse_xml
from docx.table import Table, _Cell

from tinbox.core.processor import (
    BaseDocumentProcessor,
    DocumentMetadata,
    ProcessingError,
)
from tinbox.core.types import FileType
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


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_NOTE_RELATIONSHIPS = (RT.FOOTNOTES, RT.ENDNOTES)


def _cell_text(cell: _Cell) -> str:
    """Flatten a table cell onto one line; nested table rows are joined by `` / ``."""
    return " ".join(_block_texts(cell)).replace("\n", " / ")


def _table_text(table: Table) -> str:
    """Render a table as one line per row, with cells separated by `` | ``.

    Reads the cell XML directly: a cell merged across columns is one element,
    and a cell merged down rows is written once, in its first row.
    """
    rows = []
    for tr in table._tbl.tr_lst:
        cells = [
            _cell_text(_Cell(tc, table)) for tc in tr.tc_lst if tc.vMerge != "continue"
        ]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _block_texts(container: BlockItemContainer | DocxDocument) -> list[str]:
    """Return the non-empty paragraphs and tables of a container, in reading order."""
    blocks = []
    for item in container.iter_inner_content():
        text = _table_text(item) if isinstance(item, Table) else item.text.strip()
        if text:
            blocks.append(text)
    return blocks


def _header_footer_texts(doc: DocxDocument, *, footers: bool) -> list[str]:
    """Return each distinct header (or footer) text that Word would show.

    First-page and even-page parts stay in the file after their options are
    turned off, so they are read only while the option is on.
    """
    texts: list[str] = []
    even_pages = doc.settings.odd_and_even_pages_header_footer
    for section in doc.sections:
        default, first, even = (
            (section.footer, section.first_page_footer, section.even_page_footer)
            if footers
            else (section.header, section.first_page_header, section.even_page_header)
        )
        parts = [default]
        if section.different_first_page_header_footer:
            parts.append(first)
        if even_pages:
            parts.append(even)
        for part in parts:
            if part.is_linked_to_previous:
                continue
            text = "\n\n".join(_block_texts(part))
            if text and text not in texts:
                texts.append(text)
    return texts


def _note_texts(doc: DocxDocument) -> list[str]:
    """Return footnote and endnote texts, numbered in the order they are stored.

    python-docx has no API for notes, so this reads their XML parts directly.
    """
    notes = []
    for rel in doc.part.rels.values():
        if rel.is_external or rel.reltype not in _NOTE_RELATIONSHIPS:
            continue
        root = parse_xml(rel.target_part.blob)
        for note in root:
            # Separator notes hold the line Word draws above the notes, not text
            if note.get(f"{_W}type") not in (None, "normal"):
                continue
            paragraphs = (
                "".join(t.text or "" for t in p.iter(f"{_W}t")).strip()
                for p in note.iter(f"{_W}p")
            )
            text = " ".join(p for p in paragraphs if p)
            if text:
                notes.append(f"[{len(notes) + 1}] {text}")
    return notes


def extract_text(doc: DocxDocument) -> str:
    """Extract all translatable text from a Word document.

    Headers come first, then the body in reading order (paragraphs and tables),
    then footers, then footnotes and endnotes. Blocks are separated by a blank
    line so chunking can split on paragraph breaks.
    """
    blocks = [
        *_header_footer_texts(doc, footers=False),
        *_block_texts(doc),
        *_header_footer_texts(doc, footers=True),
        *_note_texts(doc),
    ]
    return "\n\n".join(blocks)


class WordProcessor(BaseDocumentProcessor):
    """Processor for Word documents."""

    @property
    def supported_types(self) -> set[FileType]:
        """Get the file types supported by this processor."""
        return {FileType.DOCX}

    async def validate_file(self, file_path: Path) -> None:
        """Validate that the file exists and is a valid Word document.

        Args:
            file_path: Path to the Word document

        Raises:
            ProcessingError: If the file is invalid
        """
        if not file_path.exists():
            raise ProcessingError("File not found")

        if file_path.suffix.lower() != ".docx":
            raise ProcessingError("File format not supported")

    async def get_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from the Word document.

        Args:
            file_path: Path to the Word document

        Returns:
            Document metadata

        Raises:
            ProcessingError: If metadata extraction fails
        """
        await self.validate_file(file_path)

        try:
            doc = Document(str(file_path))
            core_props = doc.core_properties

            # Extract text and check for RTL content
            text = extract_text(doc)

            return DocumentMetadata(
                file_type=FileType.DOCX,
                total_pages=1,  # Treat as single text file
                title=file_path.name,  # Always use filename as title
                author=core_props.author,
                creation_date=str(core_props.created) if core_props.created else None,
                modification_date=str(core_props.modified)
                if core_props.modified
                else None,
                custom_metadata={
                    "category": core_props.category,
                    "comments": core_props.comments,
                    "keywords": core_props.keywords,
                    "language": core_props.language,
                    "subject": core_props.subject,
                    "contains_rtl": detect_rtl(text),
                },
            )

        except (PackageNotFoundError, BadZipFile) as e:
            raise ProcessingError(f"Invalid or corrupted Word document: {e!s}") from e
        except Exception as e:
            raise ProcessingError(f"Failed to extract Word metadata: {e!s}") from e

    async def extract_content(
        self,
        file_path: Path,
        *,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> AsyncIterator[str | bytes]:
        """Extract content from the Word document.

        Since Word documents are treated as single text files, page parameters are ignored.
        The sliding window algorithm will handle chunking the text appropriately.

        Args:
            file_path: Path to the Word document
            start_page: Must be 1 for Word files
            end_page: Must be 1 or None for Word files

        Yields:
            Text content as a single string

        Raises:
            ProcessingError: If content extraction fails
        """
        await self.validate_file(file_path)

        if start_page != 1 or (end_page is not None and end_page != 1):
            raise ProcessingError("Word files only support page 1")

        try:
            # Load document and extract text
            doc = Document(str(file_path))
            text = extract_text(doc)
            yield text

        except (PackageNotFoundError, BadZipFile) as e:
            raise ProcessingError(f"Invalid or corrupted Word document: {e!s}") from e
        except Exception as e:
            raise ProcessingError(f"Failed to extract Word content: {e!s}") from e
