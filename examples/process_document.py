"""Example script demonstrating document processing with Tinbox.

This script shows how to use the document processing API to:
1. Load documents of various types (PDF, DOCX, TXT)
2. Access document metadata
3. Iterate through document pages/content

Usage:
    python examples/process_document.py [path_to_document]

If no path is provided, it will process the sample Arabic document.
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console

from tinbox.core.processor import (
    load_document,
    get_processor_for_file_type,
    ProcessingError,
)
from tinbox.core.types import FileType

console = Console()


async def process_document(file_path: Path) -> None:
    """Process a document and display its content and metadata.

    Args:
        file_path: Path to the document to process
    """
    # Validate file exists
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return

    # Determine file type
    try:
        file_type = FileType(file_path.suffix.lstrip(".").lower())
    except ValueError:
        console.print(f"[red]Unsupported file type: {file_path.suffix}[/red]")
        console.print("Supported types: .pdf, .docx, .txt")
        return

    try:
        # Get processor for metadata display
        processor = get_processor_for_file_type(file_type)

        # Display metadata
        console.print("\n[bold blue]Document Metadata:[/bold blue]")
        metadata = await processor.get_metadata(file_path)
        console.print(metadata.model_dump_json(indent=2))

        # Load document content
        console.print("\n[bold blue]Loading document...[/bold blue]")
        doc = await load_document(file_path)

        console.print(f"Content type: [cyan]{doc.content_type}[/cyan]")
        console.print(f"Total pages/chunks: [cyan]{len(doc.pages)}[/cyan]")

        # Display each page
        console.print("\n[bold blue]Document Content:[/bold blue]")
        for page_num, page in enumerate(doc.pages, start=1):
            console.print(f"\n[bold]{'─' * 40}[/bold]")
            console.print(f"[bold]Page/Chunk {page_num}:[/bold]")

            if isinstance(page, bytes):
                # Binary content (e.g., PDF page as PNG image)
                console.print(
                    f"[yellow]Binary content (image): {len(page):,} bytes[/yellow]"
                )
            else:
                # Text content
                preview = page[:500] + "..." if len(page) > 500 else page
                console.print(f"[green]{preview}[/green]")

    except ProcessingError as e:
        console.print(f"[red]Processing error: {e}[/red]")

        # Provide helpful hints for common errors
        if "poppler" in str(e).lower():
            console.print("\n[yellow]Hint: PDF processing requires poppler-utils.[/yellow]")
            console.print("  macOS: brew install poppler")
            console.print("  Ubuntu: sudo apt-get install poppler-utils")
        elif "pdf2image" in str(e).lower():
            console.print("\n[yellow]Hint: Install PDF extras:[/yellow]")
            console.print("  pip install tinbox[pdf]")

    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")


async def main() -> None:
    """Main entry point."""
    # Use command-line argument or default to sample document
    if len(sys.argv) > 1:
        sample_path = Path(sys.argv[1])
    else:
        sample_path = Path("tests/data/sample_ar.docx")

    if not sample_path.exists():
        console.print(f"[red]Document not found: {sample_path}[/red]")
        console.print("\nUsage: python examples/process_document.py [path_to_document]")
        return

    console.print(f"[bold]Processing: {sample_path}[/bold]")
    await process_document(sample_path)


if __name__ == "__main__":
    asyncio.run(main())
