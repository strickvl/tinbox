"""Translation algorithms for Tinbox."""

import asyncio
from datetime import datetime
from typing import Optional

from rich.progress import Progress, TaskID

from tinbox.core.processor import DocumentContent
from tinbox.core.progress import ProgressTracker
from tinbox.core.translation.checkpoint import (
    CheckpointManager,
    TranslationState,
    should_resume,
    load_checkpoint,
    save_checkpoint,
)
from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.types import TranslationConfig, TranslationResult
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class TranslationProgress:
    """Track translation progress and statistics."""

    def __init__(
        self,
        total_pages: int,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> None:
        """Initialize translation progress tracking.

        Args:
            total_pages: Total number of pages to translate
            progress: Optional Rich progress instance
            task_id: Optional task ID for the progress bar
        """
        self.total_pages = total_pages
        self.completed_pages = 0
        self.failed_pages: list[int] = []
        self.token_usage = 0
        self.cost = 0.0
        self.start_time = datetime.now()
        self._progress = progress
        self._task_id = task_id

    @property
    def time_taken(self) -> float:
        """Get the total time taken so far."""
        return (datetime.now() - self.start_time).total_seconds()

    def update(
        self,
        page_number: int,
        response: Optional[TranslationResponse] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Update progress with page results.

        Args:
            page_number: The page number that was processed
            response: Optional translation response
            error: Optional error that occurred
        """
        if error:
            self.failed_pages.append(page_number)
            logger.error(
                f"Translation failed for page {page_number}",
                error=str(error),
                page=page_number,
            )
        else:
            self.completed_pages += 1
            if response:
                self.token_usage += response.tokens_used
                self.cost += response.cost

        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.completed_pages,
                total=self.total_pages,
            )


async def translate_document(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
) -> TranslationResponse:
    """Translate a document using the specified algorithm.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    if config.algorithm == "page":
        return await translate_page_by_page(content, config, translator, progress)
    elif config.algorithm == "sliding-window":
        return await translate_sliding_window(content, config, translator, progress)
    else:
        raise TranslationError(f"Unknown algorithm: {config.algorithm}")


async def translate_page_by_page(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
) -> TranslationResponse:
    """Translate a document page by page.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    total_tokens = 0
    total_cost = 0.0
    translated_pages = []
    task_id: Optional[TaskID] = None

    try:
        # Set up progress tracking
        if progress:
            task_id = progress.add_task(
                "Translating pages...",
                total=len(content.pages),
            )

        # Check for checkpoint
        if should_resume(config):
            checkpoint = load_checkpoint(config)
            if checkpoint:
                translated_pages = checkpoint.pages
                total_tokens = checkpoint.tokens_used
                total_cost = checkpoint.cost
                if progress and task_id is not None:
                    progress.update(task_id, completed=len(translated_pages))

        # Translate remaining pages
        for i, page in enumerate(
            content.pages[len(translated_pages) :], len(translated_pages)
        ):
            # Create translation request
            request = TranslationRequest(
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                content=page,
                content_type=content.content_type,
                model=config.model,
                model_params={"model_name": config.model_name}
                if config.model_name
                else {},
            )

            # Translate page
            response = await translator.translate(request)
            translated_pages.append(response.text)
            total_tokens += response.tokens_used
            total_cost += response.cost

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            # Save checkpoint if needed
            if config.checkpoint_dir and (i + 1) % config.checkpoint_frequency == 0:
                await save_checkpoint(
                    config=config,
                    pages=translated_pages,
                    tokens_used=total_tokens,
                    cost=total_cost,
                )

        # Join pages with double newlines
        final_text = "\n\n".join(translated_pages)

        return TranslationResponse(
            text=final_text,
            tokens_used=total_tokens,
            cost=total_cost,
            time_taken=0.0,  # Will be set by caller
        )

    except Exception as e:
        raise TranslationError(f"Translation failed: {str(e)}") from e


async def translate_sliding_window(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
) -> TranslationResponse:
    """Translate a document using sliding window algorithm.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    if isinstance(content.pages[0], bytes):
        raise TranslationError(
            "Sliding window algorithm not supported for image content"
        )

    try:
        # Join all pages into single text
        text = "\n\n".join(content.pages)

        # Create translation request
        request = TranslationRequest(
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            content=text,
            content_type="text/plain",
            model=config.model,
            model_params={"model_name": config.model_name} if config.model_name else {},
        )

        # Translate entire text
        response = await translator.translate(request)
        return response

    except Exception as e:
        logger.exception("Translation failed")
        raise TranslationError(f"Translation failed: {str(e)}") from e


async def repair_seams(
    pages: list[str],
    config: TranslationConfig,
    translator: ModelInterface,
) -> str:
    """Repair seams between translated pages.

    Args:
        pages: List of translated pages
        config: Translation configuration
        translator: Model interface for translation

    Returns:
        Combined text with repaired seams

    Raises:
        TranslationError: If seam repair fails
    """
    if len(pages) <= 1:
        return pages[0] if pages else ""

    try:
        result = [pages[0]]
        for i in range(1, len(pages)):
            # Extract overlapping content
            seam = extract_seam(result[-1], pages[i], config.page_seam_overlap)
            if seam:
                # Update page with repaired seam
                result.append(update_page_with_seam(pages[i], seam))
            else:
                result.append(pages[i])

        return "\n\n".join(result)

    except Exception as e:
        raise TranslationError(f"Failed to repair seams: {str(e)}") from e


def create_windows(
    text: str,
    window_size: int,
    overlap_size: int,
) -> list[str]:
    """Create overlapping windows from text.

    Args:
        text: Input text
        window_size: Size of each window in characters
        overlap_size: Size of overlap between windows

    Returns:
        List of text windows
    """
    if not text:
        return []

    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if overlap_size < 0:
        raise ValueError("Overlap size must be non-negative")
    if overlap_size >= window_size:
        raise ValueError("Overlap size must be less than window size")

    windows = []
    start = 0

    while start < len(text):
        # Calculate end position
        end = min(start + window_size, len(text))

        # Extract window
        window = text[start:end]
        windows.append(window)

        # If we've reached the end, break
        if end == len(text):
            break

        # Move start position, ensuring we make progress
        start = end - min(overlap_size, end - start)
        if start <= 0 or start >= end:
            break

    return windows


def merge_chunks(chunks: list[str], overlap_size: int) -> str:
    """Merge translated chunks, handling overlaps.

    Args:
        chunks: List of translated chunks
        overlap_size: Size of overlap between chunks

    Returns:
        Merged text
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    # If no overlap is requested, just join the chunks
    if overlap_size == 0:
        return "".join(chunks)

    result = chunks[0]
    for i in range(1, len(chunks)):
        current_chunk = chunks[i]

        # Try to find the overlap at the end of the result and start of current chunk
        for overlap_len in range(
            min(len(result), len(current_chunk), overlap_size), 0, -1
        ):
            end_of_result = result[-overlap_len:]
            start_of_chunk = current_chunk[:overlap_len]

            if end_of_result == start_of_chunk:
                # Found overlap, append only the non-overlapping part
                result += "\n\n" + current_chunk[overlap_len:]
                break
        else:
            # No overlap found, append entire chunk
            result += "\n\n" + current_chunk

    return result


def extract_seam(text1: str, text2: str, overlap_size: int) -> str:
    """Extract overlapping content between two texts.

    Args:
        text1: First text
        text2: Second text
        overlap_size: Size of overlap to look for

    Returns:
        Overlapping content or empty string if no overlap found
    """
    if not text1 or not text2 or overlap_size <= 0:
        return ""

    # Get end of first text
    end1 = text1[-overlap_size:]
    if not end1:
        return ""

    # Look for overlap in second text
    start_pos = text2.find(end1)
    if start_pos >= 0:
        return text2[start_pos : start_pos + overlap_size]

    return ""


def update_page_with_seam(page: str, seam: str) -> str:
    """Update page text with repaired seam.

    Args:
        page: Page text
        seam: Seam content

    Returns:
        Updated page text
    """
    if not seam:
        return page

    # Find seam position
    pos = page.find(seam)
    if pos >= 0:
        # Keep text after seam
        return page[pos + len(seam) :]

    return page
