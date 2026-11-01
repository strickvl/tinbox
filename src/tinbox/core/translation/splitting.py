"""Split text into chunks or windows and build the context sent beside them."""

from __future__ import annotations

import re

from tinbox.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "build_overlap_context",
    "build_translation_context_info",
    "create_windows",
    "smart_text_split",
]


def create_windows(
    text: str,
    window_size: int,
    overlap_size: int,
) -> list[tuple[str, str]]:
    """Split text into windows, each paired with the source text just before it.

    Windows end at natural boundaries (see `smart_text_split`) and do not overlap,
    so each character is translated once. The last `overlap_size` characters of
    the preceding source text are returned separately, to be sent as context only.

    Args:
        text: Input text
        window_size: Target size of each window in characters
        overlap_size: Number of preceding characters to send as context

    Returns:
        List of (preceding_text, window) pairs
    """
    if not text:
        return []

    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if overlap_size < 0:
        raise ValueError("Overlap size must be non-negative")
    if overlap_size >= window_size:
        raise ValueError("Overlap size must be less than window size")

    windows = smart_text_split(text, window_size)
    preceding = [""]
    for window in windows[:-1]:
        preceding.append(
            (preceding[-1] + window)[-overlap_size:] if overlap_size else ""
        )
    logger.debug(f"Created {len(windows)} windows")
    return list(zip(preceding, windows, strict=True))


def build_overlap_context(preceding_text: str) -> str | None:
    """Wrap the source text before a window as context the model must not translate."""
    if not preceding_text:
        return None
    return (
        f"[PRECEDING_TEXT]\n{preceding_text}\n[/PRECEDING_TEXT]\n\n"
        "This text comes directly before the text to translate and is already "
        "translated elsewhere. Use it only to understand the text to translate; "
        "do not include it in your translation."
    )


def smart_text_split(
    text: str, target_size: int, custom_split_token: str | None = None
) -> list[str]:
    """Split text at natural boundaries or custom tokens.

    Priority order:
    1. Custom split token (if provided) - ignores target_size
    2. Paragraph breaks (\\n\\n)
    3. Sentence endings ([.!?]\\s+)
    4. Line breaks (\\n)
    5. Clause boundaries ([;:,]\\s+)
    6. Word boundaries (\\s+)

    Args:
        text: Input text to split
        target_size: Target size for each chunk (characters)
        custom_split_token: Optional custom token to split on (ignores target_size)

    Returns:
        List of text chunks

    Raises:
        ValueError: If target_size is not positive or text is empty
    """
    if not text:
        return []

    if target_size <= 0:
        raise ValueError("target_size must be positive")

    # If custom split token is provided, split on it and ignore target_size
    if custom_split_token:
        chunks = text.split(custom_split_token)
        # Remove empty chunks
        return [chunk for chunk in chunks if chunk]

    # If text is smaller than target size, return as single chunk
    if len(text) <= target_size:
        return [text]

    chunks = []
    current_pos = 0

    while current_pos < len(text):
        # Calculate the end position for this chunk
        end_pos = min(current_pos + target_size, len(text))

        # If we're at the end of the text, take the remaining text
        if end_pos == len(text):
            chunks.append(text[current_pos:end_pos])
            break

        # Try to find a good split point within the target size
        chunk_text = text[current_pos:end_pos]
        best_split_pos = len(chunk_text)  # Default to target size

        # Priority 1: Paragraph breaks (\n\n)
        paragraph_matches = list(re.finditer(r"\n\n", chunk_text))
        if paragraph_matches:
            # Take the last paragraph break within the chunk
            best_split_pos = paragraph_matches[-1].end()
        else:
            # Priority 2: Sentence endings ([.!?]\s+)
            sentence_matches = list(re.finditer(r"[.!?]\s+", chunk_text))
            if sentence_matches:
                best_split_pos = sentence_matches[-1].end()
            else:
                # Priority 3: Line breaks (\n)
                line_matches = list(re.finditer(r"\n", chunk_text))
                if line_matches:
                    best_split_pos = line_matches[-1].end()
                else:
                    # Priority 4: Clause boundaries ([;:,]\s+)
                    clause_matches = list(re.finditer(r"[;:,]\s+", chunk_text))
                    if clause_matches:
                        best_split_pos = clause_matches[-1].end()
                    else:
                        # Priority 5: Word boundaries (\s+)
                        word_matches = list(re.finditer(r"\s+", chunk_text))
                        if word_matches:
                            best_split_pos = word_matches[-1].end()
                        # If no word boundaries found, split at target size (fallback)

        # Extract the chunk up to the best split position
        actual_end = current_pos + best_split_pos
        # A split at the window's end can leave whitespace just after it; keep that
        # whitespace in this chunk so the next chunk starts at a word
        if best_split_pos == len(chunk_text):
            while actual_end < len(text) and text[actual_end].isspace():
                actual_end += 1
        chunk = text[current_pos:actual_end]

        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        current_pos = actual_end

    return chunks


def build_translation_context_info(
    source_lang: str,
    target_lang: str,
    previous_chunk: str | None = None,
    previous_translation: str | None = None,
    next_chunk: str | None = None,
) -> str | None:
    """Build context information for translation consistency using tag-based notation.

    Args:
        source_lang: Source language code
        target_lang: Target language code
        previous_chunk: Previous chunk (for context)
        previous_translation: Previous translation (for consistency)
        next_chunk: Next chunk (for context and flow)

    Returns:
        Context information string, or None if no context available
    """
    context_parts = []

    # Add previous context if both chunk and translation are available
    if previous_chunk and previous_translation:
        context_parts.append(f"[PREVIOUS_CHUNK]\n{previous_chunk}\n[/PREVIOUS_CHUNK]")
        context_parts.append(
            f"[PREVIOUS_CHUNK_TRANSLATION]\n{previous_translation}\n[/PREVIOUS_CHUNK_TRANSLATION]"
        )

    # Add next context if available
    if next_chunk:
        context_parts.append(f"[NEXT_CHUNK]\n{next_chunk}\n[/NEXT_CHUNK]")

    # Return context if we have any parts, otherwise None
    if context_parts:
        context_parts.append(
            "Use this context to maintain consistency in terminology and style."
        )
        return "\n\n".join(context_parts)

    return None
