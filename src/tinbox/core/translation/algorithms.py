"""Translation algorithms for Tinbox."""

from __future__ import annotations

from datetime import datetime

from rich.progress import Progress, TaskID

from tinbox.core.processor import DocumentContent
from tinbox.core.review import ReviewPair
from tinbox.core.translation.checkpoint import (
    CheckpointManager,
    TranslationState,
    resume_from_checkpoint,
)
from tinbox.core.translation.glossary import GlossaryManager
from tinbox.core.translation.interface import (
    ModelInterface,
    ProviderAccessError,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
)
from tinbox.core.translation.splitting import (
    build_overlap_context,
    build_translation_context_info,
    create_windows,
    smart_text_split,
)
from tinbox.core.types import TranslationConfig, TranslationProtocol
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


def _billed(error: Exception) -> tuple[float, int, int]:
    """Return the (cost, input tokens, output tokens) a provider billed for a failed call."""
    return (
        getattr(error, "cost", 0.0),
        getattr(error, "input_tokens", 0),
        getattr(error, "output_tokens", 0),
    )


def _cost_cap_message(total_cost: float, max_cost: float) -> str:
    return (
        f"Translation cost of ${total_cost:.4f} exceeded maximum cost of "
        f"${max_cost:.2f}"
    )


def _skip_remaining_pages(
    page_num: int,
    page_count: int,
    translated: dict[int, str],
    failed_pages: list[int],
    page_errors: dict[int, str],
    reason: str,
) -> int:
    """Mark untranslated pages after page_num as skipped; return how many."""
    skipped = [p for p in range(page_num + 1, page_count + 1) if p not in translated]
    for p in skipped:
        failed_pages.append(p)
        page_errors[p] = f"Skipped: {reason}"
    return len(skipped)


def _stopped_early(
    translated: list[str], total: int, reason: str
) -> tuple[str, list[int], str]:
    """Return (text, untranslated chunk numbers, warning) for a run that stopped.

    The text keeps every translated chunk and ends with a TRANSLATION_FAILED
    marker naming the chunks that were not translated, as page mode does.
    """
    first = len(translated) + 1
    span = f"Chunk {first}" if first == total else f"Chunks {first}-{total}"
    marker = (
        f"[TRANSLATION_FAILED: {span} of {total}]\nReason: {reason}\n"
        "[/TRANSLATION_FAILED]"
    )
    text = "".join(translated)
    warning = (
        f"Translation incomplete: {span.lower()} of {total} not translated. "
        f"Reason: {reason}"
    )
    return (
        f"{text}\n\n{marker}" if text else marker,
        list(range(first, total + 1)),
        warning,
    )


async def _save_chunks(
    checkpoint_manager: CheckpointManager,
    config: TranslationConfig,
    algorithm: str,
    translated: list[str],
    usage: tuple[int, int, int, float],
    start_time: datetime,
    glossary_manager: GlossaryManager | None,
) -> None:
    """Save a sliding-window or context-aware checkpoint.

    ``usage`` is (total tokens, input tokens, output tokens, cost) so far.
    """
    tokens, input_tokens, output_tokens, cost = usage
    await checkpoint_manager.save(
        TranslationState(
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            algorithm=algorithm,
            # Chunked algorithms treat the whole document as one page
            completed_pages=[1],
            failed_pages=[],
            translated_chunks={j + 1: text for j, text in enumerate(translated)},
            token_usage=tokens,
            cost=cost,
            time_taken=(datetime.now() - start_time).total_seconds(),
            glossary_entries=glossary_manager.get_current_glossary().entries
            if glossary_manager
            else {},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    )


async def translate_document(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Progress | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    glossary_manager: GlossaryManager | None = None,
) -> TranslationResponse:
    """Translate a document using the specified algorithm.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar
        checkpoint_manager: Optional checkpoint manager for saving/loading state

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    if config.algorithm == "page":
        return await translate_page_by_page(
            content,
            config,
            translator,
            progress,
            checkpoint_manager,
            glossary_manager,
        )
    elif config.algorithm == "sliding-window":
        return await translate_sliding_window(
            content,
            config,
            translator,
            progress,
            checkpoint_manager,
            glossary_manager,
        )
    elif config.algorithm == "context-aware":
        return await translate_context_aware(
            content,
            config,
            translator,
            progress,
            checkpoint_manager,
            glossary_manager,
        )
    else:
        raise TranslationError(f"Unknown algorithm: {config.algorithm}")


async def translate_page_by_page(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Progress | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    glossary_manager: GlossaryManager | None = None,
) -> TranslationResponse:
    """Translate a document page by page.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar
        checkpoint_manager: Optional checkpoint manager for saving/loading state

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    total_tokens = input_total = output_total = 0
    total_cost = 0.0
    # Use dict to track translations by actual page number (1-indexed)
    translated_pages_by_num: dict[int, str] = {}
    task_id: TaskID | None = None
    start_time = datetime.now()

    try:
        # Set up progress tracking
        if progress:
            task_id = progress.add_task(
                "Translating pages...",
                total=len(content.pages),
            )

        # Check for checkpoint and resume if available
        resume_result = await resume_from_checkpoint(checkpoint_manager, config)
        total_tokens = resume_result.total_tokens
        total_cost = resume_result.total_cost
        input_total = resume_result.total_input_tokens
        output_total = resume_result.total_output_tokens

        # Convert resumed list to dict (items are in order starting from page 1)
        # The resume_result.metadata may contain the actual page->translation mapping
        if resume_result.resumed and resume_result.metadata.get("translated_chunks"):
            # Restore from checkpoint's translated_chunks which has page numbers as keys
            translated_pages_by_num = {
                int(k): v
                for k, v in resume_result.metadata["translated_chunks"].items()
            }
        elif resume_result.translated_items:
            # Fallback: assume sequential from page 1
            for i, text in enumerate(resume_result.translated_items, start=1):
                translated_pages_by_num[i] = text

        if progress and task_id is not None and resume_result.resumed:
            progress.update(
                task_id, completed=len(translated_pages_by_num), total_cost=total_cost
            )

        # Track failed pages with their error messages
        failed_pages: list[int] = []
        page_errors: dict[int, str] = {}

        # Restore glossary state from checkpoint if enabled
        if config.use_glossary and glossary_manager and resume_result.glossary_entries:
            glossary_manager.restore_from_checkpoint(resume_result.glossary_entries)

        # If everything already translated via checkpoint, short-circuit
        if len(translated_pages_by_num) == len(content.pages):
            # Build final text in page order
            final_text = "\n\n".join(
                translated_pages_by_num[p]
                for p in sorted(translated_pages_by_num.keys())
            )
            time_taken = (datetime.now() - start_time).total_seconds()
            return TranslationResponse(
                text=final_text,
                tokens_used=total_tokens,
                cost=total_cost,
                time_taken=time_taken,
                input_tokens=input_total,
                output_tokens=output_total,
            )

        # Translate pages - iterate by actual page number (1-indexed)
        for page_num, page in enumerate(content.pages, start=1):
            # Skip already translated pages
            if page_num in translated_pages_by_num:
                continue

            try:
                # Create translation request
                request = TranslationRequest(
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    content=page,
                    context=None,  # No context for page-by-page algorithm
                    content_type=content.content_type,
                    model=config.model,
                    translation_protocol=config.translation_protocol,
                    local_model=config.local_model,
                    model_params={"model_name": config.model_name}
                    if config.model_name
                    else {},
                    glossary=glossary_manager.get_current_glossary()
                    if config.use_glossary and glossary_manager
                    else None,
                    reasoning_effort=config.reasoning_effort,
                    price_in=config.price_in,
                    price_out=config.price_out,
                )

                # Translate page
                response = await translator.translate(request)

                # Update glossary if new terms were discovered
                if response.glossary_updates and glossary_manager:
                    glossary_manager.update_glossary(response.glossary_updates)

                # Store with actual page number as key
                translated_pages_by_num[page_num] = response.text
                total_tokens += response.tokens_used
                total_cost += response.cost
                input_total += response.input_tokens
                output_total += response.output_tokens

                # Update progress
                if progress and task_id is not None:
                    progress.update(task_id, advance=1, total_cost=total_cost)

                # Save checkpoint if needed
                if (
                    checkpoint_manager
                    and config.checkpoint_dir
                    and page_num % config.checkpoint_frequency == 0
                ):
                    state = TranslationState(
                        source_lang=config.source_lang,
                        target_lang=config.target_lang,
                        algorithm="page",
                        completed_pages=sorted(translated_pages_by_num.keys()),
                        failed_pages=failed_pages,
                        translated_chunks=translated_pages_by_num.copy(),
                        token_usage=total_tokens,
                        cost=total_cost,
                        time_taken=(datetime.now() - start_time).total_seconds(),
                        glossary_entries=glossary_manager.get_current_glossary().entries
                        if glossary_manager
                        else {},
                        input_tokens=input_total,
                        output_tokens=output_total,
                    )
                    await checkpoint_manager.save(state)

            except Exception as e:
                error_msg = str(e)
                cost, in_tokens, out_tokens = _billed(e)
                total_cost += cost
                total_tokens += in_tokens + out_tokens
                input_total += in_tokens
                output_total += out_tokens
                failed_pages.append(page_num)
                page_errors[page_num] = error_msg
                if isinstance(e, ProviderAccessError):
                    skipped = _skip_remaining_pages(
                        page_num,
                        len(content.pages),
                        translated_pages_by_num,
                        failed_pages,
                        page_errors,
                        error_msg,
                    )
                    logger.error(
                        f"Stopping at page {page_num}; the provider refused access, "
                        f"so {skipped} remaining page(s) were skipped: {error_msg}"
                    )
                    break
                logger.error(f"Failed to translate page {page_num}: {error_msg}")

            if config.max_cost is not None and total_cost > config.max_cost:
                reason = _cost_cap_message(total_cost, config.max_cost)
                skipped = _skip_remaining_pages(
                    page_num,
                    len(content.pages),
                    translated_pages_by_num,
                    failed_pages,
                    page_errors,
                    reason,
                )
                logger.error(f"Stopping at page {page_num}: {reason}")
                break

        if not translated_pages_by_num:
            if failed_pages:
                raise TranslationError(
                    f"Translation failed: No pages were successfully translated. Failed pages: {failed_pages}"
                )
            else:
                raise TranslationError(
                    "Translation failed: No pages were successfully translated"
                )

        # Build warnings list
        warnings: list[str] = []
        if config.max_cost is not None and total_cost > config.max_cost:
            warnings.append(_cost_cap_message(total_cost, config.max_cost))
        if failed_pages:
            logger.warning(f"Failed pages: {failed_pages}")
            warnings.append(
                f"Translation incomplete: {len(failed_pages)} page(s) failed to translate "
                f"(pages: {', '.join(map(str, sorted(failed_pages)))})"
            )

        # Build final text with placeholders for failed pages
        # This ensures output has content for ALL pages, making failures visible
        all_page_nums = list(range(1, len(content.pages) + 1))
        page_texts = []
        for p in all_page_nums:
            if p in translated_pages_by_num:
                page_texts.append(translated_pages_by_num[p])
            elif p in page_errors:
                # Insert placeholder for failed page
                placeholder = (
                    f"[TRANSLATION_FAILED: Page {p}]\n"
                    f"Reason: {page_errors[p]}\n"
                    f"[/TRANSLATION_FAILED]"
                )
                page_texts.append(placeholder)

        final_text = "\n\n".join(page_texts)

        time_taken = (datetime.now() - start_time).total_seconds()

        return TranslationResponse(
            text=final_text,
            tokens_used=total_tokens,
            cost=total_cost,
            time_taken=time_taken,
            input_tokens=input_total,
            output_tokens=output_total,
            failed_pages=failed_pages,
            page_errors=page_errors,
            warnings=warnings,
        )

    except Exception as e:
        raise TranslationError(f"Translation failed: {e!s}", cost=total_cost) from e


async def translate_sliding_window(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Progress | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    glossary_manager: GlossaryManager | None = None,
) -> TranslationResponse:
    """Translate a document using sliding window algorithm.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar
        checkpoint_manager: Optional checkpoint manager for saving/loading state

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    logger.info("Starting sliding window translation")

    text_pages: list[str] = []
    for page in content.pages:
        if not isinstance(page, str):
            raise TranslationError(
                "Sliding window algorithm not supported for image content"
            )
        text_pages.append(page)

    start_time = datetime.now()
    total_tokens = input_total = output_total = 0
    total_cost = 0.0

    try:
        # Join all pages into single text
        text = "\n\n".join(text_pages)
        logger.debug(f"Combined text: {text}")

        # Use configured window size and overlap size, with fallbacks
        window_size = config.window_size if config.window_size is not None else 1000
        overlap_size = (
            config.overlap_size
            if config.overlap_size is not None
            else min(100, window_size // 4)
        )
        logger.info(f"Using window size: {window_size}, overlap size: {overlap_size}")

        # Create windows
        windows = create_windows(
            text,
            window_size,
            overlap_size,
        )
        logger.info(f"Created {len(windows)} windows")

        # Check for checkpoint and resume if available
        resume_result = await resume_from_checkpoint(checkpoint_manager, config)
        total_tokens = resume_result.total_tokens
        total_cost = resume_result.total_cost
        input_total = resume_result.total_input_tokens
        output_total = resume_result.total_output_tokens

        translated_windows = resume_result.translated_items
        stop_reason: str | None = None
        unsaved = False

        # Set up progress tracking
        if progress:
            task_id = progress.add_task(
                "Translating windows...",
                total=len(windows),
                completed=len(translated_windows),
                total_cost=total_cost,
            )

        # Restore glossary state from checkpoint if enabled
        if config.use_glossary and glossary_manager and resume_result.glossary_entries:
            glossary_manager.restore_from_checkpoint(resume_result.glossary_entries)

        # Translate remaining windows
        for i, (preceding_text, window) in enumerate(
            windows[len(translated_windows) :], len(translated_windows)
        ):
            logger.debug(f"Translating window {i + 1}: {window}")
            # Create translation request
            request = TranslationRequest(
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                content=window,
                context=build_overlap_context(preceding_text),
                content_type="text/plain",
                model=config.model,
                translation_protocol=config.translation_protocol,
                local_model=config.local_model,
                model_params={"model_name": config.model_name}
                if config.model_name
                else {},
                glossary=glossary_manager.get_current_glossary()
                if config.use_glossary and glossary_manager
                else None,
                reasoning_effort=config.reasoning_effort,
                price_in=config.price_in,
                price_out=config.price_out,
            )

            # Translate window
            try:
                response = await translator.translate(request)
            except Exception as e:
                cost, in_tokens, out_tokens = _billed(e)
                total_cost += cost
                total_tokens += in_tokens + out_tokens
                input_total += in_tokens
                output_total += out_tokens
                if not translated_windows:
                    raise
                # Keep the paid-for windows instead of discarding the whole run
                logger.error(f"Stopping at window {i + 1}: {e!s}")
                stop_reason = str(e)
                unsaved = unsaved or cost > 0
                break

            # Update glossary if new terms were discovered
            if response.glossary_updates and glossary_manager:
                glossary_manager.update_glossary(response.glossary_updates)

            translated_windows.append(response.text)
            total_tokens += response.tokens_used
            total_cost += response.cost
            input_total += response.input_tokens
            output_total += response.output_tokens
            unsaved = True

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1, total_cost=total_cost)

            # Save checkpoint if needed
            if (
                checkpoint_manager
                and config.checkpoint_dir
                and (i + 1) % config.checkpoint_frequency == 0
            ):
                logger.debug(f"Saving checkpoint after window {i + 1}")
                await _save_chunks(
                    checkpoint_manager,
                    config,
                    "sliding-window",
                    translated_windows,
                    (total_tokens, input_total, output_total, total_cost),
                    start_time,
                    glossary_manager,
                )
                unsaved = False

            if (
                config.max_cost is not None
                and total_cost > config.max_cost
                and i + 1 < len(windows)
            ):
                stop_reason = _cost_cap_message(total_cost, config.max_cost)
                logger.error(f"Stopping after window {i + 1}: {stop_reason}")
                break

        # Save the stopping point so a rerun continues from the next window
        if stop_reason and unsaved and checkpoint_manager and config.checkpoint_dir:
            await _save_chunks(
                checkpoint_manager,
                config,
                "sliding-window",
                translated_windows,
                (total_tokens, input_total, output_total, total_cost),
                start_time,
                glossary_manager,
            )

        warnings: list[str] = []
        failed_chunks: list[int] = []
        if stop_reason:
            final_text, failed_chunks, warning = _stopped_early(
                translated_windows, len(windows), stop_reason
            )
            warnings.append(warning)
        else:
            # Windows do not overlap, so their translations join directly
            final_text = "".join(translated_windows)
            if config.max_cost is not None and total_cost > config.max_cost:
                warnings.append(_cost_cap_message(total_cost, config.max_cost))
        logger.debug(f"Final merged text: {final_text}")

        time_taken = (datetime.now() - start_time).total_seconds()

        return TranslationResponse(
            text=final_text,
            tokens_used=total_tokens,
            cost=total_cost,
            time_taken=time_taken,
            input_tokens=input_total,
            output_tokens=output_total,
            failed_chunks=failed_chunks,
            warnings=warnings,
        )

    except Exception as e:
        logger.error(f"Error in sliding window translation: {e!s}")
        raise TranslationError(f"Translation failed: {e!s}", cost=total_cost) from e


async def translate_context_aware(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Progress | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    glossary_manager: GlossaryManager | None = None,
) -> TranslationResponse:
    """Translate using context-aware algorithm with natural boundary splitting.

    Args:
        content: The document content to translate
        config: Translation configuration
        translator: Model interface to use
        progress: Optional progress bar
        checkpoint_manager: Optional checkpoint manager for saving/loading state

    Returns:
        Translation result

    Raises:
        TranslationError: If translation fails
    """
    text_pages: list[str] = []
    for page in content.pages:
        if not isinstance(page, str):
            raise TranslationError(
                "Context-aware algorithm not supported for image content"
            )
        text_pages.append(page)

    start_time = datetime.now()
    total_tokens = input_total = output_total = 0
    total_cost = 0.0
    task_id: TaskID | None = None

    try:
        # Join all pages into single text
        text = "\n\n".join(text_pages)
        logger.info(f"Combined text length: {len(text)} characters")

        # Use configured context size with fallback
        context_size = config.context_size if config.context_size is not None else 2000
        logger.info(f"Using context size: {context_size} characters")

        # Split text using smart splitting
        chunks = smart_text_split(
            text,
            context_size,
            config.custom_split_token,
        )
        logger.debug(f"Chunks: {chunks}")
        logger.info(f"Created {len(chunks)} chunks using context-aware splitting")

        if config.preserve_review_pairs and not chunks:
            raise TranslationError(
                "Review pair preservation requires at least one non-empty source chunk"
            )

        # Check for checkpoint and resume if available
        resume_result = await resume_from_checkpoint(checkpoint_manager, config, chunks)
        if config.preserve_review_pairs and resume_result.resumed:
            raise TranslationError(
                "Review pair preservation requires a fresh translation. "
                "Choose a new checkpoint directory or omit --checkpoint-dir."
            )
        total_tokens = resume_result.total_tokens
        total_cost = resume_result.total_cost
        input_total = resume_result.total_input_tokens
        output_total = resume_result.total_output_tokens

        translated_chunks = resume_result.translated_items
        stop_reason: str | None = None
        unsaved = False
        review_pairs: list[ReviewPair] | None = (
            [] if config.preserve_review_pairs else None
        )
        previous_chunk: str | None = resume_result.metadata.get("previous_chunk")
        previous_translation: str | None = resume_result.metadata.get(
            "previous_translation"
        )

        # Set up progress tracking
        if progress:
            task_id = progress.add_task(
                "Translating chunks...",
                total=len(chunks),
                completed=len(translated_chunks),
                total_cost=total_cost,
            )

        # Restore glossary state from checkpoint if enabled
        if config.use_glossary and glossary_manager and resume_result.glossary_entries:
            glossary_manager.restore_from_checkpoint(resume_result.glossary_entries)

        # Translate remaining chunks with context
        for i, current_chunk in enumerate(
            chunks[len(translated_chunks) :], len(translated_chunks)
        ):
            logger.debug(f"Translating chunk {i + 1}/{len(chunks)}")

            # Get the next chunk if available
            next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None

            # TranslateGemma's fixed prompt has no place for neighbouring chunks
            context_info = (
                None
                if config.translation_protocol == TranslationProtocol.TRANSLATEGEMMA
                else build_translation_context_info(
                    source_lang=config.source_lang,
                    target_lang=config.target_lang,
                    previous_chunk=previous_chunk,
                    previous_translation=previous_translation,
                    next_chunk=next_chunk,
                )
            )

            # Create translation request with separate content and context
            request = TranslationRequest(
                source_lang=config.source_lang,
                target_lang=config.target_lang,
                content=current_chunk,  # Pure content only
                context=context_info,  # Separate context
                content_type="text/plain",
                model=config.model,
                translation_protocol=config.translation_protocol,
                local_model=config.local_model,
                model_params={"model_name": config.model_name}
                if config.model_name
                else {},
                glossary=glossary_manager.get_current_glossary()
                if config.use_glossary and glossary_manager
                else None,
                reasoning_effort=config.reasoning_effort,
                price_in=config.price_in,
                price_out=config.price_out,
            )

            # Translate chunk
            try:
                response = await translator.translate(request)
            except Exception as e:
                cost, in_tokens, out_tokens = _billed(e)
                total_cost += cost
                total_tokens += in_tokens + out_tokens
                input_total += in_tokens
                output_total += out_tokens
                if not translated_chunks:
                    raise
                # Keep the paid-for chunks instead of discarding the whole run
                logger.error(f"Stopping at chunk {i + 1}/{len(chunks)}: {e!s}")
                stop_reason = str(e)
                unsaved = unsaved or cost > 0
                break

            # Update glossary if new terms were discovered
            if response.glossary_updates and glossary_manager:
                glossary_manager.update_glossary(response.glossary_updates)

            translated_chunks.append(response.text)
            if review_pairs is not None:
                review_pairs.append(
                    ReviewPair(
                        unit_id=i + 1,
                        source=current_chunk,
                        translation=response.text,
                    )
                )
            total_tokens += response.tokens_used
            total_cost += response.cost
            input_total += response.input_tokens
            output_total += response.output_tokens
            unsaved = True

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1, total_cost=total_cost)

            # Save checkpoint if needed
            if (
                checkpoint_manager
                and config.checkpoint_dir
                and (i + 1) % config.checkpoint_frequency == 0
            ):
                logger.debug(f"Saving checkpoint after chunk {i + 1}")
                await _save_chunks(
                    checkpoint_manager,
                    config,
                    "context-aware",
                    translated_chunks,
                    (total_tokens, input_total, output_total, total_cost),
                    start_time,
                    glossary_manager,
                )
                unsaved = False

            if (
                config.max_cost is not None
                and total_cost > config.max_cost
                and i + 1 < len(chunks)
            ):
                stop_reason = _cost_cap_message(total_cost, config.max_cost)
                logger.error(f"Stopping after chunk {i + 1}: {stop_reason}")
                break

            # Update context for next iteration
            previous_chunk = current_chunk
            previous_translation = response.text

        # Save the stopping point so a rerun continues from the next chunk
        if stop_reason and unsaved and checkpoint_manager and config.checkpoint_dir:
            await _save_chunks(
                checkpoint_manager,
                config,
                "context-aware",
                translated_chunks,
                (total_tokens, input_total, output_total, total_cost),
                start_time,
                glossary_manager,
            )

        warnings: list[str] = []
        failed_chunks: list[int] = []
        if stop_reason:
            final_text, failed_chunks, warning = _stopped_early(
                translated_chunks, len(chunks), stop_reason
            )
            warnings.append(warning)
            # A review bundle must cover the whole source
            review_pairs = None
        else:
            # Direct concatenation (no complex merging needed)
            final_text = "".join(translated_chunks)
            if config.max_cost is not None and total_cost > config.max_cost:
                warnings.append(_cost_cap_message(total_cost, config.max_cost))
        logger.info(f"Final translated text length: {len(final_text)} characters")

        time_taken = (datetime.now() - start_time).total_seconds()

        return TranslationResponse(
            text=final_text,
            tokens_used=total_tokens,
            cost=total_cost,
            time_taken=time_taken,
            input_tokens=input_total,
            output_tokens=output_total,
            failed_chunks=failed_chunks,
            warnings=warnings,
            review_pairs=review_pairs,
        )

    except Exception as e:
        logger.error(f"Error in context-aware translation: {e!s}")
        raise TranslationError(f"Translation failed: {e!s}", cost=total_cost) from e
