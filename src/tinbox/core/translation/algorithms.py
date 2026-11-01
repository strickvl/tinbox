"""Translation algorithms for Tinbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.progress import Progress, TaskID

from tinbox.core.processor import DocumentContent
from tinbox.core.review import ReviewPair
from tinbox.core.translation.checkpoint import (
    CheckpointManager,
    ResumeResult,
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
from tinbox.core.types import (
    DEFAULT_CONTEXT_SIZE,
    TranslationConfig,
    TranslationProtocol,
)
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


def _billed(error: Exception) -> tuple[float, int, int]:
    """Return the (cost, input tokens, output tokens) a provider billed for a failed call."""
    return (
        getattr(error, "cost", 0.0),
        getattr(error, "input_tokens", 0),
        getattr(error, "output_tokens", 0),
    )


def _over_cap(config: TranslationConfig, total_cost: float) -> str | None:
    """Return why the run must stop when recorded cost already exceeds --max-cost."""
    if config.max_cost is None or total_cost <= config.max_cost:
        return None
    return (
        f"Translation cost of ${total_cost:.4f} exceeded maximum cost of "
        f"${config.max_cost:.2f}"
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


@dataclass
class _Run:
    """The bookkeeping every algorithm shares for one translation run.

    It holds the running token and cost totals, builds and sends each request,
    updates the glossary and progress bar, and writes checkpoints. Each
    algorithm keeps its own loop, its own rule for what a failed request means,
    and its own way of assembling the output.
    """

    config: TranslationConfig
    translator: ModelInterface
    progress: Progress | None
    checkpoint_manager: CheckpointManager | None
    glossary_manager: GlossaryManager | None
    start_time: datetime = field(default_factory=datetime.now)
    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    # True when the totals or translations changed since the last checkpoint
    unsaved: bool = False
    task_id: TaskID | None = None

    def add_task(self, description: str, total: int, **fields: Any) -> None:
        if self.progress:
            self.task_id = self.progress.add_task(description, total=total, **fields)

    def resume(self, result: ResumeResult) -> None:
        """Carry the checkpoint's token and cost totals into this run."""
        self.tokens, self.cost = result.total_tokens, result.total_cost
        self.input_tokens = result.total_input_tokens
        self.output_tokens = result.total_output_tokens

    def restore_glossary(self, result: ResumeResult) -> None:
        if (
            self.config.use_glossary
            and self.glossary_manager
            and result.glossary_entries
        ):
            self.glossary_manager.restore_from_checkpoint(result.glossary_entries)

    def over_cap(self) -> str | None:
        return _over_cap(self.config, self.cost)

    def request(
        self, content: str | bytes, *, context: str | None, content_type: str
    ) -> TranslationRequest:
        config = self.config
        return TranslationRequest(
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            content=content,
            context=context,
            content_type=content_type,
            model=config.model,
            translation_protocol=config.translation_protocol,
            local_model=config.local_model,
            model_params={"model_name": config.model_name} if config.model_name else {},
            glossary=self.glossary_manager.get_current_glossary()
            if config.use_glossary and self.glossary_manager
            else None,
            reasoning_effort=config.reasoning_effort,
            price_in=config.price_in,
            price_out=config.price_out,
        )

    def _add_usage(self, cost: float, input_tokens: int, output_tokens: int) -> None:
        self.cost += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    async def send(self, request: TranslationRequest) -> TranslationResponse:
        """Translate one request; a failure adds what the provider billed, then re-raises.

        The caller decides what the failure means, then passes a successful
        response to ``record``.
        """
        try:
            return await self.translator.translate(request)
        except Exception as e:
            cost, input_tokens, output_tokens = _billed(e)
            self._add_usage(cost, input_tokens, output_tokens)
            self.tokens += input_tokens + output_tokens
            self.unsaved = self.unsaved or any((cost, input_tokens, output_tokens))
            raise

    def record(self, response: TranslationResponse) -> None:
        """Apply a successful response to the glossary, totals and progress bar."""
        if response.glossary_updates and self.glossary_manager:
            self.glossary_manager.update_glossary(response.glossary_updates)
        self._add_usage(response.cost, response.input_tokens, response.output_tokens)
        self.tokens += response.tokens_used
        self.unsaved = True
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, advance=1, total_cost=self.cost)

    def show_resumed(self, completed: int) -> None:
        if self.progress and self.task_id is not None:
            self.progress.update(
                self.task_id, completed=completed, total_cost=self.cost
            )

    @property
    def checkpoints_on(self) -> bool:
        return bool(self.checkpoint_manager and self.config.checkpoint_dir)

    def checkpoint_due(self, number: int) -> bool:
        """Whether the 1-based page or chunk ``number`` ends a checkpoint interval."""
        return self.checkpoints_on and number % self.config.checkpoint_frequency == 0

    async def _save(
        self,
        algorithm: str,
        translated: dict[int, str],
        *,
        completed_pages: list[int],
        failed_pages: list[int],
    ) -> None:
        assert self.checkpoint_manager is not None
        await self.checkpoint_manager.save(
            TranslationState(
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                algorithm=algorithm,
                completed_pages=completed_pages,
                failed_pages=failed_pages,
                translated_chunks=translated,
                token_usage=self.tokens,
                cost=self.cost,
                time_taken=(datetime.now() - self.start_time).total_seconds(),
                glossary_entries=self.glossary_manager.get_current_glossary().entries
                if self.glossary_manager
                else {},
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            )
        )
        self.unsaved = False

    async def save_pages(self, translated: dict[int, str], failed: list[int]) -> None:
        await self._save(
            "page",
            translated.copy(),
            completed_pages=sorted(translated),
            failed_pages=failed,
        )

    async def save_chunks(self, algorithm: str, translated: list[str]) -> None:
        # Chunked algorithms treat the whole document as one page
        await self._save(
            algorithm,
            {j + 1: text for j, text in enumerate(translated)},
            completed_pages=[1],
            failed_pages=[],
        )

    def response(
        self,
        text: str,
        *,
        failed_pages: list[int] | None = None,
        page_errors: dict[int, str] | None = None,
        failed_chunks: list[int] | None = None,
        warnings: list[str] | None = None,
        review_pairs: list[ReviewPair] | None = None,
    ) -> TranslationResponse:
        return TranslationResponse(
            text=text,
            tokens_used=self.tokens,
            cost=self.cost,
            time_taken=(datetime.now() - self.start_time).total_seconds(),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            failed_pages=failed_pages or [],
            page_errors=page_errors or {},
            failed_chunks=failed_chunks or [],
            warnings=warnings or [],
            review_pairs=review_pairs,
        )

    async def finish_chunks(
        self,
        algorithm: str,
        translated: list[str],
        total: int,
        stop_reason: str | None,
        review_pairs: list[ReviewPair] | None = None,
    ) -> TranslationResponse:
        """Assemble a sliding-window or context-aware result.

        A run that stopped early saves its stopping point, so a rerun continues
        from the next chunk, and returns the partial text with a failure marker
        and no review pairs, because a review bundle must cover the whole source.
        """
        if stop_reason:
            if self.unsaved and self.checkpoints_on:
                await self.save_chunks(algorithm, translated)
            text, failed_chunks, warning = _stopped_early(
                translated, total, stop_reason
            )
            return self.response(text, failed_chunks=failed_chunks, warnings=[warning])
        # Chunks do not overlap, so their translations join directly
        warnings = [reason] if (reason := self.over_cap()) else []
        return self.response(
            "".join(translated), warnings=warnings, review_pairs=review_pairs
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
    run = _Run(config, translator, progress, checkpoint_manager, glossary_manager)
    # Use dict to track translations by actual page number (1-indexed)
    translated_pages_by_num: dict[int, str] = {}

    try:
        run.add_task("Translating pages...", total=len(content.pages))

        # Check for checkpoint and resume if available
        resume_result = await resume_from_checkpoint(checkpoint_manager, config)
        run.resume(resume_result)

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

        if resume_result.resumed:
            run.show_resumed(len(translated_pages_by_num))
        run.restore_glossary(resume_result)

        # Track failed pages with their error messages
        failed_pages: list[int] = []
        page_errors: dict[int, str] = {}

        # If everything already translated via checkpoint, short-circuit
        if len(translated_pages_by_num) == len(content.pages):
            return run.response(
                "\n\n".join(
                    translated_pages_by_num[p] for p in sorted(translated_pages_by_num)
                )
            )

        # Translate pages - iterate by actual page number (1-indexed)
        for page_num, page in enumerate(content.pages, start=1):
            # Skip already translated pages
            if page_num in translated_pages_by_num:
                continue

            # Checked before sending, so a resumed run already over the cap spends nothing
            if reason := run.over_cap():
                _skip_remaining_pages(
                    page_num - 1,
                    len(content.pages),
                    translated_pages_by_num,
                    failed_pages,
                    page_errors,
                    reason,
                )
                logger.error(f"Stopping before page {page_num}: {reason}")
                break

            try:
                # No context for the page-by-page algorithm
                request = run.request(
                    page, context=None, content_type=content.content_type
                )
                response = await run.send(request)
                run.record(response)
                # Store with actual page number as key
                translated_pages_by_num[page_num] = response.text
                if run.checkpoint_due(page_num):
                    await run.save_pages(translated_pages_by_num, failed_pages)

            except Exception as e:
                error_msg = str(e)
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

        if not translated_pages_by_num:
            if failed_pages:
                raise TranslationError(
                    f"Translation failed: No pages were successfully translated. Failed pages: {failed_pages}"
                )
            else:
                raise TranslationError(
                    "Translation failed: No pages were successfully translated"
                )

        # Save the pages the checkpoint frequency skipped, so a rerun retries
        # only the failed pages
        if failed_pages and run.unsaved and run.checkpoints_on:
            await run.save_pages(translated_pages_by_num, failed_pages)

        # Build warnings list
        warnings: list[str] = []
        if reason := run.over_cap():
            warnings.append(reason)
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

        return run.response(
            "\n\n".join(page_texts),
            failed_pages=failed_pages,
            page_errors=page_errors,
            warnings=warnings,
        )

    except Exception as e:
        raise TranslationError(f"Translation failed: {e!s}") from e


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

    run = _Run(config, translator, progress, checkpoint_manager, glossary_manager)

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
        translated_windows = resume_result.translated_items
        stop_reason: str | None = None

        run.resume(resume_result)
        run.add_task(
            "Translating windows...",
            total=len(windows),
            completed=len(translated_windows),
            total_cost=run.cost,
        )
        run.restore_glossary(resume_result)

        # Translate remaining windows
        for i, (preceding_text, window) in enumerate(
            windows[len(translated_windows) :], len(translated_windows)
        ):
            # Checked before sending, so a resumed run already over the cap spends nothing
            if stop_reason := run.over_cap():
                logger.error(f"Stopping before window {i + 1}: {stop_reason}")
                break
            logger.debug(f"Translating window {i + 1}: {window}")
            request = run.request(
                window,
                context=build_overlap_context(preceding_text),
                content_type="text/plain",
            )
            try:
                response = await run.send(request)
            except Exception as e:
                if not translated_windows:
                    raise
                # Keep the paid-for windows instead of discarding the whole run
                logger.error(f"Stopping at window {i + 1}: {e!s}")
                stop_reason = str(e)
                break

            run.record(response)
            translated_windows.append(response.text)
            if run.checkpoint_due(i + 1):
                logger.debug(f"Saving checkpoint after window {i + 1}")
                await run.save_chunks("sliding-window", translated_windows)

        result = await run.finish_chunks(
            "sliding-window", translated_windows, len(windows), stop_reason
        )
        logger.debug(f"Final merged text: {result.text}")
        return result

    except Exception as e:
        logger.error(f"Error in sliding window translation: {e!s}")
        raise TranslationError(f"Translation failed: {e!s}") from e


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

    run = _Run(config, translator, progress, checkpoint_manager, glossary_manager)

    try:
        # Join all pages into single text
        text = "\n\n".join(text_pages)
        logger.info(f"Combined text length: {len(text)} characters")

        # Use configured context size with fallback
        context_size = (
            config.context_size
            if config.context_size is not None
            else DEFAULT_CONTEXT_SIZE
        )
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

        translated_chunks = resume_result.translated_items
        stop_reason: str | None = None
        review_pairs: list[ReviewPair] | None = (
            [] if config.preserve_review_pairs else None
        )
        previous_chunk: str | None = resume_result.metadata.get("previous_chunk")
        previous_translation: str | None = resume_result.metadata.get(
            "previous_translation"
        )

        run.resume(resume_result)
        run.add_task(
            "Translating chunks...",
            total=len(chunks),
            completed=len(translated_chunks),
            total_cost=run.cost,
        )
        run.restore_glossary(resume_result)

        # Translate remaining chunks with context
        for i, current_chunk in enumerate(
            chunks[len(translated_chunks) :], len(translated_chunks)
        ):
            if stop_reason := run.over_cap():
                logger.error(f"Stopping before chunk {i + 1}: {stop_reason}")
                break
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

            # Content and context travel separately
            request = run.request(
                current_chunk, context=context_info, content_type="text/plain"
            )
            try:
                response = await run.send(request)
            except Exception as e:
                if not translated_chunks:
                    raise
                # Keep the paid-for chunks instead of discarding the whole run
                logger.error(f"Stopping at chunk {i + 1}/{len(chunks)}: {e!s}")
                stop_reason = str(e)
                break

            run.record(response)
            translated_chunks.append(response.text)
            if review_pairs is not None:
                review_pairs.append(
                    ReviewPair(
                        unit_id=i + 1,
                        source=current_chunk,
                        translation=response.text,
                    )
                )
            if run.checkpoint_due(i + 1):
                logger.debug(f"Saving checkpoint after chunk {i + 1}")
                await run.save_chunks("context-aware", translated_chunks)

            # Update context for next iteration
            previous_chunk = current_chunk
            previous_translation = response.text

        result = await run.finish_chunks(
            "context-aware",
            translated_chunks,
            len(chunks),
            stop_reason,
            review_pairs,
        )
        logger.info(f"Final translated text length: {len(result.text)} characters")
        return result

    except Exception as e:
        logger.error(f"Error in context-aware translation: {e!s}")
        raise TranslationError(f"Translation failed: {e!s}") from e
