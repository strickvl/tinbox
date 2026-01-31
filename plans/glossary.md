# Glossary Translation Feature Design

## Overview

This document outlines the design for implementing a Glossary feature for Tinbox that maintains consistent translation of individual terms across all translation algorithms. The system will maintain a simple glossary throughout the translation process, where each page/window/chunk (depending on algorithm) provides the glossary as context and can extend it with newly discovered important terms. The glossary will be integrated into the checkpoint system for resumability and support loading/saving from external files.

## Current State

The current translation system in Tinbox processes documents using three algorithms (page-by-page, sliding-window, and context-aware) without any mechanism for maintaining consistent terminology across chunks. This leads to several translation quality issues:

### Terminology Inconsistency Problem

In `src/tinbox/core/translation/algorithms.py`:

```python
# Current translation process for all algorithms
for chunk in chunks:
    request = TranslationRequest(
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        content=chunk,
        context=None,  # Limited context, no glossary
        content_type=content.content_type,
        model=config.model,
        model_params={"model_name": config.model_name}
    )

    response = await translator.translate(request)
    translated_chunks.append(response.text)  # No term consistency tracking
```

Issues with current approach:

- **Inconsistent terminology**: Same technical terms translated differently across chunks
- **No term learning**: Important domain-specific vocabulary not captured for reuse
- **Context limitations**: Only previous chunk context available (context-aware only)
- **No persistence**: No way to maintain terminology across translation sessions

### Limited Context Structure

In `src/tinbox/core/translation/litellm.py`:

```python
def _create_prompt(self, request: TranslationRequest) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": f"You are a professional translator..."
        }
    ]

    # Add context if provided (only for context-aware algorithm)
    if request.context:
        messages.append({
            "role": "user",
            "content": f"[TRANSLATION_CONTEXT]{request.context}[/TRANSLATION_CONTEXT]"
        })

    # No glossary context available
    messages.append({"role": "user", "content": request.content})
```

Problems:

- **No terminology guidance**: LLM has no access to previously established translations
- **Single response format**: Only returns translated text, no term extraction
- **Algorithm-specific context**: Context only available in context-aware algorithm
- **No structured output**: Cannot capture newly discovered terms

### Missing Checkpoint Integration

In `src/tinbox/core/translation/checkpoint.py`:

```python
@dataclass
class TranslationState:
    source_lang: str
    target_lang: str
    algorithm: str
    completed_pages: list[int]
    failed_pages: list[int]
    translated_chunks: dict[int, str]
    token_usage: int
    cost: float
    time_taken: float
    # No glossary state preservation
```

Limitations:

- **No glossary persistence**: Glossary state lost when resuming from checkpoint
- **Incomplete state**: Cannot fully restore translation context including learned terms
- **No term continuity**: Resumed translations start without previously learned terminology

## Proposed Solution

### 1. Architecture Decision

Implement a **Glossary system** that integrates with all translation algorithms to maintain consistent terminology. This approach:

- **Maintains term consistency**: Same source terms always translated to same target terms
- **Learns from context**: Extracts important terms during translation process
- **Integrates with all algorithms**: Works with page-by-page, sliding-window, and context-aware
- **Persists across sessions**: Saves/loads glossaries from files and checkpoints
- **Uses structured output**: LiteLLM's response_format for reliable term extraction

### 2. Core Glossary Design

#### Glossary Data Structure

```python
# New glossary types in src/tinbox/core/types.py
class GlossaryEntry(BaseModel):
    """A single glossary entry mapping source term to target translation."""

    term: str = Field(description="Term in source language")
    translation: str = Field(description="Translation in target language")

    model_config = ConfigDict(frozen=True)


class Glossary(BaseModel):
    """Collection of translation glossary entries."""

    entries: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from source terms to target translations"
    )

    def extend(self, new_entries: List[GlossaryEntry]) -> "Glossary":
        """Extend glossary with multiple entries (returns new immutable instance)."""
        updated_entries = self.entries.copy()
        for entry in new_entries:
            updated_entries[entry.term] = entry.translation
        return Glossary(entries=updated_entries)

    def to_context_string(self) -> str:
        """Convert glossary to context string for LLM."""
        if not self.entries:
            return ""

        lines = ["[GLOSSARY]"]
        for term, translation in self.entries.items():
            lines.append(f"{term} -> {translation}")
        lines.append("[/GLOSSARY]")
        return "\n".join(lines)

    model_config = ConfigDict(frozen=True)
```

#### Enhanced Translation Interface

```python
# Enhanced TranslationRequest to include optional glossary
class TranslationRequest(BaseModel):
    """Configuration for a translation request with optional glossary support."""

    source_lang: str
    target_lang: str
    content: Union[str, bytes]  # Pure content to translate (text or image bytes)
    context: Optional[str] = None  # Supporting context information for better translation
    content_type: str = Field(pattern=r"^(text|image)/.+$")
    model: ModelType
    model_params: dict = Field(default_factory=dict)  # Additional model-specific parameters
    glossary: Optional[Glossary] = Field(default=None, description="Optional glossary for consistent translations")

    model_config = ConfigDict(frozen=True, protected_namespaces=())


# Structured response format for LLM with glossary extension
class TranslationWithGlossaryResponse(BaseModel):
    """Structured response format for LLM with optional glossary extension."""

    translation: str = Field(description="The translated text")
    glossary_extension: Optional[List[GlossaryEntry]] = Field(
        default=None,
        description="New glossary entries discovered during translation (optional)"
    )

    model_config = ConfigDict(frozen=True)

# Structured response format for LLM without glossary extension
class TranslationWithoutGlossaryResponse(BaseModel):
    """Structured response format for LLM without glossary extension."""

    translation: str = Field(description="The translated text")

    model_config = ConfigDict(frozen=True)


# Enhanced TranslationResponse to include glossary updates
class TranslationResponse(BaseModel):
    """Response from a translation request with optional glossary updates."""

    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)
    glossary_updates: List[GlossaryEntry] = Field(
        default_factory=list,
        description="New glossary entries discovered during this translation"
    )

    model_config = ConfigDict(frozen=True)
```

### 3. Key Components

#### Glossary Manager

```python
# New file: src/tinbox/core/translation/glossary.py
class GlossaryManager:
    """Manages glossary state and persistence during translation."""

    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self.current_glossary = config.initial_glossary or Glossary()

    def get_current_glossary(self) -> Glossary:
        """Get the current glossary state."""
        return self.current_glossary

    def update_glossary(self, new_entries: List[GlossaryEntry]) -> None:
        """Update the current glossary with new entries."""
        if new_entries:
            self.current_glossary = self.current_glossary.extend(new_entries)

    def save_to_file(self, file_path: Path) -> None:
        """Save current glossary to a JSON file."""
        # Implementation details...

    @classmethod
    def load_from_file(cls, file_path: Path, config: TranslationConfig) -> "GlossaryManager":
        """Load glossary from a JSON file."""
        # Implementation details...
```

#### Enhanced LiteLLM Integration

```python
# Extensions to src/tinbox/core/translation/litellm.py
class LiteLLMTranslator(ModelInterface):

    def _create_prompt(self, request: TranslationRequest) -> list[dict]:
        """Enhanced prompt creation with glossary support from request."""

        # Keep the original system prompt mostly unchanged
        messages = [
            {
                "role": "system",
                "content": (
                    # ... (as before)
                ),
            }
        ]

        # Add glossary context if available
        if request.glossary and request.glossary.entries:
            glossary_context = (
                f"Use this glossary for consistent translations:\n"
                f"{request.glossary.to_context_string()}\n\n"
                f"When you encounter these terms, use the provided translations exactly. "
                f"If you encounter new important terms that would benefit from consistent translation "
                f"(technical terms, proper nouns, domain vocabulary), include them in the glossary_extension field."
            )
            messages.append({"role": "user", "content": glossary_context})

        # ... rest of the code ...

        return messages

    @completion_with_retry
    async def _make_completion_request(
        self, request: TranslationRequest, stream: bool = False
    ):
        """Make a completion request with retry logic for rate limits.

        Args:
            request: The translation request
            stream: Whether to stream the response

        Returns:
            The completion response

        Raises:
            TranslationError: If translation fails after retries
        """
        try:
            return completion(
                model=self._get_model_string(request),
                messages=self._create_prompt(request),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
                response_format=TranslationWithGlossaryResponse if request.glossary is not None else TranslationWithoutGlossaryResponse,
                drop_params=True,
                **{k: v for k, v in request.model_params.items() if k != "model_name"},
            )
        except RateLimitError as e:
            # This will be caught by the retry decorator
            raise
        except Exception as e:
            raise TranslationError(f"Translation failed: {str(e)}") from e


    async def translate(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        # Most of the translate() method stays exactly the same, with these minimal changes:

        # 1. Add glossary field to clean_request creation (line ~244):
        clean_request = TranslationRequest(
            # ... existing fields ...
            glossary=request.glossary,  # Add this line
        )

        # 2. REPLACE text extraction logic (line ~294):
        # OLD: text = response.choices[0].message.content

        # NEW: Extract text and glossary_updates from structured response
        # ... existing code ...
        if not hasattr(response.choices[0], "message") or not hasattr(
            response.choices[0].message, "parsed"
        ):
            raise TranslationError("Invalid response format")

        parsed = response.choices[0].message.parsed

        if not hasattr(parsed, 'translation'):
            raise TranslationError("Missing translation in response")

        text = parsed.translation

        glossary_updates = []

        if hasattr(parsed, 'glossary_extension') and parsed.glossary_extension:
            glossary_updates = parsed.glossary_extension

        # ... existing code ...

        # 3. Add glossary_updates to return statement (line ~322):
        return TranslationResponse(
            # ... existing fields ...
            glossary_updates=glossary_updates,  # Add this line
        )
```

#### Enhanced Checkpoint Integration

```python
# Updates to src/tinbox/core/translation/checkpoint.py
from dataclasses import dataclass, field

@dataclass
class TranslationState:
    """State of a translation task for checkpointing with glossary support."""

    source_lang: str
    target_lang: str
    algorithm: str
    completed_pages: list[int]
    failed_pages: list[int]
    translated_chunks: dict[int, str]
    token_usage: int
    cost: float
    time_taken: float
    # New glossary state
    glossary_entries: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResumeResult:
    """Result of attempting to resume from checkpoint with glossary support."""

    resumed: bool
    translated_items: List[str]
    total_tokens: int
    total_cost: float
    metadata: Dict[str, Any]
    # New glossary state
    glossary_entries: Dict[str, str] = field(default_factory=dict)


class CheckpointManager:
    """Enhanced checkpoint manager with glossary support."""

    async def save(self, state: TranslationState) -> None:
        """Enhanced save with glossary state included in TranslationState."""
        try:
            checkpoint_path = self._get_checkpoint_path()
            checkpoint_data = {
                "source_lang": state.source_lang,
                "target_lang": state.target_lang,
                "algorithm": state.algorithm,
                "completed_pages": state.completed_pages,
                "failed_pages": state.failed_pages,
                "translated_chunks": state.translated_chunks,
                "token_usage": state.token_usage,
                "cost": state.cost,
                "time_taken": state.time_taken,
                "glossary_entries": state.glossary_entries,  # New field
                "config": {
                    "source_lang": self.config.source_lang,
                    "target_lang": self.config.target_lang,
                    "model": self.config.model.value,
                    "algorithm": self.config.algorithm,
                },
            }

            # Write checkpoint atomically
            temp_path = checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.rename(checkpoint_path)

            self._logger.debug(
                f"Saved checkpoint with glossary to {checkpoint_path}",
                pages=len(state.completed_pages) + len(state.failed_pages),
                tokens=state.token_usage,
                cost=state.cost,
                glossary_terms=len(state.glossary_entries)
            )

        except Exception as e:
            self._logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    async def load(self) -> Optional[TranslationState]:
        """Enhanced load with glossary state restoration."""
        try:
            checkpoint_path = self._get_checkpoint_path()
            if not checkpoint_path.exists():
                return None

            with open(checkpoint_path) as f:
                data = json.load(f)

            # Validate checkpoint matches current config (existing validation)
            config = data.get("config", {})
            if (
                config.get("source_lang") != self.config.source_lang
                or config.get("target_lang") != self.config.target_lang
                or config.get("model") != self.config.model.value
                or config.get("algorithm") != self.config.algorithm
            ):
                self._logger.warning("Checkpoint configuration mismatch")
                return None

            # Convert string keys back to integers for translated_chunks
            translated_chunks = {}
            for key, value in data["translated_chunks"].items():
                try:
                    int_key = int(key)
                    translated_chunks[int_key] = value
                except (ValueError, TypeError):
                    translated_chunks[key] = value

            # Load glossary entries (with backward compatibility)
            glossary_entries = data.get("glossary_entries", {})

            state = TranslationState(
                source_lang=data["source_lang"],
                target_lang=data["target_lang"],
                algorithm=data["algorithm"],
                completed_pages=data["completed_pages"],
                failed_pages=data["failed_pages"],
                translated_chunks=translated_chunks,
                token_usage=data["token_usage"],
                cost=data["cost"],
                time_taken=data["time_taken"],
                glossary_entries=glossary_entries,  # New field
            )

            self._logger.debug(
                f"Loaded checkpoint with glossary from {checkpoint_path}",
                pages=len(state.completed_pages) + len(state.failed_pages),
                tokens=state.token_usage,
                cost=state.cost,
                glossary_terms=len(state.glossary_entries)
            )

            return state

        except Exception as e:
            self._logger.error(f"Failed to load checkpoint: {str(e)}")
            return None


# Enhanced resume function with glossary support
async def resume_from_checkpoint(
    checkpoint_manager: Optional[CheckpointManager],
    config: TranslationConfig,
    chunks: Optional[List[str]] = None,
) -> ResumeResult:
    """Enhanced resume function with glossary state restoration."""

    if not checkpoint_manager or not config.resume_from_checkpoint:
        return ResumeResult(
            resumed=False,
            translated_items=[],
            total_tokens=0,
            total_cost=0.0,
            metadata={},
            glossary_entries={}
        )

    logger.info("Checking for checkpoint")
    checkpoint = await checkpoint_manager.load()

    if not checkpoint or not checkpoint.translated_chunks:
        return ResumeResult(
            resumed=False,
            translated_items=[],
            total_tokens=0,
            total_cost=0.0,
            metadata={},
            glossary_entries={}
        )

    logger.info("Found valid checkpoint with glossary, resuming from saved state")

    # Load existing translated items in order
    translated_items = [
        checkpoint.translated_chunks[i]
        for i in range(1, len(checkpoint.translated_chunks) + 1)
        if i in checkpoint.translated_chunks
    ]

    # Prepare algorithm-specific metadata (no longer includes glossary)
    metadata = {}

    # For context-aware algorithm, set up context from the last completed chunk
    if config.algorithm == "context-aware" and chunks and translated_items:
        chunk_index = len(translated_items) - 1
        if chunk_index < len(chunks):
            metadata["previous_chunk"] = chunks[chunk_index]
            metadata["previous_translation"] = translated_items[-1]

    result = ResumeResult(
        resumed=True,
        translated_items=translated_items,
        total_tokens=checkpoint.token_usage,
        total_cost=checkpoint.cost,
        metadata=metadata,
        glossary_entries=checkpoint.glossary_entries  # Direct glossary field
    )

    logger.info(f"Resumed with {len(translated_items)} completed items and {len(checkpoint.glossary_entries)} glossary terms")
    return result
```

### 4. Implementation Details

#### Algorithm Integration

```python
# Updates to all algorithms in src/tinbox/core/translation/algorithms.py

async def translate_page_by_page(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    glossary_manager: Optional[GlossaryManager] = None,  # New parameter
) -> TranslationResponse:
    """Enhanced page-by-page translation with glossary support."""

    # Check for checkpoint and resume if available
    resume_result = await resume_from_checkpoint(checkpoint_manager, config)
    total_tokens = resume_result.total_tokens
    total_cost = resume_result.total_cost
    if progress and task_id is not None and resume_result.resumed:
        progress.update(task_id, completed=len(resume_result.translated_items))

    translated_pages = resume_result.translated_items

    # Initialize glossary if enabled
    current_glossary = None
    if config.use_glossary and glossary_manager:
        # Restore glossary from checkpoint if available
        if resume_result.glossary_entries:
            from tinbox.core.types import Glossary
            restored_glossary = Glossary(entries=resume_result.glossary_entries)
            glossary_manager.current_glossary = restored_glossary
        current_glossary = glossary_manager.get_current_glossary()

    # Translate remaining pages
    for i, page in enumerate(content.pages[len(translated_pages):], len(translated_pages)):
        # Create translation request with glossary
        request = TranslationRequest(
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            content=page,
            context=None,  # No context for page-by-page algorithm
            content_type=content.content_type,
            model=config.model,
            model_params={"model_name": config.model_name} if config.model_name else {},
            glossary=current_glossary,  # Include glossary in request
        )

        # Translate (glossary automatically handled by request)
        response = await translator.translate(request)

        # Update glossary with new terms if any were discovered
        if response.glossary_updates and glossary_manager:
            glossary_manager.update_glossary(response.glossary_updates)
            current_glossary = glossary_manager.get_current_glossary()

        translated_pages.append(response.text)
        total_tokens += response.tokens_used
        total_cost += response.cost

        # Save checkpoint with glossary state
        if checkpoint_manager and (i + 1) % config.checkpoint_frequency == 0:
            state = TranslationState(
                # ... existing fields ...
                glossary_entries=current_glossary.entries if current_glossary else {},
            )
            await checkpoint_manager.save(state)
```

#### CLI Integration

```python
# Updates to src/tinbox/cli.py
@app.command()
def translate(
    # ... existing parameters ...
    use_glossary: bool = typer.Option(
        False,
        "--glossary",
        help="Enable glossary for consistent term translations.",
    ),
    glossary_file: Optional[Path] = typer.Option(
        None,
        "--glossary-file",
        help="Path to existing glossary file (JSON format).",
    ),
    save_glossary: Optional[Path] = typer.Option(
        None,
        "--save-glossary",
        help="Path to save the updated glossary after translation.",
    ),
) -> None:
    """Enhanced translate command with glossary support."""

    # Initialize glossary manager
    glossary_manager = None
    if use_glossary:
        if glossary_file and glossary_file.exists():
            glossary_manager = GlossaryManager.load_from_file(glossary_file, config)
        else:
            # Create updated config with glossary enabled
            config = config.model_copy(update={"use_glossary": True})
            glossary_manager = GlossaryManager(config)

    # Run translation with glossary support
    response = asyncio.run(
        translate_document(
            content=content,
            config=config,
            translator=translator,
            progress=progress,
            checkpoint_manager=checkpoint_manager,
            glossary_manager=glossary_manager,
        )
    )

    # Save glossary if requested
    if glossary_manager and save_glossary:
        glossary_manager.save_to_file(save_glossary)
```

## Implementation Steps

### Phase 1: Core Data Structures and Types

#### 1. Update TranslationConfig Type Definitions

**File**: `src/tinbox/core/types.py`

- Add `GlossaryEntry` and `Glossary` classes with proper Pydantic validation
- Add glossary-related fields to `TranslationConfig`:
  - `use_glossary: bool = Field(default=False)`
  - `initial_glossary: Optional[Glossary] = Field(default=None)`
- Ensure immutability and proper serialization support
- Add comprehensive field descriptions and validation rules

#### 2. Enhance Translation Interface

**File**: `src/tinbox/core/translation/interface.py`

- Add `glossary` field to `TranslationRequest` class (optional)
- Add `TranslationWithGlossaryResponse` class for structured LLM output with optional glossary_extension
- Update `TranslationResponse` to include `glossary_updates` field (always present, defaults to empty list)
- Maintain backward compatibility with existing response format
- Add proper type annotations and documentation

#### 3. Create Glossary Manager

**File**: `src/tinbox/core/translation/glossary.py` (new file)

- Implement `GlossaryManager` class with state management
- Add file I/O methods for loading/saving glossaries
- Implement glossary update and merge logic
- Add comprehensive error handling and logging
- Include validation for glossary file formats

### Phase 2: LiteLLM Integration

#### 4. Enhance LiteLLM Translator

**File**: `src/tinbox/core/translation/litellm.py`

- Update `_create_prompt()` method to add glossary context as user message (keep system prompt unchanged)
- Modify `_make_completion_request()` to conditionally add `response_format` based on glossary presence
- Update `translate()` method with minimal changes:
  - Add `glossary=request.glossary` to `clean_request` creation
  - Extract `glossary_updates` from `response.choices[0].message.parsed.glossary_extension` if available
  - Add `glossary_updates` field to both `TranslationResponse` return statements
- All existing logic, error handling, and patterns remain unchanged

#### 5. Add Response Format Support

**File**: `src/tinbox/core/translation/litellm.py`

- Use LiteLLM's native Pydantic model support for structured output
- Handle `response.choices[0].message.parsed` (structured) response
- Include proper token and cost calculation for structured responses
- Add logging for glossary extraction success/failure rates

### Phase 3: Algorithm Integration

#### 6. Update Page-by-Page Algorithm

**File**: `src/tinbox/core/translation/algorithms.py`

- Add `glossary_manager` parameter to `translate_page_by_page()`
- Create `TranslationRequest` with glossary field populated from current glossary state
- Process `response.glossary_updates` to update glossary manager
- Modify checkpoint saving to include glossary state in `TranslationState`
- Modify checkpoint loading/resume to include glossary state in `ResumeResult`
- Preserve other existing logic

#### 7. Update Sliding-Window Algorithm

**File**: `src/tinbox/core/translation/algorithms.py`

- Add `glossary_manager` parameter to `translate_sliding_window()`
- Create `TranslationRequest` with glossary field populated from current glossary state
- Process `response.glossary_updates` to update glossary manager
- Modify checkpoint saving to include glossary state in `TranslationState`
- Modify checkpoint loading/resume to include glossary state in `ResumeResult`
- Preserve other existing logic

#### 8. Update Context-Aware Algorithm

**File**: `src/tinbox/core/translation/algorithms.py`

- Add `glossary_manager` parameter to `translate_context_aware()`
- Create `TranslationRequest` with glossary field populated from current glossary state
- Process `response.glossary_updates` to update glossary manager
- Modify checkpoint saving to include glossary state in `TranslationState`
- Modify checkpoint loading/resume to include glossary state in `ResumeResult`
- Preserve other existing logic

#### 9. Update Main Translation Router

**File**: `src/tinbox/core/translation/algorithms.py`

- Add `glossary_manager` parameter to `translate_document()`
- Pass glossary manager to all algorithm implementations
- Add algorithm-agnostic glossary initialization
- Maintain backward compatibility for calls without glossary

### Phase 4: Checkpoint System Integration

#### 10. Enhance Checkpoint Data Structure

**File**: `src/tinbox/core/translation/checkpoint.py`

- Add `glossary_entries` field to `TranslationState` dataclass
- Add `glossary_entries` field to `ResumeResult` dataclass (replace metadata approach)
- Update checkpoint serialization to include glossary data
- Modify checkpoint loading to restore glossary state
- Add glossary validation during checkpoint restoration

#### 11. Update Checkpoint Manager

**File**: `src/tinbox/core/translation/checkpoint.py`

- Update `save()` method to include glossary_entries from TranslationState
- Update `load()` method to return glossary state in TranslationState
- Add glossary-specific checkpoint validation
- Update `resume_from_checkpoint()` to return glossary_entries in ResumeResult (not metadata)

### Phase 5: CLI Integration

#### 12. Add CLI Parameters

**File**: `src/tinbox/cli.py`

- Add `--glossary` flag to enable glossary functionality
- Add `--glossary-file` parameter for loading existing glossaries
- Add `--save-glossary` parameter for saving updated glossaries
- Update help text and parameter descriptions
- Add parameter validation and error handling

#### 13. Update Configuration Building

**File**: `src/tinbox/cli.py`

- Modify `TranslationConfig` creation to include glossary settings
- Add glossary file loading logic in CLI command
- Initialize `GlossaryManager` based on CLI parameters
- Update translation execution to pass glossary manager
- Add glossary saving logic after successful translation

#### 14. Enhance Cost Estimation

**File**: `src/tinbox/core/cost.py`

- Add cost estimation for glossary-enabled translations
- Include context overhead from glossary in cost calculations
- Update cost display to show glossary overhead
- Add warnings for high glossary context costs

#### 15. Update Documentation

**File**: `README.md`

**Add to Features section (around line 70):**

```markdown
### 📚 Glossary Support

- **Consistent Terminology**: Maintain consistent translation of technical terms across documents
- **Term Learning**: Automatically discover and reuse domain-specific vocabulary
- **Persistent Glossaries**: Save and load glossaries across translation sessions
- **Algorithm Integration**: Works with all translation algorithms (page-by-page, sliding-window, context-aware)
```

**Add to Command-Line Options table (around line 242):**

```markdown
#### Glossary Options

| Option            | Description                                      | Example                      |
| ----------------- | ------------------------------------------------ | ---------------------------- |
| `--glossary`      | Enable glossary for consistent term translations | `--glossary`                 |
| `--glossary-file` | Path to existing glossary file (JSON format)     | `--glossary-file terms.json` |
| `--save-glossary` | Path to save updated glossary after translation  | `--save-glossary terms.json` |
```

**Add to Advanced Usage section (around line 380):**

````markdown
8. **Glossary Support for Consistent Terminology**

   ```bash
   # Enable glossary with automatic term discovery
   tinbox translate --to es --glossary --save-glossary medical_terms.json medical_document.pdf

   # Load existing glossary and extend it
   tinbox translate --to fr --glossary-file existing_terms.json --save-glossary updated_terms.json technical_doc.pdf

   # Use glossary without saving updates
   tinbox translate --to de --glossary-file company_terms.json document.docx
   ```
````

````

**Add to Tips for Best Results section (around line 216):**
```markdown
6. **For Consistent Terminology**
   - Enable glossary for technical documents: `--glossary`
   - Build domain-specific glossaries: `--save-glossary terms.json`
   - Share glossaries across projects: `--glossary-file shared_terms.json`
   - Glossary works with all algorithms and improves translation consistency
````

**Add new section for Glossary File Format (around line 380):**

````markdown
### Glossary File Format

Glossary files are stored in JSON format:

```json
...document actual structure with an example...
```
````

You can create glossary files manually or let Tinbox generate them automatically during translation.

````

### Phase 6: Testing and Validation

#### 16. Create Comprehensive Test Suite

**New Test Files**:

- `tests/test_core/test_translation/test_glossary.py` (glossary manager tests)
- `tests/test_core/test_translation/test_glossary_integration.py` (algorithm integration tests)

**Test Coverage**:

```python
# New tests in test_glossary.py
def test_glossary_entry_creation()
def test_glossary_extend_functionality()
def test_glossary_to_context_string()
def test_glossary_manager_initialization()
def test_glossary_manager_update()
def test_glossary_file_save_load()
def test_glossary_manager_edge_cases()

# New tests in test_glossary_integration.py
def test_page_by_page_with_glossary()
def test_sliding_window_with_glossary()
def test_context_aware_with_glossary()
def test_glossary_checkpoint_integration()
def test_glossary_resume_functionality()
````

#### 17. Update Existing Test Suites

**Files to Update**:

- `tests/test_core/test_translation/test_algorithms.py`
- `tests/test_core/test_translation/test_checkpoint.py`
- `tests/test_core/test_translation/test_litellm.py`
- `tests/test_cli.py`

**Test Updates**:

```python
# Updates to test_algorithms.py
def test_translate_document_with_glossary_manager()
def test_algorithm_glossary_parameter_handling()

# Updates to test_checkpoint.py
def test_checkpoint_save_with_glossary()
def test_checkpoint_load_with_glossary()
def test_resume_from_checkpoint_with_glossary()

# Updates to test_litellm.py
def test_translate_with_glossary_context()
def test_structured_response_parsing()
def test_glossary_fallback_handling()

# Updates to test_cli.py
def test_cli_glossary_parameters()
def test_glossary_file_loading()
def test_glossary_saving()
```

#### 18. Integration Testing

**Test Scenarios**:

- End-to-end translation with glossary across all algorithms
- Glossary file loading and saving functionality
- Checkpoint resume with glossary state restoration
- Large document translation with glossary growth tracking
- Error handling for corrupted glossary files
- Performance impact assessment with glossary enabled

### Phase 7: Quality Assurance

#### 19. Run Comprehensive Quality Checks

```bash
# Run all tests
pytest
```

All tests should pass.

## Benefits

### Translation Quality Improvements

- **Consistent terminology**: Same technical terms translated identically across entire documents
- **Domain adaptation**: System learns and reuses domain-specific vocabulary
- **Context preservation**: Important terms maintained across all translation algorithms
- **Quality assurance**: Reduced manual post-editing for terminology consistency

### System Reliability

- **Algorithm agnostic**: Works seamlessly with all existing translation algorithms
- **Checkpoint integration**: Glossary state preserved during interruptions and resumptions
- **Robust error handling**: Graceful degradation when glossary features fail
- **Backward compatibility**: Existing workflows continue unchanged when glossary disabled

### User Experience

- **Flexible adoption**: Optional feature that can be enabled per translation
- **Persistent learning**: Glossaries can be saved and reused across projects
- **Transparent operation**: Clear indication of glossary usage and growth
- **Cost awareness**: Accurate cost estimation including glossary overhead

## Migration Strategy

### Backward Compatibility

- **Optional feature**: Glossary functionality disabled by default
- **Existing workflows preserved**: All current CLI usage patterns continue working
- **Configuration compatibility**: Existing config files remain valid
- **Algorithm independence**: Existing algorithms work unchanged without glossary

### Gradual Adoption

- **Project-specific glossaries**: Users can enable glossary for specific translations
- **Glossary sharing**: Teams can share and maintain common glossaries
- **Documentation updates**: Clear guidance on when and how to use glossary features
- **Migration tools**: Utilities for converting existing terminology lists to glossary format

## Implementation Status

✅ **IN PROGRESS** - Implementation underway. Core types, interface, LiteLLM integration, algorithms, checkpointing, CLI, and cost estimation updated. Tests and doc polish pending.

### Implementation Checklist

- [x] **Phase 1: Core Data Structures and Types**
  - [x] Update TranslationConfig type definitions (added `use_glossary`, `initial_glossary`, `Glossary`, `GlossaryEntry`)
  - [x] Enhance translation interface (added `glossary` on `TranslationRequest`, `glossary_updates` on `TranslationResponse`, structured response models)
  - [x] Create glossary manager (`src/tinbox/core/translation/glossary.py` with load/save/update)
- [x] **Phase 2: LiteLLM Integration**
  - [x] Enhance LiteLLM translator (glossary context included in prompt)
  - [x] Add response format support (conditional `response_format` and structured parsing with fallback)
- [x] **Phase 3: Algorithm Integration**
  - [x] Update page-by-page algorithm (passes glossary, processes `glossary_updates`)
  - [x] Update sliding-window algorithm (passes glossary, processes `glossary_updates`)
  - [x] Update context-aware algorithm (passes glossary, processes `glossary_updates`)
  - [x] Update main translation router (added `glossary_manager` parameter)
- [x] **Phase 4: Checkpoint System Integration**
  - [x] Enhance checkpoint data structure (`glossary_entries` on `TranslationState`, `ResumeResult`)
  - [x] Update checkpoint manager (save/load glossary state, resume returns glossary)
- [x] **Phase 5: CLI Integration**
  - [x] Add CLI parameters (`--glossary`, `--glossary-file`, `--save-glossary`)
  - [x] Update configuration building and wire `GlossaryManager`
  - [x] Enhance cost estimation (added glossary overhead in `core/cost.py`)
  - [ ] Update documentation sections below (in progress)
- [ ] **Phase 6: Testing and Validation**
  - [ ] Create comprehensive test suite (new tests for glossary manager and integration)
  - [ ] Update existing test suites to cover glossary paths
  - [ ] Integration testing
- [ ] **Phase 7: Quality Assurance**
  - [ ] Run quality checks
  - [ ] Performance and accuracy validation

## Success Criteria

1. **Translation Quality** 🎯

   - Consistent terminology across all chunks in translated documents
   - Successful glossary extraction from LLM responses (>90% success rate)
   - Improved translation quality metrics for domain-specific documents
   - Zero regression in translation quality when glossary disabled

2. **System Integration** 🎯

   - Seamless integration with all three translation algorithms
   - Checkpoint system preserves and restores glossary state correctly
   - Glossary file I/O operations work reliably across platforms
   - No breaking changes to existing API or CLI interfaces

3. **Performance** 🎯

   - Glossary overhead adds <20% to translation time
   - Memory usage for glossary remains <10MB for typical documents
   - Cost estimation accuracy within 15% for glossary-enabled translations
   - Structured response parsing succeeds >95% of the time

4. **User Experience** 🎯
   - Intuitive CLI parameters for glossary functionality
   - Clear documentation and usage examples
   - Helpful error messages for glossary-related issues
   - Transparent reporting of glossary growth and usage
