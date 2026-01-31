# Context-Aware Translation Algorithm Design

## Overview

This document outlines the design for implementing a new "context-aware" translation algorithm for Tinbox. The system will address the fundamental issues with the current sliding window algorithm by providing translation context without creating overlapping content in the final output. This approach eliminates duplicate content while maintaining high translation quality through intelligent context provision and natural text boundary splitting.

## Current State

The existing sliding window algorithm in `translate_sliding_window` has several critical issues that impact translation quality and reliability:

### Duplicate Content Problem

In `src/tinbox/core/translation/algorithms.py`:

```python
# Current sliding window creates overlapping windows
windows = create_windows(text, window_size, overlap_size)

# Each window contains overlapping content from adjacent windows
# Window 1: [text chars 0-2000]    + [overlap chars 1800-2000]
# Window 2: [overlap chars 1800-2000] + [text chars 2000-4000]

# Translation produces different results for same overlapping content
translated_windows = []
for window in windows:
    response = await translator.translate(request)
    translated_windows.append(response.text)  # Overlapping content translated differently

# Merge attempts to find identical text (rarely works)
final_text = merge_chunks(translated_windows, overlap_size)
```

Issues with current approach:

- **Inconsistent overlap translation**: LLMs translate the same source text differently in different contexts
- **Failed merge detection**: `merge_chunks()` relies on exact string matching between translated overlaps
- **Lost or duplicated content**: When merge fails, content is either duplicated or lost entirely
- **Brittle merge logic**: Falls back to simple concatenation when overlap detection fails

### Poor Text Chunking Strategy

In `create_windows()` function:

```python
def create_windows(text: str, window_size: int, overlap_size: int) -> list[str]:
    # Cuts text at arbitrary character boundaries
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        window = text[start:end]  # May cut sentences/words mid-way
        windows.append(window)
        start = end - min(overlap_size, end - start)
```

Problems:

- **Arbitrary character cuts**: Breaks sentences, paragraphs, and semantic units
- **Poor translation context**: Incomplete sentences provide inadequate context for accurate translation
- **Inconsistent chunk quality**: Some chunks end mid-sentence, others at natural boundaries

### Current Algorithm Configuration

In `src/tinbox/core/types.py`:

```python
class TranslationConfig(BaseModel):
    algorithm: Literal["page", "sliding-window"]  # Limited options

    # Sliding window specific (problematic)
    window_size: int = Field(default=2000, description="Window size for sliding window translation")
    overlap_size: int = Field(default=200, description="Overlap size for sliding window translation")
```

Limitations:

- Only two algorithms available: "page" and "sliding-window"
- No support for custom splitting strategies
- No context-aware translation options
- Fixed overlap strategy that creates the duplicate content problem

## Proposed Solution

### 1. Architecture Decision

Implement a new **"context-aware"** translation algorithm that provides context to the LLM without translating overlapping content. This approach:

- **Eliminates duplicate content**: Only translates the main content, not the context
- **Maintains translation quality**: Provides previous slice + translation + current slice context
- **Uses natural boundaries**: Splits text at paragraphs, sentences, clauses, and words
- **Supports custom splitting**: Optional custom split tokens for precise control
- **Simplifies merging**: Direct concatenation of translated chunks (no separators needed)

### 2. Core Algorithm Design

#### Context-Aware Translation Flow

```python
# New algorithm flow:
# 1. Split text at natural boundaries (respecting context_size target)
chunks = smart_text_split(text, context_size, custom_split_token)

# 2. For each chunk, provide context but only translate the main content
for i, current_chunk in enumerate(chunks):
    context = build_translation_context(
        previous_chunk=chunks[i-1] if i > 0 else None,
        previous_translation=translated_chunks[i-1] if i > 0 else None,
        current_chunk=current_chunk
    )

    # 3. Translate only the current chunk with full context
    translation = await translator.translate(context)
    translated_chunks.append(translation)

# 4. Direct concatenation (no complex merging needed)
final_text = "".join(translated_chunks)
```

#### Context Structure

```python
def build_translation_context(previous_chunk, previous_translation, current_chunk):
    context = f"Translate the following text from {source_lang} to {target_lang}.\n\n"

    if previous_chunk and previous_translation:
        context += f"[PREVIOUS_CHUNK]\n{previous_chunk}\n[/PREVIOUS_CHUNK]\n\n"
        context += f"[PREVIOUS_CHUNK_TRANSLATION]\n{previous_translation}\n[/PREVIOUS_CHUNK_TRANSLATION]\n\n"

    context += f"[TRANSLATE_THIS]\n{current_chunk}\n[/TRANSLATE_THIS]\n\n"
    context += "Only return the translation of the text between [TRANSLATE_THIS] tags. Do not include the tags or any other content."

    return context
```

### 3. Key Components

#### Smart Text Splitting with Natural Boundaries

```python
def smart_text_split(
    text: str,
    target_size: int,
    custom_split_token: Optional[str] = None
) -> List[str]:
    """Split text at natural boundaries or custom tokens.

    Priority order:
    1. Custom split token (if provided) - ignores target_size
    2. Paragraph breaks (\n\n)
    3. Sentence endings ([.!?]\s+)
    4. Line breaks (\n)
    5. Clause boundaries ([;:,]\s+)
    6. Word boundaries (\s+)
    """
```

#### Enhanced Configuration Support

```python
class TranslationConfig(BaseModel):
    # Extended algorithm options
    algorithm: Literal["page", "sliding-window", "context-aware"]

    # Context-aware specific settings
    context_size: Optional[int] = Field(
        default=2000,
        description="Target size for context-aware chunks"
    )
    custom_split_token: Optional[str] = Field(
        default=None,
        description="Custom token to split text on (ignores context_size)"
    )

    # Deprecated but maintained for compatibility
    window_size: int = Field(default=2000)  # Used by sliding-window
    overlap_size: int = Field(default=200)  # Used by sliding-window
```

#### Updated Cost Estimation

```python
def estimate_context_aware_cost(
    estimated_tokens: int,
    model: ModelType,
    context_multiplier: float = 1.8
) -> float:
    """Estimate cost for context-aware translation.

    Context-aware algorithm uses more input tokens due to:
    - Previous chunk context
    - Previous translation context
    - Translation instructions

    Multiplier accounts for additional context overhead.
    """
    input_cost_per_1k, output_cost_per_1k = MODEL_COSTS.get(model, (0.0, 0.0))

    # Input tokens include context overhead
    input_cost = (estimated_tokens * context_multiplier / 1000) * input_cost_per_1k
    # Output tokens remain the same (only translating main content)
    output_cost = (estimated_tokens / 1000) * output_cost_per_1k

    return input_cost + output_cost
```

### 4. Implementation Details

#### New Algorithm Function

```python
async def translate_context_aware(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> TranslationResponse:
    """Translate using context-aware algorithm with natural boundary splitting."""
```

#### Updated Main Translation Router

```python
async def translate_document(
    content: DocumentContent,
    config: TranslationConfig,
    translator: ModelInterface,
    progress: Optional[Progress] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> TranslationResponse:
    if config.algorithm == "page":
        return await translate_page_by_page(...)
    elif config.algorithm == "sliding-window":
        return await translate_sliding_window(...)
    elif config.algorithm == "context-aware":
        return await translate_context_aware(...)
    else:
        raise TranslationError(f"Unknown algorithm: {config.algorithm}")
```

#### CLI Parameter Updates

```python
@app.command()
def translate(
    # ... existing parameters ...
    algorithm: str = typer.Option(
        "context-aware",  # New default
        "--algorithm",
        "-a",
        help="Translation algorithm: 'page', 'sliding-window', or 'context-aware'.",
    ),
    context_size: Optional[int] = typer.Option(
        2000,
        "--context-size",
        help="Target chunk size for context-aware algorithm (characters).",
    ),
    custom_split_token: Optional[str] = typer.Option(
        None,
        "--split-token",
        help="Custom token to split text on (context-aware only).",
    ),
):
```

## Implementation Steps

### Phase 1: Core Algorithm Implementation

#### 1. Update TranslationConfig Type Definitions

**File**: `src/tinbox/core/types.py`

- Add "context-aware" to algorithm Literal type
- Add `context_size: Optional[int]` field
- Add `custom_split_token: Optional[str]` field
- Maintain backward compatibility with existing fields
- Add proper field descriptions and validation

#### 2. Implement Smart Text Splitting

**File**: `src/tinbox/core/translation/algorithms.py`

- Create `smart_text_split()` function with natural boundary detection
- Implement priority-based splitting (custom token → paragraphs → sentences → clauses → words)
- Add proper error handling for edge cases (empty text, invalid tokens)
- Include comprehensive logging for debugging split decisions

#### 3. Implement Context-Aware Translation Function

**File**: `src/tinbox/core/translation/algorithms.py`

- Create `translate_context_aware()` async function
- Implement context building with previous chunk + translation
- Add proper prompt engineering for LLM instructions
- Handle edge cases (first chunk, last chunk, single chunk)
- Implement checkpoint support for resumability
- Add progress tracking integration

#### 4. Update Main Translation Router

**File**: `src/tinbox/core/translation/algorithms.py`

- Add "context-aware" case to `translate_document()` function
- Ensure proper error handling and algorithm validation
- Maintain backward compatibility with existing algorithms

### Phase 2: Cost Estimation Updates

#### 5. Enhance Cost Estimation Logic

**File**: `src/tinbox/core/cost.py`

- Add `estimate_context_aware_tokens()` function
- Implement context multiplier calculation (estimated 1.8x for input tokens)
- Update `estimate_cost()` function to handle context-aware algorithm
- Add algorithm-specific cost estimation routing
- Include warnings for high context overhead

#### 6. Update Cost Display Logic

**File**: `src/tinbox/cli.py`

- Update `display_cost_estimate()` to show algorithm-specific information
- Add context overhead warnings for context-aware algorithm
- Include estimated context multiplier in cost breakdown

### Phase 3: CLI Integration

#### 7. Add New CLI Parameters

**File**: `src/tinbox/cli.py`

- Add `context_size` parameter with proper validation
- Add `custom_split_token` parameter
- Update algorithm help text to include new option
- Set "context-aware" as new default algorithm
- Ensure parameter validation and error handling

#### 8. Update Configuration Building

**File**: `src/tinbox/cli.py`

- Update `TranslationConfig` creation to include new parameters
- Add parameter validation logic
- Ensure proper default value handling
- Add algorithm-specific parameter warnings

### Phase 4: Testing and Validation

#### 9. Create and Update Comprehensive Test Suite

**New Test Files**:

- `tests/test_core/test_translation/test_context_aware.py` (new algorithm tests)

**Existing Test Files to Update**:

- `tests/test_core/test_translation/test_algorithms.py` (add new context-aware algorithm tests)
- `tests/test_core/test_cost.py` (add context-aware cost estimation tests)
- `tests/test_cli.py` (add new CLI parameters and algorithm option tests)

**Test Coverage**:

```python
# New tests in test_context_aware.py
def test_smart_text_split_paragraphs()
def test_smart_text_split_sentences()
def test_smart_text_split_custom_token()
def test_smart_text_split_edge_cases()
def test_context_aware_single_chunk()
def test_context_aware_multiple_chunks()
def test_context_aware_with_custom_token()
def test_context_aware_checkpoint_resume()

# Updates to existing test_algorithms.py
def test_translate_document_context_aware()  # Add to existing test suite
def test_context_aware_algorithm_routing()   # Add to translate_document tests

# Updates to existing test_cost.py
def test_context_aware_cost_estimation()     # Add context multiplier tests
def test_cost_multiplier_calculation()       # Add algorithm-specific costs
def test_estimate_cost_context_aware()       # Add to existing estimate_cost tests

# Updates to existing test_cli.py
def test_cli_context_aware_parameters()      # Add to existing CLI tests
def test_cli_algorithm_selection()           # Update existing algorithm tests
def test_cli_custom_split_token()           # Add new parameter tests
def test_parse_model_spec_context_aware()    # Update existing parsing tests
```

#### 10. Integration Testing

**Test Scenarios**:

- End-to-end translation with various document types (PDF, DOCX, TXT)
- Custom split token functionality with real documents
- Cost estimation accuracy validation
- Performance comparison with existing algorithms
- Checkpoint and resume functionality

### Phase 5: Quality Assurance

#### 11. Run Comprehensive Quality Checks

```bash
# Type checking
python -m mypy src/

# Linting
python -m flake8 src/
python -m black --check src/

# Testing
python -m pytest tests/ -v --cov=src/
```

#### 12. Performance and Accuracy Validation

- **Translation Quality**: Compare output quality against existing algorithms
- **Performance Metrics**: Measure execution time and memory usage
- **Cost Accuracy**: Validate cost estimation against actual usage
- **Edge Case Handling**: Test with various document sizes and structures

## Benefits

### Translation Quality Improvements

- **Eliminates duplicate content**: No overlapping translations in final output
- **Better context awareness**: Previous translation context improves consistency
- **Natural text boundaries**: Preserves semantic units for better translation quality
- **Custom splitting control**: Precise control over chunk boundaries when needed

### Performance and Reliability

- **Simplified merging**: Direct concatenation eliminates complex merge logic (no separators needed)
- **Predictable output**: No merge failures or content loss
- **Better cost predictability**: More accurate cost estimation with context multiplier
- **Robust error handling**: Fewer edge cases and failure modes

### User Experience

- **Improved default algorithm**: Context-aware becomes the recommended default
- **Flexible configuration**: Support for both automatic and manual splitting
- **Better cost transparency**: Clear indication of context overhead costs
- **Backward compatibility**: Existing algorithms remain available

## Migration Strategy

### Backward Compatibility

- **Existing algorithms preserved**: "page" and "sliding-window" remain unchanged
- **Configuration compatibility**: Existing config files continue to work
- **CLI compatibility**: Existing command-line usage patterns supported
- **Gradual adoption**: Users can opt into new algorithm when ready

### Default Algorithm Change

- **New installations**: Default to "context-aware" algorithm
- **Existing users**: Continue using previously specified algorithms
- **Documentation updates**: Recommend context-aware for new projects
- **Migration guides**: Provide clear upgrade paths and comparisons

## Implementation Status

⏳ **PENDING** - Design has been completed and implementation is ready to begin.

### Implementation Checklist

- [ ] **Phase 1: Core Algorithm Implementation**
  - [ ] Update TranslationConfig type definitions
  - [ ] Implement smart text splitting function
  - [ ] Implement context-aware translation function
  - [ ] Update main translation router
- [ ] **Phase 2: Cost Estimation Updates**
  - [ ] Enhance cost estimation logic
  - [ ] Update cost display logic
- [ ] **Phase 3: CLI Integration**
  - [ ] Add new CLI parameters
  - [ ] Update configuration building
- [ ] **Phase 4: Testing and Validation**
  - [ ] Create comprehensive test suite
  - [ ] Integration testing
- [ ] **Phase 5: Quality Assurance**
  - [ ] Run quality checks
  - [ ] Performance and accuracy validation

## Success Criteria

1. **Translation Quality** 🎯

   - Zero duplicate content in final translations
   - Improved translation consistency through context awareness
   - Natural text boundaries preserved in all chunk types
   - Custom split token functionality works reliably

2. **System Reliability** 🎯

   - No merge failures or content loss incidents
   - Robust handling of edge cases (empty chunks, single chunks, etc.)
   - Checkpoint and resume functionality works correctly
   - Backward compatibility with existing algorithms maintained

3. **Performance** 🎯

   - Cost estimation accuracy within 10% of actual costs
   - Context overhead properly calculated and displayed
   - Execution time comparable to or better than sliding-window
   - Memory usage remains within acceptable bounds

4. **User Experience** 🎯
   - Intuitive CLI parameters for new algorithm options
   - Clear documentation and migration guidance
   - Helpful error messages and validation
   - Seamless integration with existing workflows

## Risk Mitigation

### Technical Risks

- **Context size management**: Monitor token usage to prevent API limits
- **LLM instruction following**: Test prompt engineering across different models
- **Natural boundary detection**: Validate splitting logic with various text types
- **Checkpoint compatibility**: Ensure new algorithm works with existing checkpoint system

### User Experience Risks

- **Breaking changes**: Maintain strict backward compatibility
- **Cost surprises**: Clear communication about context overhead
- **Migration complexity**: Provide comprehensive documentation and examples
- **Performance regressions**: Thorough performance testing before release
