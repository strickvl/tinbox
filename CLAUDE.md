# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tinbox is a CLI tool for translating large documents (especially PDFs) using LLMs. It handles model limitations, bypasses refusals, and uses multimodal capabilities for direct PDF translation without OCR.

## Development Commands

### Installation
- Use `uv pip install -e .` for development installation (not regular pip)
- Install with extras: `uv pip install -e ".[pdf,docx,dev]"`

### Testing
```bash
pytest                      # Run all tests
pytest tests/test_cli.py   # Run specific test file
pytest -k test_name        # Run specific test
pytest --cov=tinbox        # Run with coverage
```

### Code Quality
```bash
ruff check .               # Run linter
ruff format .              # Format code
mypy src/tinbox           # Type checking
black src tests           # Alternative formatter
isort src tests           # Sort imports
```

## Architecture

### Translation Pipeline
1. **Document Loading** (`core/processor/`): Handles PDF, DOCX, and text files
   - PDF: Converts pages to images for multimodal processing
   - DOCX: Extracts text while preserving structure
   - Text: Direct text processing

2. **Translation Algorithms** (`core/translation/algorithms.py`):
   - **Page-by-page**: Default for PDFs, includes seam repair between pages
   - **Sliding window**: For long text documents, maintains context overlap

3. **Model Interfaces** (`core/translation/`):
   - `interface.py`: Protocol definition for model implementations
   - `litellm.py`: LiteLLM integration for OpenAI/Anthropic/Ollama
   - Supports checkpointing for resumable translations

4. **Cost Management** (`core/cost.py`):
   - Estimates tokens and costs before translation
   - Warns about high-cost operations
   - Tracks actual usage during translation

## Key Technical Details

- **Model Specification Format**: `provider:model` (e.g., `anthropic:claude-3-sonnet`, `openai:gpt-4o`, `ollama:mistral-small`)
  - **GPT-5 Support**: Supports OpenAI's GPT-5 models (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`)
  - GPT-5 models don't support temperature/top_p parameters, use reasoning effort and verbosity instead
- **Language codes**: ISO 639-1 format with aliases (e.g., 'en', 'zh', 'es')
- **Source language 'auto'**: Accepted as valid value for automatic detection in LiteLLM translator
- **Async architecture**: Core translation operations use asyncio
- **Progress tracking**: Rich library for terminal UI
- **Pillow dependency**: Required for PDF image processing

## Common Usage Examples

```bash
# Basic translation
tinbox translate --to es document.pdf

# With specific model
tinbox translate --model openai:gpt-4o --to fr document.docx

# With GPT-5 models
tinbox translate --model openai:gpt-5 --to es document.pdf
tinbox translate --model openai:gpt-5-mini --to fr document.docx
tinbox translate --model openai:gpt-5-nano --to de document.txt

# Cost estimation only
tinbox translate --dry-run --to de large_document.pdf

# Sliding window for long texts
tinbox translate --algorithm sliding-window --to zh document.txt
```
