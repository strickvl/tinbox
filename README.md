# 🔄 Tinbox: Your Ultimate Translation Tool

**Tinbox** is a robust command-line tool designed to tackle the challenges of translating large documents using Large Language Models (LLMs). Unlike other tools, Tinbox excels in handling extensive document sizes and navigates around model limitations related to size and copyright issues, ensuring seamless and efficient translations.

## 🚀 Quick Start

```bash
# Install with all dependencies
pip install -e ".[all]"

# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Translate a text document
tinbox translate --to de --model openai:gpt-5-2025-08-07 ./examples/elara_story.txt

# Translate a PDF (requires poppler - see Installation section)
tinbox translate --to de --model openai:gpt-5-2025-08-07 --algorithm page ./examples/elara_story.pdf

# Use local models with Ollama
ollama serve  # In another terminal
tinbox translate --to de --model ollama:llama3.1:8b ./examples/elara_story.txt
```

## ✨ Key Features

- **🔍 No OCR Needed**: Directly translates PDFs using advanced multimodal models
- **📄 Smart Document Handling**: Supports PDFs, Word documents, and text files
- **📏 Large Document Handling**: Efficiently processes large PDFs and other document types
- **📌 Overcomes Model Limitations**: Bypasses common model refusals due to size or copyright concerns.
- **🧠 Intelligent Algorithms**: Context-aware translation with smart text splitting
- **🌐 Local and Cloud Support**: Works with OpenAI, Anthropic, Google, or local Ollama models
- **📚 Glossary Support**: Maintains consistent terminology across documents
- **💾 Checkpoint & Resume**: Handle large documents with automatic progress saving
- **📊 Cost Tracking**: Estimate translation costs and time and monitor during translation

## 🎯 The Problems Tinbox Solves

1. **PDF Translation Challenges**

   - Most tools require OCR, leading to formatting loss and errors
   - Tinbox uses multimodal models to directly understand PDFs as images

2. **Large Document Limitations**

   - Traditional tools often fail with large documents
   - Models frequently refuse or timeout on big files
   - Tinbox smartly splits and processes documents while maintaining context

3. **Model Refusal Issues**

   - Many models refuse translation tasks due to:
     - Copyright concerns
     - Document size limitations
     - Rate limiting
   - Tinbox's algorithms work around these limitations intelligently

4. **Quality and Consistency**
   - Smart algorithms ensure consistent translations across document sections
   - Maintains context between pages and segments
   - Uses glossary to maintain consistent terminology

## ✨ Features

### 📄 Smart Document Handling

- **PDFs**: Processed directly as images - no OCR needed!
- **Word (docx)**: Preserves formatting while translating
- **Text files**: Efficient processing for large files

### 🧠 Intelligent Translation

- **Smart Algorithms**:
  - Page-by-Page (used for PDF documents)
  - Sliding Window for long text documents (deprecated)
  - Context-Aware algorithm with smart text splitting (default for text documents)

### 🤖 Flexible Model Support

- Use powerful cloud models (GPT-5, Claude 4 Sonnet, Google Gemini, etc.)
- Run translations locally with Ollama
- Mix and match models for different tasks

### 💾 Checkpoint & Resume

- Automatically save checkpoints and resume translations

### 🌐 Language Support

- Flexible source/target language specification using ISO 639-1 codes
- Common language aliases (e.g., "en", "zh", "es")

### 📊 Cost Tracking

- Track overall translation time and token usage/cost
- Compare algorithms or model providers side-by-side
- Full progress and cost tracking during translation

## 🚀 Getting Started

## 📦 Installation

### Basic Installation

```bash
# Recommended: Install with all features (PDF, DOCX, image processing)
pip install -e ".[all]"

# Or install specific features
pip install -e ".[pdf]"     # PDF support only
pip install -e ".[docx]"    # Word document support only
pip install tinbox          # Base package only
```

### System Requirements

**For PDF processing**, install `poppler-utils`:

```bash
# macOS
brew install poppler

# Linux (Ubuntu/Debian)
sudo apt-get install poppler-utils

# Windows (Chocolatey)
choco install poppler
```

### API Keys

Set up your API keys for cloud models:

```bash
# Choose your preferred model provider
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## 🎮 Basic Usage

```bash
# Simple document translation
tinbox translate --to es --model openai:gpt-5-2025-08-07 document.txt

# Specify output file
tinbox translate --to es --model openai:gpt-5-2025-08-07 --output document_es.txt document.txt

# Specify source language
tinbox translate --from zh --to en --model openai:gpt-5-2025-08-07 document.txt

# Large documents with checkpointing
tinbox translate --to de --checkpoint-dir ./checkpoints --model openai:gpt-5-2025-08-07 large_document.txt

# With glossary
tinbox translate --to es --glossary --model openai:gpt-5-2025-08-07 document.txt

# Setting a cost limit
tinbox translate --to es --max-cost 10.00 --model openai:gpt-5-2025-08-07 document.txt

# Use model reasoning
tinbox translate --to es --reasoning-effort low --max-cost 10.00 --model openai:gpt-5-2025-08-07 document.txt
```

## 💡 Best Practices

| Document Type         | Recommended Settings                    | Notes                                                        |
| --------------------- | --------------------------------------- | ------------------------------------------------------------ |
| **PDFs**              | `--algorithm page`                      | Page-by-page processing works best                           |
| **Large Text**        | `--context-size 2000`                   | Context-aware algorithm (default) handles size intelligently |
| **Technical Docs**    | `--glossary --save-glossary terms.json` | Maintains consistent terminology                             |
| **Long Translations** | `--checkpoint-dir ./checkpoints`        | Resume interrupted translations                              |
| **Cost Control**      | `--dry-run --max-cost 5.00`             | Preview costs before translating                             |
| **Reasoning**         | `--reasoning-effort minimal`            | Faster and cheaper, only use higher levels if needed         |

## 📖 Detailed Documentation

### Command-Line Reference

| Category         | Option                   | Description                                                       | Default         |
| ---------------- | ------------------------ | ----------------------------------------------------------------- | --------------- |
| **Core**         | `--from, -f`             | Source language (auto-detect if not specified)                    | Auto-detect     |
|                  | `--to, -t`               | Target language                                                   | English         |
|                  | `--model`                | Model to use (`openai:gpt-4o`, `anthropic:claude-3-sonnet`, etc.) | Auto-select     |
|                  | `--output, -o`           | Output file (default: print to console)                           | Console         |
| **Algorithms**   | `--algorithm, -a`        | Translation algorithm (`context-aware`, `page`, `sliding-window`) | `context-aware` |
|                  | `--context-size`         | Target chunk size for context-aware algorithm                     | 2000 chars      |
|                  | `--split-token`          | Custom token to split text on                                     | None            |
| **Checkpoints**  | `--checkpoint-dir`       | Directory to store checkpoints for resuming                       | None            |
|                  | `--checkpoint-frequency` | Save checkpoint every N pages/chunks                              | 1               |
| **Glossary**     | `--glossary`             | Enable glossary for consistent terminology                        | Disabled        |
|                  | `--glossary-file`        | Path to existing glossary file (JSON)                             | None            |
|                  | `--save-glossary`        | Path to save updated glossary                                     | None            |
| **Quality**      | `--reasoning-effort`     | Model reasoning level (`minimal`, `low`, `medium`, `high`)        | `minimal`       |
| **Output**       | `--format, -F`           | Output format (`text`, `json`, `markdown`)                        | `text`          |
|                  | `--benchmark, -b`        | Include performance metrics                                       | Disabled        |
| **Cost Control** | `--dry-run`              | Preview costs without translating                                 | Disabled        |
|                  | `--max-cost`             | Maximum cost limit (USD)                                          | No limit        |

### Supported Languages

Supports all language that is supported by the selected model.

### Advanced Examples

#### Algorithm Selection

```bash
# Context-aware (default) - best for most documents
tinbox translate --to es --context-size 1500 --model openai:gpt-5-2025-08-07 document.txt

# Page-by-page - best for PDFs
tinbox translate --to es --algorithm page --model openai:gpt-5-2025-08-07 document.pdf

# Sliding window (not recommended, use context-aware instead)
tinbox translate --to es --algorithm sliding-window --window-size 3000 --model openai:gpt-5-2025-08-07 large_file.txt

# Custom splitting for structured documents
tinbox translate --to fr --split-token "---" --model openai:gpt-5-2025-08-07 structured_document.txt
```

#### Model Providers

```bash
# OpenAI models
tinbox translate --to es --model openai:gpt-4o document.pdf

# Anthropic Claude
tinbox translate --to es --model anthropic:claude-3-sonnet document.pdf

# Local models with Ollama
tinbox translate --to fr --model ollama:mistral-small document.txt

# Quality vs cost control
tinbox translate --to es --reasoning-effort high --max-cost 10.00 document.pdf
```

#### Glossary & Consistency

When glossary is enabled, it will automatically detect important terms and add them to the glossary during translation.

```bash
# Use glossary during translation
tinbox translate --to es --glossary --model openai:gpt-4o medical_doc.pdf

# Build and save terminology in the end
tinbox translate --to es --glossary --save-glossary medical_terms.json --model openai:gpt-4o medical_doc.pdf

# Use existing initial glossary
tinbox translate --to fr --glossary-file company_terms.json --model openai:gpt-4o document.docx

# Use existing glossary and save extended glossary in the end
tinbox translate --to de --glossary-file base_terms.json --save-glossary extended_terms.json --model openai:gpt-4o doc.pdf
```

**Glossary File Format**

Glossary files use JSON format for easy editing and sharing:

```json
{
  "entries": {
    "CPU": "Processeur",
    "GPU": "Carte graphique",
    "API": "Interface de programmation"
  }
}
```

#### Output Formats

```bash
# JSON with metadata and costs
tinbox translate --to es --format json --benchmark --model openai:gpt-4o document.pdf

# Markdown report
tinbox translate --to es --format markdown --model openai:gpt-4o document.pdf

# Save to file
tinbox translate --to es --output translated.txt --model openai:gpt-4o document.pdf
```

---

## 🛠 Development

### Contributing Setup

```bash
# Clone and setup development environment
git clone <repository-url>
cd tinbox
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_core/test_cost.py -v          # Cost estimation
pytest tests/test_core/test_processors/ -v     # Document processing
pytest tests/test_core/test_translation/ -v    # Translation algorithms
pytest tests/test_cli.py -v                    # CLI interface

# Coverage report
pytest --cov=tinbox --cov-report=html
```

### Project Structure

```
tinbox/
├── src/tinbox/
│   ├── cli.py              # Command-line interface
│   ├── core/
│   │   ├── cost.py         # Cost tracking
│   │   ├── processor/      # Document processors (PDF, DOCX, text)
│   │   └── translation/    # Translation algorithms
│   └── utils/              # Utility functions
└── tests/                  # Comprehensive test suite
```
