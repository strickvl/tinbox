# 🔄 Tinbox: Your Ultimate Translation Tool

**Tinbox** is a robust command-line tool designed to tackle the challenges of translating large documents, especially PDFs, using Large Language Models (LLMs). Unlike other tools, Tinbox excels in handling extensive document sizes and navigates around model limitations related to size and copyright issues, ensuring seamless and efficient translations.

**Why Choose Tinbox?**
- **Handles Large Documents**: Efficiently processes large PDFs and other document types.
- **Overcomes Model Limitations**: Bypasses common model refusals due to size or copyright concerns.
- **No OCR Needed**: Directly translates PDFs using advanced multimodal models.
- **Smart Algorithms**: Achieve optimal translation results with our intelligent algorithms.
- **Local and Cloud Support**: Use models locally or in the cloud, depending on your preference.

**Quick Start Example:**
```bash
tinbox --to es document.pdf
```

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
   - Repairs potential inconsistencies at section boundaries

![Tinbox Workflow](link_to_diagram.png)

🔍 **Key Highlights:**
- Translate PDFs without OCR using advanced AI models
- Handle documents of any size with smart splitting algorithms
- Work around common model limitations and refusals
- Track costs and performance with built-in benchmarking

## ✨ Features

### 📄 Smart Document Handling
- **PDFs**: Processed directly as images - no OCR needed!
- **Word (docx)**: Preserves formatting while translating
- **Text files**: Efficient processing for large files

### 🧠 Intelligent Translation
- **Smart Algorithms**:
  - Page-by-Page with Seam Repair (default for PDF)
  - Sliding Window for long text documents
  - Automatic context preservation between sections

### 🤖 Flexible Model Support
- Use powerful cloud models (GPT-4V, Claude 3.5 Sonnet)
- Run translations locally with Ollama
- Mix and match models for different tasks

### 🌐 Language Support
- Flexible source/target language specification using ISO 639-1 codes
- Common language aliases (e.g., 'en', 'zh', 'es')

6. 📊 **Benchmarking**  
   - Track overall translation time and token usage/cost
   - Compare algorithms or model providers side-by-side

## 🚀 Getting Started

### Quick Install

```bash
# Install base package
pip install tinbox

# For PDF support (recommended)
pip install tinbox[pdf]

# For Word document support
pip install tinbox[docx]

# Install everything
pip install tinbox[all]
```

### Basic Usage

1. **Translate a PDF to Spanish**
   ```bash
   tinbox --to es document.pdf
   ```

2. **Translate a Word document from Chinese to English**
   ```bash
   tinbox --from zh --to en document.docx
   ```

3. **Handle a large text file with custom settings**
   ```bash
   tinbox --to fr --algorithm sliding-window large_document.txt
   ```

### 💡 Tips for Best Results

1. **For Large Documents**
   - Use the sliding window algorithm: `--algorithm sliding-window`
   - Adjust window size if needed: `--window-size 3000`

2. **For PDFs**
   - The default page-by-page algorithm works best
   - No OCR needed - just point to your PDF!

3. **For Best Performance**
   - Use local models via Ollama for faster processing
   - Cloud models (GPT-4V, Claude) for highest quality

## 📖 Detailed Documentation

### Command-Line Options

#### Core Options
| Option              | Description                                           | Example                    |
|--------------------|-------------------------------------------------------|----------------------------|
| `--from, -f`       | Source language (auto-detect if not specified)        | `--from zh`               |
| `--to, -t`         | Target language (default: English)                    | `--to es`                 |
| `--model`          | Model to use for translation                          | `--model gpt-4v`          |
| `--output, -o`     | Output file (default: print to console)              | `--output translated.txt`  |

#### Algorithm Options
| Option              | Description                                           | Default                    |
|--------------------|-------------------------------------------------------|----------------------------|
| `--algorithm, -a`  | Translation algorithm (`page` or `sliding-window`)    | `page` for PDF            |
| `--window-size`    | Size of translation window                            | 2000 tokens               |
| `--overlap-size`   | Overlap between windows                               | 200 tokens                |

#### Output Format Options
| Option              | Description                                           | Example Output             |
|--------------------|-------------------------------------------------------|----------------------------|
| `--format, -F`     | Output format (text, json, markdown)                  | See examples below         |
| `--benchmark, -b`  | Include performance metrics                           | Translation time, costs    |

### Supported Languages

Common language codes (ISO 639-1):

| Code | Language    | Also Accepts |
|------|-------------|--------------|
| en   | English     | eng          |
| es   | Spanish     | spa          |
| zh   | Chinese     | chi, cmn     |
| fr   | French      | fra          |
| de   | German      | deu, ger     |
| ja   | Japanese    | jpn          |
| ko   | Korean      | kor          |
| ru   | Russian     | rus          |
| ar   | Arabic      | ara          |
| hi   | Hindi       | hin          |

### Output Format Examples

#### 1. Plain Text (Default)
```bash
tinbox translate document.pdf --to es
# Output: Translated text...
```

#### 2. JSON Output
```bash
tinbox translate document.pdf --to es --format json
```

Example response:
```json
{
  "metadata": {
    "source_lang": "en",
    "target_lang": "es",
    "model": "claude-3-sonnet",
    "algorithm": "page"
  },
  "result": {
    "text": "Translated text...",
    "tokens_used": 1500,
    "cost": 0.045,
    "time_taken": 12.5
  }
}
```

#### 3. Markdown Report
```bash
tinbox translate document.pdf --to es --format markdown
```

### Advanced Usage

1. **Handling Very Large Documents**
   ```bash
   tinbox --to es --algorithm sliding-window \
          --window-size 3000 --overlap-size 300 \
          large_document.pdf
   ```

2. **Using Local Models**
   ```bash
   tinbox --to fr --model ollama:mistral-small document.txt
   ```

3. **Benchmarking Different Models**
   ```bash
   tinbox --to de --benchmark --model gpt-4v document.pdf
   ```

## 🛠 Project Structure

```
tinbox/
├── src/
│   └── tinbox/
│       ├── cli.py                 # Command-line interface
│       ├── core/                  # Core functionality
│       │   ├── cost.py           # Cost tracking
│       │   ├── processor/        # Document processors
│       │   └── translation/      # Translation algorithms
│       └── utils/                # Utilities
└── tests/                        # Test suite
```

## 🔜 Future Plans

1. **Enhanced Output Formats**
   - PDF output with original formatting
   - Word document export
   - HTML with parallel text

2. **Advanced Features**
   - AI-powered section detection
   - Custom terminology support
   - Interactive translation review
   - Domain-specific model fine-tuning

3. **Performance Improvements**
   - Parallel processing
   - Better caching
   - Reduced API costs
