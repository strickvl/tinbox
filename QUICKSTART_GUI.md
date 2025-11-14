# Tinbox GUI - Quick Start Guide

Get the Tinbox Electron GUI up and running in 5 minutes!

## Installation

### Step 1: Install Python Dependencies

```bash
# Install Tinbox with GUI support
uv pip install -e ".[gui,pdf,docx]"
```

This installs:
- Tinbox core
- FastAPI for the REST API server
- Uvicorn for serving the API
- PDF and DOCX support

### Step 2: Install Node Dependencies

```bash
cd electron
npm install
```

## Running the App

### Development Mode

From the `electron/` directory:

```bash
npm run dev
```

This will:
1. Start the FastAPI backend on `http://127.0.0.1:8765`
2. Start the Vite dev server on `http://localhost:5173`
3. Launch the Electron app

The app should open automatically with hot reload enabled.

## First Use

### 1. Configure API Keys

Click the **settings icon** (⚙️) in the top right and enter your API keys:

- **OpenAI**: For GPT-4, GPT-5 models
- **Anthropic**: For Claude models
- **Google**: For Gemini models (optional)

### 2. Set Default Preferences

In Settings, configure:
- **Default Model**: e.g., `openai:gpt-4o-mini` (recommended for most users)
- **Source Language**: `auto` for auto-detection
- **Target Language**: e.g., `en` for English
- **Algorithm**: `page-by-page` for PDFs (default)
- **Output Format**: `text` (default)

Click **Save Settings**.

### 3. Translate Your First Document

1. **Drag and drop** a PDF, DOCX, or TXT file into the window
   - Or click the drop zone to browse for files

2. **Review the cost estimate** that appears
   - Shows estimated tokens and cost
   - Click "OK" to proceed

3. **Watch the progress** in real-time
   - Progress bar shows current page
   - Token count and cost update live
   - Multiple files process in queue

4. **Download the result** when complete
   - Click the download icon (⬇️)
   - Choose where to save the translation

## Tips

### Batch Processing
- Drop multiple files at once
- They'll queue automatically
- Max 2 concurrent translations (configurable)

### Cost Management
- Start with `gpt-4o-mini` or `gpt-5-nano` for testing
- Cost estimates help avoid surprises
- View actual cost per job in the progress card

### File Support
- **PDF**: Converted to images, works with scanned docs
- **DOCX**: Text extraction maintains structure
- **TXT**: Direct text processing

### Keyboard Shortcuts
- **Cmd/Ctrl + ,**: Open settings (coming soon)
- **Cmd/Ctrl + Q**: Quit app

## Troubleshooting

### "Failed to start Python server"

**Solution**: Make sure you installed the GUI dependencies:
```bash
uv pip install -e ".[gui]"
```

### "API key not found"

**Solution**: Either:
1. Add key in Settings UI, or
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

### Port 8765 already in use

**Solution**:
1. Stop other Tinbox instances
2. Or change port in `electron/main.js` (line 13)

### Translation stuck at 0%

**Solution**:
1. Check that your API key is valid
2. Ensure you have sufficient API credits
3. Check console for error messages

## Next Steps

- Explore different models and compare quality
- Try both translation algorithms
- Experiment with output formats
- Check the full README for advanced features

## Need Help?

- Full documentation: `electron/README.md`
- Python backend: `src/tinbox/api/`
- React frontend: `electron/src/`
- Report issues: https://github.com/anthropics/tinbox/issues

Happy translating! 🎉
