# Tinbox GUI

A modern, minimalist Electron-based graphical interface for Tinbox document translation.

## Features

- 🎯 **Drag & Drop**: Simply drag files into the window to translate
- 📦 **Batch Processing**: Translate multiple documents at once
- 📊 **Real-time Progress**: Live updates via WebSocket connection
- 💰 **Cost Estimation**: Preview translation costs before starting
- ⚙️ **Settings Management**: Securely store API keys and preferences
- 🎨 **Minimalist Design**: Clean, modern UI with Tailwind CSS

## Prerequisites

- Node.js 18+ and npm
- Python 3.9+ with Tinbox installed
- API keys for your chosen LLM provider (OpenAI, Anthropic, etc.)

## Installation

### 1. Install Python Dependencies

First, install Tinbox with GUI support:

```bash
# From the root directory
cd ..
uv pip install -e ".[gui,pdf,docx]"
```

### 2. Install Node Dependencies

```bash
cd electron
npm install
```

## Development

Run the app in development mode with hot reload:

```bash
npm run dev
```

This will:
1. Start the Vite dev server for React (port 5173)
2. Start the FastAPI backend server (port 8765)
3. Launch Electron with dev tools

## Building

### Build for Development
```bash
npm run build
```

### Package for Distribution

**macOS:**
```bash
npm run electron:build:mac
```

**Windows:**
```bash
npm run electron:build:win
```

**Linux:**
```bash
npm run electron:build:linux
```

Packaged apps will be in `dist-electron/`.

## Architecture

### Frontend (Electron + React)
- **Electron**: Desktop app framework
- **React**: UI components
- **TypeScript**: Type safety
- **Tailwind CSS**: Styling
- **Zustand**: State management
- **Vite**: Build tool

### Backend (Python + FastAPI)
- **FastAPI**: REST API server
- **WebSockets**: Real-time progress updates
- **Tinbox Core**: Translation engine
- **LiteLLM**: Multi-provider LLM interface

### Communication Flow

```
┌─────────────┐         HTTP/WebSocket         ┌──────────────┐
│   Electron  │ ◄──────────────────────────► │   FastAPI    │
│   (React)   │    localhost:8765              │   Server     │
└─────────────┘                                └──────────────┘
                                                       │
                                                       ▼
                                               ┌──────────────┐
                                               │   Tinbox     │
                                               │    Core      │
                                               └──────────────┘
                                                       │
                                                       ▼
                                               ┌──────────────┐
                                               │     LLM      │
                                               │  (OpenAI,    │
                                               │  Anthropic)  │
                                               └──────────────┘
```

## API Endpoints

The FastAPI server exposes:

- `GET /api/models` - List available models
- `GET /api/languages` - List supported languages
- `POST /api/estimate-cost` - Estimate translation cost
- `POST /api/validate-config` - Validate API configuration
- `POST /api/translate` - Start translation job
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job status
- `DELETE /api/jobs/{job_id}` - Cancel job
- `WS /ws/progress/{job_id}` - Real-time progress updates

## Configuration

### API Keys

API keys are stored securely using Electron Store. Set them in the Settings panel (gear icon):

- **OpenAI**: For GPT-4, GPT-5 models
- **Anthropic**: For Claude models
- **Google**: For Gemini models

### Default Settings

Configure default values for:
- Model (e.g., `openai:gpt-4o-mini`)
- Source language (default: auto-detect)
- Target language
- Translation algorithm (page-by-page or sliding-window)
- Output format (text, markdown, or JSON)

## Usage

1. **Launch the app**
2. **Click the settings icon** to configure API keys
3. **Drag and drop files** or click the drop zone to select files
4. **Review cost estimate** and confirm
5. **Monitor progress** in real-time
6. **Download translations** when complete

## Supported File Types

- PDF (`.pdf`)
- Microsoft Word (`.docx`)
- Plain text (`.txt`)

## Troubleshooting

### Backend won't start
- Ensure Python backend is installed: `uv pip install -e ".[gui]"`
- Check that port 8765 is not in use
- Look at Electron console logs for errors

### "API key not found" error
- Open Settings and enter your API key
- Alternatively, set environment variables:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`

### Files not processing
- Check file format (PDF, DOCX, TXT only)
- Ensure model is configured correctly
- Check backend logs for errors

## Development Tips

### Hot Reload
Both React and Python support hot reload:
- React: Changes auto-refresh via Vite
- Python: Restart the app to reload backend

### Debugging
- Press `Cmd+Option+I` (Mac) or `Ctrl+Shift+I` (Windows/Linux) to open DevTools
- Backend logs appear in the terminal and DevTools console

### State Management
The app uses Zustand for state management. See `src/store/useStore.ts`.

### Adding New Models
Update `MODEL_PRICING` in `src/tinbox/api/server.py` with pricing info.

## Contributing

This GUI is part of the Tinbox project. See the main README for contribution guidelines.

## License

MIT License - see LICENSE file in the root directory.
