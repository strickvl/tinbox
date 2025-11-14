const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const Store = require('electron-store');

// Initialize electron-store for settings
const store = new Store();

let mainWindow;
let pythonProcess;

// Start Python FastAPI server
function startPythonServer() {
  const pythonPath = process.env.PYTHON_PATH || 'python3';
  const serverScript = path.join(__dirname, '..', 'src', 'tinbox', 'api', 'server.py');

  console.log('Starting Python server...');

  pythonProcess = spawn(pythonPath, ['-m', 'uvicorn', 'tinbox.api.server:app', '--host', '127.0.0.1', '--port', '8765'], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env },
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python server exited with code ${code}`);
  });

  // Wait a bit for server to start
  return new Promise((resolve) => setTimeout(resolve, 2000));
}

// Stop Python server
function stopPythonServer() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#ffffff',
    show: false, // Don't show until ready
  });

  // Load the app
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC handlers
ipcMain.handle('get-setting', (event, key) => {
  return store.get(key);
});

ipcMain.handle('set-setting', (event, key, value) => {
  store.set(key, value);
  return true;
});

ipcMain.handle('delete-setting', (event, key) => {
  store.delete(key);
  return true;
});

ipcMain.handle('open-file-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: options?.filters || [
      { name: 'Documents', extensions: ['pdf', 'docx', 'txt'] },
      { name: 'All Files', extensions: ['*'] }
    ],
  });
  return result.filePaths;
});

ipcMain.handle('save-file-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: options?.defaultPath || 'translated.txt',
    filters: options?.filters || [
      { name: 'Text Files', extensions: ['txt'] },
      { name: 'Markdown Files', extensions: ['md'] },
      { name: 'JSON Files', extensions: ['json'] },
      { name: 'All Files', extensions: ['*'] }
    ],
  });
  return result.filePath;
});

// App lifecycle
app.on('ready', async () => {
  await startPythonServer();
  createWindow();
});

app.on('window-all-closed', () => {
  stopPythonServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('before-quit', () => {
  stopPythonServer();
});
