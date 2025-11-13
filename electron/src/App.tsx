import { useEffect, useState } from 'react';
import { useStore } from './store/useStore';
import { useTranslation } from './hooks/useTranslation';
import { TinboxAPI } from './utils/api';
import { DropZone } from './components/DropZone';
import { TranslationQueue } from './components/TranslationQueue';
import { SettingsModal } from './components/SettingsModal';
import { Settings } from './components/Icons';

export default function App() {
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const setSettingsOpen = useStore((state) => state.setSettingsOpen);
  const settings = useStore((state) => state.settings);
  const setModels = useStore((state) => state.setModels);
  const setLanguages = useStore((state) => state.setLanguages);
  const updateSettings = useStore((state) => state.updateSettings);

  const { startTranslation, estimateCost } = useTranslation();

  // Load models and languages on mount
  useEffect(() => {
    async function loadData() {
      try {
        const [models, languages] = await Promise.all([
          TinboxAPI.getModels(),
          TinboxAPI.getLanguages(),
        ]);
        setModels(models);
        setLanguages(languages);
      } catch (error) {
        console.error('Failed to load data:', error);
      }
    }
    loadData();
  }, [setModels, setLanguages]);

  // Load settings from Electron store on mount
  useEffect(() => {
    async function loadSettings() {
      const saved = await window.electron.getSetting('settings');
      if (saved) {
        updateSettings(saved);
      }
    }
    loadSettings();
  }, [updateSettings]);

  // Set API keys in environment
  useEffect(() => {
    // In a real app, you'd need to pass these to the Python backend
    // For now, the backend will read from environment variables
    console.log('API keys configured:', Object.keys(settings.apiKeys));
  }, [settings.apiKeys]);

  const handleFilesSelected = async (files: string[]) => {
    setSelectedFiles(files);

    // Show cost estimate for first file
    if (files.length > 0) {
      const estimate = await estimateCost(files[0], settings.defaultModel);
      if (estimate) {
        const confirmMessage =
          files.length === 1
            ? `Estimated cost: $${estimate.estimated_cost.toFixed(4)}\n` +
              `Tokens: ${estimate.estimated_tokens.toLocaleString()}\n` +
              (estimate.num_pages ? `Pages: ${estimate.num_pages}\n` : '') +
              (estimate.warnings.length > 0 ? `\nWarnings:\n${estimate.warnings.join('\n')}` : '')
            : `Processing ${files.length} files. First file estimate:\n` +
              `Cost: $${estimate.estimated_cost.toFixed(4)}\n` +
              `Tokens: ${estimate.estimated_tokens.toLocaleString()}`;

        if (!confirm(`${confirmMessage}\n\nContinue with translation?`)) {
          setSelectedFiles([]);
          return;
        }
      }
    }

    // Start translation jobs
    setIsProcessing(true);
    for (const file of files) {
      await startTranslation({
        file_path: file,
        source_lang: settings.defaultSourceLang,
        target_lang: settings.defaultTargetLang,
        model: settings.defaultModel,
        algorithm: settings.defaultAlgorithm,
        output_format: settings.defaultOutputFormat,
      });
    }
    setIsProcessing(false);
    setSelectedFiles([]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Tinbox</h1>
            <p className="text-sm text-gray-500">Document Translation</p>
          </div>

          <button
            onClick={() => setSettingsOpen(true)}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            title="Settings"
          >
            <Settings className="w-6 h-6" />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="space-y-8">
          {/* Drop Zone */}
          <DropZone
            onFilesSelected={handleFilesSelected}
            disabled={isProcessing}
          />

          {/* Translation Queue */}
          <TranslationQueue />
        </div>
      </main>

      {/* Settings Modal */}
      <SettingsModal />
    </div>
  );
}
