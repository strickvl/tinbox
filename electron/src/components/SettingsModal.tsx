import { useEffect, useState } from 'react';
import { useStore } from '../store/useStore';
import { X } from './Icons';

export function SettingsModal() {
  const isOpen = useStore((state) => state.isSettingsOpen);
  const setOpen = useStore((state) => state.setSettingsOpen);
  const settings = useStore((state) => state.settings);
  const updateSettings = useStore((state) => state.updateSettings);
  const models = useStore((state) => state.models);
  const languages = useStore((state) => state.languages);

  const [localSettings, setLocalSettings] = useState(settings);

  useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  const handleSave = async () => {
    updateSettings(localSettings);

    // Save to Electron store
    await window.electron.setSetting('settings', localSettings);

    setOpen(false);
  };

  const handleApiKeyChange = (provider: string, value: string) => {
    setLocalSettings({
      ...localSettings,
      apiKeys: {
        ...localSettings.apiKeys,
        [provider]: value,
      },
    });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 animate-fade-in">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Settings</h2>
          <button
            onClick={() => setOpen(false)}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4 overflow-y-auto max-h-[calc(90vh-140px)]">
          {/* API Keys Section */}
          <section className="mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-3">API Keys</h3>
            <div className="space-y-4">
              {/* OpenAI */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  OpenAI API Key
                </label>
                <input
                  type="password"
                  value={localSettings.apiKeys.openai || ''}
                  onChange={(e) => handleApiKeyChange('openai', e.target.value)}
                  placeholder="sk-..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  For GPT-4, GPT-5, and other OpenAI models
                </p>
              </div>

              {/* Anthropic */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Anthropic API Key
                </label>
                <input
                  type="password"
                  value={localSettings.apiKeys.anthropic || ''}
                  onChange={(e) => handleApiKeyChange('anthropic', e.target.value)}
                  placeholder="sk-ant-..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  For Claude models
                </p>
              </div>

              {/* Google/Gemini */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Google API Key
                </label>
                <input
                  type="password"
                  value={localSettings.apiKeys.google || ''}
                  onChange={(e) => handleApiKeyChange('google', e.target.value)}
                  placeholder="..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  For Gemini models
                </p>
              </div>
            </div>
          </section>

          {/* Default Settings Section */}
          <section className="mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-3">Default Settings</h3>
            <div className="space-y-4">
              {/* Default Model */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Default Model
                </label>
                <select
                  value={localSettings.defaultModel}
                  onChange={(e) =>
                    setLocalSettings({ ...localSettings, defaultModel: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  {models.map((model) => (
                    <option key={model.full_name} value={model.full_name}>
                      {model.full_name} - ${model.input_cost_per_1m}/{model.output_cost_per_1m} per 1M
                    </option>
                  ))}
                </select>
              </div>

              {/* Source Language */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Default Source Language
                </label>
                <select
                  value={localSettings.defaultSourceLang}
                  onChange={(e) =>
                    setLocalSettings({
                      ...localSettings,
                      defaultSourceLang: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  <option value="auto">Auto-detect</option>
                  {languages.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Target Language */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Default Target Language
                </label>
                <select
                  value={localSettings.defaultTargetLang}
                  onChange={(e) =>
                    setLocalSettings({
                      ...localSettings,
                      defaultTargetLang: e.target.value,
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  {languages.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Algorithm */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Default Algorithm
                </label>
                <select
                  value={localSettings.defaultAlgorithm}
                  onChange={(e) =>
                    setLocalSettings({
                      ...localSettings,
                      defaultAlgorithm: e.target.value as 'page' | 'sliding-window',
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  <option value="page">Page-by-page (recommended for PDFs)</option>
                  <option value="sliding-window">Sliding window (for long texts)</option>
                </select>
              </div>

              {/* Output Format */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Default Output Format
                </label>
                <select
                  value={localSettings.defaultOutputFormat}
                  onChange={(e) =>
                    setLocalSettings({
                      ...localSettings,
                      defaultOutputFormat: e.target.value as 'text' | 'json' | 'markdown',
                    })
                  }
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  <option value="text">Text</option>
                  <option value="markdown">Markdown</option>
                  <option value="json">JSON</option>
                </select>
              </div>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 bg-gray-50">
          <button
            onClick={() => setOpen(false)}
            className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700 transition-colors"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}
