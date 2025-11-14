import { create } from 'zustand';
import type { Job, ModelInfo, Language } from '../types/api';

interface Settings {
  defaultModel: string;
  defaultSourceLang: string;
  defaultTargetLang: string;
  defaultAlgorithm: 'page' | 'sliding-window';
  defaultOutputFormat: 'text' | 'json' | 'markdown';
  apiKeys: Record<string, string>;
}

interface AppState {
  // Jobs
  jobs: Job[];
  addJob: (job: Job) => void;
  updateJob: (jobId: string, updates: Partial<Job>) => void;
  removeJob: (jobId: string) => void;
  clearCompletedJobs: () => void;

  // Models and languages
  models: ModelInfo[];
  languages: Language[];
  setModels: (models: ModelInfo[]) => void;
  setLanguages: (languages: Language[]) => void;

  // Settings
  settings: Settings;
  updateSettings: (updates: Partial<Settings>) => void;
  setApiKey: (provider: string, key: string) => void;

  // UI state
  isSettingsOpen: boolean;
  setSettingsOpen: (open: boolean) => void;
}

const defaultSettings: Settings = {
  defaultModel: 'openai:gpt-4o-mini',
  defaultSourceLang: 'auto',
  defaultTargetLang: 'en',
  defaultAlgorithm: 'page',
  defaultOutputFormat: 'text',
  apiKeys: {},
};

export const useStore = create<AppState>((set) => ({
  // Jobs
  jobs: [],
  addJob: (job) => set((state) => ({ jobs: [job, ...state.jobs] })),
  updateJob: (jobId, updates) =>
    set((state) => ({
      jobs: state.jobs.map((job) =>
        job.job_id === jobId ? { ...job, ...updates } : job
      ),
    })),
  removeJob: (jobId) =>
    set((state) => ({
      jobs: state.jobs.filter((job) => job.job_id !== jobId),
    })),
  clearCompletedJobs: () =>
    set((state) => ({
      jobs: state.jobs.filter(
        (job) => job.status !== 'completed' && job.status !== 'failed'
      ),
    })),

  // Models and languages
  models: [],
  languages: [],
  setModels: (models) => set({ models }),
  setLanguages: (languages) => set({ languages }),

  // Settings
  settings: defaultSettings,
  updateSettings: (updates) =>
    set((state) => ({
      settings: { ...state.settings, ...updates },
    })),
  setApiKey: (provider, key) =>
    set((state) => ({
      settings: {
        ...state.settings,
        apiKeys: { ...state.settings.apiKeys, [provider]: key },
      },
    })),

  // UI state
  isSettingsOpen: false,
  setSettingsOpen: (open) => set({ isSettingsOpen: open }),
}));
