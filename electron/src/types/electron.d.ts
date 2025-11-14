export interface ElectronAPI {
  getSetting: (key: string) => Promise<any>;
  setSetting: (key: string, value: any) => Promise<boolean>;
  deleteSetting: (key: string) => Promise<boolean>;
  openFileDialog: (options?: {
    filters?: Array<{ name: string; extensions: string[] }>;
  }) => Promise<string[]>;
  saveFileDialog: (options?: {
    defaultPath?: string;
    filters?: Array<{ name: string; extensions: string[] }>;
  }) => Promise<string | undefined>;
}

declare global {
  interface Window {
    electron: ElectronAPI;
    API_BASE_URL: string;
  }
}

export {};
