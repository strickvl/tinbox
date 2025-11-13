import type {
  Job,
  TranslateRequest,
  CostEstimate,
  ModelInfo,
  Language,
} from '../types/api';

const API_BASE = window.API_BASE_URL;

export class TinboxAPI {
  static async getModels(): Promise<ModelInfo[]> {
    const response = await fetch(`${API_BASE}/api/models`);
    if (!response.ok) throw new Error('Failed to fetch models');
    const data = await response.json();
    return data.models;
  }

  static async getLanguages(): Promise<Language[]> {
    const response = await fetch(`${API_BASE}/api/languages`);
    if (!response.ok) throw new Error('Failed to fetch languages');
    const data = await response.json();
    return data.languages;
  }

  static async estimateCost(
    filePath: string,
    model: string
  ): Promise<CostEstimate> {
    const response = await fetch(`${API_BASE}/api/estimate-cost`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_path: filePath,
        model,
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to estimate cost');
    }
    return response.json();
  }

  static async validateConfig(
    model: string,
    apiKey?: string
  ): Promise<{ valid: boolean; message: string; provider: string }> {
    const response = await fetch(`${API_BASE}/api/validate-config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        api_key: apiKey,
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to validate config');
    }
    return response.json();
  }

  static async startTranslation(request: TranslateRequest): Promise<Job> {
    const response = await fetch(`${API_BASE}/api/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to start translation');
    }
    return response.json();
  }

  static async getJob(jobId: string): Promise<Job> {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get job');
    }
    return response.json();
  }

  static async getAllJobs(): Promise<Job[]> {
    const response = await fetch(`${API_BASE}/api/jobs`);
    if (!response.ok) throw new Error('Failed to get jobs');
    return response.json();
  }

  static async cancelJob(jobId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to cancel job');
    }
  }

  static createProgressWebSocket(
    jobId: string,
    onProgress: (progress: any) => void,
    onClose?: () => void
  ): WebSocket {
    const ws = new WebSocket(`ws://127.0.0.1:8765/ws/progress/${jobId}`);

    ws.onmessage = (event) => {
      const progress = JSON.parse(event.data);
      onProgress(progress);
    };

    ws.onclose = () => {
      if (onClose) onClose();
    };

    return ws;
  }
}
