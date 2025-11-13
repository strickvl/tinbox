export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface TranslateRequest {
  file_path: string;
  source_lang: string;
  target_lang: string;
  model: string;
  algorithm: 'page' | 'sliding-window';
  output_format: 'text' | 'json' | 'markdown';
  output_path?: string;
}

export interface CostEstimate {
  estimated_tokens: number;
  estimated_cost: number;
  num_pages?: number;
  warnings: string[];
}

export interface ProgressUpdate {
  job_id: string;
  status: JobStatus;
  current_page: number;
  total_pages: number;
  tokens_used: number;
  cost_so_far: number;
  time_elapsed: number;
  message?: string;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  request: TranslateRequest;
  result?: {
    translated_text: string;
    tokens_used: number;
    cost: number;
    time_elapsed: number;
    output_path?: string;
  };
  error?: string;
  progress?: ProgressUpdate;
}

export interface ModelInfo {
  provider: string;
  model_name: string;
  full_name: string;
  input_cost_per_1m?: number;
  output_cost_per_1m?: number;
}

export interface Language {
  code: string;
  name: string;
}
