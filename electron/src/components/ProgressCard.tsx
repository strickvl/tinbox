import { useEffect } from 'react';
import { useStore } from '../store/useStore';
import { useProgressWebSocket } from '../hooks/useWebSocket';
import type { Job } from '../types/api';
import { Check, AlertCircle, X, Download } from './Icons';
import { TinboxAPI } from '../utils/api';

interface ProgressCardProps {
  job: Job;
}

export function ProgressCard({ job }: ProgressCardProps) {
  const updateJob = useStore((state) => state.updateJob);
  const removeJob = useStore((state) => state.removeJob);

  // Connect to WebSocket for progress updates
  useProgressWebSocket(
    job.status === 'running' ? job.job_id : null,
    (progress) => {
      updateJob(job.job_id, { progress });
    }
  );

  const progress = job.progress;
  const progressPercent = progress
    ? Math.round((progress.current_page / progress.total_pages) * 100)
    : 0;

  const handleCancel = async () => {
    try {
      await TinboxAPI.cancelJob(job.job_id);
      updateJob(job.job_id, { status: 'cancelled' });
    } catch (error) {
      console.error('Failed to cancel job:', error);
    }
  };

  const handleDownload = async () => {
    if (!job.result?.translated_text) return;

    const outputPath = await window.electron.saveFileDialog({
      defaultPath: `translated.${job.request.output_format}`,
      filters: [
        { name: 'Text Files', extensions: ['txt'] },
        { name: 'Markdown Files', extensions: ['md'] },
        { name: 'JSON Files', extensions: ['json'] },
      ],
    });

    if (outputPath) {
      // Save the file (in a real app, you'd write the content)
      alert(`Translation saved to ${outputPath}`);
    }
  };

  const getStatusColor = () => {
    switch (job.status) {
      case 'completed':
        return 'text-green-600';
      case 'failed':
        return 'text-red-600';
      case 'cancelled':
        return 'text-gray-500';
      case 'running':
        return 'text-blue-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStatusIcon = () => {
    switch (job.status) {
      case 'completed':
        return <Check className="w-5 h-5 text-green-600" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      case 'cancelled':
        return <X className="w-5 h-5 text-gray-500" />;
      default:
        return null;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 animate-slide-up">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-medium text-gray-900 truncate">
            {job.request.file_path.split('/').pop()}
          </h3>
          <p className="text-xs text-gray-500 mt-1">
            {job.request.source_lang} → {job.request.target_lang} • {job.request.model}
          </p>
        </div>

        <div className="flex items-center gap-2 ml-3">
          {getStatusIcon()}
          {job.status === 'running' && (
            <button
              onClick={handleCancel}
              className="text-gray-400 hover:text-gray-600 transition-colors"
              title="Cancel"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          {job.status === 'completed' && (
            <button
              onClick={handleDownload}
              className="text-primary-600 hover:text-primary-700 transition-colors"
              title="Download"
            >
              <Download className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={() => removeJob(job.job_id)}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            title="Remove"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Progress bar */}
      {job.status === 'running' && progress && (
        <div className="mb-3">
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span>
              Page {progress.current_page} of {progress.total_pages}
            </span>
            <span>{progressPercent}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
            <div
              className="bg-primary-600 h-full transition-all duration-300 ease-out"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      )}

      {/* Status info */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span className={getStatusColor()}>
          {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
        </span>

        {progress && (
          <div className="flex gap-3">
            {progress.tokens_used > 0 && (
              <span>{progress.tokens_used.toLocaleString()} tokens</span>
            )}
            {progress.cost_so_far > 0 && (
              <span>${progress.cost_so_far.toFixed(4)}</span>
            )}
          </div>
        )}

        {job.result && (
          <div className="flex gap-3">
            <span>{job.result.tokens_used.toLocaleString()} tokens</span>
            <span>${job.result.cost.toFixed(4)}</span>
            <span>{job.result.time_elapsed.toFixed(1)}s</span>
          </div>
        )}
      </div>

      {/* Error message */}
      {job.error && (
        <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
          {job.error}
        </div>
      )}
    </div>
  );
}
