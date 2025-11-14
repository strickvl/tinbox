import { useStore } from '../store/useStore';
import { ProgressCard } from './ProgressCard';

export function TranslationQueue() {
  const jobs = useStore((state) => state.jobs);
  const clearCompletedJobs = useStore((state) => state.clearCompletedJobs);

  if (jobs.length === 0) {
    return null;
  }

  const completedCount = jobs.filter((j) => j.status === 'completed' || j.status === 'failed').length;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">
          Translations ({jobs.length})
        </h2>
        {completedCount > 0 && (
          <button
            onClick={clearCompletedJobs}
            className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
          >
            Clear completed ({completedCount})
          </button>
        )}
      </div>

      <div className="space-y-3">
        {jobs.map((job) => (
          <ProgressCard key={job.job_id} job={job} />
        ))}
      </div>
    </div>
  );
}
