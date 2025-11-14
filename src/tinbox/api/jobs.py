"""Job queue management for batch processing."""

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .models import JobResponse, JobStatus, ProgressUpdate, TranslateRequest


class Job:
    """Represents a translation job."""

    def __init__(self, request: TranslateRequest):
        self.id = str(uuid4())
        self.request = request
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress: Optional[ProgressUpdate] = None
        self.task: Optional[asyncio.Task] = None

    def to_response(self) -> JobResponse:
        """Convert job to API response."""
        return JobResponse(
            job_id=self.id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            request=self.request,
            result=self.result,
            error=self.error,
            progress=self.progress,
        )

    def update_progress(
        self,
        current_page: int = 0,
        total_pages: int = 0,
        tokens_used: int = 0,
        cost_so_far: float = 0.0,
        time_elapsed: float = 0.0,
        message: Optional[str] = None,
    ):
        """Update job progress."""
        self.progress = ProgressUpdate(
            job_id=self.id,
            status=self.status,
            current_page=current_page,
            total_pages=total_pages,
            tokens_used=tokens_used,
            cost_so_far=cost_so_far,
            time_elapsed=time_elapsed,
            message=message,
        )


class JobQueue:
    """Manages translation job queue."""

    def __init__(self, max_concurrent: int = 2):
        self.jobs: dict[str, Job] = {}
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._progress_callbacks: dict[str, list[Callable]] = {}

    def create_job(self, request: TranslateRequest) -> str:
        """Create a new job and add to queue."""
        job = Job(request)
        self.jobs[job.id] = job
        return job.id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list[Job]:
        """Get all jobs."""
        return list(self.jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self.get_job(job_id)
        if not job:
            return False

        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True

        if job.status == JobStatus.RUNNING and job.task:
            job.task.cancel()
            job.status = JobStatus.CANCELLED
            return True

        return False

    def register_progress_callback(self, job_id: str, callback: Callable):
        """Register a callback for job progress updates."""
        if job_id not in self._progress_callbacks:
            self._progress_callbacks[job_id] = []
        self._progress_callbacks[job_id].append(callback)

    def unregister_progress_callbacks(self, job_id: str):
        """Unregister all callbacks for a job."""
        if job_id in self._progress_callbacks:
            del self._progress_callbacks[job_id]

    async def _notify_progress(self, job_id: str):
        """Notify all callbacks about progress update."""
        job = self.get_job(job_id)
        if not job or not job.progress:
            return

        callbacks = self._progress_callbacks.get(job_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(job.progress)
                else:
                    callback(job.progress)
            except Exception:
                pass  # Ignore callback errors

    async def execute_job(self, job_id: str, executor: Callable):
        """Execute a job with the provided executor function."""
        job = self.get_job(job_id)
        if not job:
            return

        async with self.semaphore:
            try:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()

                # Progress callback wrapper
                async def progress_callback(**kwargs):
                    job.update_progress(**kwargs)
                    await self._notify_progress(job_id)

                # Execute the translation
                result = await executor(job.request, progress_callback)

                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now()
                job.result = result
                await self._notify_progress(job_id)

            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                job.error = "Job cancelled by user"
                await self._notify_progress(job_id)
                raise

            except Exception as e:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
                await self._notify_progress(job_id)

    async def start_job(self, job_id: str, executor: Callable):
        """Start executing a job in the background."""
        job = self.get_job(job_id)
        if not job:
            return

        job.task = asyncio.create_task(self.execute_job(job_id, executor))


# Global job queue instance
job_queue = JobQueue(max_concurrent=2)
