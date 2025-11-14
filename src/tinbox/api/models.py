"""Pydantic models for API requests and responses."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a translation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TranslateRequest(BaseModel):
    """Request to translate a document."""

    file_path: str = Field(..., description="Path to the file to translate")
    source_lang: str = Field("auto", description="Source language code (ISO 639-1)")
    target_lang: str = Field(..., description="Target language code (ISO 639-1)")
    model: str = Field("openai:gpt-4o-mini", description="Model to use (provider:model)")
    algorithm: Literal["page", "sliding-window"] = Field(
        "page", description="Translation algorithm"
    )
    output_format: Literal["text", "json", "markdown"] = Field(
        "text", description="Output format"
    )
    output_path: Optional[str] = Field(None, description="Optional output path")


class CostEstimateRequest(BaseModel):
    """Request to estimate translation cost."""

    file_path: str = Field(..., description="Path to the file to estimate")
    model: str = Field("openai:gpt-4o-mini", description="Model to use (provider:model)")


class CostEstimateResponse(BaseModel):
    """Response with cost estimation."""

    estimated_tokens: int = Field(..., description="Estimated token count")
    estimated_cost: float = Field(..., description="Estimated cost in USD")
    num_pages: Optional[int] = Field(None, description="Number of pages (for PDFs)")
    warnings: list[str] = Field(default_factory=list, description="Cost warnings")


class ProgressUpdate(BaseModel):
    """Progress update for a translation job."""

    job_id: str
    status: JobStatus
    current_page: int = 0
    total_pages: int = 0
    tokens_used: int = 0
    cost_so_far: float = 0.0
    time_elapsed: float = 0.0
    message: Optional[str] = None


class JobResponse(BaseModel):
    """Response with job information."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    request: TranslateRequest
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[ProgressUpdate] = None


class ModelInfo(BaseModel):
    """Information about a model."""

    provider: str
    model_name: str
    full_name: str
    input_cost_per_1m: Optional[float] = None
    output_cost_per_1m: Optional[float] = None


class ModelsResponse(BaseModel):
    """Response with available models."""

    models: list[ModelInfo]


class LanguagesResponse(BaseModel):
    """Response with supported languages."""

    languages: list[dict[str, str]]


class ValidateConfigRequest(BaseModel):
    """Request to validate API configuration."""

    model: str = Field(..., description="Model to validate (provider:model)")
    api_key: Optional[str] = Field(None, description="Optional API key to test")


class ValidateConfigResponse(BaseModel):
    """Response from config validation."""

    valid: bool
    message: str
    provider: str
