"""FastAPI server for Tinbox GUI."""

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core import translate_document
from ..core.cost import estimate_translation_cost
from ..core.output import OutputFormat, create_handler
from ..core.processor import load_document
from ..core.translation.litellm import LiteLLMTranslator
from ..core.types import TranslationConfig, ModelType
from ..utils.language import SUPPORTED_LANGUAGES, normalize_language_code

from .jobs import job_queue, Job
from .models import (
    CostEstimateRequest,
    CostEstimateResponse,
    JobResponse,
    LanguagesResponse,
    ModelInfo,
    ModelsResponse,
    TranslateRequest,
    ValidateConfigRequest,
    ValidateConfigResponse,
)

# Create FastAPI app
app = FastAPI(
    title="Tinbox API",
    description="REST API for Tinbox document translation",
    version="1.0.0",
)

# Configure CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model pricing data
MODEL_PRICING = {
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-5": {"input": 1.25, "output": 10.00},
    "openai:gpt-5-mini": {"input": 0.25, "output": 2.00},
    "openai:gpt-5-nano": {"input": 0.05, "output": 0.40},
    "anthropic:claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "anthropic:claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "anthropic:claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Tinbox API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "translate": "/api/translate",
            "cost_estimate": "/api/estimate-cost",
            "models": "/api/models",
            "languages": "/api/languages",
            "jobs": "/api/jobs",
            "progress": "/ws/progress/{job_id}",
        },
    }


@app.get("/api/models", response_model=ModelsResponse)
async def get_models():
    """Get list of available models with pricing."""
    models = []
    for full_name, pricing in MODEL_PRICING.items():
        provider, model_name = full_name.split(":", 1)
        models.append(
            ModelInfo(
                provider=provider,
                model_name=model_name,
                full_name=full_name,
                input_cost_per_1m=pricing["input"],
                output_cost_per_1m=pricing["output"],
            )
        )
    return ModelsResponse(models=models)


@app.get("/api/languages", response_model=LanguagesResponse)
async def get_languages():
    """Get list of supported languages."""
    languages = []
    for code, name in SUPPORTED_LANGUAGES.items():
        languages.append({"code": code, "name": name})
    # Sort by name
    languages.sort(key=lambda x: x["name"])
    return LanguagesResponse(languages=languages)


@app.post("/api/estimate-cost", response_model=CostEstimateResponse)
async def estimate_cost(request: CostEstimateRequest):
    """Estimate translation cost for a document."""
    try:
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Load document to count pages/tokens
        content = await load_document(file_path)

        # Parse model
        provider, model_name = request.model.split(":", 1)
        model_type = ModelType(provider.upper())

        # Get pricing
        pricing = MODEL_PRICING.get(request.model)
        if not pricing:
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

        # Estimate tokens (rough estimate: 1 token ≈ 4 chars for text, or count images)
        if content.content_type == "text/plain":
            total_text = "".join(str(p) for p in content.pages)
            estimated_tokens = len(total_text) // 4
        else:
            # For images, estimate based on image size (GPT-4V style)
            estimated_tokens = len(content.pages) * 1000  # ~1000 tokens per image

        # Assume output is similar length (translation)
        input_tokens = estimated_tokens
        output_tokens = int(estimated_tokens * 1.1)  # 10% longer for translation

        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        # Generate warnings
        warnings = []
        if total_cost > 10.0:
            warnings.append(f"High cost estimate: ${total_cost:.2f}")
        if len(content.pages) > 50:
            warnings.append(f"Large document: {len(content.pages)} pages")

        return CostEstimateResponse(
            estimated_tokens=input_tokens + output_tokens,
            estimated_cost=total_cost,
            num_pages=len(content.pages) if content.content_type != "text/plain" else None,
            warnings=warnings,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/validate-config", response_model=ValidateConfigResponse)
async def validate_config(request: ValidateConfigRequest):
    """Validate API configuration and credentials."""
    try:
        provider, model_name = request.model.split(":", 1)

        # Check if API key is set
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }

        env_var = env_var_map.get(provider)
        if env_var:
            api_key = request.api_key or os.getenv(env_var)
            if not api_key:
                return ValidateConfigResponse(
                    valid=False,
                    message=f"API key not found. Set {env_var} environment variable or provide in settings.",
                    provider=provider,
                )

        # For Ollama, just check if model format is correct
        if provider == "ollama":
            return ValidateConfigResponse(
                valid=True,
                message="Ollama model format valid. Ensure Ollama is running locally.",
                provider=provider,
            )

        return ValidateConfigResponse(
            valid=True,
            message=f"Configuration valid for {provider}",
            provider=provider,
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid model format. Use 'provider:model' (e.g., 'openai:gpt-4o-mini')",
        )


@app.post("/api/translate", response_model=JobResponse)
async def start_translation(request: TranslateRequest):
    """Start a translation job."""
    try:
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Validate languages
        if request.source_lang != "auto":
            try:
                normalize_language_code(request.source_lang)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        try:
            normalize_language_code(request.target_lang)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create job
        job_id = job_queue.create_job(request)

        # Start job execution in background
        async def execute_translation(req: TranslateRequest, progress_callback):
            """Execute the translation."""
            # Load document
            content = await load_document(Path(req.file_path))

            # Create translator
            provider, model_name = req.model.split(":", 1)
            model_type = ModelType(provider.upper())
            translator = LiteLLMTranslator(model_type, model_name)

            # Create config
            config = TranslationConfig(
                source_lang=req.source_lang,
                target_lang=req.target_lang,
                model=req.model,
                algorithm=req.algorithm,
            )

            # Track progress
            start_time = time.time()
            total_pages = len(content.pages)

            class ProgressTracker:
                def __init__(self):
                    self.current_page = 0
                    self.tokens_used = 0
                    self.cost = 0.0

                async def update(self, page: int = 0, tokens: int = 0, cost: float = 0.0):
                    self.current_page = page
                    self.tokens_used += tokens
                    self.cost += cost
                    elapsed = time.time() - start_time
                    await progress_callback(
                        current_page=self.current_page,
                        total_pages=total_pages,
                        tokens_used=self.tokens_used,
                        cost_so_far=self.cost,
                        time_elapsed=elapsed,
                    )

            tracker = ProgressTracker()

            # Translate document
            response = await translate_document(
                content=content,
                config=config,
                translator=translator,
            )

            # Update final progress
            await tracker.update(
                page=total_pages,
                tokens=response.tokens_used,
                cost=response.cost,
            )

            # Save output if path specified
            if req.output_path:
                output_format = OutputFormat(req.output_format.upper())
                handler = create_handler(output_format)
                handler.write(response.text, Path(req.output_path))

            return {
                "translated_text": response.text,
                "tokens_used": response.tokens_used,
                "cost": response.cost,
                "time_elapsed": response.time_elapsed,
                "output_path": req.output_path,
            }

        # Start the job
        await job_queue.start_job(job_id, execute_translation)

        # Return job info
        job = job_queue.get_job(job_id)
        return job.to_response()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs", response_model=list[JobResponse])
async def get_all_jobs():
    """Get all jobs."""
    jobs = job_queue.get_all_jobs()
    return [job.to_response() for job in jobs]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and progress."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_response()


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job."""
    success = job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    return {"message": "Job cancelled", "job_id": job_id}


@app.websocket("/ws/progress/{job_id}")
async def progress_websocket(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()

    job = job_queue.get_job(job_id)
    if not job:
        await websocket.close(code=1008, reason="Job not found")
        return

    # Send initial state
    if job.progress:
        await websocket.send_json(job.progress.model_dump())

    # Register callback for progress updates
    async def send_progress(progress):
        try:
            await websocket.send_json(progress.model_dump())
        except Exception:
            pass

    job_queue.register_progress_callback(job_id, send_progress)

    try:
        # Keep connection alive until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        job_queue.unregister_progress_callbacks(job_id)


def start_server(host: str = "127.0.0.1", port: int = 8765):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
