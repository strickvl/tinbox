"""Cost estimation utilities."""

from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from tinbox.core.types import FileType, ModelType


class CostLevel(str, Enum):
    """Cost level classification."""

    LOW = "low"  # < $1
    MEDIUM = "medium"  # $1-$5
    HIGH = "high"  # $5-$20
    VERY_HIGH = "very_high"  # > $20


# Approximate costs per 1K tokens (as of March 2024)
# Note: GPT-5 models (gpt-5, gpt-5-mini, gpt-5-nano) have different pricing
# Check https://openai.com/pricing for current GPT-5 pricing
MODEL_COSTS: Dict[ModelType, float] = {
    ModelType.OPENAI: 0.03,  # Default: $0.03 per 1K input tokens, $0.06 per 1K output tokens
    ModelType.ANTHROPIC: 0.003,  # $0.003 per 1K input tokens, $0.015 per 1K output tokens
    ModelType.OLLAMA: 0.0,  # Free for local models
}


def estimate_document_tokens(file_path: Path) -> int:
    """Estimate the number of tokens in a document.

    Args:
        file_path: Path to the document

    Returns:
        Estimated number of tokens

    Note:
        These are rough estimates:
        - PDF: 500 tokens per page
        - DOCX: 1.3 tokens per word, rounded up
        - TXT: 1 token per 4 characters, rounded up
    """
    file_type = FileType(file_path.suffix.lstrip(".").lower())

    # Read sample of document to estimate tokens
    if file_type == FileType.PDF:
        # For PDFs, estimate based on number of pages
        import pypdf

        with open(file_path, "rb") as f:
            pdf = pypdf.PdfReader(f)
            # Rough estimate: 500 tokens per page
            return len(pdf.pages) * 500
    elif file_type == FileType.DOCX:
        # For DOCX, estimate based on word count
        from docx import Document

        doc = Document(file_path)
        word_count = sum(len(p.text.split()) for p in doc.paragraphs)
        # Rough estimate: 1.3 tokens per word, rounded up
        return int(word_count * 1.3 + 0.999)  # Round up by adding 0.999
    else:  # TXT
        # For text files, estimate based on character count
        text = file_path.read_text()
        # Rough estimate: 1 token per 4 characters, rounded up
        return -(-len(text) // 4)  # Ceiling division


def get_cost_level(cost: float) -> CostLevel:
    """Get the cost level classification.

    Args:
        cost: Estimated cost in USD

    Returns:
        Cost level classification
    """
    if cost < 1.0:
        return CostLevel.LOW
    elif cost < 5.0:
        return CostLevel.MEDIUM
    elif cost < 20.0:
        return CostLevel.HIGH
    else:
        return CostLevel.VERY_HIGH


class CostEstimate:
    """Cost estimate for a translation task."""

    def __init__(
        self,
        estimated_tokens: int,
        estimated_cost: float,
        estimated_time: float,
        warnings: list[str],
    ) -> None:
        """Initialize cost estimate.

        Args:
            estimated_tokens: Estimated number of tokens
            estimated_cost: Estimated cost in USD
            estimated_time: Estimated time in seconds
            warnings: List of warning messages
        """
        self.estimated_tokens = estimated_tokens
        self.estimated_cost = estimated_cost
        self.estimated_time = estimated_time
        self.warnings = warnings
        self.cost_level = get_cost_level(estimated_cost)


def estimate_cost(
    file_path: Path,
    model: ModelType,
    *,
    max_cost: Optional[float] = None,
) -> CostEstimate:
    """Estimate the cost of translating a document.

    Args:
        file_path: Path to the document
        model: Model to use for translation
        max_cost: Optional maximum cost threshold

    Returns:
        CostEstimate object with token count, cost, and warnings
    """
    estimated_tokens = estimate_document_tokens(file_path)
    cost_per_1k = MODEL_COSTS.get(model, 0.0)
    estimated_cost = (estimated_tokens / 1000) * cost_per_1k

    # Estimate time (very rough estimate)
    # Assume 5 tokens/second for cloud models, 20 tokens/second for local
    tokens_per_second = 20 if model == ModelType.OLLAMA else 5
    estimated_time = estimated_tokens / tokens_per_second

    warnings = []

    # Generate warnings
    if model != ModelType.OLLAMA:
        if estimated_tokens > 50000:  # More than 50K tokens
            warnings.append(
                f"Large document detected ({estimated_tokens:,} tokens). "
                "Consider using Ollama for better performance and no cost."
            )

        if max_cost and estimated_cost > max_cost:
            warnings.append(
                f"Estimated cost (${estimated_cost:.2f}) exceeds maximum "
                f"threshold (${max_cost:.2f})"
            )

    return CostEstimate(
        estimated_tokens=estimated_tokens,
        estimated_cost=estimated_cost,
        estimated_time=estimated_time,
        warnings=warnings,
    )
