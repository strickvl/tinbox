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


# Approximate costs per 1K tokens (as of September 2025)
# Format: (input_cost_per_1k, output_cost_per_1k)
MODEL_COSTS: Dict[ModelType, tuple[float, float]] = {
    ModelType.OPENAI: (0.00125, 0.01),  # $0.00125 per 1K input tokens, $0.01 per 1K output tokens (GPT-5)
    ModelType.ANTHROPIC: (0.003, 0.015),  # $0.003 per 1K input tokens, $0.015 per 1K output tokens (Sonnet 4)
    ModelType.GEMINI: (0.00125, 0.01),  # $0.00125 per 1K input tokens, $0.01 per 1K output tokens (Gemini 2.5 Pro)
    ModelType.OLLAMA: (0.0, 0.0),  # Free for local models
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


def estimate_context_aware_tokens(
    estimated_tokens: int,
    context_multiplier: float = 4
) -> int:
    """Estimate input tokens for context-aware translation.

    Context-aware algorithm uses more input tokens due to:
    - Previous chunk context
    - Previous translation context
    - Translation instructions

    Args:
        estimated_tokens: Base estimated tokens from document
        context_multiplier: Multiplier to account for context overhead

    Returns:
        Estimated input tokens including context overhead
    """
    return int(estimated_tokens * context_multiplier)


def estimate_cost(
    file_path: Path,
    model: ModelType,
    *,
    algorithm: str = "page",
    max_cost: Optional[float] = None,
    use_glossary: bool = False,
    reasoning_effort: str = "minimal",
) -> CostEstimate:
    """Estimate the cost of translating a document.

    Args:
        file_path: Path to the document
        model: Model to use for translation
        algorithm: Translation algorithm to use
        max_cost: Optional maximum cost threshold
        use_glossary: Whether glossary is enabled
        reasoning_effort: Model reasoning effort level

    Returns:
        CostEstimate object with token count, cost, and warnings
    """
    estimated_tokens = estimate_document_tokens(file_path)
    input_cost_per_1k, output_cost_per_1k = MODEL_COSTS.get(model, (0.0, 0.0))
    
    # Calculate input tokens based on algorithm
    if algorithm == "context-aware":
        input_tokens = estimate_context_aware_tokens(estimated_tokens)
        output_tokens = estimated_tokens  # Output tokens remain the same
    else:
        # For page and sliding-window algorithms, input and output tokens are roughly equal
        input_tokens = estimated_tokens
        output_tokens = estimated_tokens
    
    # Add general prompt overhead (system prompt, etc.):
    prompt_factor = 0.03
    input_tokens += input_tokens * prompt_factor
    
    # Add glossary overhead: assume a 20% overhead as per testing
    glossary_factor = 0.20
    if use_glossary:
        glossary_overhead_tokens = (input_tokens + output_tokens) * glossary_factor
        input_tokens += glossary_overhead_tokens

    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    estimated_cost = input_cost + output_cost

    estimated_total_tokens = input_tokens + output_tokens

    # Estimate time (very rough estimate)
    # Assume 30 tokens/second for cloud models, 20 tokens/second for local
    tokens_per_second = 20 if model == ModelType.OLLAMA else 30
    estimated_time = output_tokens / tokens_per_second

    warnings = []

    # Generate warnings
    if model != ModelType.OLLAMA:
        if estimated_total_tokens > 50000:  # More than 50K tokens
            warnings.append(
                f"Large document detected ({estimated_total_tokens:,} tokens). "
                "Consider using Ollama for no cost."
            )

        if algorithm == "context-aware":
            context_overhead = input_tokens - estimated_tokens
            warnings.append(
                f"Context-aware algorithm uses additional input tokens for context "
                f"(+{context_overhead:,.0f} tokens, ~{context_overhead / estimated_tokens * 100:.0f}% overhead). "
                f"This improves translation quality but increases cost."
            )

        if use_glossary:
            warnings.append(
                f"Glossary enabled adds input token overhead (~{glossary_factor * 100:.0f}% of total tokens)."
            )

        if max_cost and estimated_cost > max_cost:
            warnings.append(
                f"Estimated cost (${estimated_cost:.2f}) exceeds maximum "
                f"threshold (${max_cost:.2f})"
            )

    # Reasoning effort warning applies to all models (even free ones like Ollama)
    if reasoning_effort != "minimal":
        warnings.append(
            f"Reasoning effort is '{reasoning_effort}', which means cost and time estimations are unreliable and will be much higher. "
            f"Make sure to set a --max-cost and keep an eye on the live cost and time predictions in the progress bar."
        )

    return CostEstimate(
        estimated_tokens=estimated_total_tokens,
        estimated_cost=estimated_cost,
        estimated_time=estimated_time,
        warnings=warnings,
    )
