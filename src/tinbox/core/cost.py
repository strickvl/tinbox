"""Cost estimation utilities."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path

from tinbox.core.types import FileType, ModelType, file_type_for_path


class CostLevel(StrEnum):
    """Cost level classification."""

    LOW = "low"  # < $1
    MEDIUM = "medium"  # $1-$5
    HIGH = "high"  # $5-$20
    VERY_HIGH = "very_high"  # > $20


# Fallback list prices in USD per 1M tokens (input, output), used only when the
# genai-prices catalog does not know the model name. Checked against the
# providers' pricing pages on 1 November 2026: OpenAI gpt-6.1-sol (prompts up to
# 272K tokens), Anthropic claude-sonnet-5-5, Gemini gemini-3.8-flash (until
# 31 December 2026).
PROVIDER_DEFAULT_PRICES: dict[ModelType, tuple[float, float]] = {
    ModelType.OPENAI: (2.0, 10.0),
    ModelType.ANTHROPIC: (2.0, 10.0),
    ModelType.GEMINI: (0.75, 3.75),
    ModelType.OLLAMA: (0.0, 0.0),
}

# Provider IDs in the genai-prices catalog
_CATALOG_PROVIDERS: dict[ModelType, str] = {
    ModelType.OPENAI: "openai",
    ModelType.ANTHROPIC: "anthropic",
    ModelType.GEMINI: "google",
}


def is_local_model(model: ModelType, *, local: bool = False) -> bool:
    """Return whether local pricing is selected explicitly or through Ollama."""
    return model == ModelType.OLLAMA or local


def _catalog_price(
    model: ModelType, model_name: str | None, input_tokens: int, output_tokens: int
) -> tuple[float, float] | None:
    """Return (input, output) cost in USD from genai-prices, or None if unknown.

    genai-prices ships its price data in the package and reads no network here.
    It applies long-context tiers and dated price changes for the given usage.
    """
    provider_id = _CATALOG_PROVIDERS.get(model)
    if provider_id is None or not model_name:
        return None
    # Imported here so commands that never price a model skip loading the catalog
    from genai_prices import Usage, calc_price

    try:
        price = calc_price(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            model_ref=model_name,
            provider_id=provider_id,
        )
    except LookupError:
        return None
    return float(price.input_price), float(price.output_price)


def _check_custom_prices(price_in: float | None, price_out: float | None) -> bool:
    """Return whether custom prices are set, requiring both or neither."""
    if (price_in is None) != (price_out is None):
        raise ValueError("--price-in and --price-out must be given together")
    return price_in is not None


def price_per_million(
    model: ModelType,
    model_name: str | None = None,
    *,
    price_in: float | None = None,
    price_out: float | None = None,
) -> tuple[float, float, bool]:
    """Return (input, output) USD per 1M tokens and whether the price is known.

    Custom prices win. Otherwise the catalog's short-prompt rate for
    ``model_name`` is used; an unknown or missing model name falls back to
    PROVIDER_DEFAULT_PRICES and reports the price as unknown.
    """
    if _check_custom_prices(price_in, price_out):
        return float(price_in or 0.0), float(price_out or 0.0), True
    # Price a short request so long-context tiers do not apply to the rate
    catalog = _catalog_price(model, model_name, 1000, 1000)
    if catalog is not None:
        return catalog[0] * 1000, catalog[1] * 1000, True
    default_in, default_out = PROVIDER_DEFAULT_PRICES.get(model, (0.0, 0.0))
    return default_in, default_out, False


def calculate_usage_cost(
    model: ModelType,
    input_tokens: int,
    output_tokens: int,
    *,
    model_name: str | None = None,
    local: bool = False,
    price_in: float | None = None,
    price_out: float | None = None,
) -> float:
    """Calculate the cost in USD of one request's token usage.

    Args:
        model: Model provider type
        input_tokens: Number of input/prompt tokens used
        output_tokens: Number of output/completion tokens used
        model_name: Provider model name, looked up in the genai-prices catalog
        local: Whether the model runs locally without API charges
        price_in: Custom input price in USD per 1M tokens (with price_out)
        price_out: Custom output price in USD per 1M tokens (with price_in)

    Returns:
        Cost in USD
    """
    if is_local_model(model, local=local):
        return 0.0
    if _check_custom_prices(price_in, price_out):
        rate_in, rate_out = float(price_in or 0.0), float(price_out or 0.0)
    else:
        catalog = _catalog_price(model, model_name, input_tokens, output_tokens)
        if catalog is not None:
            return catalog[0] + catalog[1]
        rate_in, rate_out = PROVIDER_DEFAULT_PRICES.get(model, (0.0, 0.0))
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


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
        - Plain text: 1 token per 4 characters, rounded up
    """
    file_type = file_type_for_path(file_path)

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

        doc = Document(str(file_path))
        word_count = sum(len(p.text.split()) for p in doc.paragraphs)
        # Rough estimate: 1.3 tokens per word, rounded up
        return int(word_count * 1.3 + 0.999)  # Round up by adding 0.999
    else:  # Plain text
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


# Estimated cost in USD at or above which the CLI asks before translating
CONFIRMATION_COST_THRESHOLD = 1.0


class CostEstimate:
    """Cost estimate for a translation task."""

    def __init__(
        self,
        estimated_tokens: int,
        estimated_cost: float,
        estimated_time: float,
        warnings: list[str],
        *,
        requires_confirmation: bool | None = None,
    ) -> None:
        """Initialize cost estimate.

        Args:
            estimated_tokens: Estimated number of tokens
            estimated_cost: Estimated cost in USD
            estimated_time: Estimated time in seconds
            warnings: List of warning messages
            requires_confirmation: Whether the CLI should ask before translating.
                Defaults to whether the cost reaches CONFIRMATION_COST_THRESHOLD.
        """
        self.estimated_tokens = estimated_tokens
        self.estimated_cost = estimated_cost
        self.estimated_time = estimated_time
        self.warnings = warnings
        self.requires_confirmation = (
            estimated_cost >= CONFIRMATION_COST_THRESHOLD
            if requires_confirmation is None
            else requires_confirmation
        )
        self.cost_level = get_cost_level(estimated_cost)


def estimate_context_aware_tokens(
    estimated_tokens: int, context_multiplier: float = 4
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
    return math.ceil(estimated_tokens * context_multiplier)


def estimate_cost(
    file_path: Path,
    model: ModelType,
    *,
    algorithm: str = "page",
    max_cost: float | None = None,
    use_glossary: bool = False,
    reasoning_effort: str = "minimal",
    local: bool = False,
    model_name: str | None = None,
    price_in: float | None = None,
    price_out: float | None = None,
) -> CostEstimate:
    """Estimate the cost of translating a document.

    Args:
        file_path: Path to the document
        model: Model to use for translation
        algorithm: Translation algorithm to use
        max_cost: Optional maximum cost threshold
        use_glossary: Whether glossary is enabled
        reasoning_effort: Model reasoning effort level
        local: Whether the model runs locally without API charges
        model_name: Provider model name used to look up its price
        price_in: Custom input price in USD per 1M tokens (with price_out)
        price_out: Custom output price in USD per 1M tokens (with price_in)

    Returns:
        CostEstimate object with token count, cost, and warnings
    """
    estimated_tokens = estimate_document_tokens(file_path)
    local_model = is_local_model(model, local=local)

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
    input_tokens = math.ceil(input_tokens * (1 + prompt_factor))

    # Add glossary overhead: assume a 20% overhead as per testing
    glossary_factor = 0.20
    if use_glossary:
        glossary_overhead_tokens = math.ceil(
            (input_tokens + output_tokens) * glossary_factor
        )
        input_tokens += glossary_overhead_tokens

    rate_in, rate_out, price_known = price_per_million(
        model, model_name, price_in=price_in, price_out=price_out
    )
    estimated_cost = (
        0.0
        if local_model
        else (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000
    )

    estimated_total_tokens = input_tokens + output_tokens

    # Estimate time (very rough estimate)
    # Assume 30 tokens/second for cloud models, 20 tokens/second for local
    tokens_per_second = 20 if local_model else 30
    estimated_time = output_tokens / tokens_per_second

    warnings = []

    if algorithm == "context-aware" and estimated_tokens:
        context_overhead = input_tokens - estimated_tokens
        warnings.append(
            f"Context-aware algorithm uses additional input tokens for context "
            f"(approximately +{context_overhead:,} tokens, "
            f"~{context_overhead * 100 // estimated_tokens}% overhead). "
            "Additional input tokens may increase runtime; these estimates are approximate."
        )

    # Price warnings apply only to cloud providers.
    if not local_model:
        if model_name and not price_known:
            warnings.append(
                f"Price for {model_name} unknown, using {model.value} default "
                f"(${rate_in:.2f} input / ${rate_out:.2f} output per 1M tokens). "
                "Pass --price-in and --price-out to set the real price."
            )

        if estimated_total_tokens > 50000:  # More than 50K tokens
            warnings.append(
                f"Large document detected ({estimated_total_tokens:,} tokens). "
                "Consider using Ollama for no cost."
            )

        if use_glossary:
            warnings.append(
                f"Glossary enabled adds input token overhead (~{glossary_factor * 100:.0f}% of total tokens)."
            )

        if max_cost is not None and estimated_cost > max_cost:
            warnings.append(
                f"Estimated cost (${estimated_cost:.2f}) exceeds maximum "
                f"threshold (${max_cost:.2f})"
            )

    if reasoning_effort != "minimal":
        if local_model:
            warnings.append(
                f"Reasoning effort is '{reasoning_effort}', so token and time "
                "estimates are approximate and may be higher."
            )
        else:
            warnings.append(
                f"Reasoning effort is '{reasoning_effort}', which means cost and time estimations are unreliable and will be much higher. "
                f"Make sure to set a --max-cost and keep an eye on the live cost and time predictions in the progress bar."
            )

    return CostEstimate(
        estimated_tokens=estimated_total_tokens,
        estimated_cost=estimated_cost,
        estimated_time=estimated_time,
        warnings=warnings,
        # Ask only when real money is at stake or the estimate is unreliable;
        # cheap cloud runs and local models proceed without a prompt.
        requires_confirmation=(
            not local_model and estimated_cost >= CONFIRMATION_COST_THRESHOLD
        )
        or reasoning_effort != "minimal",
    )
