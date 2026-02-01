"""Core data types for the Tinbox translation tool."""

from enum import Enum
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FileType(str, Enum):
    """Supported file types for translation."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class ModelType(str, Enum):
    """Supported LLM model providers."""

    OPENAI = "openai"  # OpenAI models (GPT-4, etc.)
    ANTHROPIC = "anthropic"  # Anthropic models (Claude)
    OLLAMA = "ollama"  # Local models via Ollama
    GEMINI = "gemini"  # Google's Gemini models


class TranslationConfig(BaseModel):
    """Configuration for translation tasks."""

    # Basic settings
    source_lang: str
    target_lang: str
    model: ModelType
    model_name: str = Field(
        description="Specific model name (e.g., 'gpt-4o' for OpenAI, 'claude-3-sonnet' for Anthropic)",
    )
    algorithm: Literal["page", "sliding-window", "context-aware"]
    input_file: Path
    output_file: Optional[Path] = None

    # UI and progress settings
    verbose: bool = Field(
        default=False,
        description="Whether to show detailed progress information",
    )
    progress_callback: Optional[Callable[[int], None]] = Field(
        default=None,
        description="Callback function to update progress (receives tokens processed)",
    )

    # Cost control settings
    max_cost: Optional[float] = Field(
        default=None,
        description="Maximum cost threshold in USD",
        ge=0.0,
    )
    force: bool = Field(
        default=False,
        description="Whether to skip cost and size warnings",
    )

    # Algorithm-specific settings
    window_size: int = Field(
        default=2000,
        gt=0,
        description="Window size for sliding window translation",
    )
    overlap_size: int = Field(
        default=200,
        gt=0,
        description="Overlap size for sliding window translation",
    )

    # Context-aware algorithm specific settings
    context_size: Optional[int] = Field(
        default=2000,
        gt=0,
        description="Target size for context-aware chunks (characters)",
    )
    custom_split_token: Optional[str] = Field(
        default=None,
        description="Custom token to split text on (context-aware only, ignores context_size)",
    )

    # Checkpoint settings
    checkpoint_dir: Optional[Path] = Field(
        default=None,
        description="Directory to store checkpoints",
    )
    checkpoint_frequency: int = Field(
        default=1,
        gt=0,
        description="Save checkpoint every N pages/chunks",
    )
    resume_from_checkpoint: bool = Field(
        default=True,
        description="Whether to try resuming from checkpoint",
    )

    # Glossary settings (optional feature; disabled by default)
    use_glossary: bool = Field(
        default=False,
        description="Enable glossary for consistent term translations.",
    )

    # Model reasoning settings
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="minimal",
        description="Model reasoning effort level. Higher levels improve quality but increase cost and time significantly.",
    )

    model_config = ConfigDict(
        frozen=True,  # Make config immutable
        arbitrary_types_allowed=True,  # Allow Callable type for progress_callback
        protected_namespaces=(),  # Allow model_* field names
    )


class TranslationResult(BaseModel):
    """Result of a translation operation."""

    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)

    # Failed page tracking (for page-by-page algorithm)
    failed_pages: list[int] = Field(
        default_factory=list,
        description="List of page numbers that failed to translate",
    )
    page_errors: dict[int, str] = Field(
        default_factory=dict,
        description="Mapping from page number to error message",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings during translation",
    )

    model_config = ConfigDict(frozen=True)


# ----------------------
# Glossary data types
# ----------------------


class GlossaryEntry(BaseModel):
    """A single glossary entry mapping source term to target translation."""

    term: str = Field(description="Term in source language")
    translation: str = Field(description="Translation in target language")

    model_config = ConfigDict(frozen=True)


class Glossary(BaseModel):
    """Collection of translation glossary entries."""

    entries: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from source terms to target translations",
    )

    def extend(self, new_entries: list[GlossaryEntry]) -> "Glossary":
        """Extend glossary with multiple entries and return new instance."""
        if not new_entries:
            return Glossary(entries=self.entries.copy())
        updated_entries = self.entries.copy()
        for entry in new_entries:
            # Last write wins; overwrite existing to ensure latest consistent mapping
            updated_entries[entry.term] = entry.translation
        return Glossary(entries=updated_entries)

    def to_context_string(self) -> str:
        """Convert glossary to context string for LLM consumption."""
        if not self.entries:
            return "[The glossary is still empty… add terms as they are encountered.]"
        lines: list[str] = ["[GLOSSARY]"]
        for term, translation in self.entries.items():
            lines.append(f"{term} -> {translation}")
        lines.append("[/GLOSSARY]")
        return "\n".join(lines)

    model_config = ConfigDict(frozen=True)
