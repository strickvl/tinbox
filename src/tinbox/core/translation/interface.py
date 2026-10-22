"""Translation interface definitions."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from tinbox.core.types import Glossary, GlossaryEntry, ModelType


class TranslationError(Exception):
    """Base class for translation-related errors."""

    pass


class TranslationRequest(BaseModel):
    """Configuration for a translation request."""

    source_lang: str
    target_lang: str
    content: str | bytes  # Pure content to translate (text or image bytes)
    context: str | None = None  # Supporting context information for better translation
    content_type: str = Field(pattern=r"^(text|image)/.+$")
    model: ModelType
    model_params: dict = Field(
        default_factory=dict
    )  # Additional model-specific parameters
    glossary: Glossary | None = Field(
        default=None,
        description="Optional glossary for consistent translations",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="minimal",
        description="Model reasoning effort level. Higher levels improve quality but increase cost and time significantly.",
    )

    model_config = ConfigDict(frozen=True, protected_namespaces=())


class TranslationResponse(BaseModel):
    """Response from a translation request (also used as aggregate response from algorithms)."""

    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)
    glossary_updates: list[GlossaryEntry] = Field(
        default_factory=list,
        description="New glossary entries discovered during this translation",
    )
    # Aggregate response fields (populated by algorithms, not individual translations)
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


class ModelInterface(Protocol):
    """Protocol for interacting with LLMs."""

    async def translate(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Translate content using the model.

        Args:
            request: The translation request configuration

        Returns:
            A single translation response

        Raises:
            TranslationError: If translation fails
        """
        ...

    async def validate_model(self) -> bool:
        """Check if the model is available and properly configured.

        Returns:
            True if the model is available and configured correctly
        """
        ...


class TranslationWithGlossaryResponse(BaseModel):
    """Structured LLM response when glossary is enabled."""

    translation: str = Field(description="The translated text")
    glossary_extension: list[GlossaryEntry] | None = Field(
        default=None,
        description="New glossary entries discovered during translation (optional)",
    )

    model_config = ConfigDict(frozen=True)


class TranslationWithoutGlossaryResponse(BaseModel):
    """Structured LLM response when glossary is not enabled."""

    translation: str = Field(description="The translated text")

    model_config = ConfigDict(frozen=True)
