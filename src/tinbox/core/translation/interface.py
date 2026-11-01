"""Translation interface definitions."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from tinbox.core.review import ReviewPair
from tinbox.core.types import Glossary, GlossaryEntry, ModelType, TranslationProtocol


class TranslationError(Exception):
    """Base class for translation-related errors.

    ``cost`` and the token counts record what the provider billed for the failed
    request, for example a reply rejected for its stop reason or invalid JSON.
    """

    def __init__(
        self,
        *args: object,
        cost: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        super().__init__(*args)
        self.cost = cost
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class ProviderAccessError(TranslationError):
    """The provider refused the account, so no later request in this run can work.

    Raised for bad API keys, missing permissions, and exhausted quota.
    """


class TranslationRequest(BaseModel):
    """Configuration for a translation request."""

    source_lang: str
    target_lang: str
    content: str | bytes  # Pure content to translate (text or image bytes)
    context: str | None = None  # Supporting context information for better translation
    content_type: str = Field(pattern=r"^(text|image)/.+$")
    model: ModelType
    translation_protocol: TranslationProtocol = TranslationProtocol.CHAT
    local_model: bool = False
    model_params: dict = Field(
        default_factory=dict
    )  # Additional model-specific parameters
    glossary: Glossary | None = Field(
        default=None,
        description="Optional glossary for consistent translations",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="minimal",
        description="Model reasoning effort level. Provider mapping varies; higher effort may increase time and cost.",
    )
    price_in: float | None = None  # Custom USD per 1M input tokens
    price_out: float | None = None  # Custom USD per 1M output tokens

    model_config = ConfigDict(frozen=True, protected_namespaces=())


class TranslationResponse(BaseModel):
    """Response from a translation request (also used as aggregate response from algorithms)."""

    text: str
    tokens_used: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    time_taken: float = Field(ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
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
    failed_chunks: list[int] = Field(
        default_factory=list,
        description="Chunk or window numbers left untranslated after a stop",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings during translation",
    )
    review_pairs: list[ReviewPair] | None = Field(
        default=None,
        description="Source and translation chunks captured by an opted-in algorithm",
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
