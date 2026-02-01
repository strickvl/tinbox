"""Language code handling and validation for Tinbox."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LanguageCode(str, Enum):
    """ISO 639-1 language codes with common aliases."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE_SIMPLIFIED = "zh"
    CHINESE_TRADITIONAL = "zh-tw"
    ARABIC = "ar"
    HINDI = "hi"
    BENGALI = "bn"
    DUTCH = "nl"
    TURKISH = "tr"
    VIETNAMESE = "vi"
    THAI = "th"
    INDONESIAN = "id"
    POLISH = "pl"
    UKRAINIAN = "uk"
    CZECH = "cs"
    SWEDISH = "sv"
    DANISH = "da"
    FINNISH = "fi"
    GREEK = "el"
    HEBREW = "he"
    AUTO = "auto"  # Special case for auto-detection


# Common aliases mapping to ISO 639-1 codes
LANGUAGE_ALIASES: dict[str, str] = {
    # English aliases
    "eng": "en",
    "english": "en",
    # Spanish aliases
    "spa": "es",
    "spanish": "es",
    "español": "es",
    # French aliases
    "fra": "fr",
    "french": "fr",
    "français": "fr",
    # German aliases
    "deu": "de",
    "ger": "de",
    "german": "de",
    "deutsch": "de",
    # Chinese aliases
    "chi": "zh",
    "zho": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "chinese-simplified": "zh",
    "chinese-traditional": "zh-tw",
    # Japanese aliases
    "jpn": "ja",
    "japanese": "ja",
    "日本語": "ja",
    # Korean aliases
    "kor": "ko",
    "korean": "ko",
    "한국어": "ko",
    # Russian aliases
    "rus": "ru",
    "russian": "ru",
    "русский": "ru",
    # Auto-detection aliases
    "auto": "auto",
    "detect": "auto",
    "automatic": "auto",
}


class LanguageError(Exception):
    """Exception raised for language-related errors."""

    pass


class LanguageSupport(BaseModel):
    """Language support configuration and metadata."""

    code: LanguageCode
    name: str
    native_name: str | None = None
    supported_models: list[str] = Field(default_factory=list)
    bidirectional: bool = True  # Whether the language can be both source and target

    model_config = ConfigDict(frozen=True)


def normalize_language_code(code: str) -> str:
    """Normalize a language code or alias to its ISO 639-1 form.

    Args:
        code: A language code or alias (e.g., 'en', 'eng', 'english')

    Returns:
        The normalized ISO 639-1 language code

    Raises:
        LanguageError: If the language code is not recognized
    """
    normalized = code.lower().strip()

    # Check if it's already a valid ISO 639-1 code
    try:
        return LanguageCode(normalized).value
    except ValueError:
        pass

    # Check aliases
    if normalized in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[normalized]

    raise LanguageError(
        f"Unsupported language code: {code}. "
        "Please use a valid ISO 639-1 code or common language name."
    )


def validate_language_pair(source: str, target: str) -> tuple[str, str]:
    """Validate a source-target language pair.

    Args:
        source: Source language code or alias
        target: Target language code or alias

    Returns:
        Tuple of normalized (source, target) language codes

    Raises:
        LanguageError: If either language code is invalid or the pair is not supported
    """
    try:
        norm_source = normalize_language_code(source)
        norm_target = normalize_language_code(target)
    except LanguageError as e:
        raise LanguageError(f"Invalid language code: {e!s}")

    # Special handling for auto-detection
    if norm_source == "auto":
        return norm_source, norm_target

    # Ensure both languages are different
    if norm_source == norm_target:
        raise LanguageError(f"Source and target languages are the same: {norm_source}")

    return norm_source, norm_target
