"""Render TranslateGemma's text template for OpenAI-compatible completions."""

from tinbox.core.translation.interface import TranslationError, TranslationRequest
from tinbox.core.types import ModelType
from tinbox.utils.language import LanguageCode, normalize_language_code


def build_translategemma_prompt(
    request: TranslationRequest, content: str | bytes
) -> str:
    """Render the single-user text form of Google's TranslateGemma template.

    https://huggingface.co/mlx-community/translategemma-27b-it-4bit/blob/main/chat_template.jinja
    """
    if (
        request.model != ModelType.OPENAI
        or not request.content_type.startswith("text/")
        or not isinstance(content, str)
        or request.context
        or request.glossary is not None
        or request.reasoning_effort != "minimal"
    ):
        raise TranslationError(
            "TranslateGemma requires an openai: text model, an explicit source language, "
            "and no context, glossary, or reasoning options"
        )

    def language(code: str) -> tuple[str, str]:
        normalized = normalize_language_code(code)
        if normalized == "auto":
            raise TranslationError(
                "TranslateGemma requires an explicit source language"
            )
        if normalized in {"zh", "zh-tw"}:
            return "Chinese", "zh-TW" if normalized == "zh-tw" else "zh"
        return LanguageCode(normalized).name.title(), normalized

    source, source_code = language(request.source_lang)
    target, target_code = language(request.target_lang)
    return (
        f"<bos><start_of_turn>user\nYou are a professional {source} ({source_code}) "
        f"to {target} ({target_code}) translator. Your goal is to accurately convey "
        f"the meaning and nuances of the original {source} text while adhering to "
        f"{target} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target} translation, without any additional explanations "
        f"or commentary. Please translate the following {source} text into {target}:\n\n\n"
        f"{content.strip()}<end_of_turn>\n<start_of_turn>model\n"
    )
