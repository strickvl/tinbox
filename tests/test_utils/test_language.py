"""Tests for the language handling module."""

import pytest

from tinbox.utils.language import (
    LanguageCode,
    LanguageError,
    normalize_language_code,
    validate_language_pair,
)


def test_language_code_enum():
    """Test the LanguageCode enum."""
    assert LanguageCode.ENGLISH.value == "en"
    assert LanguageCode.SPANISH.value == "es"
    assert LanguageCode.CHINESE_SIMPLIFIED.value == "zh"
    assert LanguageCode.AUTO.value == "auto"


@pytest.mark.parametrize(
    "input_code,expected",
    [
        ("en", "en"),
        ("eng", "en"),
        ("english", "en"),
        ("English", "en"),
        ("zh", "zh"),
        ("chinese", "zh"),
        ("mandarin", "zh"),
        ("auto", "auto"),
        ("detect", "auto"),
        ("español", "es"),
        ("русский", "ru"),
        ("la", "la"),
        ("lat", "la"),
        ("Latin", "la"),
        ("fr\u200b", "fr"),  # Trailing zero-width space from a pasted command
        ("\u200bfr", "fr"),
        ("fr\u2060", "fr"),  # Word joiner
    ],
)
def test_normalize_language_code_valid(input_code: str, expected: str):
    """Test normalizing valid language codes and aliases."""
    assert normalize_language_code(input_code) == expected


@pytest.mark.parametrize(
    "invalid_code",
    [
        "invalid",
        "xx",
        "123",
        "",
        "klingon",
    ],
)
def test_normalize_language_code_invalid(invalid_code: str):
    """Test normalizing invalid language codes."""
    with pytest.raises(LanguageError):
        normalize_language_code(invalid_code)


def test_normalize_language_code_error_shows_hidden_characters():
    """Characters that survive normalization are visible in the error message."""
    with pytest.raises(LanguageError, match=r"'f\\u0433'"):
        normalize_language_code("f\u0433")  # Cyrillic ghe looks like a Latin r


@pytest.mark.parametrize(
    "source,target,expected",
    [
        ("en", "es", ("en", "es")),
        ("english", "spanish", ("en", "es")),
        ("auto", "zh", ("auto", "zh")),
        ("detect", "japanese", ("auto", "ja")),
        ("русский", "english", ("ru", "en")),
    ],
)
def test_validate_language_pair_valid(
    source: str, target: str, expected: tuple[str, str]
):
    """Test validating valid language pairs."""
    assert validate_language_pair(source, target) == expected


@pytest.mark.parametrize(
    "source,target",
    [
        ("en", "en"),  # Same language
        ("english", "eng"),  # Same language through aliases
        ("invalid", "es"),  # Invalid source
        ("en", "invalid"),  # Invalid target
        ("", "es"),  # Empty source
        ("en", ""),  # Empty target
    ],
)
def test_validate_language_pair_invalid(source: str, target: str):
    """Test validating invalid language pairs."""
    with pytest.raises(LanguageError):
        validate_language_pair(source, target)
