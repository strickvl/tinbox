"""Tests for cost estimation functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tinbox.core.cost import (
    CostLevel,
    estimate_context_aware_tokens,
    estimate_cost,
    estimate_document_tokens,
    get_cost_level,
)
from tinbox.core.types import FileType, ModelType


def test_cost_level_thresholds():
    """Test cost level threshold calculations."""
    assert get_cost_level(0.5) == CostLevel.LOW
    assert get_cost_level(2.0) == CostLevel.MEDIUM
    assert get_cost_level(10.0) == CostLevel.HIGH
    assert get_cost_level(25.0) == CostLevel.VERY_HIGH


@pytest.mark.parametrize(
    "file_type,content,expected_tokens",
    [
        (
            FileType.PDF,
            b"PDF content",  # Mock PDF with 2 pages
            1000,  # 2 pages * 500 tokens per page
        ),
        (
            FileType.DOCX,
            "This is a test document",  # 5 words
            7,  # 5 words * 1.3 tokens per word, rounded up
        ),
        (
            FileType.TXT,
            "This is a test",  # 16 characters
            4,  # 16 characters / 4, rounded up
        ),
    ],
)
def test_estimate_document_tokens(tmp_path, file_type, content, expected_tokens):
    """Test token estimation for different file types."""
    file_path = tmp_path / f"test.{file_type.value}"

    if file_type == FileType.PDF:
        # Create a mock PDF file
        file_path.write_bytes(content)
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]  # 2 pages
        with patch("pypdf.PdfReader", return_value=mock_pdf) as mock_reader:
            mock_reader.return_value = mock_pdf
            tokens = estimate_document_tokens(file_path)
            assert tokens == expected_tokens
    elif file_type == FileType.DOCX:
        # Create a mock DOCX file
        file_path.write_bytes(b"dummy docx content")
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text=content)]
        with patch("docx.Document", return_value=mock_doc):
            tokens = estimate_document_tokens(file_path)
            assert tokens == expected_tokens
    else:  # TXT
        file_path.write_text(content)
        tokens = estimate_document_tokens(file_path)
        assert tokens == expected_tokens


@pytest.mark.parametrize(
    "model,tokens,expected_cost,expected_warnings",
    [
        (
            ModelType.OPENAI,
            100_000,
            1.12875,  # 100K input tokens + 3% overhead = 103K input, 100K output -> (103 * 0.00125) + (100 * 0.01) = 0.12875 + 1.0 = 1.12875
            [
                "Large document detected (203,000 tokens). Consider using Ollama for no cost."
            ],
        ),
        (
            ModelType.ANTHROPIC,
            25_000,
            0.45225,  # 25K input tokens + 3% overhead = 25.75K input, 25K output -> (25.75 * 0.003) + (25 * 0.015) = 0.07725 + 0.375 = 0.45225
            [],
        ),
        (
            ModelType.OLLAMA,
            200_000,
            0.0,  # Local model, no cost
            [],
        ),
    ],
)
def test_estimate_cost(tmp_path, model, tokens, expected_cost, expected_warnings):
    """Test cost estimation for different models and token counts."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Test content")

    with patch(
        "tinbox.core.cost.estimate_document_tokens",
        return_value=tokens,
    ):
        estimate = estimate_cost(file_path, model)
        # estimated_tokens now includes input + output tokens
        # For page algorithm: input tokens + 3% overhead + output tokens
        expected_input_with_overhead = tokens + (tokens * 0.03)
        expected_total_tokens = expected_input_with_overhead + tokens
        assert estimate.estimated_tokens == expected_total_tokens
        assert (
            abs(estimate.estimated_cost - expected_cost) < 0.01
        )  # Allow small floating-point differences
        assert all(
            any(expected in warning for warning in estimate.warnings)
            for expected in expected_warnings
        )


def test_cost_threshold_warning():
    """Test cost threshold warning."""
    file_path = Path("test.txt")
    max_cost = 0.5

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=100_000):
        estimate = estimate_cost(
            file_path,
            ModelType.OPENAI,
            max_cost=max_cost,
        )
        assert any(
            f"Estimated cost ($1.13) exceeds maximum threshold (${max_cost:.2f})"
            in warning
            for warning in estimate.warnings
        )


def test_ollama_suggestion():
    """Test Ollama suggestion for large documents."""
    file_path = Path("test.txt")

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=100_000):
        estimate = estimate_cost(file_path, ModelType.OPENAI)
        assert any(
            "Consider using Ollama for no cost" in warning
            for warning in estimate.warnings
        )


def test_estimate_context_aware_tokens():
    """Test context-aware token estimation."""
    base_tokens = 1000

    # Test with default multiplier (4.0)
    result = estimate_context_aware_tokens(base_tokens)
    assert result == 4000

    # Test with custom multiplier
    result = estimate_context_aware_tokens(base_tokens, context_multiplier=2.0)
    assert result == 2000

    # Test with fractional tokens
    result = estimate_context_aware_tokens(1500, context_multiplier=1.5)
    assert result == 2250


@pytest.mark.parametrize(
    "algorithm,expected_input_tokens,expected_output_tokens",
    [
        ("page", 1000, 1000),
        ("sliding-window", 1000, 1000),
        ("context-aware", 4000, 1000),  # 4.0x input tokens for context overhead
    ],
)
def test_estimate_cost_by_algorithm(
    tmp_path, algorithm, expected_input_tokens, expected_output_tokens
):
    """Test cost estimation for different algorithms."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Test content")

    base_tokens = 1000

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=base_tokens):
        estimate = estimate_cost(file_path, ModelType.OPENAI, algorithm=algorithm)

        # Account for prompt overhead (3%)
        if algorithm == "context-aware":
            # Context-aware: 4000 input tokens + 3% prompt overhead
            adjusted_input_tokens = expected_input_tokens + (
                expected_input_tokens * 0.03
            )
        else:
            # Page/sliding-window: 1000 input tokens + 3% prompt overhead
            adjusted_input_tokens = expected_input_tokens + (
                expected_input_tokens * 0.03
            )

        # For OpenAI: input $0.00125/1K, output $0.01/1K
        expected_input_cost = (adjusted_input_tokens / 1000) * 0.00125
        expected_output_cost = (expected_output_tokens / 1000) * 0.01
        expected_total_cost = expected_input_cost + expected_output_cost

        assert abs(estimate.estimated_cost - expected_total_cost) < 0.001
        # estimated_tokens now includes input + output tokens
        expected_total_tokens = adjusted_input_tokens + expected_output_tokens
        assert estimate.estimated_tokens == expected_total_tokens


def test_context_aware_cost_warning():
    """Test context-aware algorithm cost warning."""
    file_path = Path("test.txt")

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=1000):
        estimate = estimate_cost(file_path, ModelType.OPENAI, algorithm="context-aware")

        # Should have context overhead warning
        context_warnings = [
            warning
            for warning in estimate.warnings
            if "Context-aware algorithm uses additional input tokens" in warning
        ]
        assert len(context_warnings) == 1

        # Should mention the overhead amount
        warning = context_warnings[0]
        # Context-aware tokens: 4000, plus 3% prompt overhead = 4120
        # Overhead: 4120 - 1000 = 3120
        # Percentage: 3120/1000 * 100 = 312%
        assert "+3,120 tokens" in warning
        assert "~312% overhead" in warning


def test_context_aware_no_warning_for_ollama():
    """Test that context-aware algorithm doesn't warn for Ollama."""
    file_path = Path("test.txt")

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=1000):
        estimate = estimate_cost(file_path, ModelType.OLLAMA, algorithm="context-aware")

        # Should not have context overhead warning for free models
        context_warnings = [
            warning
            for warning in estimate.warnings
            if "Context-aware algorithm uses additional input tokens" in warning
        ]
        assert len(context_warnings) == 0


def test_glossary_cost_overhead():
    """Test glossary adds proper cost overhead."""
    file_path = Path("test.txt")
    base_tokens = 1000

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=base_tokens):
        # Test without glossary
        estimate_no_glossary = estimate_cost(
            file_path, ModelType.OPENAI, use_glossary=False
        )

        # Test with glossary
        estimate_with_glossary = estimate_cost(
            file_path, ModelType.OPENAI, use_glossary=True
        )

        # With glossary should cost more
        assert (
            estimate_with_glossary.estimated_cost > estimate_no_glossary.estimated_cost
        )

        # Should have glossary warning
        glossary_warnings = [
            warning
            for warning in estimate_with_glossary.warnings
            if "Glossary enabled adds input token overhead" in warning
        ]
        assert len(glossary_warnings) == 1
        assert "~20% of total tokens" in glossary_warnings[0]


def test_glossary_no_warning_for_ollama():
    """Test that glossary doesn't warn for Ollama."""
    file_path = Path("test.txt")

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=1000):
        estimate = estimate_cost(file_path, ModelType.OLLAMA, use_glossary=True)

        # Should not have glossary warning for free models
        glossary_warnings = [
            warning
            for warning in estimate.warnings
            if "Glossary enabled adds input token overhead" in warning
        ]
        assert len(glossary_warnings) == 0


def test_reasoning_effort_minimal_no_warning():
    """Test that minimal reasoning effort doesn't produce warnings."""
    test_file = Path("test.txt")

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=1000):
        estimate = estimate_cost(
            test_file, ModelType.OPENAI, reasoning_effort="minimal"
        )

    # Should not have reasoning effort warning for minimal
    reasoning_warnings = [
        warning for warning in estimate.warnings if "Reasoning effort is" in warning
    ]
    assert len(reasoning_warnings) == 0


@pytest.mark.parametrize("reasoning_effort", ["low", "medium", "high"])
def test_reasoning_effort_warning(reasoning_effort):
    """Test that non-minimal reasoning effort produces warnings."""
    test_file = Path("test.txt")

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=1000):
        estimate = estimate_cost(
            test_file, ModelType.OPENAI, reasoning_effort=reasoning_effort
        )

    # Should have reasoning effort warning for non-minimal
    reasoning_warnings = [
        warning for warning in estimate.warnings if "Reasoning effort is" in warning
    ]
    assert len(reasoning_warnings) == 1
    assert f"Reasoning effort is '{reasoning_effort}'" in reasoning_warnings[0]
    assert "cost and time estimations are unreliable" in reasoning_warnings[0]
    assert "--max-cost" in reasoning_warnings[0]


def test_reasoning_effort_with_ollama():
    """Test that reasoning effort warning appears even with Ollama (free models)."""
    test_file = Path("test.txt")

    with patch("tinbox.core.cost.estimate_document_tokens", return_value=1000):
        estimate = estimate_cost(test_file, ModelType.OLLAMA, reasoning_effort="high")

    # Should have reasoning effort warning even for free models
    reasoning_warnings = [
        warning for warning in estimate.warnings if "Reasoning effort is" in warning
    ]
    assert len(reasoning_warnings) == 1
    assert "Reasoning effort is 'high'" in reasoning_warnings[0]
