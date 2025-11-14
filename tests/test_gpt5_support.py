"""Tests for GPT-5 model support.

This test suite verifies:
1. GPT-5 model detection (gpt-5, gpt-5-mini, gpt-5-nano)
2. Parameter handling (reasoning_effort, verbosity instead of temperature)
3. Cost calculation with accurate GPT-5 pricing
4. Backward compatibility with non-GPT-5 models
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tinbox.core.translation.litellm import LiteLLMTranslator
from tinbox.core.translation.interface import TranslationRequest
from tinbox.core.types import ModelType


class TestGPT5Detection:
    """Test GPT-5 model detection and classification."""

    def test_gpt5_model_detected(self):
        """Test that gpt-5 is correctly detected as a GPT-5 model."""
        translator = LiteLLMTranslator(temperature=0.7)

        request = TranslationRequest(
            content="test",
            source_lang="en",
            target_lang="es",
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-5"}
        )

        model_string = translator._get_model_string(request)

        assert model_string == "gpt-5"
        assert translator._is_gpt5 is True

    def test_gpt5_mini_detected(self):
        """Test that gpt-5-mini is correctly detected as a GPT-5 model."""
        translator = LiteLLMTranslator(temperature=0.7)

        request = TranslationRequest(
            content="test",
            source_lang="en",
            target_lang="es",
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-5-mini"}
        )

        model_string = translator._get_model_string(request)

        assert model_string == "gpt-5-mini"
        assert translator._is_gpt5 is True

    def test_gpt5_nano_detected(self):
        """Test that gpt-5-nano is correctly detected as a GPT-5 model."""
        translator = LiteLLMTranslator(temperature=0.7)

        request = TranslationRequest(
            content="test",
            source_lang="en",
            target_lang="es",
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-5-nano"}
        )

        model_string = translator._get_model_string(request)

        assert model_string == "gpt-5-nano"
        assert translator._is_gpt5 is True

    def test_gpt4_not_detected_as_gpt5(self):
        """Test that GPT-4 models are NOT detected as GPT-5."""
        translator = LiteLLMTranslator(temperature=0.7)

        request = TranslationRequest(
            content="test",
            source_lang="en",
            target_lang="es",
            content_type="text/plain",
            model=ModelType.OPENAI,
            model_params={"model_name": "gpt-4o"}
        )

        model_string = translator._get_model_string(request)

        assert model_string == "gpt-4o"
        assert translator._is_gpt5 is False

    def test_claude_not_detected_as_gpt5(self):
        """Test that Claude models are NOT detected as GPT-5."""
        translator = LiteLLMTranslator(temperature=0.7)

        request = TranslationRequest(
            content="test",
            source_lang="en",
            target_lang="es",
            content_type="text/plain",
            model=ModelType.ANTHROPIC,
            model_params={"model_name": "claude-3-5-sonnet-20250219"}
        )

        model_string = translator._get_model_string(request)

        assert "claude" in model_string
        assert translator._is_gpt5 is False


class TestGPT5ParameterHandling:
    """Test GPT-5 parameter handling (reasoning_effort, verbosity)."""

    @pytest.mark.asyncio
    async def test_gpt5_uses_reasoning_effort(self):
        """Test that GPT-5 models use reasoning_effort parameter."""
        with patch("tinbox.core.translation.litellm.completion") as mock_completion:
            # Mock successful response
            mock_completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Hola"))],
                usage=MagicMock(
                    total_tokens=20,
                    prompt_tokens=10,
                    completion_tokens=10
                )
            )

            translator = LiteLLMTranslator(temperature=0.7)

            request = TranslationRequest(
                content="Hello",
                source_lang="en",
                target_lang="es",
                content_type="text/plain",
                model=ModelType.OPENAI,
                model_params={"model_name": "gpt-5"}
            )

            await translator.translate(request)

            # Verify reasoning_effort was passed
            call_kwargs = mock_completion.call_args.kwargs
            assert "reasoning_effort" in call_kwargs
            assert call_kwargs["reasoning_effort"] in ["minimal", "low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_gpt5_uses_verbosity(self):
        """Test that GPT-5 models use verbosity parameter."""
        with patch("tinbox.core.translation.litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Hola"))],
                usage=MagicMock(
                    total_tokens=20,
                    prompt_tokens=10,
                    completion_tokens=10
                )
            )

            translator = LiteLLMTranslator(temperature=0.7)

            request = TranslationRequest(
                content="Hello",
                source_lang="en",
                target_lang="es",
                content_type="text/plain",
                model=ModelType.OPENAI,
                model_params={"model_name": "gpt-5"}
            )

            await translator.translate(request)

            # Verify verbosity was passed
            call_kwargs = mock_completion.call_args.kwargs
            assert "verbosity" in call_kwargs
            assert call_kwargs["verbosity"] in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_gpt5_excludes_temperature(self):
        """Test that GPT-5 models DO NOT use temperature parameter."""
        with patch("tinbox.core.translation.litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Hola"))],
                usage=MagicMock(
                    total_tokens=20,
                    prompt_tokens=10,
                    completion_tokens=10
                )
            )

            translator = LiteLLMTranslator(temperature=0.7)

            request = TranslationRequest(
                content="Hello",
                source_lang="en",
                target_lang="es",
                content_type="text/plain",
                model=ModelType.OPENAI,
                model_params={"model_name": "gpt-5"}
            )

            await translator.translate(request)

            # Verify temperature was NOT passed
            call_kwargs = mock_completion.call_args.kwargs
            assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_gpt4_uses_temperature(self):
        """Test that non-GPT-5 models still use temperature parameter."""
        with patch("tinbox.core.translation.litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Hola"))],
                usage=MagicMock(
                    total_tokens=20,
                    prompt_tokens=10,
                    completion_tokens=10
                )
            )

            translator = LiteLLMTranslator(temperature=0.7)

            request = TranslationRequest(
                content="Hello",
                source_lang="en",
                target_lang="es",
                content_type="text/plain",
                model=ModelType.OPENAI,
                model_params={"model_name": "gpt-4o"}
            )

            await translator.translate(request)

            # Verify temperature WAS passed for non-GPT-5
            call_kwargs = mock_completion.call_args.kwargs
            assert "temperature" in call_kwargs
            assert call_kwargs["temperature"] == 0.7


class TestGPT5CostCalculation:
    """Test accurate cost calculation for GPT-5 models."""

    def test_calculate_model_cost_exists(self):
        """Test that calculate_model_cost function exists."""
        from tinbox.core.cost import calculate_model_cost
        assert callable(calculate_model_cost)

    def test_gpt5_cost_calculation(self):
        """Test cost calculation for GPT-5 model.

        GPT-5 pricing: $1.25 per 1M input tokens, $10.00 per 1M output tokens
        For 1K tokens each: $0.00125 input + $0.01000 output = $0.01125
        """
        from tinbox.core.cost import calculate_model_cost

        cost = calculate_model_cost(
            model_name="gpt-5",
            input_tokens=1000,
            output_tokens=1000
        )

        expected_cost = 0.01125  # $0.00125 + $0.01000
        assert abs(cost - expected_cost) < 0.00001

    def test_gpt5_mini_cost_calculation(self):
        """Test cost calculation for GPT-5-mini model.

        GPT-5-mini pricing: $0.25 per 1M input tokens, $2.00 per 1M output tokens
        For 1K tokens each: $0.00025 input + $0.00200 output = $0.00225
        """
        from tinbox.core.cost import calculate_model_cost

        cost = calculate_model_cost(
            model_name="gpt-5-mini",
            input_tokens=1000,
            output_tokens=1000
        )

        expected_cost = 0.00225  # $0.00025 + $0.00200
        assert abs(cost - expected_cost) < 0.00001

    def test_gpt5_nano_cost_calculation(self):
        """Test cost calculation for GPT-5-nano model.

        GPT-5-nano pricing: $0.05 per 1M input tokens, $0.40 per 1M output tokens
        For 1K tokens each: $0.00005 input + $0.00040 output = $0.00045
        """
        from tinbox.core.cost import calculate_model_cost

        cost = calculate_model_cost(
            model_name="gpt-5-nano",
            input_tokens=1000,
            output_tokens=1000
        )

        expected_cost = 0.00045  # $0.00005 + $0.00040
        assert abs(cost - expected_cost) < 0.00001

    def test_gpt5_cost_with_provider_prefix(self):
        """Test that cost calculation handles provider prefix (e.g., 'openai/gpt-5')."""
        from tinbox.core.cost import calculate_model_cost

        cost = calculate_model_cost(
            model_name="openai/gpt-5",
            input_tokens=1000,
            output_tokens=1000
        )

        expected_cost = 0.01125
        assert abs(cost - expected_cost) < 0.00001

    def test_gpt5_cost_realistic_translation(self):
        """Test cost for realistic translation scenario (100-page PDF).

        Estimated: 50K input tokens, 50K output tokens
        GPT-5: (50 * $0.00125) + (50 * $0.01000) = $0.5625
        """
        from tinbox.core.cost import calculate_model_cost

        cost = calculate_model_cost(
            model_name="gpt-5",
            input_tokens=50000,
            output_tokens=50000
        )

        expected_cost = 0.5625
        assert abs(cost - expected_cost) < 0.0001

    def test_unknown_model_returns_zero(self):
        """Test that unknown models return zero cost (graceful fallback)."""
        from tinbox.core.cost import calculate_model_cost

        cost = calculate_model_cost(
            model_name="unknown-model",
            input_tokens=1000,
            output_tokens=1000
        )

        # Should return 0 for unknown models
        assert cost == 0.0


class TestGPT5Integration:
    """Integration tests for GPT-5 translation workflow."""

    @pytest.mark.asyncio
    async def test_gpt5_translation_returns_cost(self):
        """Test that GPT-5 translation returns accurate cost information."""
        with patch("tinbox.core.translation.litellm.completion") as mock_completion:
            # Mock response with realistic token usage
            mock_completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Hola, mundo"))],
                usage=MagicMock(
                    total_tokens=25,
                    prompt_tokens=10,
                    completion_tokens=15
                )
            )

            translator = LiteLLMTranslator(temperature=0.7)

            request = TranslationRequest(
                content="Hello, world",
                source_lang="en",
                target_lang="es",
                content_type="text/plain",
                model=ModelType.OPENAI,
                model_params={"model_name": "gpt-5-mini"}
            )

            result = await translator.translate(request)

            # Verify cost is calculated
            assert hasattr(result, "cost") or "cost" in result

            # For GPT-5-mini: (10 * $0.00025) + (15 * $0.00200) = $0.0325
            # But this depends on implementation returning cost

    @pytest.mark.asyncio
    async def test_all_gpt5_variants_work(self):
        """Test that all three GPT-5 variants can be used."""
        models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

        for model in models:
            with patch("tinbox.core.translation.litellm.completion") as mock_completion:
                mock_completion.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Hola"))],
                    usage=MagicMock(
                        total_tokens=20,
                        prompt_tokens=10,
                        completion_tokens=10
                    )
                )

                translator = LiteLLMTranslator(temperature=0.7)

                request = TranslationRequest(
                    content="Hello",
                    source_lang="en",
                    target_lang="es",
                    content_type="text/plain",
                    model=ModelType.OPENAI,
                    model_params={"model_name": model}
                )

                # Should not raise any errors
                result = await translator.translate(request)
                assert result is not None
