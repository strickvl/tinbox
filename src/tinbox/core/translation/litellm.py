"""LiteLLM-based translation implementation."""

import base64
from datetime import datetime
from typing import Union
import io
import json
from PIL import Image

from litellm import completion, completion_cost
from litellm.exceptions import RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from tinbox.core.translation.interface import (
    ModelInterface,
    TranslationError,
    TranslationRequest,
    TranslationResponse,
    TranslationWithGlossaryResponse,
    TranslationWithoutGlossaryResponse,
)
from tinbox.core.types import ModelType, GlossaryEntry
from tinbox.utils.chunks import extract_whitespace_formatting
from tinbox.utils.logging import get_logger

logger = get_logger(__name__)

# Configure retry decorator for rate limit handling
completion_with_retry = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(
        multiplier=1, min=4, max=120
    ),  # Start at 4s, double each time, max 60s
    stop=stop_after_attempt(10),  # Maximum 5 attempts
    before_sleep=before_sleep_log(
        logger, log_level=20
    ),  # Log retry attempts at INFO level
)


class LiteLLMTranslator(ModelInterface):
    """LiteLLM-based implementation of the model interface."""

    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 100000,
    ) -> None:
        """Initialize the translator.

        Args:
            temperature: Model temperature (randomness). Defaults to 0.3.
            max_tokens: Maximum tokens in response. Defaults to 100000.
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._logger = logger

    def _get_model_string(self, request: TranslationRequest) -> str:
        """Get the model string for LiteLLM.

        Args:
            request: The translation request

        Returns:
            Model string for LiteLLM

        Raises:
            TranslationError: If model name is missing
        """
        if not request.model_params.get("model_name"):
            raise TranslationError("No model name provided")

        model_name = request.model_params["model_name"]

        # If model name already includes provider prefix, use it directly
        if "/" in model_name:  # Using / instead of : as that's what litellm expects
            return model_name

        # Handle provider-specific model strings
        if request.model == ModelType.OLLAMA:
            return f"ollama/{model_name}"
        elif request.model == ModelType.OPENAI:
            return model_name  # OpenAI models use their names directly
        elif request.model == ModelType.ANTHROPIC:
            # Anthropic models need anthropic/ prefix according to litellm docs
            # Replace : with / if present in the model name
            clean_name = model_name.replace(":", "/")
            return f"anthropic/{clean_name}"
        elif request.model == ModelType.GEMINI:
            # Gemini models need gemini/ prefix according to litellm docs
            # Replace : with / if present in the model name
            clean_name = model_name.replace(":", "/")
            return f"gemini/{clean_name}"
        else:
            raise TranslationError(f"Unsupported model provider: {request.model}")

    def _create_prompt(self, request: TranslationRequest) -> list[dict]:
        """Create the prompt for the model with context support.

        Args:
            request: The translation request

        Returns:
            List of message dictionaries for the model
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. Translate the following content "
                    f"from '{request.source_lang}' to '{request.target_lang}'. "
                    f"Maintain the original formatting and structure (including whitespaces, line breaks, etc.). "
                    f"Include ALL markup/formatting in the translation. But do not fix tags or formatting errors - "
                    f"you only receive chunks of text to translate and later chunks might contain the 'missing' tags. "
                    f"Translate only the content, do not add any explanations or notes. "
                    f"Do not add any commentary or notes to the translation. It is extremely "
                    f"important that the only output you give is the translation of the content. "
                    f"IMPORTANT: Your translation should ALWAYS(!) be in '{request.target_lang}' language."
                ),
            }
        ]

        # Add context if provided (with tag-based notation)
        if request.context:
            messages.append({
                "role": "user",
                "content": f"[TRANSLATION_CONTEXT]{request.context}[/TRANSLATION_CONTEXT]"
            })

        # Add glossary context if available
        if request.glossary:
            glossary_context = (
                "Use this glossary for consistent translations:\n"
                f"{request.glossary.to_context_string()}\n\n"
                "When you encounter these terms, use the provided translations. "
                "If you encounter new important terms that benefit from consistent translation "
                "(technical terms, proper nouns, domain vocabulary, names, etc.), include them in the glossary_extension field in your response. "
                "Only include terms that are important for consistent translation."
            )
            messages.append({"role": "user", "content": glossary_context})

        if request.content_type.startswith("text/"):
            # Text content
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[TRANSLATE_THIS]{request.content}[/TRANSLATE_THIS]\n\n"
                        f"Only return the translation in '{request.target_lang}' language of the text between the [TRANSLATE_THIS]-tags (including ALL markup/formatting and line-breaks). "
                        f"Do not include any other content. \n\nTranslation in '{request.target_lang}' language without commentary:"
                    ),
                }
            )
        else:
            # Image content
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Translate the contents of the image from {request.source_lang} to {request.target_lang}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(request.content).decode()}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Translation without commentary:",
                        },
                    ],
                }
            )
        
        logger.debug("Messages: ", messages=messages)

        return messages

    @completion_with_retry
    async def _make_completion_request(
        self, request: TranslationRequest
    ):
        """Make a completion request with retry logic for rate limits.

        Args:
            request: The translation request

        Returns:
            The completion response

        Raises:
            TranslationError: If translation fails after retries
        """
        try:
            return completion(
                model=self._get_model_string(request),
                messages=self._create_prompt(request),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
                response_format=TranslationWithGlossaryResponse if request.glossary else TranslationWithoutGlossaryResponse,
                reasoning_effort=request.reasoning_effort,
                drop_params=True,
                **{k: v for k, v in request.model_params.items() if k != "model_name"},
            )
        except RateLimitError as e:
            # This will be caught by the retry decorator
            raise
        except Exception as e:
            raise TranslationError(f"Translation failed: {str(e)}") from e

    async def translate(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Translate content using LiteLLM with whitespace preservation.

        Args:
            request: The translation request configuration

        Returns:
            A single translation response

        Raises:
            TranslationError: If translation fails
        """
        start_time = datetime.now()

        try:
            # Handle empty content - return as-is since there's nothing to translate
            if not request.content or (
                isinstance(request.content, str) and not request.content.strip()
            ):
                logger.info("Empty content, returning as-is")

                end_time = datetime.now()
                time_taken = (end_time - start_time).total_seconds()
                
                return TranslationResponse(
                    text=request.content if isinstance(request.content, str) else "",
                    tokens_used=0,
                    cost=0.0,
                    time_taken=time_taken,
                )

            # Extract whitespace from content
            content_prefix, clean_content, content_suffix = extract_whitespace_formatting(request.content)

            logger.debug("Content prefix: ", prefix=content_prefix)
            logger.debug("Content suffix: ", suffix=content_suffix)
            logger.debug("Clean content: ", content=clean_content)

            # Create updated request with clean content
            clean_request = TranslationRequest(
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                content=clean_content,
                context=request.context,
                content_type=request.content_type,
                model=request.model,
                model_params=request.model_params,
                glossary=request.glossary,
                reasoning_effort=request.reasoning_effort,
            )

            # Validate language codes (2 or 3 letter codes, or 'auto' for source)
            if request.source_lang != "auto" and (not request.source_lang.isalpha() or len(request.source_lang) not in [
                2,
                3,
            ]):
                raise TranslationError("Translation failed: Invalid language code")
            if not request.target_lang.isalpha() or len(request.target_lang) not in [
                2,
                3,
            ]:
                raise TranslationError("Translation failed: Invalid language code")

            # Validate image content
            if request.content_type.startswith("image/"):
                try:
                    Image.open(io.BytesIO(clean_content))
                except Exception as e:
                    raise TranslationError(
                        "Translation failed: Invalid image data"
                    ) from e

            # Handle translation with whitespace restoration
            try:
                logger.debug(f"Making completion request", request=clean_request)

                response = await self._make_completion_request(clean_request)
                
                logger.debug(f"Completion request made", response=response)

                if not hasattr(response, "choices") or not response.choices:
                    raise TranslationError("No response from model")

                if not hasattr(response.choices[0], "finish_reason") or not response.choices[0].finish_reason == "stop":
                    raise TranslationError(f"Invalid finish reason from model: {response.choices[0].finish_reason}, expected 'stop'")

                # Require structured response - parse JSON from content
                text: str
                glossary_updates = []
                
                if not hasattr(response.choices[0], "message") or not response.choices[0].message.content:
                    raise TranslationError("Invalid response format: missing message content")
                
                try:
                    parsed_content = json.loads(response.choices[0].message.content)
                except (json.JSONDecodeError, AttributeError) as e:
                    raise TranslationError(f"Invalid JSON response format: {str(e)}")
                
                if not isinstance(parsed_content, dict) or "translation" not in parsed_content:
                    raise TranslationError("Missing translation field in JSON response")
                
                text = parsed_content["translation"]
                
                # Parse glossary updates if present
                if "glossary_extension" in parsed_content and parsed_content["glossary_extension"]:
                    glossary_data = parsed_content["glossary_extension"]
                    if isinstance(glossary_data, list):
                        glossary_updates = [
                            GlossaryEntry(term=entry["term"], translation=entry["translation"])
                            for entry in glossary_data
                            if isinstance(entry, dict) and "term" in entry and "translation" in entry
                        ]

                if not text or not text.strip():
                    raise TranslationError("No content returned from model")

                # Get actual token usage from LiteLLM response
                tokens = getattr(response.usage, "total_tokens", 0)
                
                # Get actual cost from LiteLLM response
                cost = 0.0
                if hasattr(response, "_hidden_params") and "response_cost" in response._hidden_params:
                    cost = response._hidden_params["response_cost"]
                else:
                    # Fallback: calculate cost using LiteLLM's completion_cost function
                    try:
                        cost = completion_cost(completion_response=response)
                    except Exception:
                        # If all else fails, cost remains 0.0
                        pass
                
                # Calculate time taken
                time_taken = (datetime.now() - start_time).total_seconds()

                # Restore whitespace
                final_text = content_prefix + text + content_suffix

                logger.debug("Glossary updates: ", glossary_updates=glossary_updates)
                logger.debug("Final text: ", final_text=final_text)

                return TranslationResponse(
                    text=final_text,
                    tokens_used=tokens,
                    cost=cost,
                    time_taken=time_taken,
                    glossary_updates=glossary_updates,
                )
            except TranslationError:
                raise
            except Exception as e:
                self._logger.error(f"Translation failed: {str(e)}")
                raise TranslationError(f"Translation failed: {str(e)}") from e

        except TranslationError:
            # Re-raise TranslationError without wrapping
            raise
        except Exception as e:
            # Catch any remaining errors
            self._logger.error(f"Translation failed: {str(e)}")
            raise TranslationError(f"Translation failed: {str(e)}") from e

    async def validate_model(self) -> bool:
        """Verify the model is available and properly configured.

        Returns:
            True if the model is available and configured correctly
        """
        try:
            # Make a minimal test request
            response = completion(
                model="gpt-3.5-turbo",  # Use a simple model for validation
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return hasattr(response, "choices") and len(response.choices) > 0
        except Exception as e:
            self._logger.error(f"Model validation failed: {str(e)}")
            return False
