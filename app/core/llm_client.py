
"""Provider-independent text interface with a Gemini adapter.

Client creation is lazy: startup and uploads do not require credentials.
Provider response bodies and keys are never included in application errors.
"""
import os
from time import perf_counter
from functools import lru_cache
from typing import Optional, Protocol

import httpx
from google import genai
from google.genai import errors, types

from app.utils.logger import get_logger
from app.config import config
from app.utils.request_budget import RequestBudgetExceeded, provider_request_budget
from app.utils.prompt_budget import PromptBudgetError, ensure_prompt_size


logger = get_logger(__name__)


def _finish_reason_name(value: object) -> str | None:
    """Return the SDK enum name without depending on its string rendering.

    ``google-genai`` exposes ``FinishReason`` as an enum, while generated
    response models and test doubles can expose either the enum itself or an
    object with a ``name`` attribute.  Its ``str`` representation is not a
    stable wire contract, so do not use it for a safety decision.
    """
    name = getattr(value, "name", value)
    if not isinstance(name, str):
        return None
    return name.rsplit(".", 1)[-1].upper()


def _log_provider_failure(exc):
    logger.warning(
        "AI request failed | error_type=%s | http_status=%s",
        type(exc).__name__,
        getattr(exc, "status_code", None),
    )


class LLMClientError(Exception):
    """Base error exposed to application callers."""


class LLMCredentialsError(LLMClientError):
    """Missing or rejected credentials."""


class LLMAccessDeniedError(LLMClientError):
    """Provider or model access denied."""


class LLMThrottlingError(LLMClientError):
    """Provider quota exhausted; retry later."""


class LLMAPIError(LLMClientError):
    """Provider unavailable or request rejected."""


class LLMResponseFormatError(LLMClientError):
    """Empty, malformed or truncated completion."""


class LLMClient(Protocol):
    """Implement this boundary to add providers without changing callers."""

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...


class GeminiClient:
    """Text-only Gemini completions, with no tools or provider fallback."""

    def __init__(self, model_id: Optional[str] = None):
        self._model_id = model_id or config.llm_model_id

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        try:
            ensure_prompt_size(prompt, system_prompt)
        except PromptBudgetError as exc:
            raise LLMResponseFormatError(str(exc)) from None

        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise LLMCredentialsError(
                "Configure GEMINI_API_KEY privately in deployment secrets."
            )

        try:
            provider_request_budget.acquire()
        except RequestBudgetExceeded as exc:
            raise LLMThrottlingError(str(exc)) from None

        request_config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        timeout_ms = (
            config.llm_connect_timeout_seconds
            + config.llm_read_timeout_seconds
        ) * 1000

        started = perf_counter()
        try:
            # Retries are disabled so timeouts and quota errors do not
            # multiply token use. A new user request is the retry boundary.
            with genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ) as client:
                response = client.models.generate_content(
                    model=self._model_id,
                    contents=prompt,
                    config=request_config,
                )
        except errors.APIError as exc:
            _log_provider_failure(exc)
            status = getattr(exc, "code", None)

            if status == 401:
                raise LLMCredentialsError(
                    "The provider rejected the API key."
                ) from None
            if status == 403:
                raise LLMAccessDeniedError(
                    "The configured model is not permitted for this account."
                ) from None
            if status == 429:
                raise LLMThrottlingError(
                    "The provider quota was reached. Try again later."
                ) from None

            raise LLMAPIError(
                "The provider request failed. Check service status and model configuration."
            ) from None
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            _log_provider_failure(exc)
            raise LLMAPIError(
                "The provider request failed. Check service status and model configuration."
            ) from None
        except Exception as exc:
            # Keep unexpected SDK failures sanitized; do not expose request
            # contents, credentials, or provider response bodies.
            _log_provider_failure(exc)
            raise LLMAPIError(
                "The provider request failed. Check service status and model configuration."
            ) from None

        try:
            content = response.text
            candidates = response.candidates or []
            finish_reason = candidates[0].finish_reason if candidates else None

            # Gemini reports STOP for a normally completed answer. Reject
            # blocked, truncated, empty, or otherwise incomplete responses.
            if (
                not isinstance(content, str)
                or not content.strip()
                or _finish_reason_name(finish_reason) != "STOP"
            ):
                raise LLMResponseFormatError(
                    "The provider returned no complete text answer."
                )

            usage = getattr(response, "usage_metadata", None)
            logger.info(
                "AI request completed | duration_ms=%.1f | prompt_tokens=%s | completion_tokens=%s",
                (perf_counter() - started) * 1000,
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
            )
            return content.strip()
        except LLMResponseFormatError:
            raise
        except (AttributeError, IndexError, TypeError):
            raise LLMResponseFormatError(
                "The provider returned an invalid response."
            ) from None


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Select explicitly; unsupported providers never fall back silently."""
    if config.llm_provider != "gemini":
        raise LLMAPIError("Unsupported LLM_PROVIDER. Configure gemini.")
    return GeminiClient()
