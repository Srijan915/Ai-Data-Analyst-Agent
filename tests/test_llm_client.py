
"""Gemini adapter contract tests use an offline SDK mock."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from google.genai import errors

from app.core import llm_client as llm


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-not-a-real-key")
    calls = []
    responses = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "test-only-not-a-real-key"
            options = kwargs["http_options"]
            assert options.timeout == 50000
            assert options.retry_options.attempts == 1
            self.models = SimpleNamespace(generate_content=self.generate_content)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def generate_content(self, **kwargs):
            calls.append(kwargs)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(llm.genai, "Client", FakeClient)
    return calls, responses


def completion(content=" SELECT 1 ", reason="STOP"):
    return SimpleNamespace(
        text=content,
        candidates=[
            SimpleNamespace(finish_reason=SimpleNamespace(name=reason))
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=3,
            candidates_token_count=2,
        ),
    )


def test_request_and_response_contract(transport):
    calls, responses = transport
    responses.append(completion())

    result = llm.GeminiClient(model_id="configured-model").generate_text(
        "question", "rules", 128, 0
    )

    assert result == "SELECT 1"
    assert calls == [{
        "model": "configured-model",
        "contents": "question",
        "config": calls[0]["config"],
    }]
    assert calls[0]["config"].system_instruction == "rules"
    assert calls[0]["config"].max_output_tokens == 128
    assert calls[0]["config"].temperature == 0


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, llm.LLMCredentialsError),
        (403, llm.LLMAccessDeniedError),
        (429, llm.LLMThrottlingError),
        (400, llm.LLMAPIError),
        (404, llm.LLMAPIError),
        (500, llm.LLMAPIError),
    ],
)
def test_safe_errors_without_retries(transport, status, expected):
    calls, responses = transport
    responses.append(errors.APIError(
        code=status,
        response_json={"error": {"message": "private data secret-value"}},
    ))

    with pytest.raises(expected) as error:
        llm.GeminiClient().generate_text("question")

    assert "secret-value" not in str(error.value)
    assert len(calls) == 1
    assert error.value.__suppress_context__


def test_timeout_without_retry(transport):
    calls, responses = transport
    responses.append(httpx.ReadTimeout("private details"))

    with pytest.raises(llm.LLMAPIError, match="provider request failed"):
        llm.GeminiClient().generate_text("question")

    assert len(calls) == 1


@pytest.mark.parametrize(
    "content,reason",
    [
        (None, "STOP"),
        ("", "STOP"),
        ("partial", "MAX_TOKENS"),
        ("x", "SAFETY"),
    ],
)
def test_rejects_incomplete_responses(transport, content, reason):
    transport[1].append(completion(content, reason))

    with pytest.raises(llm.LLMResponseFormatError):
        llm.GeminiClient().generate_text("question")


def test_missing_key_is_lazy_and_local(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = llm.GeminiClient()
    sdk = Mock()
    monkeypatch.setattr(llm.genai, "Client", sdk)

    with pytest.raises(llm.LLMCredentialsError):
        client.generate_text("question")

    sdk.assert_not_called()


@pytest.mark.parametrize("prompt", [None, "", "   "])
def test_empty_prompt_is_local(prompt):
    with pytest.raises(ValueError):
        llm.GeminiClient().generate_text(prompt)


def test_unknown_provider_never_falls_back(monkeypatch):
    llm.get_llm_client.cache_clear()
    monkeypatch.setattr(
        llm, "config", SimpleNamespace(llm_provider="unknown")
    )

    with pytest.raises(llm.LLMAPIError):
        llm.get_llm_client()

    llm.get_llm_client.cache_clear()


def test_default_model_has_no_provider_specific_reasoning_fields(transport):
    calls, responses = transport
    responses.append(completion())

    llm.GeminiClient().generate_text("question")

    assert "include_reasoning" not in calls[0]
    assert "reasoning_effort" not in calls[0]


def test_malformed_response_is_safe(transport):
    transport[1].append(SimpleNamespace(text=None, candidates=None))

    with pytest.raises(llm.LLMResponseFormatError):
        llm.GeminiClient().generate_text("question")
