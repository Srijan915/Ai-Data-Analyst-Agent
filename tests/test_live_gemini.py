"""One explicitly opt-in provider check; default test runs make no API calls."""
import os

import pytest

from app.core.llm_client import GeminiClient


@pytest.mark.live
@pytest.mark.skipif(os.getenv("RUN_LIVE_GEMINI_TEST") != "1", reason="Set RUN_LIVE_GEMINI_TEST=1 to opt in")
def test_live_gemini_returns_text():
    if "GEMINI_API_KEY" not in os.environ:
        pytest.skip("GEMINI_API_KEY is unavailable in the process environment")
    response = GeminiClient().generate_text("Reply with only the word READY.", max_tokens=32)
    assert isinstance(response, str) and response.strip()
