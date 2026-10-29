import os
from pathlib import Path

import pytest

from app.config import AppConfig, _load_project_dotenv


@pytest.mark.parametrize(
    "setting",
    [
        "llm_connect_timeout_seconds",
        "llm_read_timeout_seconds",
        "max_upload_size_mb",
        "max_query_result_rows",
        "query_timeout_seconds",
        "max_concurrent_analyses",
        "duckdb_thread_limit",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, -1])
def test_config_rejects_non_positive_operational_limits(
    setting,
    invalid_value,
):
    with pytest.raises(ValueError, match="must be greater than zero"):
        AppConfig(**{setting: invalid_value})


@pytest.mark.parametrize(
    "setting",
    [
        "app_name",
        "app_env",
        "llm_provider",
        "llm_model_id",
        "duckdb_memory_limit",
    ],
)
@pytest.mark.parametrize("invalid_value", ["", "   "])
def test_config_rejects_blank_required_text_values(
    setting,
    invalid_value,
):
    with pytest.raises(ValueError, match="must be a non-empty string"):
        AppConfig(**{setting: invalid_value})


def test_config_accepts_defaults():
    config = AppConfig()

    assert config.max_concurrent_analyses == 1
    assert config.llm_connect_timeout_seconds == 5
    assert config.llm_read_timeout_seconds == 45


def test_project_dotenv_is_loaded_from_explicit_root_not_current_directory(
    monkeypatch,
    tmp_path,
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text(
        "GEMINI_API_KEY=test-key-from-project-root\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert _load_project_dotenv(project_root)
    assert os.getenv("GEMINI_API_KEY") == "test-key-from-project-root"


def test_config_reads_gemini_environment_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL_ID", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_CONNECT_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("LLM_READ_TIMEOUT_SECONDS", "50")

    config = AppConfig.from_env()

    assert config.llm_provider == "gemini"
    assert config.llm_model_id == "gemini-2.5-flash"
    assert config.llm_connect_timeout_seconds == 7
    assert config.llm_read_timeout_seconds == 50
