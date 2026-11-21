"""Smoke tests for the Streamlit application entry point."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_FILE = PROJECT_ROOT / "app" / "main.py"


def test_initial_streamlit_screen_renders_without_errors() -> None:
    """The unauthenticated, no-dataset state must render without external provider access."""

    app = AppTest.from_file(str(MAIN_FILE))
    app.run(timeout=15)

    assert not app.exception
    assert len(app.title) == 1
    assert app.title[0].value == "QueryPilotAI"
    assert len(app.info) == 1
    assert "Upload a CSV, XLSX, TXT or PDF file" in app.info[0].value
    assert len(app.text_area) == 0
    assert len(app.button) >= 2
    rendered = " ".join(item.value for item in app.markdown)
    assert "Python executable:" not in rendered
    assert "Pip show output:" not in rendered
