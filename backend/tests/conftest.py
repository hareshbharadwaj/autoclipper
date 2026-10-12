"""Global test configuration and fixtures for AutoClipper tests."""
import pytest
from autoclipper import config


@pytest.fixture(autouse=True)
def _isolate_gemini_api_key(monkeypatch):
    """Ensure tests run deterministically without accidental live Gemini API calls."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
