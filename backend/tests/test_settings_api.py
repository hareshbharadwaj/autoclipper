"""The settings API must surface credentials that need re-entry after a key change."""
from types import SimpleNamespace

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import app as app_module
from autoclipper import db


def test_get_settings_reports_undecryptable_fields_without_leaking_values():
    legacy_cipher = Fernet(Fernet.generate_key()).encrypt(b"legacy-secret").decode("utf-8")

    def _user():
        return SimpleNamespace(
            id=1,
            username="migration-user",
            groq_key=db.encrypt_value("active-secret"),
            gemini_key=legacy_cipher,
            youtube_api_key=None,
            gemini_model="gemini-3.5-flash",
            whisper_model="whisper-large-v3-turbo",
            youtube_cookies=None,
        )

    app_module.app.dependency_overrides[app_module.get_current_user] = _user
    try:
        with TestClient(app_module.app) as client:
            response = client.get("/api/auth/settings")
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["undecryptable_fields"] == ["gemini_key"]
    assert payload["groq_key"] == "active-secret"
    assert payload["gemini_key"] is None
    body = response.text
    assert "legacy-secret" not in body
    assert legacy_cipher not in body
