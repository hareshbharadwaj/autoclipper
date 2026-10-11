from types import SimpleNamespace
import urllib.error

from fastapi.testclient import TestClient

import app as app_module
from autoclipper import db as user_db


def test_youtube_explore_resolves_encrypted_user_key_and_ignores_query_secrets(monkeypatch):
    captured = {}
    user = SimpleNamespace(
        username="explore-test",
        youtube_api_key=user_db.encrypt_value("settings-youtube-key"),
        gemini_key=user_db.encrypt_value("settings-gemini-key"),
        gemini_model="user-gemini-model",
    )
    app_module.app.dependency_overrides[app_module.get_current_user] = lambda: user
    monkeypatch.setattr(app_module.config, "YOUTUBE_API_KEY", "server-fallback-key")

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(app_module, "fetch_trending", fake_fetch)
    try:
        with TestClient(app_module.app) as client:
            response = client.get(
                "/api/trending/youtube?region=IN&category=general&enrich=false"
                "&youtube_key=attacker-query-key&gemini_key=attacker-gemini-key"
            )
        assert response.status_code == 200, response.text
        assert captured["api_key"] == "settings-youtube-key"
        assert captured["region"] == "IN"
        assert "attacker-query-key" not in response.text
        assert "attacker-gemini-key" not in response.text
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


def test_youtube_explore_uses_server_key_when_user_has_no_key(monkeypatch):
    captured = {}
    user = SimpleNamespace(
        username="explore-fallback-test",
        youtube_api_key=None,
        gemini_key=None,
        gemini_model=None,
    )
    app_module.app.dependency_overrides[app_module.get_current_user] = lambda: user
    monkeypatch.setattr(app_module.config, "YOUTUBE_API_KEY", "server-youtube-key")
    monkeypatch.setattr(app_module, "fetch_trending", lambda **kwargs: captured.update(kwargs) or [])
    try:
        with TestClient(app_module.app) as client:
            response = client.get("/api/trending/youtube?enrich=false")
        assert response.status_code == 200, response.text
        assert captured["api_key"] == "server-youtube-key"
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


def test_youtube_explore_missing_key_returns_actionable_503(monkeypatch):
    user = SimpleNamespace(
        username="explore-missing-key-test",
        youtube_api_key=None,
        gemini_key=None,
        gemini_model=None,
    )
    app_module.app.dependency_overrides[app_module.get_current_user] = lambda: user
    monkeypatch.setattr(app_module.config, "YOUTUBE_API_KEY", None)
    monkeypatch.setattr(
        app_module,
        "fetch_trending",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fetch must not run without a key")),
    )
    try:
        with TestClient(app_module.app) as client:
            response = client.get("/api/trending/youtube?enrich=false")
        assert response.status_code == 503
        assert "YouTube Data API key" in response.json()["detail"]
        assert "Settings" in response.json()["detail"]
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


def test_youtube_explore_reports_when_stored_key_cannot_be_decrypted(monkeypatch):
    user = SimpleNamespace(
        username="explore-undecryptable-key-test",
        youtube_api_key="unreadable-encrypted-value",
        gemini_key=None,
        gemini_model=None,
    )
    app_module.app.dependency_overrides[app_module.get_current_user] = lambda: user
    monkeypatch.setattr(app_module.config, "YOUTUBE_API_KEY", None)
    try:
        with TestClient(app_module.app) as client:
            response = client.get("/api/trending/youtube?enrich=false")
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "stored for this account but cannot be decrypted" in detail
        assert "AUTOCLIPPER_SECRET" in detail
        assert "unreadable-encrypted-value" not in response.text
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


def test_youtube_provider_errors_are_safe_502_responses(monkeypatch):
    youtube_key = "do-not-return-this-key"
    user = SimpleNamespace(
        username="explore-provider-error-test",
        youtube_api_key=user_db.encrypt_value(youtube_key),
        gemini_key=None,
        gemini_model=None,
    )
    app_module.app.dependency_overrides[app_module.get_current_user] = lambda: user
    monkeypatch.setattr(app_module.config, "YOUTUBE_API_KEY", None)

    def fail_fetch(**_kwargs):
        raise urllib.error.HTTPError(
            f"https://www.googleapis.com/youtube/v3/videos?key={youtube_key}",
            403,
            "Forbidden",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(app_module, "fetch_trending", fail_fetch)
    try:
        with TestClient(app_module.app) as client:
            response = client.get("/api/trending/youtube?enrich=false")
        assert response.status_code == 502
        assert "API rejected" in response.json()["detail"]
        assert youtube_key not in response.text
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)
