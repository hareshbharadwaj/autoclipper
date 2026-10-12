import os
import stat

import pytest
from cryptography.fernet import Fernet

from autoclipper import db


def test_local_fernet_key_is_persisted_with_private_permissions(tmp_path):
    key_path = tmp_path / "autoclipper.fernet-key"

    first_key = db._persistent_fernet_key(str(key_path))
    second_key = db._persistent_fernet_key(str(key_path))
    ciphertext = Fernet(first_key).encrypt(b"synthetic test value")

    assert first_key == second_key
    assert Fernet(second_key).decrypt(ciphertext) == b"synthetic test value"
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600


def test_corrupt_persistent_fernet_key_fails_closed(tmp_path):
    key_path = tmp_path / "autoclipper.fernet-key"
    key_path.write_text("not-a-fernet-key", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unreadable or invalid"):
        db._persistent_fernet_key(str(key_path))


def test_legacy_ephemeral_key_ciphertext_is_reported_not_silently_migrated():
    """Ciphertext from a previous random in-memory key must be flagged, not hidden."""
    from types import SimpleNamespace

    legacy_key = Fernet.generate_key()  # stands in for the old per-process key
    legacy_cipher = Fernet(legacy_key).encrypt(b"legacy-groq-secret").decode("utf-8")
    active_cipher = db.encrypt_value("active-gemini-secret")

    user = SimpleNamespace(
        groq_key=legacy_cipher,
        gemini_key=active_cipher,
        youtube_api_key=None,
        youtube_cookies="",
        youtube_token=legacy_cipher,
        facebook_token=None,
    )

    assert db.decrypt_value(legacy_cipher) is None
    assert db.decrypt_value_with_status(legacy_cipher) == ("undecryptable", None)
    assert db.decrypt_value_with_status(active_cipher) == ("ok", "active-gemini-secret")
    assert db.decrypt_value_with_status(None) == ("empty", None)
    assert db.undecryptable_fields(user) == ["groq_key", "youtube_token"]


def test_unchanged_autoclipper_secret_keeps_credentials_decryptable(monkeypatch):
    monkeypatch.setenv("AUTOCLIPPER_SECRET", "stable-production-secret")
    first_key = db._fernet_key()
    ciphertext = Fernet(first_key).encrypt(b"saved-credential")
    # Simulate a restart: deriving again from the same secret must still decrypt.
    second_key = db._fernet_key()
    assert first_key == second_key
    assert Fernet(second_key).decrypt(ciphertext) == b"saved-credential"
    assert db.ENCRYPTION_KEY_SOURCE == "env_secret"


def test_empty_key_file_from_interrupted_creation_is_regenerated(tmp_path):
    key_path = tmp_path / "autoclipper.fernet-key"
    key_path.write_bytes(b"")  # process died between create and write

    key = db._persistent_fernet_key(str(key_path))
    Fernet(key)  # valid key material
    assert key_path.read_bytes().strip() == key
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600
    # A simulated restart reuses the regenerated key (persistence).
    assert db._persistent_fernet_key(str(key_path)) == key


def test_key_file_source_tracking_and_restart_persistence(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOCLIPPER_SECRET", raising=False)
    monkeypatch.setenv("AUTOCLIPPER_SECRET_FILE", str(tmp_path / "dev.fernet-key"))
    monkeypatch.setattr(db, "ENCRYPTION_KEY_SOURCE", db.ENCRYPTION_KEY_SOURCE)

    first_key = db._fernet_key()
    assert db.ENCRYPTION_KEY_SOURCE == "key_file_created"
    ciphertext = Fernet(first_key).encrypt(b"dev-credential")

    second_key = db._fernet_key()  # simulated restart
    assert db.ENCRYPTION_KEY_SOURCE == "key_file_existing"
    assert second_key == first_key
    assert Fernet(second_key).decrypt(ciphertext) == b"dev-credential"


def test_report_encryption_status_lists_users_needing_reentry(tmp_path, monkeypatch):
    import json

    from sqlmodel import Session, SQLModel, create_engine

    engine = create_engine(
        f"sqlite:///{tmp_path / 'status.db'}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(db, "engine", engine)

    legacy_cipher = Fernet(Fernet.generate_key()).encrypt(b"legacy-secret").decode("utf-8")
    with Session(engine) as session:
        session.add(db.User(
            username="legacy-user", password_hash="x",
            groq_key=legacy_cipher, gemini_key=db.encrypt_value("active-secret"),
        ))
        session.add(db.User(username="clean-user", password_hash="y"))
        session.commit()

    summary = db.report_encryption_status()
    assert summary["users_with_undecryptable_credentials"] == ["legacy-user"]
    assert summary["undecryptable_field_count"] == 1
    # Never leak key material or ciphertext into the report/logs.
    assert "legacy-secret" not in json.dumps(summary)
    assert "active-secret" not in json.dumps(summary)
    assert legacy_cipher not in json.dumps(summary)


def test_init_db_reports_encryption_status_on_startup(tmp_path, monkeypatch):
    from sqlmodel import SQLModel, create_engine

    engine = create_engine(
        f"sqlite:///{tmp_path / 'init.db'}", connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(db, "engine", engine)
    calls = []
    monkeypatch.setattr(db, "report_encryption_status", lambda: calls.append(True) or {})
    db.init_db()
    assert calls == [True]
