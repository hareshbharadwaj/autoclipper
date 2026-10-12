"""SQLite-backed user store with at-rest encryption for API keys.

Each user owns their own API keys (Groq, Gemini, YouTube) and pipeline
preferences. Keys are encrypted with Fernet before being written to the
database. Configure AUTOCLIPPER_SECRET to a stable value in production and
share it across restarts/replicas. Local development falls back to a private,
persistent key file next to the SQLite database (or AUTOCLIPPER_SECRET_FILE).
"""
import base64
import os
import hashlib
from datetime import datetime, timezone
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from sqlmodel import Field, Session, SQLModel, create_engine, select

from . import config

# Keep the default database/key sidecar beside this module's SQLite database.
DB_PATH = os.environ.get(
    "AUTOCLIPPER_DB",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autoclipper.db")),
)

# Where the active Fernet key came from: "env_secret", "key_file_existing",
# or "key_file_created". Set by _fernet_key() at import time; used to explain
# upgrade behavior (see report_encryption_status).
ENCRYPTION_KEY_SOURCE = "env_secret"

# User columns that hold Fernet ciphertext. Used to tell an operator exactly
# which credentials must be re-saved when the encryption key changed.
ENCRYPTED_USER_FIELDS = (
    "groq_key",
    "gemini_key",
    "youtube_api_key",
    "youtube_cookies",
    "youtube_token",
    "facebook_token",
)


# --- Encryption ----------------------------------------------------------
def _persistent_fernet_key(path: str) -> bytes:
    """Read or atomically create a private Fernet key file for local development.

    An existing but empty key file is treated as an interrupted creation (the
    process died between creating the path and writing the key) and is
    regenerated atomically, so the next start cannot fail on a blank file.
    """
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "rb") as handle:
            existing = handle.read().strip()
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise RuntimeError(
            "The persistent AutoClipper encryption key file is unreadable. "
            "Restore it or configure AUTOCLIPPER_SECRET."
        ) from exc

    if existing:
        try:
            Fernet(existing)  # fail closed if the key file is corrupt
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                "The persistent AutoClipper encryption key file is unreadable or invalid. "
                "Restore it or configure AUTOCLIPPER_SECRET."
            ) from exc
        if os.name == "posix":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        return existing

    # Create (or regenerate an empty placeholder) atomically: write the key to
    # a temp file in the same directory, fsync it, then os.replace() over the
    # final path so readers never observe a partially written key.
    key = Fernet.generate_key()
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            "Cannot create a persistent AutoClipper encryption key file. "
            "Configure AUTOCLIPPER_SECRET or AUTOCLIPPER_SECRET_FILE."
        ) from exc
    return key


def _fernet_key() -> bytes:
    """Use a stable environment secret, or a persistent private key file locally."""
    global ENCRYPTION_KEY_SOURCE
    secret = os.environ.get("AUTOCLIPPER_SECRET")
    if secret:
        ENCRYPTION_KEY_SOURCE = "env_secret"
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)
    key_path = os.environ.get("AUTOCLIPPER_SECRET_FILE") or f"{DB_PATH}.fernet-key"
    existed = os.path.isfile(key_path) and os.path.getsize(key_path) > 0
    key = _persistent_fernet_key(key_path)
    ENCRYPTION_KEY_SOURCE = "key_file_existing" if existed else "key_file_created"
    return key


_fernet = Fernet(_fernet_key())


def encrypt_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return _fernet.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return _fernet.decrypt(value.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        return None


def decrypt_value_with_status(value: Optional[str]):
    """Return ("empty" | "ok" | "undecryptable", plaintext_or_None).

    Unlike decrypt_value() this distinguishes a field that was never set from
    ciphertext that cannot be decrypted with the active key (for example after
    an upgrade from a build that used an ephemeral random key).
    """
    if not value:
        return "empty", None
    try:
        return "ok", _fernet.decrypt(value.encode("utf-8")).decode("utf-8")
    except Exception:  # noqa: BLE001 - InvalidToken and any decode failure
        return "undecryptable", None


def undecryptable_fields(user) -> list:
    """Names of the user's stored credential fields that cannot be decrypted.

    These are credentials saved under a previous encryption key; the values
    are gone and must be entered again. Never returns (or logs) the values.
    """
    fields = []
    for field in ENCRYPTED_USER_FIELDS:
        status, _ = decrypt_value_with_status(getattr(user, field, None))
        if status == "undecryptable":
            fields.append(field)
    return fields


def report_encryption_status() -> dict:
    """Log (and return) a safe summary of key provenance and migration impact.

    Called from init_db() on startup. Reports which users hold credentials
    that the active key cannot decrypt, so operators are told explicitly which
    credentials must be re-saved instead of seeing silent empty values.
    """
    from .utils import logger

    summary = {
        "key_source": ENCRYPTION_KEY_SOURCE,
        "users_with_undecryptable_credentials": [],
        "undecryptable_field_count": 0,
    }
    if ENCRYPTION_KEY_SOURCE == "key_file_created":
        logger.warning(
            "AUTOCLIPPER_SECRET is not set; created a persistent local encryption key "
            "file beside the database. Set AUTOCLIPPER_SECRET in production so every "
            "restart/replica shares one stable key."
        )
    try:
        with Session(engine) as session:
            users = session.exec(select(User)).all()
        for user in users:
            fields = undecryptable_fields(user)
            if fields:
                summary["users_with_undecryptable_credentials"].append(user.username)
                summary["undecryptable_field_count"] += len(fields)
                logger.warning(
                    "User '%s' has %d stored credential field(s) encrypted with a "
                    "previous key that can no longer be decrypted: %s. They must be "
                    "saved again in Settings; other data is unaffected.",
                    user.username, len(fields), ", ".join(fields),
                )
    except Exception as exc:  # noqa: BLE001 - startup report must never crash boot
        logger.warning("Encryption status check could not inspect the database (%s).", type(exc).__name__)
    return summary


# --- Engine --------------------------------------------------------------
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def init_db() -> None:
    """Create tables. Idempotent; safe to call on startup."""
    SQLModel.metadata.create_all(engine)
    # Migrate: add facebook_token column if missing (for existing DBs)
    with engine.connect() as conn:
        try:
            conn.exec_driver_sql("ALTER TABLE user ADD COLUMN facebook_token TEXT")
            conn.commit()
        except Exception:
            pass  # column already exists
    # Surface any credentials that the active encryption key cannot decrypt
    # (e.g. after upgrading from a build that used an ephemeral random key).
    report_encryption_status()


def get_session():
    with Session(engine) as session:
        yield session


# --- Models --------------------------------------------------------------
class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    groq_key: str | None = None
    gemini_key: str | None = None
    youtube_api_key: str | None = None
    gemini_model: str | None = None
    whisper_model: str | None = None
    youtube_cookies: str | None = None
    youtube_token: str | None = None  # encrypted JSON of the OAuth credentials
    facebook_token: str | None = None  # encrypted JSON of Facebook OAuth token
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserCreate(SQLModel):
    username: str
    password: str


class UserLogin(SQLModel):
    username: str
    password: str


class UserSettings(SQLModel):
    """Settings the user can read/update (keys are returned decrypted)."""
    groq_key: str | None = None
    gemini_key: str | None = None
    youtube_api_key: str | None = None
    gemini_model: str | None = None
    whisper_model: str | None = None
    youtube_cookies: str | None = None


class UserSettingsUpdate(SQLModel):
    model_config = {"populate_by_name": True}

    groq_key: str | None = None
    gemini_key: str | None = None
    youtube_api_key: str | None = None
    gemini_model: str | None = None
    whisper_model: str | None = None
    youtube_cookies: str | None = None


class UserPublic(SQLModel):
    id: int
    username: str
    created_at: datetime


# --- Helpers -------------------------------------------------------------
def get_user_by_username(session: Session, username: str) -> Optional[User]:
    return session.exec(select(User).where(User.username == username)).first()


def create_user(session: Session, data: UserCreate, password_hash: str) -> User:
    user = User(username=data.username, password_hash=password_hash)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def settings_for(user: User) -> UserSettings:
    """Return the user's settings with keys decrypted for display."""
    return UserSettings(
        groq_key=decrypt_value(user.groq_key),
        gemini_key=decrypt_value(user.gemini_key),
        youtube_api_key=decrypt_value(user.youtube_api_key),
        gemini_model=user.gemini_model,
        whisper_model=user.whisper_model,
        youtube_cookies=decrypt_value(user.youtube_cookies),
    )


def apply_settings(user: User, data: UserSettingsUpdate) -> None:
    """Persist settings, encrypting any key fields that were provided."""
    if data.groq_key is not None:
        user.groq_key = encrypt_value(data.groq_key)
    if data.gemini_key is not None:
        user.gemini_key = encrypt_value(data.gemini_key)
    if data.youtube_api_key is not None:
        user.youtube_api_key = encrypt_value(data.youtube_api_key)
    if data.gemini_model is not None:
        user.gemini_model = data.gemini_model
    if data.whisper_model is not None:
        user.whisper_model = data.whisper_model
    if data.youtube_cookies is not None:
        user.youtube_cookies = encrypt_value(data.youtube_cookies)


def save_youtube_token(user: User, token_json: str) -> None:
    """Encrypt and store the user's YouTube OAuth token (JSON string)."""
    user.youtube_token = encrypt_value(token_json)


def load_youtube_token(user: User) -> Optional[str]:
    """Return the user's YouTube OAuth token JSON, decrypted (or None)."""
    return decrypt_value(user.youtube_token)


def save_facebook_token(user: User, token_data: dict) -> None:
    """Encrypt and store the user's Facebook OAuth token (dict serialized to JSON)."""
    import json
    user.facebook_token = encrypt_value(json.dumps(token_data))


def load_facebook_token(user: User) -> Optional[dict]:
    """Return the user's Facebook OAuth token dict, decrypted (or None)."""
    import json
    raw = decrypt_value(user.facebook_token)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
