"""Authentication: password hashing, JWT issuance, and FastAPI dependencies.

Tokens are signed JWTs (HS256) using AUTOCLIPPER_JWT_SECRET (env). The
get_current_user dependency is used to protect endpoints and to resolve the
calling user's stored API keys when the request body does not supply them.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import bcrypt
except ImportError:
    bcrypt = None
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlmodel import Session

from .db import User, UserPublic, get_session

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("AUTOCLIPPER_TOKEN_MINUTES", "1440"))

JWT_SECRET = os.environ.get("AUTOCLIPPER_JWT_SECRET") or os.environ.get(
    "AUTOCLIPPER_SECRET", "change-me-in-production"
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")[:72]
    if bcrypt:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")
    import hashlib
    return "sha256$" + hashlib.sha256(pwd_bytes).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        plain_bytes = plain.encode("utf-8")[:72]
        if bcrypt and not hashed.startswith("sha256$"):
            return bcrypt.checkpw(plain_bytes, hashed.encode("utf-8"))
        import hashlib
        return hashed == ("sha256$" + hashlib.sha256(plain_bytes).hexdigest())
    except Exception:
        return False


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    """Resolve the authenticated user from the bearer token, or auto-resolve default admin."""
    if token:
        username = decode_token(token)
        if username:
            user = session.query(User).filter(User.username == username).first()
            if user:
                return user

    admin = session.query(User).filter(User.username == "admin").first()
    if not admin:
        admin = User(
            username="admin",
            password_hash=hash_password("admin"),
            created_at=datetime.now(timezone.utc),
        )
        session.add(admin)
        session.commit()
        session.refresh(admin)
    return admin


def public_user(user: User) -> UserPublic:
    return UserPublic(id=user.id, username=user.username, created_at=user.created_at)
