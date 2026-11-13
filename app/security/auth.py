"""Authentication: hardcoded demo users (brief Option A) and JWT tokens.

Role and clearance always come from the signed token. Nothing the user types
in chat, and nothing the LLM produces, can change them.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum

import bcrypt
import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.config import get_settings
from app.errors import AuthenticationError
from app.logging import bind_context, get_logger

log = get_logger(__name__)

JWT_ALGORITHM = "HS256"
TOKEN_TTL = timedelta(hours=8)


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


# Document access levels each role may read (see docs/decisions.md and POL-002).
ROLE_CLEARANCE: dict[Role, list[str]] = {
    Role.VIEWER: ["public", "internal"],
    Role.ANALYST: ["public", "internal", "confidential"],
    Role.ADMIN: ["public", "internal", "confidential", "restricted"],
}


class User(BaseModel):
    user_id: str
    display_name: str
    department: str
    role: Role

    @property
    def clearance(self) -> list[str]:
        return ROLE_CLEARANCE[self.role]


class _UserRecord(BaseModel):
    user: User
    password_hash: str


# Demo users. Passwords are listed in README "Demo users"; only bcrypt hashes live here.
_USERS: dict[str, _UserRecord] = {
    "viewer1": _UserRecord(
        user=User(user_id="viewer1", display_name="Nimal (Branch Ops)", department="operations", role=Role.VIEWER),
        password_hash="$2b$10$jagVZ3QVkaB7Mfo4A8kNyemlhVrQp6SyHuqcEJddFDiyloWEXcPMO",
    ),
    "analyst1": _UserRecord(
        user=User(user_id="analyst1", display_name="Ayesha (Payments Analyst)", department="payments", role=Role.ANALYST),
        password_hash="$2b$10$4vRZqvoj8rb5Ix0uDKdGNOcUQAdqL05Sv6krZsS.kB5vU59hxwmb6",
    ),
    "admin1": _UserRecord(
        user=User(user_id="admin1", display_name="Ruwan (IT Admin)", department="it", role=Role.ADMIN),
        password_hash="$2b$10$jSmLvZdCfKcOdfGPklFpdOd/1a5/4W4JductWOyJHMBklW.WoRpqq",
    ),
}

# Checked when the username is unknown, so response time does not reveal which usernames exist.
_DUMMY_HASH = "$2b$10$M9YlbE8m/L5O7RMpArDg7.b.ZBkf3LKGmfquRxYzzxxSq4vYAFjlu"


def authenticate(username: str, password: str) -> User:
    record = _USERS.get(username)
    hashed = record.password_hash if record else _DUMMY_HASH
    password_ok = bcrypt.checkpw(password.encode(), hashed.encode())
    if not record or not password_ok:
        log.warning("login_failed", username=username, reason="unknown_user" if not record else "bad_password")
        raise AuthenticationError("Invalid username or password.")
    log.info("login_succeeded", user_id=record.user.user_id, role=record.user.role.value)
    return record.user


def create_token(user: User) -> tuple[str, int]:
    """Return (token, expires_in_seconds)."""
    now = datetime.now(timezone.utc)
    claims = {
        "sub": user.user_id,
        "role": user.role.value,
        "clearance": user.clearance,
        "iat": now,
        "exp": now + TOKEN_TTL,
    }
    secret = get_settings().jwt_secret.get_secret_value()
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM), int(TOKEN_TTL.total_seconds())


def decode_token(token: str) -> User:
    secret = get_settings().jwt_secret.get_secret_value()
    try:
        claims = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Session expired. Please log in again.") from None
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid session. Please log in again.") from None

    # Re-load the user from the server-side list: a token for a removed user is rejected,
    # and role/clearance are taken from the server, not trusted blindly from the token.
    record = _USERS.get(claims.get("sub", ""))
    if not record or record.user.role.value != claims.get("role"):
        raise AuthenticationError("Invalid session. Please log in again.")
    return record.user


_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User:
    """FastAPI dependency: every protected endpoint gets the logged-in user from here."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError()
    user = decode_token(credentials.credentials)
    # From here on, every log line in this request carries the user.
    bind_context(user_id=user.user_id, role=user.role.value)
    # Also kept on the request so the HTTP middleware can add it to the request log line
    # (the middleware runs outside the endpoint's context).
    request.state.user_id, request.state.role = user.user_id, user.role.value
    return user
