import base64
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

_bearer_scheme = HTTPBearer(auto_error=False)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(f"{data}{padding}")


def _sign(message: bytes, secret_key: str) -> str:
    digest = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).digest()
    return _b64url_encode(digest)


def create_access_token(
    *,
    username: str,
    ttl_seconds: int | None = None,
    token_type: str = "access",
) -> tuple[str, int]:
    settings = get_settings()
    now = int(time.time())
    expires_in = ttl_seconds or settings.access_token_ttl_seconds
    payload = {
        "sub": username,
        "type": token_type,
        "iat": now,
        "exp": now + expires_in,
    }
    header = {"alg": "HS256", "typ": "JWT"}
    header_part = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_part = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    signature = _sign(signing_input, settings.auth_secret_key)
    return f"{header_part}.{payload_part}.{signature}", expires_in


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        header_part, payload_part, signature = token.split(".")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid access token",
        ) from exc

    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    expected_signature = _sign(signing_input, settings.auth_secret_key)
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid access token signature",
        )

    try:
        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid access token payload",
        ) from exc

    exp = payload.get("exp")
    if not isinstance(exp, int) or exp <= int(time.time()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="access token expired",
        )
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid access token type",
        )
    return payload


def verify_admin_credentials(username: str, password: str) -> bool:
    settings = get_settings()
    expected_username = settings.auth_admin_username
    expected_password = settings.auth_admin_password or ""
    return hmac.compare_digest(username, expected_username) and hmac.compare_digest(
        password,
        expected_password,
    )


def get_optional_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, str] | None:
    settings = get_settings()
    token = credentials.credentials if credentials else request.cookies.get(
        settings.session_cookie_name
    )
    if not token:
        return None

    payload = decode_access_token(token)
    username = payload.get("sub")
    if not isinstance(username, str) or not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid access token subject",
        )
    return {"username": username, "role": "admin"}


def require_api_auth(user: dict[str, str] | None = Depends(get_optional_current_user)) -> dict[str, str] | None:
    settings = get_settings()
    if not settings.auth_enabled:
        return {"username": "anonymous", "role": "system"}
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )
    return user
