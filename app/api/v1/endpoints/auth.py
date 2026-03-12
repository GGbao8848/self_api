from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    get_optional_current_user,
    verify_admin_credentials,
)
from app.schemas.auth import AuthStatusResponse, AuthUser, LoginRequest, LoginResponse, LogoutResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, response: Response) -> LoginResponse:
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(status_code=400, detail="auth is disabled")
    if not settings.auth_admin_password:
        raise HTTPException(status_code=500, detail="auth admin password is not configured")
    if not verify_admin_credentials(payload.username, payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")

    access_token, expires_in = create_access_token(username=payload.username)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=access_token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=expires_in,
        path="/",
    )
    return LoginResponse(
        access_token=access_token,
        expires_in=expires_in,
        user=AuthUser(username=payload.username),
    )


@router.post("/refresh", response_model=LoginResponse)
def refresh(
    response: Response,
    user: dict[str, str] | None = Depends(get_optional_current_user),
) -> LoginResponse:
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(status_code=400, detail="auth is disabled")
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")

    access_token, expires_in = create_access_token(username=user["username"])
    response.set_cookie(
        key=settings.session_cookie_name,
        value=access_token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=expires_in,
        path="/",
    )
    return LoginResponse(
        access_token=access_token,
        expires_in=expires_in,
        user=AuthUser(username=user["username"], role=user["role"]),
    )


@router.post("/logout", response_model=LogoutResponse)
def logout(response: Response) -> LogoutResponse:
    response.delete_cookie(key=get_settings().session_cookie_name, path="/")
    return LogoutResponse()


@router.get("/me", response_model=AuthStatusResponse)
def me(user: dict[str, str] | None = Depends(get_optional_current_user)) -> AuthStatusResponse:
    settings = get_settings()
    if user is None:
        return AuthStatusResponse(authenticated=False, auth_enabled=settings.auth_enabled)
    return AuthStatusResponse(
        authenticated=True,
        auth_enabled=settings.auth_enabled,
        user=AuthUser(username=user["username"], role=user["role"]),
    )
