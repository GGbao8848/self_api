from pydantic import BaseModel, Field


class AuthUser(BaseModel):
    username: str
    role: str = "admin"


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    status: str = "ok"
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthUser


class LogoutResponse(BaseModel):
    status: str = "ok"


class AuthStatusResponse(BaseModel):
    status: str = "ok"
    authenticated: bool
    auth_enabled: bool
    user: AuthUser | None = None
