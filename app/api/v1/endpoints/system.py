import socket
from urllib.parse import urlparse

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.schemas.system import SystemComponentStatus, SystemInfoResponse, SystemStatusResponse

router = APIRouter(tags=["system"])


def _check_storage() -> SystemComponentStatus:
    settings = get_settings()
    try:
        storage_root = settings.resolved_storage_root
        storage_root.mkdir(parents=True, exist_ok=True)
        probe = storage_root / ".readiness_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return SystemComponentStatus(name="storage", status="degraded", detail=str(exc))
    return SystemComponentStatus(name="storage", status="ok", detail=str(storage_root))


def _check_auth() -> SystemComponentStatus:
    settings = get_settings()
    if not settings.auth_enabled:
        return SystemComponentStatus(name="auth", status="not_configured", detail="auth disabled")
    if not settings.auth_admin_password:
        return SystemComponentStatus(
            name="auth",
            status="degraded",
            detail="SELF_API_AUTH_ADMIN_PASSWORD is not configured",
        )
    if settings.auth_secret_key == "change-me-in-production":
        return SystemComponentStatus(
            name="auth",
            status="degraded",
            detail="SELF_API_AUTH_SECRET_KEY is using the default value",
        )
    return SystemComponentStatus(name="auth", status="ok", detail="configured")


def _check_socket_target(name: str, raw_url: str | None, default_port: int) -> SystemComponentStatus:
    if not raw_url:
        return SystemComponentStatus(name=name, status="not_configured", detail="not configured")

    parsed = urlparse(raw_url)
    host = parsed.hostname
    port = parsed.port or default_port
    if not host:
        return SystemComponentStatus(name=name, status="degraded", detail=f"invalid url: {raw_url}")

    try:
        with socket.create_connection((host, port), timeout=1.5):
            pass
    except OSError as exc:
        return SystemComponentStatus(name=name, status="degraded", detail=str(exc))
    return SystemComponentStatus(name=name, status="ok", detail=f"{host}:{port}")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/liveness")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readiness", response_model=SystemStatusResponse)
def readiness() -> SystemStatusResponse | JSONResponse:
    settings = get_settings()
    components = [
        _check_storage(),
        _check_auth(),
        _check_socket_target("postgres", settings.postgres_dsn, 5432),
        _check_socket_target("redis", settings.redis_url, 6379),
        _check_socket_target("s3", settings.s3_endpoint_url, 9000),
    ]
    overall = "degraded" if any(item.status == "degraded" for item in components) else "ok"
    response = SystemStatusResponse(status=overall, components=components)
    if overall == "degraded":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )
    return response


@router.get("/info", response_model=SystemInfoResponse)
def info() -> SystemInfoResponse:
    settings = get_settings()
    return SystemInfoResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        app_env=settings.app_env,
        api_v1_prefix=settings.api_v1_prefix,
        auth_enabled=settings.auth_enabled,
        storage_root=str(settings.resolved_storage_root),
        file_access_roots=[str(path) for path in settings.resolved_file_access_roots],
    )
