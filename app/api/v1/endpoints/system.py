import socket
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.schemas.system import (
    SystemComponentStatus,
    SystemInfoResponse,
    SystemStatusResponse,
    ValidateYoloEnvRequest,
    ValidateYoloEnvResponse,
)
from app.services.yolo_env import validate_yolo_env

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
        status_name = "degraded" if settings.is_production_env else "not_configured"
        return SystemComponentStatus(name="auth", status=status_name, detail="auth disabled")
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


def _check_public_base_url() -> SystemComponentStatus:
    settings = get_settings()
    public_base_url = settings.normalized_public_base_url
    if not public_base_url:
        status_name = "degraded" if settings.is_production_env else "not_configured"
        return SystemComponentStatus(
            name="public_base_url",
            status=status_name,
            detail="SELF_API_PUBLIC_BASE_URL is not configured",
        )

    parsed = urlparse(public_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return SystemComponentStatus(
            name="public_base_url",
            status="degraded",
            detail=f"invalid public base url: {public_base_url}",
        )
    return SystemComponentStatus(name="public_base_url", status="ok", detail=public_base_url)


def _check_file_access() -> SystemComponentStatus:
    settings = get_settings()
    if not settings.restrict_file_access:
        status_name = "degraded" if settings.is_production_env else "not_configured"
        return SystemComponentStatus(
            name="file_access",
            status=status_name,
            detail="file access restriction disabled",
        )

    if not settings.has_explicit_file_access_roots:
        status_name = "degraded" if settings.is_production_env else "not_configured"
        return SystemComponentStatus(
            name="file_access",
            status=status_name,
            detail="SELF_API_FILE_ACCESS_ROOTS is not explicitly configured",
        )

    missing_roots = [
        str(path) for path in settings.resolved_file_access_roots if not path.exists()
    ]
    if missing_roots:
        return SystemComponentStatus(
            name="file_access",
            status="degraded",
            detail=f"configured roots do not exist: {', '.join(missing_roots)}",
        )
    return SystemComponentStatus(
        name="file_access",
        status="ok",
        detail=", ".join(str(path) for path in settings.resolved_file_access_roots),
    )


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
        _check_public_base_url(),
        _check_file_access(),
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
        public_base_url=settings.normalized_public_base_url,
        auth_enabled=settings.auth_enabled,
        session_cookie_secure=settings.session_cookie_secure,
        restrict_file_access=settings.restrict_file_access,
        explicit_file_access_roots=settings.has_explicit_file_access_roots,
        storage_root=str(settings.resolved_storage_root),
        file_access_roots=[str(path) for path in settings.resolved_file_access_roots],
        cors_allow_origins=settings.cors_allow_origin_list,
    )


@router.post("/validate-yolo-env", response_model=ValidateYoloEnvResponse)
def validate_yolo_runtime(payload: ValidateYoloEnvRequest) -> ValidateYoloEnvResponse:
    try:
        result = validate_yolo_env(payload.yolo_train_env)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ValidateYoloEnvResponse(
        yolo_train_env=payload.yolo_train_env,
        **result,
    )
