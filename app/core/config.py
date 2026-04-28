from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Image Preprocess API"
    app_version: str = "0.1.0"
    app_env: str = "dev"
    api_v1_prefix: str = "/api/v1"
    public_base_url: str | None = None
    log_level: str = "INFO"
    restrict_file_access: bool = True
    file_access_roots: str | None = None
    storage_root: str = "./storage"
    auth_enabled: bool = False
    auth_admin_username: str = "admin"
    auth_admin_password: str | None = None
    auth_secret_key: str = "change-me-in-production"
    access_token_ttl_seconds: int = 3600
    session_cookie_name: str = "self_api_session"
    session_cookie_secure: bool = False
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    postgres_dsn: str | None = None
    redis_url: str | None = None
    s3_endpoint_url: str | None = None
    max_upload_size_mb: int = 512
    # 供与 n8n 集成时使用（.env：SELF_API_N8N_BASE_URL / SELF_API_N8N_API_KEY）
    n8n_base_url: str | None = None
    n8n_api_key: str | None = None
    publish_project_root_dir: str | None = None
    remote_sftp_host: str | None = None
    remote_sftp_project_root_dir: str | None = None
    remote_sftp_username: str | None = None
    remote_sftp_private_key_path: str | None = None
    remote_sftp_port: int = 22

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SELF_API_",
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def is_production_env(self) -> bool:
        return self.app_env.strip().lower() in {"prod", "production"}

    @property
    def normalized_public_base_url(self) -> str | None:
        raw = (self.public_base_url or "").strip()
        if not raw:
            return None
        return raw.rstrip("/")

    @property
    def has_explicit_file_access_roots(self) -> bool:
        raw = self.file_access_roots
        if not raw:
            return False
        return any(item.strip() for item in raw.split(","))

    @property
    def resolved_storage_root(self) -> Path:
        return (self.project_root / self.storage_root).resolve()

    @property
    def resolved_file_access_roots(self) -> list[Path]:
        if not self.restrict_file_access:
            return []

        raw = self.file_access_roots
        if raw:
            roots = []
            for item in raw.split(","):
                text = item.strip()
                if not text:
                    continue
                roots.append(Path(text).expanduser().resolve())
            if roots:
                return roots

        return [self.project_root.resolve()]

    @property
    def cors_allow_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return max(self.max_upload_size_mb, 1) * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
