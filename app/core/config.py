from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Image Preprocess API"
    app_version: str = "0.1.0"
    app_env: str = "dev"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SELF_API_")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
