from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


def test_system_info_and_readiness(isolated_runtime) -> None:
    get_settings.cache_clear()

    with TestClient(app) as client:
        info_resp = client.get("/api/v1/info")
        assert info_resp.status_code == 200
        assert info_resp.json()["storage_root"].endswith("tmp_datasets/test_storage")

        readiness_resp = client.get("/api/v1/readiness")
        assert readiness_resp.status_code == 200
        assert readiness_resp.json()["status"] == "ok"


def test_readiness_degraded_when_auth_enabled_without_password(monkeypatch, isolated_runtime) -> None:
    monkeypatch.setenv("SELF_API_AUTH_ENABLED", "true")
    monkeypatch.delenv("SELF_API_AUTH_ADMIN_PASSWORD", raising=False)
    get_settings.cache_clear()

    with TestClient(app) as client:
        readiness_resp = client.get("/api/v1/readiness")
        assert readiness_resp.status_code == 503
        assert readiness_resp.json()["status"] == "degraded"
