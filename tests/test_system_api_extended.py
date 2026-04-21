from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


def test_system_info_and_readiness(isolated_runtime) -> None:
    get_settings.cache_clear()

    with TestClient(app) as client:
        info_resp = client.get("/api/v1/info")
        assert info_resp.status_code == 200
        assert info_resp.json()["storage_root"].endswith("tmp_datasets/test_storage")
        assert info_resp.json()["public_base_url"] is None
        assert info_resp.json()["restrict_file_access"] is True
        assert info_resp.json()["explicit_file_access_roots"] is False

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


def test_readiness_degraded_in_production_without_public_base_url(
    monkeypatch,
    isolated_runtime,
) -> None:
    monkeypatch.setenv("SELF_API_APP_ENV", "prod")
    monkeypatch.setenv("SELF_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("SELF_API_AUTH_ADMIN_PASSWORD", "secret-pass")
    monkeypatch.setenv("SELF_API_AUTH_SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("SELF_API_FILE_ACCESS_ROOTS", "./tmp_datasets")
    get_settings.cache_clear()

    with TestClient(app) as client:
        readiness_resp = client.get("/api/v1/readiness")
        assert readiness_resp.status_code == 503
        payload = readiness_resp.json()
        assert payload["status"] == "degraded"
        component_by_name = {item["name"]: item for item in payload["components"]}
        assert component_by_name["public_base_url"]["status"] == "degraded"
        assert "SELF_API_PUBLIC_BASE_URL" in component_by_name["public_base_url"]["detail"]


def test_validate_yolo_env_endpoint(monkeypatch, isolated_runtime) -> None:
    get_settings.cache_clear()
    monkeypatch.setattr(
        "app.api.v1.endpoints.system.validate_yolo_env",
        lambda env: {
            "status": "ok",
            "mode": "python_path",
            "resolved_python": "/tmp/fake/python",
            "command": "python -c ...",
            "exit_code": 0,
            "stdout": "ok\n",
            "stderr": "",
        },
    )
    with TestClient(app) as client:
        resp = client.post("/api/v1/validate-yolo-env", json={"yolo_train_env": "/tmp/fake/env"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "python_path"
