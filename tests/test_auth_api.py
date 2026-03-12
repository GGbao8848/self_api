from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


def test_auth_login_me_logout(monkeypatch, isolated_runtime) -> None:
    monkeypatch.setenv("SELF_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("SELF_API_AUTH_ADMIN_PASSWORD", "secret-pass")
    monkeypatch.setenv("SELF_API_AUTH_SECRET_KEY", "unit-test-secret")
    get_settings.cache_clear()

    with TestClient(app) as client:
        unauth_me = client.get("/api/v1/auth/me")
        assert unauth_me.status_code == 200
        assert unauth_me.json()["authenticated"] is False
        assert unauth_me.json()["auth_enabled"] is True

        unauthorized_tasks = client.get("/api/v1/tasks")
        assert unauthorized_tasks.status_code == 401

        login_resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "secret-pass"},
        )
        assert login_resp.status_code == 200
        login_data = login_resp.json()
        assert login_data["user"]["username"] == "admin"
        assert login_data["access_token"]

        me_resp = client.get("/api/v1/auth/me")
        assert me_resp.status_code == 200
        assert me_resp.json()["authenticated"] is True
        assert me_resp.json()["user"]["username"] == "admin"

        tasks_resp = client.get("/api/v1/tasks")
        assert tasks_resp.status_code == 200

        logout_resp = client.post("/api/v1/auth/logout")
        assert logout_resp.status_code == 200

        me_after_logout = client.get("/api/v1/auth/me")
        assert me_after_logout.status_code == 200
        assert me_after_logout.json()["authenticated"] is False


def test_auth_login_rejects_invalid_credentials(monkeypatch, isolated_runtime) -> None:
    monkeypatch.setenv("SELF_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("SELF_API_AUTH_ADMIN_PASSWORD", "secret-pass")
    monkeypatch.setenv("SELF_API_AUTH_SECRET_KEY", "unit-test-secret")
    get_settings.cache_clear()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrong-pass"},
        )
        assert response.status_code == 401
