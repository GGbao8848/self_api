import json

from app.api.v1.endpoints import tasks as tasks_endpoint
from app.core.config import get_settings


def test_train_ui_page(client) -> None:
    response = client.get("/train-ui")
    assert response.status_code == 200
    assert "模型训练任务提交台" in response.text
    assert "launch-training-workflow" in response.text


def test_launch_training_workflow_endpoint(client, monkeypatch) -> None:
    monkeypatch.setenv("SELF_API_N8N_BASE_URL", "http://127.0.0.1:5678")
    get_settings.cache_clear()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"message": "Workflow got started"}).encode("utf-8")

        def getcode(self):
            return 200

    def fake_urlopen(request, timeout=0):
        assert request.full_url == "http://127.0.0.1:5678/webhook/self-api-train"
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["project_name"] == "TVDS"
        assert payload["detector_name"] == "nzxj_louyou"
        assert payload["remote_project_root_dir"] == "/workspace_root"
        return FakeResponse()

    monkeypatch.setattr(tasks_endpoint, "urlopen", fake_urlopen)

    response = client.post(
        "/api/v1/tasks/launch-training-workflow",
        json={
            "self_api_url": "http://192.168.1.73:8666",
            "workspace_root_dir": "/workspace_root",
            "project_name": "TVDS",
            "detector_name": "nzxj_louyou",
            "original_dataset_dir": "/raw/data",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_mode"] == "local"
    assert data["webhook_url"] == "http://127.0.0.1:5678/webhook/self-api-train"
    assert data["upstream_status_code"] == 200
    assert data["upstream_response"]["message"] == "Workflow got started"


def test_launch_training_workflow_explicit_local_strips_remote_for_webhook(
    client, monkeypatch
) -> None:
    """run_target=local 时即使用户填了 SSH，发往 n8n 的 body 仍应为空远端字段。"""
    monkeypatch.setenv("SELF_API_N8N_BASE_URL", "http://127.0.0.1:5678")
    get_settings.cache_clear()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"{}"

        def getcode(self):
            return 200

    def fake_urlopen(request, timeout=0):
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["remote_host"] is None
        assert payload["remote_username"] is None
        assert payload["remote_private_key_path"] is None
        assert payload["remote_project_root_dir"] is None
        return FakeResponse()

    monkeypatch.setattr(tasks_endpoint, "urlopen", fake_urlopen)

    response = client.post(
        "/api/v1/tasks/launch-training-workflow",
        json={
            "self_api_url": "http://192.168.1.73:8666",
            "workspace_root_dir": "/workspace_root",
            "project_name": "TVDS",
            "detector_name": "nzxj_louyou",
            "original_dataset_dir": "/raw/data",
            "run_target": "local",
            "remote_host": "192.168.2.1",
            "remote_username": "u",
            "remote_private_key_path": "/key",
        },
    )
    assert response.status_code == 200
    assert response.json()["workflow_mode"] == "local"


def test_launch_training_workflow_remote_requires_ssh(client, monkeypatch) -> None:
    monkeypatch.setenv("SELF_API_N8N_BASE_URL", "http://127.0.0.1:5678")
    get_settings.cache_clear()

    response = client.post(
        "/api/v1/tasks/launch-training-workflow",
        json={
            "self_api_url": "http://192.168.1.73:8666",
            "workspace_root_dir": "/workspace_root",
            "project_name": "TVDS",
            "detector_name": "nzxj_louyou",
            "original_dataset_dir": "/raw/data",
            "run_target": "remote_sftp",
            "remote_host": "192.168.2.1",
        },
    )
    assert response.status_code == 400
