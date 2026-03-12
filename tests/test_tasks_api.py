import time

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.task_manager import ensure_current_task_active, submit_task


def _wait_for_task(client: TestClient, task_id: str, timeout_seconds: float = 5.0) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["state"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"task not finished in time: {task_id}")


def test_tasks_list_and_artifacts(case_dir, isolated_runtime) -> None:
    get_settings.cache_clear()

    output_dir = case_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.txt").write_text("done", encoding="utf-8")

    task_id = submit_task(
        "unit_artifact_task",
        runner=lambda: {"output_dir": str(output_dir)},
    )

    with TestClient(app) as client:
        task_payload = _wait_for_task(client, task_id)
        assert task_payload["state"] == "succeeded"
        assert task_payload["artifacts"]

        list_resp = client.get("/api/v1/tasks")
        assert list_resp.status_code == 200
        items = list_resp.json()["items"]
        assert any(item["task_id"] == task_id for item in items)


def test_task_cancel_endpoint(isolated_runtime) -> None:
    get_settings.cache_clear()

    def _runner() -> dict[str, str]:
        for _ in range(40):
            ensure_current_task_active()
            time.sleep(0.05)
        return {"output_dir": "."}

    task_id = submit_task("unit_cancel_task", runner=_runner)

    with TestClient(app) as client:
        cancel_resp = client.post(f"/api/v1/tasks/{task_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["task"]["cancellation_requested"] is True

        task_payload = _wait_for_task(client, task_id)
        assert task_payload["state"] == "cancelled"
