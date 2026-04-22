import time

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.services.task_manager import cancel_task, ensure_current_task_active, get_task, report_progress, submit_task


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


def test_tasks_list_and_artifacts(case_dir, isolated_runtime, client: TestClient) -> None:
    get_settings.cache_clear()

    output_dir = case_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.txt").write_text("done", encoding="utf-8")

    task_id = submit_task(
        "unit_artifact_task",
        runner=lambda: {"output_dir": str(output_dir)},
    )

    task_payload = _wait_for_task(client, task_id)
    assert task_payload["state"] == "succeeded"
    assert task_payload["artifacts"]

    list_resp = client.get("/api/v1/tasks")
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert any(item["task_id"] == task_id for item in items)


def test_task_cancel_endpoint(isolated_runtime, client: TestClient) -> None:
    get_settings.cache_clear()

    def _runner() -> dict[str, str]:
        for _ in range(40):
            ensure_current_task_active()
            time.sleep(0.05)
        return {"output_dir": "."}

    task_id = submit_task("unit_cancel_task", runner=_runner)

    cancel_resp = client.post(f"/api/v1/tasks/{task_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["task"]["cancellation_requested"] is True

    task_payload = _wait_for_task(client, task_id)
    assert task_payload["state"] == "cancelled"


def test_task_status_exposes_runtime_progress(isolated_runtime, client: TestClient) -> None:
    get_settings.cache_clear()

    def _runner() -> dict[str, str]:
        for index in range(1, 4):
            ensure_current_task_active()
            report_progress(
                current=index,
                total=3,
                unit="file",
                stage="copy",
                message=f"processed {index}/3 files",
                indeterminate=False,
            )
            time.sleep(0.02)
        return {"output_dir": "."}

    task_id = submit_task("unit_progress_task", runner=_runner)
    payload = _wait_for_task(client, task_id)

    assert payload["state"] == "succeeded"
    assert payload["progress"]["percent"] == 100
    assert payload["progress"]["stage"] == "completed"


def test_task_cancel_removes_pending_queued_task(isolated_runtime) -> None:
    get_settings.cache_clear()

    first_started = False
    second_started = False

    def _first_runner() -> dict[str, str]:
        nonlocal first_started
        first_started = True
        for _ in range(20):
            ensure_current_task_active()
            time.sleep(0.05)
        return {"output_dir": "."}

    def _second_runner() -> dict[str, str]:
        nonlocal second_started
        second_started = True
        return {"output_dir": "."}

    first_task_id = submit_task(
        "unit_serial_task",
        runner=_first_runner,
        queue_key="unit_serial_queue",
    )
    second_task_id = submit_task(
        "unit_serial_task",
        runner=_second_runner,
        queue_key="unit_serial_queue",
    )

    deadline = time.time() + 1
    second_task_payload: dict | None = None
    while time.time() < deadline:
        second_task_payload = get_task(second_task_id)
        if second_task_payload["state"] == "pending":
            break
        time.sleep(0.02)
    assert first_started is True
    assert second_task_payload is not None
    assert second_task_payload["state"] == "pending"
    assert second_task_payload["queue_position"] == 1

    cancelled = cancel_task(second_task_id)
    assert cancelled is not None
    assert cancelled["state"] == "cancelled"

    stop_at = time.time() + 3
    first_result = None
    second_result = None
    while time.time() < stop_at:
        first_result = get_task(first_task_id)
        second_result = get_task(second_task_id)
        if (
            first_result is not None
            and second_result is not None
            and first_result["state"] in {"succeeded", "failed", "cancelled"}
            and second_result["state"] in {"succeeded", "failed", "cancelled"}
        ):
            break
        time.sleep(0.02)

    assert first_result is not None
    assert second_result is not None
    assert first_result["state"] == "succeeded"
    assert second_result["state"] == "cancelled"
    assert second_started is False
