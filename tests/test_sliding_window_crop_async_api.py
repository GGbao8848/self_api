import time
from pathlib import Path
from urllib.error import HTTPError

from fastapi.testclient import TestClient

from app.services import task_manager
from tests.data_helpers import create_image


def _wait_task_done(
    client: TestClient,
    task_id: str,
    *,
    timeout_seconds: float = 5.0,
) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/v1/preprocess/tasks/{task_id}")
        assert response.status_code == 200
        task_data = response.json()
        if task_data["state"] in {"succeeded", "failed"}:
            return task_data
        time.sleep(0.05)
    raise AssertionError(f"task did not complete in {timeout_seconds} seconds: {task_id}")


def _wait_callback_done(
    client: TestClient,
    task_id: str,
    *,
    timeout_seconds: float = 5.0,
) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/v1/preprocess/tasks/{task_id}")
        assert response.status_code == 200
        task_data = response.json()
        if task_data["callback_state"] in {"succeeded", "failed"}:
            return task_data
        time.sleep(0.05)
    raise AssertionError(
        f"callback did not finish in {timeout_seconds} seconds: {task_id}"
    )


def test_sliding_window_crop_async_success(client: TestClient, case_dir: Path) -> None:
    input_dir = case_dir / "input"
    output_dir = case_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    create_image(input_dir / "sample.png", color=(255, 0, 0), size=(10, 10))

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_width": 4,
        "window_height": 4,
        "stride_x": 4,
        "stride_y": 4,
        "include_partial_edges": False,
        "recursive": True,
        "keep_subdirs": True,
        "output_format": "png",
    }

    submit_resp = client.post("/api/v1/preprocess/sliding-window-crop/async", json=payload)
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()
    assert submit_data["status"] == "accepted"
    assert submit_data["task_type"] == "sliding_window_crop"
    assert submit_data["status_url"].endswith(f"/api/v1/preprocess/tasks/{submit_data['task_id']}")
    assert submit_data["callback_url"] is None

    task_data = _wait_task_done(client, submit_data["task_id"])
    assert task_data["state"] == "succeeded"
    assert task_data["error"] is None
    assert task_data["callback_state"] == "succeeded"
    assert task_data["result"]["generated_crops"] == 4
    assert task_data["result"]["processed_images"] == 1
    assert len(list(output_dir.rglob("*.png"))) == 4


def test_sliding_window_crop_async_failed_task(client: TestClient, case_dir: Path) -> None:
    payload = {
        "input_dir": str(case_dir / "missing_input"),
        "output_dir": str(case_dir / "output"),
        "window_width": 4,
        "window_height": 4,
        "stride_x": 4,
        "stride_y": 4,
        "include_partial_edges": False,
        "recursive": True,
        "keep_subdirs": True,
        "output_format": "png",
    }

    submit_resp = client.post("/api/v1/preprocess/sliding-window-crop/async", json=payload)
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["task_id"]

    task_data = _wait_task_done(client, task_id)
    assert task_data["state"] == "failed"
    assert "input_dir does not exist" in task_data["error"]
    assert task_data["result"] is None


def test_sliding_window_crop_async_uses_public_base_url(
    case_dir: Path,
    monkeypatch,
    isolated_runtime,
) -> None:
    from app.core.config import get_settings
    from app.main import app

    monkeypatch.setenv("SELF_API_PUBLIC_BASE_URL", "https://self-api.example.com")
    get_settings.cache_clear()

    input_dir = case_dir / "input"
    output_dir = case_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    create_image(input_dir / "sample.png", color=(255, 0, 0), size=(10, 10))

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_width": 4,
        "window_height": 4,
        "stride_x": 4,
        "stride_y": 4,
        "include_partial_edges": False,
        "recursive": True,
        "keep_subdirs": True,
        "output_format": "png",
    }

    with TestClient(app) as client:
        submit_resp = client.post("/api/v1/preprocess/sliding-window-crop/async", json=payload)

    assert submit_resp.status_code == 202
    assert submit_resp.json()["status_url"].startswith(
        "https://self-api.example.com/api/v1/preprocess/tasks/"
    )


def test_get_preprocess_task_status_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/preprocess/tasks/not_found")
    assert response.status_code == 404


def test_sliding_window_crop_async_callback_success(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    input_dir = case_dir / "input"
    output_dir = case_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    create_image(input_dir / "sample.png", color=(255, 0, 0), size=(10, 10))

    callback_calls: list[dict] = []

    def _fake_post_callback(callback_url: str, payload: dict, timeout: float) -> int:
        callback_calls.append(
            {"callback_url": callback_url, "payload": payload, "timeout": timeout}
        )
        return 204

    monkeypatch.setattr(task_manager, "_post_callback", _fake_post_callback)

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_width": 4,
        "window_height": 4,
        "stride_x": 4,
        "stride_y": 4,
        "include_partial_edges": False,
        "recursive": True,
        "keep_subdirs": True,
        "output_format": "png",
        "callback_url": "http://127.0.0.1:9999/callback",
    }

    submit_resp = client.post("/api/v1/preprocess/sliding-window-crop/async", json=payload)
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["task_id"]
    assert submit_resp.json()["callback_url"] == "http://127.0.0.1:9999/callback"

    task_data = _wait_task_done(client, task_id)
    assert task_data["state"] == "succeeded"
    task_data = _wait_callback_done(client, task_id)
    assert task_data["callback_state"] == "succeeded"
    assert task_data["callback_status_code"] == 204
    assert task_data["callback_error"] is None
    assert task_data["callback_sent_at"] is not None
    assert len(task_data["callback_events"]) == 1
    assert [event["state"] for event in task_data["callback_events"]] == ["succeeded"]
    assert all(event["callback_url"] == "http://127.0.0.1:9999/callback" for event in task_data["callback_events"])
    assert all(event["status_code"] == 204 for event in task_data["callback_events"])
    assert all(event["method"] == "POST" for event in task_data["callback_events"])
    assert all(event["success"] is True for event in task_data["callback_events"])

    assert len(callback_calls) == 1
    assert [call["payload"]["state"] for call in callback_calls] == ["succeeded"]
    assert callback_calls[0]["callback_url"] == "http://127.0.0.1:9999/callback"
    assert callback_calls[-1]["payload"]["task_id"] == task_id


def test_sliding_window_crop_async_callback_failed_delivery(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    input_dir = case_dir / "input"
    output_dir = case_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    create_image(input_dir / "sample.png", color=(255, 0, 0), size=(10, 10))

    callback_calls: list[dict] = []

    def _fake_post_callback(callback_url: str, payload: dict, timeout: float) -> int:
        callback_calls.append(
            {"callback_url": callback_url, "payload": payload, "timeout": timeout}
        )
        raise TimeoutError("callback timeout")

    monkeypatch.setattr(task_manager, "_post_callback", _fake_post_callback)

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_width": 4,
        "window_height": 4,
        "stride_x": 4,
        "stride_y": 4,
        "include_partial_edges": False,
        "recursive": True,
        "keep_subdirs": True,
        "output_format": "png",
        "callback_url": "http://127.0.0.1:9999/callback",
    }

    submit_resp = client.post("/api/v1/preprocess/sliding-window-crop/async", json=payload)
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["task_id"]

    task_data = _wait_task_done(client, task_id)
    assert task_data["state"] == "succeeded"
    task_data = _wait_callback_done(client, task_id)
    assert task_data["callback_state"] == "failed"
    assert task_data["callback_status_code"] is None
    assert "callback timeout" in task_data["callback_error"]
    assert len(task_data["callback_events"]) == 1
    assert [event["state"] for event in task_data["callback_events"]] == ["succeeded"]
    assert all(event["callback_url"] == "http://127.0.0.1:9999/callback" for event in task_data["callback_events"])
    assert all(event["status_code"] is None for event in task_data["callback_events"])
    assert all(event["method"] == "POST" for event in task_data["callback_events"])
    assert all(event["success"] is False for event in task_data["callback_events"])
    assert all("callback timeout" in event["error"] for event in task_data["callback_events"])
    assert len(callback_calls) == 1
    assert [call["payload"]["state"] for call in callback_calls] == ["succeeded"]


def test_sliding_window_crop_async_callback_fallback_to_get(
    client: TestClient,
    case_dir: Path,
    monkeypatch,
) -> None:
    input_dir = case_dir / "input"
    output_dir = case_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    create_image(input_dir / "sample.png", color=(255, 0, 0), size=(10, 10))

    post_calls: list[dict] = []
    get_calls: list[dict] = []

    def _fake_post_callback(callback_url: str, payload: dict, timeout: float) -> int:
        post_calls.append(
            {"callback_url": callback_url, "payload": payload, "timeout": timeout}
        )
        raise HTTPError(callback_url, 405, "Method Not Allowed", None, None)

    def _fake_get_callback(callback_url: str, timeout: float) -> int:
        get_calls.append({"callback_url": callback_url, "timeout": timeout})
        return 200

    monkeypatch.setattr(task_manager, "_post_callback", _fake_post_callback)
    monkeypatch.setattr(task_manager, "_get_callback", _fake_get_callback)

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_width": 4,
        "window_height": 4,
        "stride_x": 4,
        "stride_y": 4,
        "include_partial_edges": False,
        "recursive": True,
        "keep_subdirs": True,
        "output_format": "png",
        "callback_url": "http://127.0.0.1:9999/callback",
    }

    submit_resp = client.post("/api/v1/preprocess/sliding-window-crop/async", json=payload)
    assert submit_resp.status_code == 202
    task_id = submit_resp.json()["task_id"]

    task_data = _wait_task_done(client, task_id)
    assert task_data["state"] == "succeeded"
    task_data = _wait_callback_done(client, task_id)
    assert task_data["callback_state"] == "succeeded"
    assert task_data["callback_status_code"] == 200
    assert len(task_data["callback_events"]) == 1
    assert [event["state"] for event in task_data["callback_events"]] == ["succeeded"]
    assert all(event["method"] == "GET" for event in task_data["callback_events"])
    assert len(post_calls) == 1
    assert len(get_calls) == 1
