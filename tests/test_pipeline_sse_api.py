"""Pipeline SSE `/pipeline/{run_id}/events` 端点测试。

复用 test_pipeline_graph_api.patch_all_nodes fixture，使 pipeline 在
stub 节点下可控制地暂停/推进，便于观察 SSE 事件流。
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tests.test_pipeline_graph_api import patch_all_nodes  # noqa: F401


BASE_RUN_PAYLOAD = {
    "original_dataset": "/tmp/fake_dataset",
    "detector_name": "sse_detector",
    "project_root_dir": "/tmp/fake_workspace",
    "execution_mode": "local",
    "yolo_train_env": "pytest_env",
    "yolo_train_epochs": 1,
    "yolo_train_imgsz": 320,
    "self_api_url": "http://127.0.0.1:1",
}


def _parse_sse(raw: str) -> list[dict[str, str]]:
    """极简 SSE 解析：把文本流拆成 [{event, data}, ...]。"""
    events: list[dict[str, str]] = []
    for block in raw.split("\n\n"):
        if not block.strip() or block.strip().startswith(":"):
            continue
        ev: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                ev["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                ev["data"] = line[len("data:"):].strip()
        if ev:
            events.append(ev)
    return events


def test_sse_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/pipeline/no-such-run/events")
    assert response.status_code == 404


def test_sse_emits_snapshot_and_end_for_completed_run(
    client: TestClient, patch_all_nodes: None
) -> None:
    """full_access run 启动后立刻 completed，SSE 首帧就应返回 snapshot + end。"""
    payload = {**BASE_RUN_PAYLOAD, "full_access": True}
    run_id = client.post("/api/v1/pipeline/run", json=payload).json()["run_id"]

    with client.stream(
        "GET",
        f"/api/v1/pipeline/{run_id}/events",
        params={"poll_interval": 0.1, "max_duration": 5.0},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        raw = ""
        for chunk in resp.iter_text():
            raw += chunk
            if "event: end" in raw:
                break

    events = _parse_sse(raw)
    event_types = [e.get("event") for e in events]
    assert "snapshot" in event_types
    assert "end" in event_types

    snapshot = next(e for e in events if e["event"] == "snapshot")
    snap_data = json.loads(snapshot["data"])
    assert snap_data["run_id"] == run_id
    assert snap_data["completed"] is True
    assert snap_data["interrupted"] is False

    end = next(e for e in events if e["event"] == "end")
    assert json.loads(end["data"])["reason"] == "completed"


def test_sse_emits_snapshot_for_paused_run(
    client: TestClient, patch_all_nodes: None
) -> None:
    """默认 gate 下 run 在 discover_classes 暂停，SSE 应推送 interrupted=true 的 snapshot。"""
    run_id = client.post("/api/v1/pipeline/run", json=BASE_RUN_PAYLOAD).json()["run_id"]

    with client.stream(
        "GET",
        f"/api/v1/pipeline/{run_id}/events",
        params={"poll_interval": 0.1, "max_duration": 1.0},
    ) as resp:
        assert resp.status_code == 200
        raw = ""
        for chunk in resp.iter_text():
            raw += chunk
            if "event: snapshot" in raw:
                break
        resp.close()

    events = _parse_sse(raw)
    snapshots = [e for e in events if e["event"] == "snapshot"]
    assert snapshots, "至少应推送一次 snapshot"
    snap = json.loads(snapshots[0]["data"])
    assert snap["run_id"] == run_id
    assert snap["interrupted"] is True
    assert snap["completed"] is False
    assert snap["step_results"]["healthcheck"]["status"] == "ok"
    assert "discover_classes" not in snap["step_results"]


def test_sse_deduplicates_identical_signatures(
    client: TestClient, patch_all_nodes: None
) -> None:
    """状态未变化时不应重复发 snapshot；超时后应以 end 收尾。"""
    run_id = client.post("/api/v1/pipeline/run", json=BASE_RUN_PAYLOAD).json()["run_id"]

    with client.stream(
        "GET",
        f"/api/v1/pipeline/{run_id}/events",
        params={"poll_interval": 0.1, "max_duration": 0.5},
    ) as resp:
        raw = "".join(resp.iter_text())

    events = _parse_sse(raw)
    snapshots = [e for e in events if e["event"] == "snapshot"]
    ends = [e for e in events if e["event"] == "end"]

    assert len(snapshots) == 1, f"重复状态应去重，但得到 {len(snapshots)} 条"
    assert ends and json.loads(ends[0]["data"])["reason"] == "timeout"
