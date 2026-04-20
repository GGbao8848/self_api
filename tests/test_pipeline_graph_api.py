"""LangGraph pipeline REST 端点测试。

测试策略：
- healthcheck 失败路径：使用不可达的 self_api_url，验证 run 立即终止
- gate/interrupt/resume：monkeypatch 健康检查 + 耗时节点，聚焦在 gate 机制
- full_access：跳过所有 manual gate，一次 invoke 即完成
- abort：interrupt 暂停时发送 abort，流程终止
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.graph import nodes as graph_nodes
from app.graph.state import DEFAULT_GATES, PipelineState


BASE_RUN_PAYLOAD = {
    "original_dataset": "/tmp/fake_dataset",
    "detector_name": "pytest_detector",
    "project_root_dir": "/tmp/fake_workspace",
    "execution_mode": "local",
    "yolo_train_env": "pytest_env",
    "yolo_train_epochs": 1,
    "yolo_train_imgsz": 320,
    "self_api_url": "http://127.0.0.1:1",
}


def _make_stub(step: str) -> Any:
    """返回一个使用 _maybe_interrupt + _set_step_result 的通用节点替身。"""

    def _stub(state: PipelineState) -> dict:
        review = {"step": step, "hint": f"stub review for {step}"}
        user = graph_nodes._maybe_interrupt(state, step, review)
        if user and user.get("decision") == "abort":
            return {
                **graph_nodes._set_step_result(
                    state,
                    step,
                    {"status": "skipped", "summary": "aborted", "data": {}},
                ),
                "error": f"aborted at {step}",
                "completed": True,
            }
        override = (user or {}).get("params_override") or {}
        update: dict = graph_nodes._set_step_result(
            state,
            step,
            {"status": "ok", "summary": f"{step} stub done", "data": dict(override)},
        )
        if step == "review_result":
            update["completed"] = True
        return update

    return _stub


@pytest.fixture
def patch_all_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """把所有节点替换为通用 stub，确保 interrupt / gate 机制可以被独立测试。"""
    step_to_attr = {
        "healthcheck": "node_healthcheck",
        "discover_classes": "node_discover_classes",
        "xml_to_yolo": "node_xml_to_yolo",
        "review_labels": "node_review_labels",
        "split_dataset": "node_split_dataset",
        "crop_augment": "node_crop_augment",
        "publish_transfer": "node_publish_transfer",
        "train": "node_train",
        "poll_train": "node_poll_train",
        "review_result": "node_review_result",
    }
    for step, attr in step_to_attr.items():
        monkeypatch.setattr(graph_nodes, attr, _make_stub(step))

    import app.graph.pipeline as pipeline_module
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    g = StateGraph(PipelineState)
    for step, attr in step_to_attr.items():
        g.add_node(step, getattr(graph_nodes, attr))
    g.set_entry_point("healthcheck")
    edges = [
        ("healthcheck", "discover_classes"),
        ("discover_classes", "xml_to_yolo"),
        ("xml_to_yolo", "review_labels"),
        ("review_labels", "split_dataset"),
        ("split_dataset", "crop_augment"),
        ("crop_augment", "publish_transfer"),
        ("publish_transfer", "train"),
        ("train", "poll_train"),
        ("poll_train", "review_result"),
    ]
    for src, dst in edges:
        g.add_conditional_edges(
            src,
            pipeline_module._should_abort,
            {"end": END, "continue": dst},
        )
    g.add_edge("review_result", END)

    new_compiled = g.compile(checkpointer=MemorySaver())

    import app.api.v1.endpoints.pipeline as pipeline_endpoint
    monkeypatch.setattr(pipeline_endpoint, "compiled_graph", new_compiled)


def test_pipeline_run_healthcheck_failure_terminates(client: TestClient) -> None:
    """真实 healthcheck 对不可达地址失败 → 流程立即 completed 且 error 写入。"""
    response = client.post("/api/v1/pipeline/run", json=BASE_RUN_PAYLOAD)
    assert response.status_code == 202
    data = response.json()

    assert data["run_id"]
    assert data["completed"] is True
    assert data["interrupted"] is False
    assert data["error"] and "healthcheck" in data["error"]
    assert data["step_results"]["healthcheck"]["status"] == "failed"


def test_pipeline_default_gates_pause_at_discover_classes(
    client: TestClient, patch_all_nodes: None
) -> None:
    """默认 gate 下，healthcheck 通过 → discover_classes 被 interrupt 暂停。"""
    response = client.post("/api/v1/pipeline/run", json=BASE_RUN_PAYLOAD)
    assert response.status_code == 202
    data = response.json()

    assert data["interrupted"] is True
    assert data["completed"] is False
    assert data["error"] is None
    assert data["step_results"]["healthcheck"]["status"] == "ok"
    assert "discover_classes" not in data["step_results"]


def test_pipeline_full_access_runs_to_completion(
    client: TestClient, patch_all_nodes: None
) -> None:
    """full_access=True 跳过所有 manual gate，一次 invoke 即走到 END。"""
    payload = {**BASE_RUN_PAYLOAD, "full_access": True}
    response = client.post("/api/v1/pipeline/run", json=payload)
    assert response.status_code == 202
    data = response.json()

    assert data["completed"] is True
    assert data["interrupted"] is False
    assert data["error"] is None
    for step in DEFAULT_GATES:
        assert step in data["step_results"]
        assert data["step_results"][step]["status"] == "ok"


def test_pipeline_confirm_advances_to_next_gate(
    client: TestClient, patch_all_nodes: None
) -> None:
    """在 discover_classes 暂停 → confirm（带 params_override）→ 推进到 review_labels。"""
    run_response = client.post("/api/v1/pipeline/run", json=BASE_RUN_PAYLOAD)
    run_id = run_response.json()["run_id"]
    assert run_response.json()["interrupted"] is True

    confirm = client.post(
        f"/api/v1/pipeline/{run_id}/confirm",
        json={
            "decision": "confirm",
            "params_override": {"class_name_map": {"louyou1": "louyou"}},
        },
    )
    assert confirm.status_code == 200
    data = confirm.json()

    assert data["interrupted"] is True
    assert data["completed"] is False
    assert data["step_results"]["discover_classes"]["status"] == "ok"
    assert data["step_results"]["discover_classes"]["data"] == {
        "class_name_map": {"louyou1": "louyou"},
    }
    assert data["step_results"]["xml_to_yolo"]["status"] == "ok"
    assert "review_labels" not in data["step_results"]


def test_pipeline_abort_at_interrupt_stops_pipeline(
    client: TestClient, patch_all_nodes: None
) -> None:
    """interrupt 暂停时发送 abort → pipeline 标记 completed 且带 error。"""
    run_id = client.post("/api/v1/pipeline/run", json=BASE_RUN_PAYLOAD).json()["run_id"]

    abort = client.post(f"/api/v1/pipeline/{run_id}/abort")
    assert abort.status_code == 200
    data = abort.json()

    assert data["completed"] is True
    assert data["error"] and "discover_classes" in data["error"]
    assert data["step_results"]["discover_classes"]["status"] == "skipped"


def test_pipeline_status_404_for_unknown_run(client: TestClient) -> None:
    response = client.get("/api/v1/pipeline/unknown-run-id")
    assert response.status_code == 404


def test_pipeline_confirm_rejects_when_not_paused(
    client: TestClient, patch_all_nodes: None
) -> None:
    """full_access 跑完后再 confirm → 409（pipeline 不在等待状态）。"""
    payload = {**BASE_RUN_PAYLOAD, "full_access": True}
    run_id = client.post("/api/v1/pipeline/run", json=payload).json()["run_id"]

    confirm = client.post(
        f"/api/v1/pipeline/{run_id}/confirm",
        json={"decision": "confirm", "params_override": {}},
    )
    assert confirm.status_code == 409


def test_pipeline_step_gate_override_to_auto(
    client: TestClient, patch_all_nodes: None
) -> None:
    """step_gates.discover_classes.mode=auto → 不再暂停在 discover_classes。"""
    payload = {
        **BASE_RUN_PAYLOAD,
        "step_gates": {
            "discover_classes": {"mode": "auto"},
            "review_labels": {"mode": "auto"},
        },
    }
    response = client.post("/api/v1/pipeline/run", json=payload)
    data = response.json()

    assert data["interrupted"] is True
    assert data["step_results"]["discover_classes"]["status"] == "ok"
    assert data["step_results"]["xml_to_yolo"]["status"] == "ok"
    assert data["step_results"]["review_labels"]["status"] == "ok"
    assert data["step_results"]["split_dataset"]["status"] == "ok"
    assert data["step_results"]["crop_augment"]["status"] == "ok"
    assert "publish_transfer" not in data["step_results"]
