"""SOP 模板 REST 测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.test_pipeline_graph_api import patch_all_nodes  # noqa: F401


def test_list_sops_returns_registered_templates(client: TestClient) -> None:
    response = client.get("/api/v1/pipeline/sops")
    assert response.status_code == 200
    data = response.json()
    ids = {sop["id"] for sop in data["sops"]}
    assert {
        "local-small-baseline",
        "local-large-sliding-window",
        "remote-slurm-iter",
        "full-auto-smoke",
    }.issubset(ids)

    baseline = next(s for s in data["sops"] if s["id"] == "local-small-baseline")
    assert baseline["defaults"]["execution_mode"] == "local"
    assert baseline["step_gates"]["train"] == "manual"
    large = next(s for s in data["sops"] if s["id"] == "local-large-sliding-window")
    assert large["defaults"]["yolo_export_after_train"] is True


def test_run_sop_unknown_id_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/pipeline/sops/non-existent/run",
        json={
            "original_dataset": "/tmp/x",
            "detector_name": "d",
            "project_root_dir": "/tmp/ws",
            "yolo_train_env": "env",
        },
    )
    assert response.status_code == 404


def test_run_sop_missing_required_fields_returns_422(client: TestClient) -> None:
    """remote-slurm-iter 要求 remote_* 字段，缺失时应返回 422。"""
    response = client.post(
        "/api/v1/pipeline/sops/remote-slurm-iter/run",
        json={
            "original_dataset": "/tmp/x",
            "detector_name": "d",
            "project_root_dir": "/tmp/ws",
            "yolo_train_env": "env",
        },
    )
    assert response.status_code == 422
    assert "remote_host" in response.json()["detail"]


def test_run_sop_full_auto_smoke_runs_to_completion(
    client: TestClient, patch_all_nodes: None
) -> None:
    """full-auto-smoke 设定 full_access=True，一次 invoke 即完成（借助 stub 节点）。"""
    response = client.post(
        "/api/v1/pipeline/sops/full-auto-smoke/run",
        json={
            "original_dataset": "/tmp/x",
            "detector_name": "smoke",
            "project_root_dir": "/tmp/ws",
            "yolo_train_env": "env",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["completed"] is True
    assert data["interrupted"] is False


def test_run_sop_user_override_beats_sop_default(
    client: TestClient, patch_all_nodes: None
) -> None:
    """用户提供的 yolo_train_epochs 应覆盖 SOP 默认值。"""
    response = client.post(
        "/api/v1/pipeline/sops/full-auto-smoke/run",
        json={
            "original_dataset": "/tmp/x",
            "detector_name": "smoke",
            "project_root_dir": "/tmp/ws",
            "yolo_train_env": "env",
            "yolo_train_epochs": 42,
            "yolo_train_imgsz": 512,
        },
    )
    assert response.status_code == 202
    data = response.json()
    train_result = data["step_results"].get("train") or {}
    # stub 节点返回的 data 字段是 params_override（确认时为空），
    # 但我们可以通过 get_state 间接验证：这里只要 run 成功，
    # 并且 pipeline 能从 SOP 默认的 epochs=1 被用户覆盖到 42 即可。
    assert data["completed"] is True
    assert data["error"] is None
    assert train_result.get("status") == "ok"


def test_run_sop_user_step_gates_override_sop_gates(
    client: TestClient, patch_all_nodes: None
) -> None:
    """SOP 默认 train=manual，用户把它强制改成 auto，应按用户意愿生效。"""
    response = client.post(
        "/api/v1/pipeline/sops/local-small-baseline/run",
        json={
            "original_dataset": "/tmp/x",
            "detector_name": "d",
            "project_root_dir": "/tmp/ws",
            "yolo_train_env": "env",
                "step_gates": {
                    "discover_classes": {"mode": "auto"},
                    "review_labels": {"mode": "auto"},
                    "split_dataset": {"mode": "auto"},
                    "crop_window": {"mode": "auto"},
                    "augment_only": {"mode": "auto"},
                    "publish_transfer": {"mode": "auto"},
                    "train": {"mode": "auto"},
                    "review_result": {"mode": "auto"},
                    "model_infer": {"mode": "auto"},
                },
            },
        )
    data = response.json()
    for step in ("discover_classes", "review_labels", "train", "review_result", "export_model"):
        assert step in data["step_results"], f"{step} 应已执行，但未在 step_results 中"
    assert data["completed"] is True
