"""SOP 模板 REST 测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.test_pipeline_graph_api import patch_all_nodes  # noqa: F401


def test_list_sops_returns_registered_templates(client: TestClient) -> None:
    response = client.get("/api/v1/pipeline/sops")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sops"]) == 1
    sop = data["sops"][0]
    assert sop["id"] == "local-large-sliding-window"
    assert sop["defaults"]["execution_mode"] == "local"
    assert sop["defaults"]["yolo_export_after_train"] is True
    assert sop["review_profile_default"] == "balanced"
    assert set(sop["review_profiles"].keys()) == {"strict", "balanced", "auto"}


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


def test_run_sop_user_override_beats_sop_default(
    client: TestClient, patch_all_nodes: None
) -> None:
    """用户提供的 yolo_train_epochs 应覆盖 SOP 默认值。"""
    response = client.post(
        "/api/v1/pipeline/sops/local-large-sliding-window/run",
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
    assert data["interrupted"] is True
    assert data["error"] is None
    assert data["pending_review"]["step"] == "discover_classes"


def test_run_sop_user_step_gates_override_sop_gates(
    client: TestClient, patch_all_nodes: None
) -> None:
    """SOP 默认 train=manual，用户把它强制改成 auto，应按用户意愿生效。"""
    response = client.post(
        "/api/v1/pipeline/sops/local-large-sliding-window/run",
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
    for step in ("discover_classes", "review_labels", "split_dataset", "crop_window", "augment_only", "train", "review_result"):
        assert step in data["step_results"], f"{step} 应已执行，但未在 step_results 中"
    assert data["completed"] is True


def test_run_sop_review_profile_auto_runs_without_interrupt(
    client: TestClient, patch_all_nodes: None
) -> None:
    response = client.post(
        "/api/v1/pipeline/sops/local-large-sliding-window/run",
        json={
            "original_dataset": "/tmp/x",
            "detector_name": "auto",
            "project_root_dir": "/tmp/ws",
            "yolo_train_env": "env",
            "review_profile": "auto",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["completed"] is True
    assert data["interrupted"] is False


def test_run_sop_review_profile_strict_pauses_at_healthcheck(
    client: TestClient, patch_all_nodes: None
) -> None:
    response = client.post(
        "/api/v1/pipeline/sops/local-large-sliding-window/run",
        json={
            "original_dataset": "/tmp/x",
            "detector_name": "strict",
            "project_root_dir": "/tmp/ws",
            "yolo_train_env": "env",
            "review_profile": "strict",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["interrupted"] is True
    assert data["pending_review"]["step"] == "healthcheck"
