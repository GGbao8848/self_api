from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.schemas.preprocess import RemoteSbatchYoloTrainResponse


def test_remote_sbatch_yolo_train_sync_mocked(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(payload):
        return RemoteSbatchYoloTrainResponse(
            yaml_path="/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
            project_root_dir="/mnt/usrhome/sk/ndata/TVDS",
            target_host="172.31.1.9",
            target_port=22,
            project="/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect",
            name="nzxj_louyou_20260417_1430",
            command="sbatch --parsable ...",
            job_id="12345",
            stdout_path="/mnt/usrhome/sk/ndata/TVDS/logs/slurm-%j.out",
            stderr_path="/mnt/usrhome/sk/ndata/TVDS/logs/slurm-%j.err",
            stdout="12345",
            stderr="",
        )

    monkeypatch.setattr(
        "app.api.v1.endpoints.preprocess.run_remote_sbatch_yolo_train",
        fake_run,
    )

    r = client.post(
        "/api/v1/preprocess/remote-sbatch-yolo-train",
        json={
            "host": "172.31.1.9",
            "yaml_path": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
            "project_root_dir": "/mnt/usrhome/sk/ndata/TVDS",
            "project": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect",
            "name": "nzxj_louyou_20260417_1430",
            "yolo_train_env": "yolo_pose",
            "username": "sk",
            "private_key_path": "~/.ssh/id_ed25519",
        },
    )
    assert r.status_code == 200
    assert r.json()["job_id"] == "12345"


def test_remote_sbatch_yolo_train_async_accepts(client: TestClient) -> None:
    r = client.post(
        "/api/v1/preprocess/remote-sbatch-yolo-train/async",
        json={
            "host": "172.31.1.9",
            "yaml_path": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
            "project_root_dir": "/mnt/usrhome/sk/ndata/TVDS",
            "project": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect",
            "name": "nzxj_louyou_20260417_1430",
            "yolo_train_env": "yolo_pose",
            "username": "sk",
            "private_key_path": "~/.ssh/id_ed25519",
        },
    )
    assert r.status_code == 202
    assert r.json()["task_type"] == "remote_sbatch_yolo_train"


def test_remote_slurm_route_removed(client: TestClient) -> None:
    r = client.post("/api/v1/preprocess/remote-slurm-yolo-train", json={})
    assert r.status_code == 404


def test_remote_sbatch_yolo_train_rejects_name_not_matching_yaml_stem(
    client: TestClient,
) -> None:
    r = client.post(
        "/api/v1/preprocess/remote-sbatch-yolo-train",
        json={
            "host": "172.31.1.9",
            "yaml_path": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
            "project_root_dir": "/mnt/usrhome/sk/ndata/TVDS",
            "project": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/detect",
            "name": "wrong_name",
            "yolo_train_env": "yolo_pose",
            "username": "sk",
            "private_key_path": "~/.ssh/id_ed25519",
        },
    )
    assert r.status_code == 422


def test_remote_sbatch_yolo_train_rejects_non_training_bucket(
    client: TestClient,
) -> None:
    r = client.post(
        "/api/v1/preprocess/remote-sbatch-yolo-train",
        json={
            "host": "172.31.1.9",
            "yaml_path": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
            "project_root_dir": "/mnt/usrhome/sk/ndata/TVDS",
            "project": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/runs/predict",
            "name": "nzxj_louyou_20260417_1430",
            "yolo_train_env": "yolo_pose",
            "username": "sk",
            "private_key_path": "~/.ssh/id_ed25519",
        },
    )
    assert r.status_code == 422


def test_remote_sbatch_yolo_train_rejects_project_prefix_not_matching_yaml(
    client: TestClient,
) -> None:
    r = client.post(
        "/api/v1/preprocess/remote-sbatch-yolo-train",
        json={
            "host": "172.31.1.9",
            "yaml_path": "/mnt/usrhome/sk/ndata/TVDS/nzxj_louyou/datasets/nzxj_louyou_20260417_1430/nzxj_louyou_20260417_1430.yaml",
            "project_root_dir": "/mnt/usrhome/sk/ndata/TVDS",
            "project": "/mnt/usrhome/sk/ndata/TVDS/other_detector/runs/detect",
            "name": "nzxj_louyou_20260417_1430",
            "yolo_train_env": "yolo_pose",
            "username": "sk",
            "private_key_path": "~/.ssh/id_ed25519",
        },
    )
    assert r.status_code == 422
