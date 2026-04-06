from __future__ import annotations

from subprocess import CompletedProcess

import pytest
from fastapi.testclient import TestClient

from app.services.yolo_train import project_and_name_from_yaml


def test_project_and_name_from_yaml() -> None:
    project, name = project_and_name_from_yaml(
        "/Users/me/self_api/TVDS/dog_cat_pig/dataset/dataset.yaml"
    )
    assert project.endswith("dog_cat_pig/runs/train")
    assert "/dataset/" not in project
    assert name == "dataset"


def test_project_and_name_from_yaml_requires_dataset_segment() -> None:
    with pytest.raises(ValueError, match="/dataset/"):
        project_and_name_from_yaml("/tmp/foo/bar.yaml")


def test_yolo_train_sync_mocked(
    client: TestClient, case_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = case_dir / "dog_cat_pig" / "dataset" / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")
    cwd = case_dir / "work"
    cwd.mkdir()

    def fake_run(cmd: list[str], **kwargs: object) -> CompletedProcess[str]:
        assert kwargs.get("cwd") == str(cwd.resolve())
        assert "conda" in cmd[0] or cmd[0] == "conda"
        return CompletedProcess(cmd, 0, stdout="train ok\n", stderr="")

    monkeypatch.setattr("app.services.yolo_train.subprocess.run", fake_run)

    r = client.post(
        "/api/v1/preprocess/yolo-train",
        json={
            "yaml_path": str(yaml_path.resolve()),
            "project_root_dir": str(cwd.resolve()),
            "yolo_train_env": "yolo_pose",
            "model": "yolo11s.pt",
            "epochs": 100,
            "imgsz": 640,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["exit_code"] == 0
    assert data["name"] == "dataset"
    assert "runs/train" in data["project"]
    assert "yolo train" in data["command"] or "yolo" in data["command"]


def test_yolo_train_missing_yaml(client: TestClient, case_dir) -> None:
    missing = case_dir / "dog_cat_pig" / "dataset" / "missing.yaml"
    cwd = case_dir
    r = client.post(
        "/api/v1/preprocess/yolo-train",
        json={
            "yaml_path": str(missing),
            "project_root_dir": str(cwd.resolve()),
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 400


def test_yolo_train_async_accepts(client: TestClient, case_dir) -> None:
    yaml_path = case_dir / "dog_cat_pig" / "dataset" / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")
    cwd = case_dir
    r = client.post(
        "/api/v1/preprocess/yolo-train/async",
        json={
            "yaml_path": str(yaml_path.resolve()),
            "project_root_dir": str(cwd.resolve()),
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 202
    assert r.json()["task_type"] == "yolo_train"
