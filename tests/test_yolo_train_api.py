from __future__ import annotations

from subprocess import CompletedProcess

import pytest
from fastapi.testclient import TestClient
from pathlib import Path


def test_yolo_train_sync_mocked(
    client: TestClient, case_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector_dir = case_dir / "nzxj_louyou"
    yaml_path = detector_dir / "datasets" / "nzxj_louyou_20260417_1430" / "nzxj_louyou_20260417_1430.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")
    cwd = case_dir / "work"
    cwd.mkdir()
    project = detector_dir / "runs" / "detect"

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
            "project": str(project.resolve()),
            "name": "nzxj_louyou_20260417_1430",
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
    assert data["name"] == "nzxj_louyou_20260417_1430"
    assert data["project"].endswith("runs/detect")
    assert "yolo train" in data["command"] or "yolo" in data["command"]


def test_yolo_train_resolves_model_from_parent_dirs(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.schemas.preprocess import YoloTrainRequest
    from app.services.yolo_train import run_yolo_train

    yaml_path = case_dir / "demo" / "datasets" / "demo" / "demo.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")

    cwd = case_dir / "level1" / "level2" / "level3" / "work"
    cwd.mkdir(parents=True)
    model_path = case_dir / "level1" / "yolo11s.pt"
    model_path.write_text("weights", encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> CompletedProcess[str]:
        assert f"model={model_path.resolve()}" in cmd
        return CompletedProcess(cmd, 0, stdout="train ok\n", stderr="")

    monkeypatch.setattr("app.services.yolo_train.subprocess.run", fake_run)

    response = run_yolo_train(
        YoloTrainRequest(
            yaml_path=str(yaml_path.resolve()),
            project_root_dir=str(cwd.resolve()),
            project=str((case_dir / "demo" / "runs" / "detect").resolve()),
            name="demo",
            yolo_train_env="yolo_pose",
            model="yolo11s.pt",
        )
    )
    assert response.status == "ok"


def test_yolo_train_keeps_model_name_when_not_found(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.schemas.preprocess import YoloTrainRequest
    from app.services.yolo_train import run_yolo_train

    yaml_path = case_dir / "demo" / "datasets" / "demo" / "demo.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")

    cwd = case_dir / "work"
    cwd.mkdir()

    def fake_run(cmd: list[str], **kwargs: object) -> CompletedProcess[str]:
        assert "model=yolo11s.pt" in cmd
        return CompletedProcess(cmd, 0, stdout="train ok\n", stderr="")

    monkeypatch.setattr("app.services.yolo_train.subprocess.run", fake_run)

    response = run_yolo_train(
        YoloTrainRequest(
            yaml_path=str(yaml_path.resolve()),
            project_root_dir=str(cwd.resolve()),
            project=str((case_dir / "demo" / "runs" / "detect").resolve()),
            name="demo",
            yolo_train_env="yolo_pose",
            model="yolo11s.pt",
        )
    )
    assert response.status == "ok"


def test_yolo_train_missing_yaml(client: TestClient, case_dir) -> None:
    detector_dir = case_dir / "nzxj_louyou"
    missing = detector_dir / "datasets" / "nzxj_louyou_20260417_1430" / "nzxj_louyou_20260417_1430.yaml"
    cwd = case_dir
    r = client.post(
        "/api/v1/preprocess/yolo-train",
        json={
            "yaml_path": str(missing),
            "project_root_dir": str(cwd.resolve()),
            "project": str((detector_dir / "runs" / "detect").resolve()),
            "name": "nzxj_louyou_20260417_1430",
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 400


def test_yolo_train_async_accepts(client: TestClient, case_dir) -> None:
    detector_dir = case_dir / "nzxj_louyou"
    yaml_path = detector_dir / "datasets" / "nzxj_louyou_20260417_1430" / "nzxj_louyou_20260417_1430.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")
    cwd = case_dir
    r = client.post(
        "/api/v1/preprocess/yolo-train/async",
        json={
            "yaml_path": str(yaml_path.resolve()),
            "project_root_dir": str(cwd.resolve()),
            "project": str((detector_dir / "runs" / "detect").resolve()),
            "name": "nzxj_louyou_20260417_1430",
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 202
    assert r.json()["task_type"] == "yolo_train"


def test_yolo_train_requires_explicit_project_and_name(
    client: TestClient, case_dir: Path
) -> None:
    yaml_path = case_dir / "demo" / "datasets" / "demo" / "demo.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")
    cwd = case_dir
    r = client.post(
        "/api/v1/preprocess/yolo-train",
        json={
            "yaml_path": str(yaml_path.resolve()),
            "project_root_dir": str(cwd.resolve()),
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 422


def test_yolo_train_rejects_name_not_matching_yaml_stem(
    client: TestClient, case_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector_dir = case_dir / "nzxj_louyou"
    yaml_path = detector_dir / "datasets" / "nzxj_louyou_20260417_1430" / "nzxj_louyou_20260417_1430.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")
    cwd = case_dir / "work"
    cwd.mkdir()

    def fake_run(cmd: list[str], **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(cmd, 0, stdout="train ok\n", stderr="")

    monkeypatch.setattr("app.services.yolo_train.subprocess.run", fake_run)

    r = client.post(
        "/api/v1/preprocess/yolo-train",
        json={
            "yaml_path": str(yaml_path.resolve()),
            "project_root_dir": str(cwd.resolve()),
            "project": str((detector_dir / "runs" / "detect").resolve()),
            "name": "wrong_name",
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 422


def test_yolo_train_rejects_non_training_bucket(client: TestClient, case_dir) -> None:
    detector_dir = case_dir / "nzxj_louyou"
    yaml_path = detector_dir / "datasets" / "nzxj_louyou_20260417_1430" / "nzxj_louyou_20260417_1430.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")
    cwd = case_dir

    r = client.post(
        "/api/v1/preprocess/yolo-train",
        json={
            "yaml_path": str(yaml_path.resolve()),
            "project_root_dir": str(cwd.resolve()),
            "project": str((detector_dir / "runs" / "val").resolve()),
            "name": "nzxj_louyou_20260417_1430",
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 422


def test_yolo_train_rejects_project_prefix_not_matching_yaml(client: TestClient, case_dir) -> None:
    yaml_path = (
        case_dir / "nzxj_louyou" / "datasets" / "nzxj_louyou_20260417_1430" / "nzxj_louyou_20260417_1430.yaml"
    )
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")

    r = client.post(
        "/api/v1/preprocess/yolo-train",
        json={
            "yaml_path": str(yaml_path.resolve()),
            "project_root_dir": str(case_dir.resolve()),
            "project": str((case_dir / "other_detector" / "runs" / "detect").resolve()),
            "name": "nzxj_louyou_20260417_1430",
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 422
