from __future__ import annotations
from subprocess import CompletedProcess

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app


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


def test_yolo_train_async_uses_serial_queue(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.v1.endpoints.preprocess import yolo_train_async
    from app.schemas.preprocess import YoloTrainAsyncRequest

    detector_dir = case_dir / "nzxj_louyou"
    yaml_path = detector_dir / "datasets" / "nzxj_louyou_20260417_1430" / "nzxj_louyou_20260417_1430.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("names: []\n", encoding="utf-8")

    cwd = case_dir / "work"
    cwd.mkdir()
    project = detector_dir / "runs" / "detect"

    captured: dict[str, object] = {}

    def fake_submit_task(
        task_type: str,
        runner: object,
        *,
        callback_url: str | None = None,
        callback_timeout_seconds: float = 10.0,
        queue_key: str | None = None,
    ) -> str:
        captured["task_type"] = task_type
        captured["callback_url"] = callback_url
        captured["callback_timeout_seconds"] = callback_timeout_seconds
        captured["queue_key"] = queue_key
        captured["runner"] = runner
        return "queued-task-1"

    monkeypatch.setattr("app.api.v1.endpoints.preprocess.submit_task", fake_submit_task)

    class _FakeRequest:
        app = app

        @staticmethod
        def url_for(route_name: str, **path_params: str) -> str:
            return f"http://testserver{app.url_path_for(route_name, **path_params)}"

    submit_resp = yolo_train_async(
        YoloTrainAsyncRequest(
            yaml_path=str(yaml_path.resolve()),
            project_root_dir=str(cwd.resolve()),
            project=str(project.resolve()),
            name="nzxj_louyou_20260417_1430",
            yolo_train_env="yolo_pose",
        ),
        _FakeRequest(),
    )

    assert submit_resp.task_id == "queued-task-1"
    assert captured["task_type"] == "yolo_train"
    assert captured["queue_key"] == "local_yolo_train"


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
