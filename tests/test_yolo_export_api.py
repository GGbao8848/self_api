from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _prepare_best_pt_with_args(case_dir: Path) -> tuple[Path, Path, Path]:
    run_dir = case_dir / "detector" / "runs" / "detect" / "exp1"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_text("fake-pt", encoding="utf-8")
    args_yaml = run_dir / "args.yaml"
    args_yaml.write_text("imgsz: 1280\ndata: /data/demo_dataset.yaml\n", encoding="utf-8")
    work_dir = case_dir / "work"
    work_dir.mkdir(exist_ok=True)
    return best_pt, args_yaml, work_dir


def test_yolo_export_sync_uses_args_yaml_and_renames_output(
    client: TestClient, case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    best_pt, _, work_dir = _prepare_best_pt_with_args(case_dir)
    default_export = best_pt.with_suffix(".torchscript")
    default_export.write_text("ts", encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> CompletedProcess[str]:
        assert kwargs.get("cwd") == str(work_dir.resolve())
        assert "yolo" in cmd
        assert "export" in cmd
        assert f"model={best_pt.resolve()}" in cmd
        assert "format=torchscript" in cmd
        assert "half=True" in cmd
        assert "imgsz=1280" in cmd
        return CompletedProcess(cmd, 0, stdout="export ok\n", stderr="")

    monkeypatch.setattr("app.services.yolo_export.subprocess.run", fake_run)

    r = client.post(
        "/api/v1/preprocess/yolo-export",
        json={
            "best_pt_path": str(best_pt.resolve()),
            "project_root_dir": str(work_dir.resolve()),
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["imgsz"] == 1280
    assert data["dataset_yaml"] == "demo_dataset.yaml"
    assert data["export_file_path"].endswith("demo_dataset.torchscript")
    assert (best_pt.parent / "demo_dataset.torchscript").exists()
    assert not default_export.exists()


def test_yolo_export_missing_args_yaml_returns_400(client: TestClient, case_dir: Path) -> None:
    weights_dir = case_dir / "detector" / "runs" / "detect" / "exp1" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_text("fake-pt", encoding="utf-8")

    r = client.post(
        "/api/v1/preprocess/yolo-export",
        json={
            "best_pt_path": str(best_pt.resolve()),
            "project_root_dir": str(case_dir.resolve()),
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 400


def test_yolo_export_async_accepts(client: TestClient, case_dir: Path) -> None:
    best_pt, _, work_dir = _prepare_best_pt_with_args(case_dir)
    r = client.post(
        "/api/v1/preprocess/yolo-export/async",
        json={
            "best_pt_path": str(best_pt.resolve()),
            "project_root_dir": str(work_dir.resolve()),
            "yolo_train_env": "yolo_pose",
        },
    )
    assert r.status_code == 202
    assert r.json()["task_type"] == "yolo_export"


def test_yolo_export_async_uses_serial_queue(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.v1.endpoints.preprocess import yolo_export_async
    from app.schemas.preprocess import YoloExportAsyncRequest

    best_pt, _, work_dir = _prepare_best_pt_with_args(case_dir)
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
        return "queued-task-export-1"

    monkeypatch.setattr("app.api.v1.endpoints.preprocess.submit_task", fake_submit_task)

    class _FakeRequest:
        app = app

        @staticmethod
        def url_for(route_name: str, **path_params: str) -> str:
            return f"http://testserver{app.url_path_for(route_name, **path_params)}"

    submit_resp = yolo_export_async(
        YoloExportAsyncRequest(
            best_pt_path=str(best_pt.resolve()),
            project_root_dir=str(work_dir.resolve()),
            yolo_train_env="yolo_pose",
        ),
        _FakeRequest(),
    )
    assert submit_resp.task_id == "queued-task-export-1"
    assert captured["task_type"] == "yolo_export"
    assert captured["queue_key"] == "local_yolo_train"
