from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from fastapi.testclient import TestClient

def test_yolo_infer_sync_mocked(client: TestClient, case_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = case_dir / "images"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.jpg").write_bytes(b"fake")
    model = case_dir / "m.torchscript"
    model.write_bytes(b"fake")
    out_project = case_dir / "det" / "runs" / "infer"
    out_dir = out_project / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.yaml").write_text("x: 1\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "total_images": 1,
                "detected_images": 1,
                "no_detect_images": 0,
                "result_images": 1,
                "labels_written": 1,
            }
        ),
        encoding="utf-8",
    )

    def fake_run(cmd: list[str], **kwargs: object) -> CompletedProcess[str]:
        assert "conda" in cmd[0]
        assert "run" in cmd
        assert "python" in cmd
        assert "--model-path" in cmd
        return CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr("app.services.yolo_infer.subprocess.run", fake_run)

    r = client.post(
        "/api/v1/preprocess/yolo-infer",
        json={
            "yolo_train_env": "yolo_pose",
            "model_path": str(model.resolve()),
            "source_path": str(src.resolve()),
            "project": str(out_project.resolve()),
            "name": "demo",
            "imgsz": 800,
            "conf": 0.4,
            "iou": 0.1,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["total_images"] == 1
    assert data["detected_images"] == 1
    assert data["output_dir"].endswith("/runs/infer/demo")


def test_yolo_infer_async_accepts(client: TestClient, case_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = case_dir / "images"
    src.mkdir(parents=True, exist_ok=True)
    (src / "a.jpg").write_bytes(b"fake")
    model = case_dir / "m.torchscript"
    model.write_bytes(b"fake")
    out_project = case_dir / "det" / "runs" / "infer"
    out_dir = out_project / "demo_async"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.yaml").write_text("x: 1\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "total_images": 1,
                "detected_images": 1,
                "no_detect_images": 0,
                "result_images": 1,
                "labels_written": 1,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "app.services.yolo_infer.subprocess.run",
        lambda cmd, **kwargs: CompletedProcess(cmd, 0, stdout="ok\n", stderr=""),
    )

    r = client.post(
        "/api/v1/preprocess/yolo-infer/async",
        json={
            "yolo_train_env": "yolo_pose",
            "model_path": str(model.resolve()),
            "source_path": str(src.resolve()),
            "project": str(out_project.resolve()),
            "name": "demo_async",
        },
    )
    assert r.status_code == 202
    assert r.json()["task_type"] == "yolo_infer"
