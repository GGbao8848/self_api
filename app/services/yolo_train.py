"""Run Ultralytics YOLO CLI training inside a conda environment."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from app.schemas.preprocess import YoloTrainRequest, YoloTrainResponse


def run_yolo_train(request: YoloTrainRequest) -> YoloTrainResponse:
    yaml_p = Path(request.yaml_path).expanduser().resolve()
    if not yaml_p.is_file():
        raise ValueError(f"yaml_path is not a file: {yaml_p}")

    root = Path(request.project_root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project_root_dir is not a directory: {root}")

    cmd: list[str] = [
        "conda",
        "run",
        "-n",
        request.yolo_train_env,
        "--no-capture-output",
        "yolo",
        "train",
        f"model={request.model}",
        f"data={str(yaml_p)}",
        f"epochs={request.epochs}",
        f"imgsz={request.imgsz}",
        f"project={request.project}",
        f"name={request.name}",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )

    command_display = shlex.join(cmd)

    return YoloTrainResponse(
        status="ok" if proc.returncode == 0 else "failed",
        command=command_display,
        cwd=str(root),
        project=request.project,
        name=request.name,
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
