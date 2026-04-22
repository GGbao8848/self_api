"""Run Ultralytics YOLO CLI training inside a conda environment."""

from __future__ import annotations

import shlex
import subprocess
import re
from pathlib import Path

from app.schemas.preprocess import YoloTrainRequest, YoloTrainResponse
from app.services.yolo_env import resolve_env_python
from app.services.yolo_model_resolver import resolve_local_yolo_model
from app.services.task_manager import report_progress


_EPOCH_PATTERN = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")


def run_yolo_train(request: YoloTrainRequest) -> YoloTrainResponse:
    yaml_p = Path(request.yaml_path).expanduser().resolve()
    if not yaml_p.is_file():
        raise ValueError(f"yaml_path is not a file: {yaml_p}")

    root = Path(request.project_root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project_root_dir is not a directory: {root}")

    resolved_model = resolve_local_yolo_model(request.model, cwd=root)

    env_python = resolve_env_python(request.yolo_train_env)
    if env_python:
        cmd: list[str] = [
            env_python,
            "-m",
            "ultralytics",
            "train",
            f"model={resolved_model}",
            f"data={str(yaml_p)}",
            f"epochs={request.epochs}",
            f"imgsz={request.imgsz}",
            f"project={request.project}",
            f"name={request.name}",
        ]
    else:
        cmd = [
            "conda",
            "run",
            "-n",
            request.yolo_train_env,
            "--no-capture-output",
            "yolo",
            "train",
            f"model={resolved_model}",
            f"data={str(yaml_p)}",
            f"epochs={request.epochs}",
            f"imgsz={request.imgsz}",
            f"project={request.project}",
            f"name={request.name}",
        ]
    if request.batch is not None:
        cmd.append(f"batch={request.batch}")

    report_progress(
        percent=0,
        current=0,
        total=request.epochs,
        unit="epoch",
        stage="train",
        message="starting yolo training",
        indeterminate=False,
    )

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        stdout_lines.append(raw_line)
        match = _EPOCH_PATTERN.search(raw_line)
        if not match:
            continue
        current_epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        if total_epochs <= 0:
            continue
        report_progress(
            current=current_epoch,
            total=total_epochs,
            unit="epoch",
            stage="train",
            message=f"training epoch {current_epoch}/{total_epochs}",
            indeterminate=False,
        )
    stderr_text = proc.stderr.read() if proc.stderr is not None else ""
    return_code = proc.wait()
    stdout_text = "".join(stdout_lines)

    command_display = shlex.join(cmd)

    return YoloTrainResponse(
        status="ok" if return_code == 0 else "failed",
        command=command_display,
        cwd=str(root),
        project=request.project,
        name=request.name,
        exit_code=return_code,
        stdout=stdout_text,
        stderr=stderr_text,
    )
