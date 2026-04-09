"""Run Slurm-oriented YOLO CLI training inside a conda environment."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from app.schemas.preprocess import YoloTrainRequest, YoloTrainResponse


def project_and_name_from_yaml(yaml_path: str) -> tuple[str, str]:
    """Derive `project` and `name` from yaml path."""
    p = Path(yaml_path).expanduser()
    normalized = str(p).replace("\\", "/")
    marker = "/dataset/"
    if marker not in normalized:
        raise ValueError(
            f"yaml_path must contain {marker!r} segment (got {yaml_path!r})"
        )
    prefix = normalized.split(marker, 1)[0]
    project = str(Path(prefix) / "runs" / "train")
    name = Path(normalized).stem
    return project, name


def run_slurm_yolo_train(request: YoloTrainRequest) -> YoloTrainResponse:
    yaml_p = Path(request.yaml_path).expanduser().resolve()
    if not yaml_p.is_file():
        raise ValueError(f"yaml_path is not a file: {yaml_p}")

    root = Path(request.project_root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project_root_dir is not a directory: {root}")

    project, name = project_and_name_from_yaml(str(yaml_p))

    cmd: list[str] = [
        "subyolo",
        "train",
        f"model={request.model}",
        f"data={str(yaml_p)}",
        f"epochs={request.epochs}",
        f"imgsz={request.imgsz}",
        f"project={project}",
        f"name={name}",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )

    return YoloTrainResponse(
        status="ok" if proc.returncode == 0 else "failed",
        command=shlex.join(cmd),
        cwd=str(root),
        project=project,
        name=name,
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
