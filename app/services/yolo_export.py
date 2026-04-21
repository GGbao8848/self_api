"""Export trained YOLO best.pt to torchscript with args.yaml metadata."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import yaml

from app.schemas.preprocess import YoloExportRequest, YoloExportResponse
from app.services.yolo_env import resolve_env_python


def _extract_imgsz(args_obj: dict[str, object], args_yaml_path: Path) -> int:
    raw = args_obj.get("imgsz")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], int):
        return raw[0]
    raise ValueError(f"imgsz missing or invalid in args.yaml: {args_yaml_path}")


def _extract_dataset_stem(args_obj: dict[str, object], args_yaml_path: Path) -> tuple[str, str]:
    data_value = args_obj.get("data")
    if not isinstance(data_value, str) or not data_value.strip():
        raise ValueError(f"data missing or invalid in args.yaml: {args_yaml_path}")
    data_text = data_value.strip()
    return Path(data_text).name, Path(data_text).stem


def run_yolo_export(request: YoloExportRequest) -> YoloExportResponse:
    best_pt_path = Path(request.best_pt_path).expanduser().resolve()
    if not best_pt_path.is_file():
        raise ValueError(f"best_pt_path is not a file: {best_pt_path}")
    if best_pt_path.name != "best.pt":
        raise ValueError(f"best_pt_path must point to best.pt: {best_pt_path}")

    root = Path(request.project_root_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project_root_dir is not a directory: {root}")

    args_yaml_path = best_pt_path.parent.parent / "args.yaml"
    if not args_yaml_path.is_file():
        raise ValueError(f"args.yaml not found near weights dir: {args_yaml_path}")

    args_obj = yaml.safe_load(args_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(args_obj, dict):
        raise ValueError(f"args.yaml content must be a mapping: {args_yaml_path}")

    imgsz = _extract_imgsz(args_obj, args_yaml_path)
    dataset_yaml_name, dataset_stem = _extract_dataset_stem(args_obj, args_yaml_path)
    export_file_path = best_pt_path.parent / f"{dataset_stem}.torchscript"
    if export_file_path.exists() and not request.overwrite:
        raise ValueError(f"export file already exists: {export_file_path}")

    env_python = resolve_env_python(request.yolo_train_env)
    if env_python:
        cmd: list[str] = [
            env_python,
            "-m",
            "ultralytics",
            "export",
            f"model={str(best_pt_path)}",
            "format=torchscript",
            "half=True",
            f"imgsz={imgsz}",
        ]
    else:
        cmd = [
            "conda",
            "run",
            "-n",
            request.yolo_train_env,
            "--no-capture-output",
            "yolo",
            "export",
            f"model={str(best_pt_path)}",
            "format=torchscript",
            "half=True",
            f"imgsz={imgsz}",
        ]
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        default_export = best_pt_path.with_suffix(".torchscript")
        if default_export.exists() and default_export != export_file_path:
            if export_file_path.exists():
                export_file_path.unlink()
            default_export.rename(export_file_path)

    return YoloExportResponse(
        status="ok" if proc.returncode == 0 else "failed",
        command=shlex.join(cmd),
        cwd=str(root),
        best_pt_path=str(best_pt_path),
        args_yaml_path=str(args_yaml_path),
        imgsz=imgsz,
        dataset_yaml=dataset_yaml_name,
        export_file_path=str(export_file_path),
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
