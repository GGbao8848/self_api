"""Run standardized torchscript inference in conda env."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from app.schemas.preprocess import YoloInferRequest, YoloInferResponse
from app.services.task_manager import report_progress
from app.services.yolo_env import resolve_env_python


_PROGRESS_PREFIX = "__SELF_API_PROGRESS__"


def run_yolo_infer(request: YoloInferRequest) -> YoloInferResponse:
    model_path = Path(request.model_path).expanduser().resolve()
    if not model_path.is_file():
        raise ValueError(f"model_path is not a file: {model_path}")
    output_dir = Path(request.project).expanduser().resolve() / request.name
    runner = Path(__file__).with_name("yolo_infer_runner.py").resolve()
    sources = [str(Path(p).expanduser().resolve()) for p in (request.source_paths or [])]

    env_python = resolve_env_python(request.yolo_train_env)
    if env_python:
        cmd: list[str] = [
            env_python,
            str(runner),
            "--model-path",
            str(model_path),
            "--project",
            str(Path(request.project).expanduser().resolve()),
            "--name",
            request.name,
            "--imgsz",
            str(request.imgsz),
            "--conf",
            str(request.conf),
            "--iou",
            str(request.iou),
            "--recursive",
            "1" if request.recursive else "0",
            "--save-labels",
            "1" if request.save_labels else "0",
            "--save-no-detect",
            "1" if request.save_no_detect else "0",
            "--add-conf-prefix",
            "1" if request.add_conf_prefix else "0",
            "--draw-label",
            "1" if request.draw_label else "0",
            "--overwrite",
            "1" if request.overwrite else "0",
        ]
    else:
        cmd = [
            "conda",
            "run",
            "-n",
            request.yolo_train_env,
            "--no-capture-output",
            "python",
            str(runner),
            "--model-path",
            str(model_path),
            "--project",
            str(Path(request.project).expanduser().resolve()),
            "--name",
            request.name,
            "--imgsz",
            str(request.imgsz),
            "--conf",
            str(request.conf),
            "--iou",
            str(request.iou),
            "--recursive",
            "1" if request.recursive else "0",
            "--save-labels",
            "1" if request.save_labels else "0",
            "--save-no-detect",
            "1" if request.save_no_detect else "0",
            "--add-conf-prefix",
            "1" if request.add_conf_prefix else "0",
            "--draw-label",
            "1" if request.draw_label else "0",
            "--overwrite",
            "1" if request.overwrite else "0",
        ]
    if request.device:
        cmd.extend(["--device", request.device])
    for src in sources:
        cmd.extend(["--source", src])
    if request.classes:
        cmd.extend(["--classes", ",".join(str(c) for c in request.classes)])

    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(request.project).expanduser().resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        if line.startswith(_PROGRESS_PREFIX):
            payload = json.loads(line[len(_PROGRESS_PREFIX):])
            report_progress(
                current=payload.get("current"),
                total=payload.get("total"),
                unit=payload.get("unit"),
                stage=payload.get("stage"),
                message=payload.get("message"),
                indeterminate=payload.get("indeterminate"),
            )
            continue
        stdout_lines.append(raw_line)
    stderr_text = proc.stderr.read() if proc.stderr is not None else ""
    return_code = proc.wait()
    stdout_text = "".join(stdout_lines)

    if return_code != 0:
        raise ValueError(
            f"yolo infer failed (exit_code={return_code}): {stderr_text or stdout_text or shlex.join(cmd)}"
        )

    summary_path = output_dir / "summary.json"
    run_args_path = output_dir / "args.yaml"
    if not summary_path.is_file():
        raise ValueError(f"infer summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return YoloInferResponse(
        status=summary.get("status", "ok"),
        model_path=str(model_path),
        output_dir=str(output_dir),
        run_args_path=str(run_args_path),
        summary_path=str(summary_path),
        total_images=int(summary.get("total_images", 0)),
        detected_images=int(summary.get("detected_images", 0)),
        no_detect_images=int(summary.get("no_detect_images", 0)),
        result_images=int(summary.get("result_images", 0)),
        labels_written=int(summary.get("labels_written", 0)),
        classes_filter=request.classes,
    )
