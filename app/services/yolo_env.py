"""Helpers for resolving and validating yolo_train_env."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def resolve_env_python(yolo_train_env: str) -> str | None:
    """Resolve env spec to python executable path when absolute/local path is provided.

    Supported input:
    - conda env name (e.g. "v11_sk") -> returns None
    - env directory (e.g. "/.../envs/v11_sk") -> returns "/.../envs/v11_sk/bin/python"
    - python executable path (e.g. "/.../envs/v11_sk/bin/python") -> returns itself
    """
    text = (yolo_train_env or "").strip()
    if not text:
        return None
    if "/" not in text and "\\" not in text and not text.startswith("."):
        return None

    p = Path(text).expanduser().resolve()
    if p.is_file():
        return str(p)
    if p.is_dir():
        py = p / "bin" / "python"
        if py.is_file():
            return str(py)
        raise ValueError(f"yolo_train_env directory missing python: {py}")
    raise ValueError(f"yolo_train_env path does not exist: {p}")


def validate_yolo_env(yolo_train_env: str) -> dict[str, object]:
    """Validate if required runtime deps exist in target environment."""
    env_python = resolve_env_python(yolo_train_env)
    if env_python:
        cmd = [
            env_python,
            "-c",
            "import ultralytics, torch, cv2, yaml, numpy; print('ok')",
        ]
        mode = "python_path"
    else:
        cmd = [
            "conda",
            "run",
            "-n",
            yolo_train_env,
            "python",
            "-c",
            "import ultralytics, torch, cv2, yaml, numpy; print('ok')",
        ]
        mode = "conda_env_name"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "mode": mode,
        "resolved_python": env_python,
        "command": shlex.join(cmd),
        "exit_code": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }
