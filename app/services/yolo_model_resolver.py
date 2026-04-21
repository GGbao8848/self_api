"""Helpers for resolving YOLO base model paths before training."""

from __future__ import annotations

import shlex
from pathlib import Path


def _should_search_parent_dirs(model: str) -> bool:
    text = model.strip()
    if not text:
        return False

    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "ftp://", "s3://", "gs://", "hf://")):
        return False

    candidate = Path(text).expanduser()
    return not candidate.is_absolute() and candidate.suffix.lower() == ".pt"


def resolve_local_yolo_model(model: str, *, cwd: Path, max_parent_levels: int = 3) -> str:
    text = model.strip()
    if not text or not _should_search_parent_dirs(text):
        return model

    relative_candidate = Path(text).expanduser()
    search_dir = cwd.resolve()
    for _ in range(max_parent_levels + 1):
        candidate = search_dir / relative_candidate
        if candidate.is_file():
            return str(candidate.resolve())
        if search_dir.parent == search_dir:
            break
        search_dir = search_dir.parent

    return model


def build_remote_yolo_model_resolver_shell(model: str, *, max_parent_levels: int = 3) -> str:
    text = model.strip()
    if not text or not _should_search_parent_dirs(text):
        return f'resolved_model={shlex.quote(model)}'

    search_bases = ["."]
    for level in range(1, max_parent_levels + 1):
        search_bases.append("/".join([".."] * level))
    search_bases_expr = " ".join(shlex.quote(base) for base in search_bases)

    return (
        f"model_input={shlex.quote(model)}; "
        "resolved_model=\"$model_input\"; "
        f"for base in {search_bases_expr}; do "
        "candidate=\"$base/$model_input\"; "
        "if [ -f \"$candidate\" ]; then "
        "resolved_model=\"$candidate\"; "
        "break; "
        "fi; "
        "done"
    )
