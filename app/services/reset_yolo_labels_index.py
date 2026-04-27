from pathlib import Path

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    ResetYoloLabelIndexRequest,
    ResetYoloLabelIndexResponse,
)
from app.services.task_manager import ensure_current_task_active


def _rewrite_label_line(line: str) -> tuple[str, bool, bool]:
    stripped = line.strip()
    if not stripped:
        return line, False, False

    parts = stripped.split()
    if len(parts) < 5:
        return line, False, True

    try:
        int(parts[0])
    except ValueError:
        return line, False, True

    if parts[0] == "0":
        return line, False, False

    parts[0] = "0"
    trailing_newline = "\n" if line.endswith("\n") else ""
    return f"{' '.join(parts)}{trailing_newline}", True, False


def _iter_label_files(labels_dir: Path, recursive: bool) -> list[Path]:
    iterator = labels_dir.rglob("*.txt") if recursive else labels_dir.glob("*.txt")
    return sorted(path for path in iterator if path.is_file())


def _resolve_input_and_labels_dirs(input_dir: str) -> tuple[Path, list[Path]]:
    resolved_input_dir = resolve_safe_path(
        input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )
    if resolved_input_dir.name == "labels":
        return resolved_input_dir, [resolved_input_dir]

    labels_dir = resolved_input_dir / "labels"
    if labels_dir.exists() and labels_dir.is_dir():
        return resolved_input_dir, [labels_dir]

    nested_labels_dirs = sorted(
        path for path in resolved_input_dir.rglob("labels") if path.is_dir()
    )
    if nested_labels_dirs:
        return resolved_input_dir, nested_labels_dirs

    raise ValueError(
        "input_dir must be a labels directory, a dataset root containing labels/, "
        f"or a parent directory containing nested labels folders: {resolved_input_dir}"
    )


def run_reset_yolo_labels_index(
    request: ResetYoloLabelIndexRequest,
) -> ResetYoloLabelIndexResponse:
    input_dir, labels_dirs = _resolve_input_and_labels_dirs(request.input_dir)
    label_files = [
        label_file
        for labels_dir in labels_dirs
        for label_file in _iter_label_files(labels_dir, request.recursive)
    ]
    modified_label_files = 0
    unchanged_label_files = 0
    changed_lines = 0
    skipped_invalid_lines = 0

    for label_path in label_files:
        ensure_current_task_active()
        original_lines = label_path.read_text(encoding="utf-8").splitlines(keepends=True)

        rewritten_lines: list[str] = []
        file_changed = False
        for line in original_lines:
            new_line, changed, invalid = _rewrite_label_line(line)
            rewritten_lines.append(new_line)
            if changed:
                file_changed = True
                changed_lines += 1
            if invalid:
                skipped_invalid_lines += 1

        if file_changed:
            label_path.write_text("".join(rewritten_lines), encoding="utf-8")
            modified_label_files += 1
        else:
            unchanged_label_files += 1

    return ResetYoloLabelIndexResponse(
        input_dir=str(input_dir),
        labels_dir=str(labels_dirs[0]),
        labels_dirs=[str(labels_dir) for labels_dir in labels_dirs],
        total_label_files=len(label_files),
        modified_label_files=modified_label_files,
        unchanged_label_files=unchanged_label_files,
        changed_lines=changed_lines,
        skipped_invalid_lines=skipped_invalid_lines,
    )
