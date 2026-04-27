from collections import Counter
from pathlib import Path

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    RewriteYoloLabelIndicesRequest,
    RewriteYoloLabelIndicesResponse,
    ScanYoloLabelIndicesRequest,
    ScanYoloLabelIndicesResponse,
    YoloLabelIndexCount,
)
from app.services.task_manager import ensure_current_task_active


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


def _parse_label_index(line: str) -> tuple[int | None, bool]:
    stripped = line.strip()
    if not stripped:
        return None, False

    parts = stripped.split()
    if len(parts) < 5:
        return None, True

    try:
        label_index = int(parts[0])
    except ValueError:
        return None, True

    return label_index, False


def scan_yolo_label_indices(
    request: ScanYoloLabelIndicesRequest,
) -> ScanYoloLabelIndicesResponse:
    input_dir, labels_dirs = _resolve_input_and_labels_dirs(request.input_dir)
    label_files = [
        label_file
        for labels_dir in labels_dirs
        for label_file in _iter_label_files(labels_dir, request.recursive)
    ]

    counter: Counter[int] = Counter()
    skipped_invalid_lines = 0

    for label_path in label_files:
        ensure_current_task_active()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            label_index, invalid = _parse_label_index(line)
            if invalid:
                skipped_invalid_lines += 1
            elif label_index is not None:
                counter[label_index] += 1

    indices = [
        YoloLabelIndexCount(index=label_index, count=counter[label_index])
        for label_index in sorted(counter)
    ]
    return ScanYoloLabelIndicesResponse(
        input_dir=str(input_dir),
        labels_dir=str(labels_dirs[0]),
        labels_dirs=[str(labels_dir) for labels_dir in labels_dirs],
        total_label_files=len(label_files),
        total_objects=sum(counter.values()),
        skipped_invalid_lines=skipped_invalid_lines,
        indices=indices,
    )


def rewrite_yolo_label_indices(
    request: RewriteYoloLabelIndicesRequest,
) -> RewriteYoloLabelIndicesResponse:
    input_dir, labels_dirs = _resolve_input_and_labels_dirs(request.input_dir)
    label_files = [
        label_file
        for labels_dir in labels_dirs
        for label_file in _iter_label_files(labels_dir, request.recursive)
    ]
    mapping = request.mapping
    default_target_index = request.default_target_index

    modified_label_files = 0
    unchanged_label_files = 0
    changed_lines = 0
    skipped_invalid_lines = 0
    total_objects = 0

    for label_path in label_files:
        ensure_current_task_active()
        original_lines = label_path.read_text(encoding="utf-8").splitlines(keepends=True)

        rewritten_lines: list[str] = []
        file_changed = False

        for line in original_lines:
            stripped = line.strip()
            if not stripped:
                rewritten_lines.append(line)
                continue

            parts = stripped.split()
            if len(parts) < 5:
                rewritten_lines.append(line)
                skipped_invalid_lines += 1
                continue

            try:
                current_index = int(parts[0])
            except ValueError:
                rewritten_lines.append(line)
                skipped_invalid_lines += 1
                continue

            total_objects += 1
            target_index = mapping.get(current_index, default_target_index)
            if target_index is None or target_index == current_index:
                rewritten_lines.append(line)
                continue

            parts[0] = str(target_index)
            trailing_newline = "\n" if line.endswith("\n") else ""
            rewritten_lines.append(f"{' '.join(parts)}{trailing_newline}")
            file_changed = True
            changed_lines += 1

        if file_changed:
            label_path.write_text("".join(rewritten_lines), encoding="utf-8")
            modified_label_files += 1
        else:
            unchanged_label_files += 1

    return RewriteYoloLabelIndicesResponse(
        input_dir=str(input_dir),
        labels_dir=str(labels_dirs[0]),
        labels_dirs=[str(labels_dir) for labels_dir in labels_dirs],
        total_label_files=len(label_files),
        total_objects=total_objects,
        modified_label_files=modified_label_files,
        unchanged_label_files=unchanged_label_files,
        changed_lines=changed_lines,
        skipped_invalid_lines=skipped_invalid_lines,
        mapping={str(key): value for key, value in sorted(mapping.items())},
        default_target_index=default_target_index,
    )
