import random
import shutil
from pathlib import Path

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    SplitYoloDatasetRequest,
    SplitYoloDatasetResponse,
    SplitYoloFileDetail,
)
from app.services.task_manager import ensure_current_task_active
from app.utils.images import normalize_extensions

_SPLIT_DIR_NAMES = {"train", "val", "test"}


def _compute_split_counts(total: int, ratios: list[tuple[str, float]]) -> dict[str, int]:
    if total <= 0:
        return {name: 0 for name, _ in ratios}

    ratio_sum = sum(ratio for _, ratio in ratios)
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to a positive value")

    normalized = [(name, ratio / ratio_sum) for name, ratio in ratios]
    raw_counts = [(name, total * ratio) for name, ratio in normalized]
    base_counts = {name: int(raw) for name, raw in raw_counts}
    remainder = total - sum(base_counts.values())

    fractions = sorted(
        [
            (name, raw - int(raw), idx)
            for idx, (name, raw) in enumerate(raw_counts)
        ],
        key=lambda item: (-item[1], item[2]),
    )
    for idx in range(remainder):
        base_counts[fractions[idx][0]] += 1

    return base_counts


def _copy_or_move(src: Path, dst: Path, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def run_split_yolo_dataset(request: SplitYoloDatasetRequest) -> SplitYoloDatasetResponse:
    input_dir = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )

    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"images directory does not exist: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise ValueError(f"labels directory does not exist: {labels_dir}")

    output_dir = (
        resolve_safe_path(request.output_dir, field_name="output_dir")
        if request.output_dir
        else (input_dir / "split_dataset").resolve()
    )
    if output_dir == input_dir and not request.copy_files:
        raise ValueError("output_dir cannot equal input_dir when copy_files=False")

    normalized_exts = normalize_extensions(request.extensions)
    iterator = images_dir.rglob("*") if request.recursive else images_dir.glob("*")
    image_paths = []
    for path in iterator:
        ensure_current_task_active()
        if not path.is_file() or path.suffix.lower() not in normalized_exts:
            continue
        rel = path.relative_to(images_dir)
        if request.ignore_existing_split_dirs and rel.parts and rel.parts[0] in _SPLIT_DIR_NAMES:
            continue
        image_paths.append(path)
    image_paths = sorted(image_paths)

    details: list[SplitYoloFileDetail] = []
    pairs: list[tuple[Path, Path | None, Path]] = []
    skipped_images = 0

    for image_path in image_paths:
        ensure_current_task_active()
        rel_path = image_path.relative_to(images_dir)
        label_path = (labels_dir / rel_path).with_suffix(".txt")

        if request.require_label and not label_path.exists():
            skipped_images += 1
            details.append(
                SplitYoloFileDetail(
                    source_image=str(image_path),
                    source_label=str(label_path),
                    skipped_reason="label file not found",
                )
            )
            continue

        pairs.append((image_path, label_path if label_path.exists() else None, rel_path))

    working_pairs = list(pairs)
    if request.shuffle:
        random.Random(request.seed).shuffle(working_pairs)

    if request.mode == "train_only":
        split_counts = {"train": len(working_pairs), "val": 0, "test": 0}
        ordered_splits = ["train"]
    elif request.mode == "train_val":
        split_counts = _compute_split_counts(
            total=len(working_pairs),
            ratios=[("train", request.train_ratio), ("val", request.val_ratio)],
        )
        split_counts["test"] = 0
        ordered_splits = ["train", "val"]
    else:
        split_counts = _compute_split_counts(
            total=len(working_pairs),
            ratios=[
                ("train", request.train_ratio),
                ("val", request.val_ratio),
                ("test", request.test_ratio),
            ],
        )
        ordered_splits = ["train", "val", "test"]

    split_assignments: dict[str, list[tuple[Path, Path | None, Path]]] = {}
    offset = 0
    for split_name in ordered_splits:
        count = split_counts[split_name]
        split_assignments[split_name] = working_pairs[offset : offset + count]
        offset += count

    train_images = 0
    val_images = 0
    test_images = 0

    for split_name in ordered_splits:
        for source_image, source_label, rel_path in split_assignments[split_name]:
            ensure_current_task_active()
            target_rel = rel_path if request.keep_subdirs else Path(rel_path.name)
            if request.output_layout == "split_first":
                target_image = output_dir / split_name / "images" / target_rel
                target_label = (
                    output_dir / split_name / "labels" / target_rel.with_suffix(".txt")
                )
            else:
                target_image = output_dir / "images" / split_name / target_rel
                target_label = (
                    output_dir / "labels" / split_name / target_rel.with_suffix(".txt")
                )

            if (target_image.exists() or target_label.exists()) and not request.overwrite:
                skipped_images += 1
                details.append(
                    SplitYoloFileDetail(
                        source_image=str(source_image),
                        source_label=str(source_label) if source_label else None,
                        split=split_name,
                        target_image=str(target_image),
                        target_label=str(target_label),
                        skipped_reason="target file already exists",
                    )
                )
                continue

            if request.overwrite:
                if target_image.exists() and target_image.is_file():
                    target_image.unlink()
                if target_label.exists() and target_label.is_file():
                    target_label.unlink()

            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)

            try:
                _copy_or_move(source_image, target_image, request.copy_files)
                if source_label:
                    _copy_or_move(source_label, target_label, request.copy_files)
                else:
                    target_label.write_text("", encoding="utf-8")
            except OSError as exc:
                skipped_images += 1
                details.append(
                    SplitYoloFileDetail(
                        source_image=str(source_image),
                        source_label=str(source_label) if source_label else None,
                        split=split_name,
                        target_image=str(target_image),
                        target_label=str(target_label),
                        skipped_reason=f"failed to copy/move: {exc}",
                    )
                )
                continue

            if split_name == "train":
                train_images += 1
            elif split_name == "val":
                val_images += 1
            elif split_name == "test":
                test_images += 1

            details.append(
                SplitYoloFileDetail(
                    source_image=str(source_image),
                    source_label=str(source_label) if source_label else None,
                    split=split_name,
                    target_image=str(target_image),
                    target_label=str(target_label),
                )
            )

    copied_classes_file = None
    classes_src = input_dir / "classes.txt"
    if classes_src.exists() and classes_src.is_file():
        classes_dst = output_dir / "classes.txt"
        classes_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(classes_src, classes_dst)
        copied_classes_file = str(classes_dst)

    return SplitYoloDatasetResponse(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        mode=request.mode,
        total_images=len(image_paths),
        paired_images=len(pairs),
        skipped_images=skipped_images,
        train_images=train_images,
        val_images=val_images,
        test_images=test_images,
        copied_classes_file=copied_classes_file,
        details=details,
    )
