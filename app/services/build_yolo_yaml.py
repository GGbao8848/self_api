from __future__ import annotations

from pathlib import Path

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import BuildYoloYamlRequest, BuildYoloYamlResponse
from app.services.task_manager import ensure_current_task_active
from app.utils.images import normalize_extensions

_DEFAULT_SPLITS = ("train", "val", "test")


def _read_classes(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    lines: list[str] = []
    for line in raw:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _yaml_quote_scalar(value: str) -> str:
    if value == "":
        return '""'
    unsafe = (
        ":",
        "\\",
        '"',
        "'",
        "{",
        "}",
        "[",
        "]",
        "#",
        "&",
        "*",
        "!",
        "|",
        ">",
        "%",
        "@",
        "`",
    )
    if any(c in value for c in unsafe) or value.strip() != value or "\n" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _apply_prefix_abs(path_str: str, from_p: str, to_p: str) -> str:
    """Replace absolute prefix from_p with to_p; join with Path to avoid // from trailing-slash + leading-slash."""
    from_norm = from_p.replace("\\", "/")
    to_norm = to_p.replace("\\", "/")
    if not path_str.startswith(from_norm):
        raise ValueError(
            f"path {path_str!r} does not start with path_prefix_replace_from {from_norm!r}"
        )
    suffix = path_str[len(from_norm) :].lstrip("/")
    if not suffix:
        return str(Path(to_norm))
    return (Path(to_norm) / suffix).as_posix()


def _build_yaml_lines(
    *,
    split_abs_paths: dict[str, list[str]],
    included_order: list[str],
    class_names: list[str],
) -> str:
    lines: list[str] = []
    for split in included_order:
        split_paths = split_abs_paths[split]
        if len(split_paths) == 1:
            lines.append(f"{split}: {_yaml_quote_scalar(split_paths[0])}")
            continue
        lines.append(f"{split}:")
        for split_path in split_paths:
            lines.append(f"  - {_yaml_quote_scalar(split_path)}")

    nc = len(class_names)
    lines.append(f"nc: {nc}")
    lines.append("names:")
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {_yaml_quote_scalar(name)}")
    return "\n".join(lines) + "\n"


def _dir_has_images(images_dir: Path, exts: set[str]) -> bool:
    if not images_dir.is_dir():
        return False
    if not any(images_dir.iterdir()):
        return False
    return any(
        p.is_file() and p.suffix.lower() in exts for p in images_dir.iterdir()
    )


def _collect_images_dirs(
    *,
    split_root: Path,
    images_subdir: str,
    exts: set[str],
) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()

    def add_if_valid(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        if _dir_has_images(resolved, exts):
            seen.add(resolved)
            found.append(resolved)

    add_if_valid(split_root)
    candidates = sorted(
        split_root.rglob(images_subdir),
        key=lambda path: (len(path.relative_to(split_root).parts), path.as_posix()),
    )
    for candidate in candidates:
        if candidate.is_dir():
            add_if_valid(candidate)
    return found


def _scan_splits_for_layout(
    *,
    root: Path,
    split_names: list[str],
    images_subdir: str,
    layout: str,
    exts: set[str],
) -> tuple[list[str], dict[str, list[Path]]]:
    included: list[str] = []
    split_dirs: dict[str, list[Path]] = {}
    for sp in split_names:
        if layout == "split_first":
            split_root = root / sp
        else:
            split_root = root / images_subdir / sp
        if not split_root.is_dir():
            continue
        matched = _collect_images_dirs(
            split_root=split_root,
            images_subdir=images_subdir,
            exts=exts,
        )
        if matched:
            included.append(sp)
            split_dirs[sp] = matched
    return included, split_dirs


def _pick_effective_root_and_layout(
    root_input: Path,
    split_names: list[str],
    images_subdir: str,
    exts: set[str],
) -> tuple[Path, str, list[str], dict[str, list[Path]]]:
    """
    Try, in order: <input>/dataset, <input>, <input>/yolo_split — each with split_first then images_first.
    Prefer more matched splits; ties favor earlier candidate (dataset before root before yolo_split),
    then split_first before images_first.
    """
    candidates: list[Path] = [
        root_input / "dataset",
        root_input,
        root_input / "yolo_split",
    ]
    best_key: tuple[int, int, int] | None = None
    best: tuple[Path, str, list[str], dict[str, list[Path]]] | None = None

    for prio, cand in enumerate(candidates):
        if not cand.is_dir():
            continue
        for li, layout in enumerate(("split_first", "images_first")):
            included, split_dirs = _scan_splits_for_layout(
                root=cand,
                split_names=split_names,
                images_subdir=images_subdir,
                layout=layout,
                exts=exts,
            )
            score = len(included)
            if score == 0:
                continue
            key = (score, -prio, -li)
            if best_key is None or key > best_key:
                best_key = key
                best = (cand, layout, included, split_dirs)

    if best is None:
        raise ValueError(
            f"no valid splits under {root_input} (tried subdirs dataset, ., yolo_split) with "
            f"layout train/images or images/train under {images_subdir!r} (checked splits: {split_names})"
        )
    return best


def _resolve_classes_file(
    request: BuildYoloYamlRequest,
    *,
    effective_root: Path,
    root_input: Path,
) -> Path:
    if request.classes_file is not None:
        return resolve_safe_path(
            request.classes_file,
            field_name="classes_file",
            must_exist=True,
            expect_file=True,
        )
    for p in (
        effective_root / "classes.txt",
        root_input / "classes.txt",
        root_input / "yolo_split" / "classes.txt",
        root_input / "dataset" / "classes.txt",
    ):
        if p.is_file():
            return p.resolve()
    raise ValueError(
        f"classes.txt not found; set classes_file or place classes.txt under {effective_root} "
        f"or {root_input}"
    )


def run_build_yolo_yaml(request: BuildYoloYamlRequest) -> BuildYoloYamlResponse:
    ensure_current_task_active()
    root_input = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )

    split_names = list(request.split_names) if request.split_names else list(_DEFAULT_SPLITS)
    images_subdir = request.images_subdir_name.strip() or "images"
    exts = normalize_extensions(None)

    effective_root, _layout_mode, included, split_dirs = _pick_effective_root_and_layout(
        root_input,
        split_names,
        images_subdir,
        exts,
    )

    classes_path = _resolve_classes_file(
        request,
        effective_root=effective_root,
        root_input=root_input,
    )

    class_names = _read_classes(classes_path)
    if not class_names:
        raise ValueError(f"no class names in {classes_path}")

    from_prefix = request.path_prefix_replace_from
    to_prefix = request.path_prefix_replace_to
    if (from_prefix is None) != (to_prefix is None):
        raise ValueError(
            "path_prefix_replace_from and path_prefix_replace_to must be both set or both omitted"
        )

    split_abs_paths: dict[str, list[str]] = {}
    for sp in included:
        split_abs_paths[sp] = []
        for abs_dir in split_dirs[sp]:
            path_str = abs_dir.as_posix()
            if from_prefix is not None and to_prefix is not None:
                path_str = _apply_prefix_abs(path_str, from_prefix, to_prefix)
            split_abs_paths[sp].append(path_str)

    output_yaml = resolve_safe_path(
        request.output_yaml_path,
        field_name="output_yaml_path",
        must_exist=False,
    )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    content = _build_yaml_lines(
        split_abs_paths=split_abs_paths,
        included_order=included,
        class_names=class_names,
    )
    output_yaml.write_text(content, encoding="utf-8")

    first_path = split_abs_paths[included[0]][0] if included else ""

    return BuildYoloYamlResponse(
        output_yaml_path=str(output_yaml),
        path_in_yaml=first_path,
        dataset_root=str(effective_root.resolve()),
        splits_included=included,
        classes_count=len(class_names),
    )
