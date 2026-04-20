"""YOLO data.yaml 构建工具（模块内部实现）。

注意：对外 REST API `/api/v1/preprocess/build-yolo-yaml` 已于 2026-04 **弃用并移除**，
请改用 `POST /api/v1/preprocess/publish-yolo-dataset`——该端点在落盘数据集版本的同时
内置生成 `<version>.yaml`。

本模块保留的原因是 `publish_yolo_dataset` 服务内部仍复用这里的工具函数
（`_pick_effective_root_and_layout`、`_publish_dataset_tree`、`run_build_yolo_yaml` 等）。
除非被 publish 调用，不要从新代码里 import 本模块。
"""
from __future__ import annotations

import shutil
import re
from datetime import datetime
from pathlib import Path

import yaml

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import BuildYoloYamlRequest, BuildYoloYamlResponse
from app.services.remote_transfer import read_sftp_file_text
from app.services.task_manager import ensure_current_task_active
from app.utils.images import normalize_extensions

_DEFAULT_SPLITS = ("train", "val", "test")

_YAML_META_KEYS = frozenset({"nc", "names", "path", "yaml_file", "download", "roboflow"})


def _looks_remote_last_yaml(s: str) -> bool:
    t = s.strip()
    if t.startswith(("sftp://", "ssh://")):
        return True
    if re.match(r"^[^@/\s]+@[^:]+:.+", t):
        return True
    if re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*:/", t) and not re.match(r"^[a-zA-Z]:[/\\]", t):
        return True
    return False


def _normalize_split_entry(v: object) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)]


def _split_paths_from_yaml_text(text: str) -> dict[str, list[str]]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if k in _YAML_META_KEYS:
            continue
        paths = _normalize_split_entry(v)
        if paths:
            out[k] = paths
    return out


def _order_included_from_merged(
    merged: dict[str, list[str]],
    preferred: list[str],
) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for k in preferred:
        if k in merged and k not in seen:
            order.append(k)
            seen.add(k)
    for k in sorted(set(merged.keys()) - seen):
        order.append(k)
    return order


def _merge_split_path_dicts(
    last_paths: dict[str, list[str]],
    new_paths: dict[str, list[str]],
    preferred_key_order: list[str],
) -> dict[str, list[str]]:
    all_keys = set(last_paths) | set(new_paths)
    order: list[str] = []
    seen: set[str] = set()
    for k in preferred_key_order:
        if k in all_keys and k not in seen:
            order.append(k)
            seen.add(k)
    for k in sorted(all_keys - seen):
        order.append(k)

    merged: dict[str, list[str]] = {}
    for k in order:
        combined: list[str] = []
        dup: set[str] = set()
        for p in last_paths.get(k, []) + new_paths.get(k, []):
            if p not in dup:
                dup.add(p)
                combined.append(p)
        if combined:
            merged[k] = combined
    return merged


def _load_last_yaml_text(request: BuildYoloYamlRequest) -> tuple[str, str]:
    assert request.last_yaml is not None
    if _looks_remote_last_yaml(request.last_yaml):
        if not request.sftp_username or not request.sftp_private_key_path:
            raise ValueError(
                "sftp_username and sftp_private_key_path are required when last_yaml is a remote SFTP path"
            )
        text = read_sftp_file_text(
            request.last_yaml,
            username=request.sftp_username,
            private_key_path=request.sftp_private_key_path,
            port=request.sftp_port,
        )
        return text, "sftp"
    path = resolve_safe_path(
        request.last_yaml,
        field_name="last_yaml",
        must_exist=True,
        expect_file=True,
    )
    return path.read_text(encoding="utf-8"), "local"


def _read_classes(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    lines: list[str] = []
    for line in raw:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _class_names_from_yaml_data(data: dict) -> list[str]:
    """Build ordered class name list from Ultralytics-style names: list or dict of id -> name."""
    names = data.get("names")
    if names is None:
        return []
    if isinstance(names, list):
        return [str(x) for x in names if str(x).strip()]
    if isinstance(names, dict):

        def key_idx(k: object) -> int:
            try:
                return int(k)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return 0

        pairs = sorted(names.items(), key=lambda kv: key_idx(kv[0]))
        return [str(v) for _, v in pairs]
    return []


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


def _collect_split_images_dirs_recursive(
    *,
    root: Path,
    split_names: list[str],
    images_subdir: str,
    exts: set[str],
) -> dict[str, list[Path]]:
    split_set = set(split_names)
    found: dict[str, list[Path]] = {split: [] for split in split_names}
    seen: dict[str, set[Path]] = {split: set() for split in split_names}

    def add(split: str, candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved in seen[split]:
            return
        if not _dir_has_images(resolved, exts):
            return
        seen[split].add(resolved)
        found[split].append(resolved)

    for candidate in sorted(root.rglob("*"), key=lambda path: path.as_posix()):
        if not candidate.is_dir():
            continue

        # split_first style anywhere in the tree: .../<split>/images
        if candidate.name == images_subdir and candidate.parent.name in split_set:
            add(candidate.parent.name, candidate)
            continue

        # images_first style anywhere in the tree: .../images/<split>
        if candidate.name in split_set and candidate.parent.name == images_subdir:
            add(candidate.name, candidate)

    return {split: paths for split, paths in found.items() if paths}


def _scan_splits_for_layout(
    *,
    root: Path,
    split_names: list[str],
    images_subdir: str,
    layout: str,
    exts: set[str],
) -> tuple[list[str], dict[str, list[Path]], int]:
    included: list[str] = []
    split_dirs: dict[str, list[Path]] = {}
    direct_match_splits = 0
    recursive_split_dirs = _collect_split_images_dirs_recursive(
        root=root,
        split_names=split_names,
        images_subdir=images_subdir,
        exts=exts,
    )

    for sp in split_names:
        if layout == "split_first":
            split_root = root / sp
        else:
            split_root = root / images_subdir / sp

        matched: list[Path] = []
        seen: set[Path] = set()
        had_direct_match = False

        if split_root.is_dir():
            for path in _collect_images_dirs(
                split_root=split_root,
                images_subdir=images_subdir,
                exts=exts,
            ):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                matched.append(resolved)
                had_direct_match = True

        for path in recursive_split_dirs.get(sp, []):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            matched.append(resolved)

        if matched:
            if had_direct_match:
                direct_match_splits += 1
            included.append(sp)
            split_dirs[sp] = matched
    return included, split_dirs, direct_match_splits


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
            included, split_dirs, direct_match_splits = _scan_splits_for_layout(
                root=cand,
                split_names=split_names,
                images_subdir=images_subdir,
                layout=layout,
                exts=exts,
            )
            score = len(included)
            if score == 0:
                continue
            key = (score, direct_match_splits, -prio, -li)
            if best_key is None or key > best_key:
                best_key = key
                best = (cand, layout, included, split_dirs)

    if best is None:
        raise ValueError(
            f"no valid splits under {root_input} (tried subdirs dataset, ., yolo_split) with "
            f"layout train/images or images/train under {images_subdir!r} (checked splits: {split_names})"
        )
    return best


def _resolve_classes_file_optional(
    request: BuildYoloYamlRequest,
    *,
    effective_root: Path,
    root_input: Path,
) -> Path | None:
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
        root_input.parent / "classes.txt",
        root_input / "yolo_split" / "classes.txt",
        root_input / "dataset" / "classes.txt",
    ):
        if p.is_file():
            return p.resolve()
    return None


def _default_dataset_version(detector_name: str) -> str:
    return f"{detector_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"


def _publish_dataset_tree(
    *,
    effective_root: Path,
    root_input: Path,
    split_dirs: dict[str, list[Path]],
    classes_path: Path | None,
    project_root_dir: Path,
    detector_name: str,
    dataset_version: str,
) -> tuple[Path, dict[str, list[Path]], Path | None]:
    dataset_dir = project_root_dir / detector_name / "datasets" / dataset_version
    if dataset_dir.exists():
        raise ValueError(f"published dataset directory already exists: {dataset_dir}")

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(effective_root, dataset_dir)

    published_split_dirs: dict[str, list[Path]] = {}
    for split, paths in split_dirs.items():
        published_split_dirs[split] = [
            (dataset_dir / path.relative_to(effective_root)).resolve()
            for path in paths
        ]

    published_classes_path: Path | None = None
    if classes_path is not None:
        candidate = dataset_dir / classes_path.name
        if classes_path.resolve() != candidate.resolve():
            shutil.copy2(classes_path, candidate)
        published_classes_path = candidate.resolve()

    return dataset_dir.resolve(), published_split_dirs, published_classes_path


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
    detector_name = request.detector_name.strip() if request.detector_name else None
    dataset_version = request.dataset_version.strip() if request.dataset_version else None

    if (request.project_root_dir is None) != (detector_name is None):
        raise ValueError("project_root_dir and detector_name must be both set or both omitted")
    if dataset_version and not (request.project_root_dir and detector_name):
        raise ValueError("dataset_version requires project_root_dir and detector_name")
    if request.project_root_dir and (request.path_prefix_replace_from or request.path_prefix_replace_to):
        raise ValueError(
            "path_prefix_replace_from/to cannot be used together with project_root_dir publishing mode"
        )

    effective_root, _layout_mode, included, split_dirs = _pick_effective_root_and_layout(
        root_input,
        split_names,
        images_subdir,
        exts,
    )

    classes_path = _resolve_classes_file_optional(
        request,
        effective_root=effective_root,
        root_input=root_input,
    )

    last_yaml_text: str | None = None
    last_yaml_source: str | None = None
    if request.last_yaml:
        last_yaml_text, last_yaml_source = _load_last_yaml_text(request)

    class_names: list[str] = []
    if classes_path is not None:
        class_names = _read_classes(classes_path)

    if class_names:
        pass
    elif last_yaml_text:
        data = yaml.safe_load(last_yaml_text)
        if not isinstance(data, dict):
            raise ValueError("last_yaml must be a YAML mapping when classes.txt is empty or missing")
        class_names = _class_names_from_yaml_data(data)
        if not class_names:
            raise ValueError(
                "no class names: classes.txt is empty or missing lines, and last_yaml has no usable names"
            )
    else:
        raise ValueError(
            "classes.txt not found or has no class names: add non-empty classes.txt, or provide "
            "last_yaml with a names section (classes.txt may be an empty file when last_yaml is set)"
        )

    from_prefix = request.path_prefix_replace_from
    to_prefix = request.path_prefix_replace_to
    if (from_prefix is None) != (to_prefix is None):
        raise ValueError(
            "path_prefix_replace_from and path_prefix_replace_to must be both set or both omitted"
        )

    published_dataset_dir: Path | None = None
    recommended_train_project: str | None = None
    recommended_train_name: str | None = None

    if request.project_root_dir and detector_name:
        project_root_dir = resolve_safe_path(
            request.project_root_dir,
            field_name="project_root_dir",
            must_exist=False,
        )
        dataset_version = dataset_version or _default_dataset_version(detector_name)
        published_dataset_dir, split_dirs, classes_path = _publish_dataset_tree(
            effective_root=effective_root,
            root_input=root_input,
            split_dirs=split_dirs,
            classes_path=classes_path,
            project_root_dir=project_root_dir,
            detector_name=detector_name,
            dataset_version=dataset_version,
        )
        effective_root = published_dataset_dir
        recommended_train_project = str((project_root_dir / detector_name / "runs" / "detect").resolve())
        recommended_train_name = dataset_version

    split_abs_paths: dict[str, list[str]] = {}
    for sp in included:
        split_abs_paths[sp] = []
        for abs_dir in split_dirs[sp]:
            path_str = abs_dir.as_posix()
            if from_prefix is not None and to_prefix is not None:
                path_str = _apply_prefix_abs(path_str, from_prefix, to_prefix)
            split_abs_paths[sp].append(path_str)

    last_yaml_merged = False
    included_order = included
    if last_yaml_text:
        last_paths = _split_paths_from_yaml_text(last_yaml_text)
        split_abs_paths = _merge_split_path_dicts(last_paths, split_abs_paths, split_names)
        last_yaml_merged = bool(last_paths)
        included_order = _order_included_from_merged(split_abs_paths, split_names)

    if request.output_yaml_path is None:
        if not (published_dataset_dir and dataset_version):
            raise ValueError("output_yaml_path is required unless project_root_dir + detector_name are provided")
        output_yaml = published_dataset_dir / f"{dataset_version}.yaml"
    else:
        output_yaml = resolve_safe_path(
            request.output_yaml_path,
            field_name="output_yaml_path",
            must_exist=False,
        )
        if published_dataset_dir and dataset_version:
            expected_yaml = (published_dataset_dir / f"{dataset_version}.yaml").resolve()
            if output_yaml != expected_yaml:
                raise ValueError(
                    "when project_root_dir + detector_name are provided, output_yaml_path must equal "
                    f"{expected_yaml}"
                )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    content = _build_yaml_lines(
        split_abs_paths=split_abs_paths,
        included_order=included_order,
        class_names=class_names,
    )
    output_yaml.write_text(content, encoding="utf-8")

    first_path = split_abs_paths[included_order[0]][0] if included_order else ""

    return BuildYoloYamlResponse(
        output_yaml_path=str(output_yaml),
        path_in_yaml=first_path,
        dataset_root=str(effective_root.resolve()),
        splits_included=included_order,
        classes_count=len(class_names),
        dataset_version=dataset_version,
        published_dataset_dir=str(published_dataset_dir) if published_dataset_dir is not None else None,
        recommended_train_project=recommended_train_project,
        recommended_train_name=recommended_train_name,
        last_yaml_merged=last_yaml_merged,
        last_yaml_source=last_yaml_source,
    )
