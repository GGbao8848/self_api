from __future__ import annotations

import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import yaml

from app.core.config import get_settings
from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    BuildYoloYamlRequest,
    PublishIncrementalYoloDatasetRequest,
    PublishYoloDatasetRequest,
    PublishYoloDatasetResponse,
    RemoteTransferRequest,
    RemoteUnzipRequest,
    ZipFolderRequest,
)
from app.services.build_yolo_yaml import (
    _DEFAULT_SPLITS,
    _build_yaml_lines,
    _class_names_from_yaml_data,
    _default_dataset_version,
    _load_last_yaml_text,
    _merge_split_path_dicts,
    _order_included_from_merged,
    _pick_effective_root_and_layout,
    _read_classes,
    _resolve_classes_file_optional,
    _split_paths_from_yaml_text,
)
from app.services.file_operations import run_zip_folder
from app.services.remote_transfer import run_remote_transfer
from app.services.remote_unzip import run_remote_unzip
from app.utils.images import normalize_extensions


def _remote_posix_uri(host: str, path: PurePosixPath, port: int) -> str:
    path_text = path.as_posix()
    if port == 22:
        return f"sftp://{host}{path_text}"
    return f"sftp://{host}:{port}{path_text}"


def _resolve_input_dirs(request: PublishYoloDatasetRequest) -> list[Path]:
    raw_paths = [request.input_dir, *(request.input_dirs or [])]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        text = raw.strip()
        if not text:
            continue
        resolved = resolve_safe_path(
            text,
            field_name="input_dir",
            must_exist=True,
            expect_directory=True,
        )
        real = resolved.resolve()
        if real in seen:
            continue
        seen.add(real)
        deduped.append(real)

    if len(deduped) == 1:
        only = deduped[0]
        only_text = str(only)
        if only_text.endswith("_aug"):
            base_candidate = Path(only_text[: -len("_aug")]).resolve()
            if base_candidate.is_dir() and base_candidate not in seen:
                deduped.insert(0, base_candidate)
                seen.add(base_candidate)
        else:
            aug_candidate = Path(f"{only_text}_aug").resolve()
            if aug_candidate.is_dir() and aug_candidate not in seen:
                deduped.append(aug_candidate)
                seen.add(aug_candidate)

    if not deduped:
        raise ValueError("input_dir or input_dirs must provide at least one dataset path")
    return deduped


def _infer_detector_name_from_last_yaml(last_yaml: str | None) -> str | None:
    ctx = _infer_publish_context_from_last_yaml(last_yaml)
    return ctx["detector_name"] if ctx else None


def _parse_remote_target_like(value: str) -> tuple[str | None, int | None, PurePosixPath]:
    text = value.strip()
    if text.startswith(("sftp://", "ssh://")):
        parsed = urlparse(text)
        return parsed.hostname, parsed.port, PurePosixPath(parsed.path or "/")
    if ":" in text and not text.startswith("/"):
        left, right = text.split(":", 1)
        if "@" in left:
            _user, host = left.rsplit("@", 1)
        else:
            host = left
        return host or None, None, PurePosixPath(right or "/")
    return None, None, PurePosixPath(text or "/")


def _infer_publish_context_from_last_yaml(last_yaml: str | None) -> dict[str, str | int | None] | None:
    if not last_yaml:
        return None
    text = last_yaml.strip().rstrip("/")
    if not text:
        return None
    host, port, path = _parse_remote_target_like(text)
    parts = path.parts
    bucket_idx = next((i for i, part in enumerate(parts) if part in {"dataset", "datasets"}), None)
    if bucket_idx is None or bucket_idx < 2 or len(parts) <= bucket_idx + 2:
        return None

    detector_name = parts[bucket_idx - 1]
    dataset_bucket = parts[bucket_idx]
    dataset_version = parts[bucket_idx + 1]
    yaml_stem = path.stem
    if dataset_version != yaml_stem:
        return None

    root_dir = PurePosixPath(*parts[: bucket_idx - 1]).as_posix() or "/"
    return {
        "remote_host": host,
        "remote_port": port,
        "remote_project_root_dir": root_dir,
        "detector_name": detector_name,
        "previous_dataset_version": dataset_version,
        "dataset_bucket": dataset_bucket,
    }


def _unique_child_name(base_name: str, used: set[str]) -> str:
    candidate = base_name.strip() or "dataset"
    if candidate not in used:
        used.add(candidate)
        return candidate
    idx = 2
    while True:
        derived = f"{candidate}_{idx}"
        if derived not in used:
            used.add(derived)
            return derived
        idx += 1


def _resolve_publish_remote_defaults(request: PublishYoloDatasetRequest) -> tuple[str | None, str | None, str | None, int]:
    settings = get_settings()
    inferred = _infer_publish_context_from_last_yaml(request.last_yaml) or {}
    return (
        request.remote_host or inferred.get("remote_host") or settings.remote_sftp_host,
        request.remote_project_root_dir or inferred.get("remote_project_root_dir") or settings.remote_sftp_project_root_dir,
        request.remote_username or settings.remote_sftp_username,
        request.remote_port or inferred.get("remote_port") or settings.remote_sftp_port,
    )


def _resolve_publish_private_key(request: PublishYoloDatasetRequest) -> str | None:
    settings = get_settings()
    return request.remote_private_key_path or settings.remote_sftp_private_key_path


def _resolve_publish_project_root_dir(request: PublishYoloDatasetRequest) -> Path:
    settings = get_settings()
    configured = (request.project_root_dir or settings.publish_project_root_dir or "").strip()
    if configured:
        return resolve_safe_path(
            configured,
            field_name="project_root_dir",
            must_exist=False,
        )
    return (settings.resolved_storage_root / "publish_workspace").resolve()


def _collect_source_specs(
    input_dirs: list[Path],
    split_names: list[str],
) -> tuple[list[dict], list[str], list[str], str]:
    exts = normalize_extensions(None)
    source_specs: list[dict] = []
    source_roots: list[str] = []
    all_class_names: list[str] = []
    first_dataset_root = ""

    for idx, root_input in enumerate(input_dirs):
        effective_root, _layout_mode, included, split_dirs = _pick_effective_root_and_layout(
            root_input,
            split_names,
            "images",
            exts,
        )
        classes_path = _resolve_classes_file_optional(
            BuildYoloYamlRequest(input_dir=str(root_input), output_yaml_path=str(root_input / "placeholder.yaml")),
            effective_root=effective_root,
            root_input=root_input,
        )
        class_names = _read_classes(classes_path) if classes_path is not None else []
        if idx == 0:
            first_dataset_root = str(effective_root.resolve())
        source_roots.append(str(effective_root.resolve()))
        source_specs.append(
            {
                "root_input": root_input,
                "effective_root": effective_root,
                "included": included,
                "split_dirs": split_dirs,
                "classes_path": classes_path,
                "class_names": class_names,
            }
        )
        if class_names:
            if not all_class_names:
                all_class_names = class_names
            elif all_class_names != class_names:
                raise ValueError(
                    "all input datasets must use identical classes.txt contents when publishing together"
                )

    return source_specs, source_roots, all_class_names, first_dataset_root


def _publish_merged_dataset_tree(
    *,
    source_specs: list[dict],
    project_root_dir: Path,
    detector_name: str,
    dataset_version: str,
    split_names: list[str],
) -> tuple[Path, dict[str, list[Path]], Path | None]:
    dataset_dir = project_root_dir / detector_name / "datasets" / dataset_version
    if dataset_dir.exists():
        raise ValueError(f"published dataset directory already exists: {dataset_dir}")

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=False)

    published_split_dirs: dict[str, list[Path]] = {split: [] for split in split_names}
    published_classes_path: Path | None = None
    used_names: set[str] = set()

    for spec in source_specs:
        effective_root: Path = spec["effective_root"]
        source_name = _unique_child_name(effective_root.name, used_names)
        target_root = dataset_dir / source_name
        shutil.copytree(effective_root, target_root)

        for split, paths in spec["split_dirs"].items():
            published_split_dirs.setdefault(split, [])
            published_split_dirs[split].extend(
                (target_root / path.relative_to(effective_root)).resolve()
                for path in paths
            )

        classes_path: Path | None = spec["classes_path"]
        if classes_path is None:
            continue
        candidate = dataset_dir / "classes.txt"
        if published_classes_path is None:
            if classes_path.resolve() != candidate.resolve():
                shutil.copy2(classes_path, candidate)
            else:
                candidate = classes_path.resolve()
            published_classes_path = candidate.resolve()

    published_split_dirs = {split: paths for split, paths in published_split_dirs.items() if paths}
    return dataset_dir.resolve(), published_split_dirs, published_classes_path


def run_publish_yolo_dataset(request: PublishYoloDatasetRequest) -> PublishYoloDatasetResponse:
    input_dirs = _resolve_input_dirs(request)
    inferred_ctx = _infer_publish_context_from_last_yaml(request.last_yaml) or {}
    detector_name = (request.detector_name or _infer_detector_name_from_last_yaml(request.last_yaml) or "").strip()
    if not detector_name:
        raise ValueError("detector_name is required unless it can be inferred from last_yaml")
    dataset_version = request.dataset_version or _default_dataset_version(detector_name)
    project_root_dir = _resolve_publish_project_root_dir(request)
    split_names = list(_DEFAULT_SPLITS)
    source_specs, source_roots, class_names, first_dataset_root = _collect_source_specs(input_dirs, split_names)

    last_yaml_text: str | None = None
    last_yaml_source: str | None = None
    if request.last_yaml:
        last_yaml_text, last_yaml_source = _load_last_yaml_text(
            BuildYoloYamlRequest(
                input_dir=str(input_dirs[0]),
                output_yaml_path=str(input_dirs[0] / "placeholder.yaml"),
                last_yaml=request.last_yaml,
                sftp_username=request.sftp_username,
                sftp_private_key_path=request.sftp_private_key_path,
                sftp_port=request.sftp_port,
            )
        )

    if not class_names and last_yaml_text:
        data = yaml.safe_load(last_yaml_text)
        if not isinstance(data, dict):
            raise ValueError("last_yaml must be a YAML mapping when classes.txt is empty or missing")
        class_names = _class_names_from_yaml_data(data)
    if not class_names:
        raise ValueError(
            "classes.txt not found or has no class names: add non-empty classes.txt, or provide "
            "last_yaml with a names section"
        )

    remote_host, remote_project_root_dir, remote_username, remote_port = _resolve_publish_remote_defaults(request)
    remote_private_key_path = _resolve_publish_private_key(request)
    if request.publish_mode == "remote_sftp":
        missing: list[str] = []
        if not remote_host:
            missing.append("remote_host")
        if not remote_project_root_dir:
            missing.append("remote_project_root_dir")
        if not remote_username:
            missing.append("remote_username")
        if not remote_private_key_path:
            missing.append("remote_private_key_path")
        if missing:
            raise ValueError(
                "publish_mode=remote_sftp requires: "
                + ", ".join(missing)
                + " (or configure matching SELF_API_REMOTE_SFTP_* entries in .env)"
            )

    staging_dataset_dir, staging_split_dirs, _published_classes_path = _publish_merged_dataset_tree(
        source_specs=source_specs,
        project_root_dir=project_root_dir,
        detector_name=detector_name,
        dataset_version=dataset_version,
        split_names=split_names,
    )

    split_abs_paths: dict[str, list[str]] = {}
    included = [split for split in split_names if staging_split_dirs.get(split)]
    for split in included:
        split_abs_paths[split] = [path.as_posix() for path in staging_split_dirs[split]]

    included_order = included
    last_yaml_merged = False
    if last_yaml_text:
        last_paths = _split_paths_from_yaml_text(last_yaml_text)
        split_abs_paths = _merge_split_path_dicts(last_paths, split_abs_paths, split_names)
        last_yaml_merged = bool(last_paths)
        included_order = _order_included_from_merged(split_abs_paths, split_names)

    staging_yaml_path = staging_dataset_dir / f"{dataset_version}.yaml"
    local_yaml_text = _build_yaml_lines(
        split_abs_paths=split_abs_paths,
        included_order=included_order,
        class_names=class_names,
    )
    staging_yaml_path.write_text(local_yaml_text, encoding="utf-8")

    if request.publish_mode == "local":
        return PublishYoloDatasetResponse(
            publish_mode="local",
            output_yaml_path=str(staging_yaml_path),
            dataset_root=first_dataset_root,
            source_dataset_roots=source_roots,
            splits_included=included_order,
            classes_count=len(class_names),
            dataset_version=dataset_version,
            published_dataset_dir=str(staging_dataset_dir),
            recommended_train_project=str((project_root_dir / detector_name / "runs" / "detect").resolve()),
            recommended_train_name=dataset_version,
            last_yaml_merged=last_yaml_merged,
            last_yaml_source=last_yaml_source,
        )

    remote_project_root = PurePosixPath(remote_project_root_dir or "/")
    dataset_bucket = str(inferred_ctx.get("dataset_bucket") or "datasets")
    remote_dataset_dir = remote_project_root / detector_name / dataset_bucket / dataset_version
    remote_train_project = (remote_project_root / detector_name / "runs" / "detect").as_posix()
    remote_yaml_path = (remote_dataset_dir / f"{dataset_version}.yaml").as_posix()

    remote_split_abs_paths: dict[str, list[str]] = {}
    for split in included:
        remote_split_abs_paths[split] = [
            (remote_dataset_dir / path.relative_to(staging_dataset_dir)).as_posix()
            for path in staging_split_dirs[split]
        ]

    if last_yaml_text:
        last_paths = _split_paths_from_yaml_text(last_yaml_text)
        remote_split_abs_paths = _merge_split_path_dicts(last_paths, remote_split_abs_paths, split_names)
        included_order = _order_included_from_merged(remote_split_abs_paths, split_names)

    staging_yaml_path.write_text(
        _build_yaml_lines(
            split_abs_paths=remote_split_abs_paths,
            included_order=included_order,
            class_names=class_names,
        ),
        encoding="utf-8",
    )

    zip_resp = run_zip_folder(
        ZipFolderRequest(
            input_dir=str(staging_dataset_dir),
            output_zip_path=str(staging_dataset_dir.parent / f"{dataset_version}.zip"),
            include_root_dir=True,
            overwrite=False,
        )
    )

    remote_datasets_parent = remote_project_root / detector_name / dataset_bucket
    transfer_resp = run_remote_transfer(
        RemoteTransferRequest(
            source_path=zip_resp.output_zip_path,
            target=_remote_posix_uri(remote_host or "", remote_datasets_parent, remote_port),
            username=remote_username,
            private_key_path=remote_private_key_path,
            port=remote_port,
            overwrite=False,
        )
    )
    run_remote_unzip(
        RemoteUnzipRequest(
            archive_path=_remote_posix_uri(
                remote_host or "",
                remote_datasets_parent / Path(zip_resp.output_zip_path).name,
                remote_port,
            ),
            output_dir=_remote_posix_uri(remote_host or "", remote_datasets_parent, remote_port),
            username=remote_username,
            private_key_path=remote_private_key_path,
            port=remote_port,
            overwrite=False,
        )
    )

    return PublishYoloDatasetResponse(
        publish_mode="remote_sftp",
        output_yaml_path=remote_yaml_path,
        dataset_root=first_dataset_root,
        source_dataset_roots=source_roots,
        splits_included=included_order,
        classes_count=len(class_names),
        dataset_version=dataset_version,
        published_dataset_dir=remote_dataset_dir.as_posix(),
        staging_published_dataset_dir=str(staging_dataset_dir),
        staging_output_yaml_path=str(staging_yaml_path),
        local_archive_path=zip_resp.output_zip_path,
        remote_target_host=remote_host,
        remote_target_port=remote_port,
        remote_archive_path=transfer_resp.target_path,
        recommended_train_project=remote_train_project,
        recommended_train_name=dataset_version,
        last_yaml_merged=last_yaml_merged,
        last_yaml_source=last_yaml_source,
    )


def run_publish_incremental_yolo_dataset(
    request: PublishIncrementalYoloDatasetRequest,
) -> PublishYoloDatasetResponse:
    local_paths = [path.strip() for path in request.local_paths if path.strip()]
    return run_publish_yolo_dataset(
        PublishYoloDatasetRequest(
            input_dir=local_paths[0],
            input_dirs=local_paths[1:] or None,
            publish_mode="remote_sftp",
            last_yaml=request.last_yaml,
        )
    )
