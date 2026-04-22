from __future__ import annotations

from pathlib import Path, PurePosixPath

import yaml

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    BuildYoloYamlRequest,
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
    _publish_dataset_tree,
    _read_classes,
    _resolve_classes_file_optional,
    _split_paths_from_yaml_text,
    run_build_yolo_yaml,
)
from app.services.file_operations import run_zip_folder
from app.services.remote_transfer import run_remote_transfer
from app.services.remote_unzip import run_remote_unzip
from app.services.task_manager import report_progress
from app.utils.images import normalize_extensions


def _remote_posix_uri(host: str, path: PurePosixPath, port: int) -> str:
    path_text = path.as_posix()
    if port == 22:
        return f"sftp://{host}{path_text}"
    return f"sftp://{host}:{port}{path_text}"


def run_publish_yolo_dataset(request: PublishYoloDatasetRequest) -> PublishYoloDatasetResponse:
    if request.publish_mode == "local":
        report_progress(percent=5, stage="build_yaml", message="building dataset yaml", indeterminate=False)
        build_resp = run_build_yolo_yaml(
            BuildYoloYamlRequest(
                input_dir=request.input_dir,
                project_root_dir=request.project_root_dir,
                detector_name=request.detector_name,
                dataset_version=request.dataset_version,
                last_yaml=request.last_yaml,
                sftp_username=request.sftp_username,
                sftp_private_key_path=request.sftp_private_key_path,
                sftp_port=request.sftp_port,
            )
        )
        report_progress(percent=100, stage="build_yaml", message="dataset yaml ready", indeterminate=False)
        return PublishYoloDatasetResponse(
            publish_mode="local",
            output_yaml_path=build_resp.output_yaml_path,
            dataset_root=build_resp.dataset_root,
            splits_included=build_resp.splits_included,
            classes_count=build_resp.classes_count,
            dataset_version=build_resp.dataset_version or Path(build_resp.output_yaml_path).stem,
            published_dataset_dir=build_resp.published_dataset_dir or build_resp.dataset_root,
            recommended_train_project=build_resp.recommended_train_project or "",
            recommended_train_name=build_resp.recommended_train_name or "",
            last_yaml_merged=build_resp.last_yaml_merged,
            last_yaml_source=build_resp.last_yaml_source,
        )

    report_progress(percent=5, stage="prepare_publish", message="preparing publish package", indeterminate=False)
    root_input = resolve_safe_path(
        request.input_dir,
        field_name="input_dir",
        must_exist=True,
        expect_directory=True,
    )
    project_root_dir = resolve_safe_path(
        request.project_root_dir,
        field_name="project_root_dir",
        must_exist=False,
    )

    split_names = list(_DEFAULT_SPLITS)
    exts = normalize_extensions(None)
    effective_root, _layout_mode, included, split_dirs = _pick_effective_root_and_layout(
        root_input,
        split_names,
        "images",
        exts,
    )

    classes_path = _resolve_classes_file_optional(
        BuildYoloYamlRequest(input_dir=request.input_dir, output_yaml_path=str(root_input / "placeholder.yaml")),
        effective_root=effective_root,
        root_input=root_input,
    )

    last_yaml_text: str | None = None
    last_yaml_source: str | None = None
    if request.last_yaml:
        last_yaml_text, last_yaml_source = _load_last_yaml_text(
            BuildYoloYamlRequest(
                input_dir=request.input_dir,
                output_yaml_path=str(root_input / "placeholder.yaml"),
                last_yaml=request.last_yaml,
                sftp_username=request.sftp_username,
                sftp_private_key_path=request.sftp_private_key_path,
                sftp_port=request.sftp_port,
            )
        )

    class_names: list[str] = []
    if classes_path is not None:
        class_names = _read_classes(classes_path)
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

    dataset_version = request.dataset_version or _default_dataset_version(request.detector_name)
    staging_dataset_dir, staging_split_dirs, _published_classes_path = _publish_dataset_tree(
        effective_root=effective_root,
        root_input=root_input,
        split_dirs=split_dirs,
        classes_path=classes_path,
        project_root_dir=project_root_dir,
        detector_name=request.detector_name,
        dataset_version=dataset_version,
    )

    remote_project_root = PurePosixPath(request.remote_project_root_dir or "/")
    remote_dataset_dir = remote_project_root / request.detector_name / "datasets" / dataset_version
    remote_train_project = (remote_project_root / request.detector_name / "runs" / "detect").as_posix()
    remote_yaml_path = (remote_dataset_dir / f"{dataset_version}.yaml").as_posix()

    split_abs_paths: dict[str, list[str]] = {}
    for split in included:
        split_abs_paths[split] = [
            (remote_dataset_dir / path.relative_to(staging_dataset_dir)).as_posix()
            for path in staging_split_dirs[split]
        ]

    included_order = included
    last_yaml_merged = False
    if last_yaml_text:
        last_paths = _split_paths_from_yaml_text(last_yaml_text)
        split_abs_paths = _merge_split_path_dicts(last_paths, split_abs_paths, split_names)
        last_yaml_merged = bool(last_paths)
        included_order = _order_included_from_merged(split_abs_paths, split_names)

    staging_yaml_path = staging_dataset_dir / f"{dataset_version}.yaml"
    staging_yaml_path.write_text(
        _build_yaml_lines(
            split_abs_paths=split_abs_paths,
            included_order=included_order,
            class_names=class_names,
        ),
        encoding="utf-8",
    )

    report_progress(percent=55, stage="zip_dataset", message="creating publish archive", indeterminate=False)
    zip_resp = run_zip_folder(
        ZipFolderRequest(
            input_dir=str(staging_dataset_dir),
            output_zip_path=str(staging_dataset_dir.parent / f"{dataset_version}.zip"),
            include_root_dir=True,
            overwrite=False,
        )
    )

    report_progress(percent=72, stage="transfer_archive", message="uploading dataset archive", indeterminate=False)
    remote_datasets_parent = remote_project_root / request.detector_name / "datasets"
    transfer_resp = run_remote_transfer(
        RemoteTransferRequest(
            source_path=zip_resp.output_zip_path,
            target=_remote_posix_uri(request.remote_host or "", remote_datasets_parent, request.remote_port),
            username=request.remote_username,
            private_key_path=request.remote_private_key_path,
            port=request.remote_port,
            overwrite=False,
        )
    )
    report_progress(percent=88, stage="unzip_archive", message="extracting remote dataset archive", indeterminate=False)
    run_remote_unzip(
        RemoteUnzipRequest(
            archive_path=_remote_posix_uri(
                request.remote_host or "",
                remote_datasets_parent / Path(zip_resp.output_zip_path).name,
                request.remote_port,
            ),
            output_dir=_remote_posix_uri(request.remote_host or "", remote_datasets_parent, request.remote_port),
            username=request.remote_username,
            private_key_path=request.remote_private_key_path,
            port=request.remote_port,
            overwrite=False,
        )
    )
    report_progress(percent=100, stage="publish_done", message="dataset publish completed", indeterminate=False)

    return PublishYoloDatasetResponse(
        publish_mode="remote_sftp",
        output_yaml_path=remote_yaml_path,
        dataset_root=str(effective_root.resolve()),
        splits_included=included_order,
        classes_count=len(class_names),
        dataset_version=dataset_version,
        published_dataset_dir=remote_dataset_dir.as_posix(),
        staging_published_dataset_dir=str(staging_dataset_dir),
        staging_output_yaml_path=str(staging_yaml_path),
        local_archive_path=zip_resp.output_zip_path,
        remote_target_host=request.remote_host,
        remote_target_port=request.remote_port,
        remote_archive_path=transfer_resp.target_path,
        recommended_train_project=remote_train_project,
        recommended_train_name=dataset_version,
        last_yaml_merged=last_yaml_merged,
        last_yaml_source=last_yaml_source,
    )
