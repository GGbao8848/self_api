from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from pydantic import BaseModel

from app.agent.types import ToolSpec
from app.schemas.preprocess import (
    AnnotateVisualizeRequest,
    AggregateNestedDatasetRequest,
    BuildYoloYamlRequest,
    CleanNestedDatasetRequest,
    CopyPathRequest,
    DiscoverLeafDirsRequest,
    MovePathRequest,
    PublishIncrementalYoloDatasetRequest,
    ResetYoloLabelIndexRequest,
    RestoreVocCropsBatchRequest,
    RewriteYoloLabelIndicesRequest,
    ScanYoloLabelIndicesRequest,
    SplitYoloDatasetRequest,
    UnzipArchiveRequest,
    VocBarCropRequest,
    XmlToYoloRequest,
    YoloAugmentRequest,
    YoloSlidingWindowCropRequest,
    ZipFolderRequest,
)
NormalizeArguments: TypeAlias = Callable[[dict], dict]
Runner: TypeAlias = Callable[[BaseModel], BaseModel]


@dataclass(frozen=True)
class AgentToolDefinition:
    name: str
    description: str
    request_model: type[BaseModel]
    runner: Runner
    argument_hint: str
    async_task: bool = False
    task_type: str | None = None
    normalize_arguments: NormalizeArguments | None = None

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            async_task=self.async_task,
            argument_hint=self.argument_hint,
        )


def _with_default_output_dir(arguments: dict, suffix: str, field_name: str = "output_dir") -> dict:
    normalized = dict(arguments)
    input_dir = normalized.get("input_dir")
    if isinstance(input_dir, str) and input_dir.strip():
        normalized.setdefault(field_name, f"{input_dir.rstrip('/')}{suffix}")
    return normalized


def _normalize_yolo_augment(arguments: dict) -> dict:
    return _with_default_output_dir(arguments, "_aug")


def _normalize_yolo_sliding_window_crop(arguments: dict) -> dict:
    return _with_default_output_dir(arguments, "_yolo-sliding-window-crop")


def _normalize_split_yolo_dataset(arguments: dict) -> dict:
    return _with_default_output_dir(arguments, "_split")


def _normalize_annotate_visualize(arguments: dict) -> dict:
    return _with_default_output_dir(arguments, "_visualized")


def _normalize_clean_nested_dataset_flat(arguments: dict) -> dict:
    normalized = _with_default_output_dir(arguments, "_cleaned_flat")
    normalized.setdefault("recursive", True)
    normalized.setdefault("pairing_mode", "images_xmls_subfolders")
    normalized.setdefault("flatten", True)
    normalized.setdefault("include_backgrounds", False)
    normalized.setdefault("copy_files", True)
    normalized.setdefault("overwrite", True)
    normalized.setdefault("images_dir_aliases", ["images", "image"])
    normalized.setdefault("xmls_dir_aliases", ["xmls", "xml"])
    return normalized


def _normalize_publish_incremental_yolo_dataset(arguments: dict) -> dict:
    normalized = dict(arguments)
    if "local_paths" not in normalized:
        local_paths: list[str] = []
        input_dir = normalized.get("input_dir")
        if isinstance(input_dir, str) and input_dir.strip():
            local_paths.append(input_dir)
        input_dirs = normalized.get("input_dirs")
        if isinstance(input_dirs, list):
            local_paths.extend(
                value for value in input_dirs if isinstance(value, str) and value.strip()
            )
        if local_paths:
            normalized["local_paths"] = local_paths
    return normalized


def _normalize_build_yolo_yaml(arguments: dict) -> dict:
    normalized = dict(arguments)
    if "output_yaml_path" in normalized:
        return normalized
    if normalized.get("project_root_dir") and normalized.get("detector_name"):
        return normalized
    input_dir = normalized.get("input_dir")
    if isinstance(input_dir, str) and input_dir.strip():
        normalized["output_yaml_path"] = f"{input_dir.rstrip('/')}/data.yaml"
    return normalized


def _normalize_zip_folder(arguments: dict) -> dict:
    normalized = dict(arguments)
    input_dir = normalized.get("input_dir")
    if (
        "output_zip_path" not in normalized
        and isinstance(input_dir, str)
        and input_dir.strip()
    ):
        stripped = input_dir.rstrip("/")
        normalized["output_zip_path"] = f"{stripped}.zip"
    return normalized


def _normalize_unzip_archive(arguments: dict) -> dict:
    normalized = dict(arguments)
    archive_path = normalized.get("archive_path")
    if "output_dir" in normalized or not isinstance(archive_path, str) or not archive_path.strip():
        return normalized
    stripped = archive_path.rstrip("/")
    normalized["output_dir"] = stripped[:-4] if stripped.lower().endswith(".zip") else f"{stripped}_unzipped"
    return normalized


def _normalize_discover_leaf_dirs(arguments: dict) -> dict:
    normalized = dict(arguments)
    normalized.setdefault("recursive", True)
    return normalized


def _normalize_aggregate_nested_dataset(arguments: dict) -> dict:
    return _with_default_output_dir(arguments, "_dataset")


def _normalize_reset_yolo_label_index(arguments: dict) -> dict:
    normalized = dict(arguments)
    normalized.setdefault("recursive", True)
    return normalized


def _normalize_voc_bar_crop(arguments: dict) -> dict:
    return _with_default_output_dir(arguments, "_voc-bar-crop")


def _normalize_restore_voc_crops_batch(arguments: dict) -> dict:
    normalized = dict(arguments)
    if "output_dir" not in normalized:
        images_dir = normalized.get("edited_crops_images_dir")
        if isinstance(images_dir, str) and images_dir.strip():
            normalized["output_dir"] = f"{images_dir.rstrip('/')}_restored"
    normalized.setdefault("recursive", False)
    normalized.setdefault("skip_unparsed_names", True)
    return normalized


def _run_xml_to_yolo(payload: BaseModel) -> BaseModel:
    from app.services.xml_to_yolo import run_xml_to_yolo

    return run_xml_to_yolo(payload)


def _run_yolo_sliding_window_crop(payload: BaseModel) -> BaseModel:
    from app.services.yolo_sliding_window import run_yolo_sliding_window_crop

    return run_yolo_sliding_window_crop(payload)


def _run_yolo_augment(payload: BaseModel) -> BaseModel:
    from app.services.yolo_augment import run_yolo_augment

    return run_yolo_augment(payload)


def _run_split_yolo_dataset(payload: BaseModel) -> BaseModel:
    from app.services.split_yolo_dataset import run_split_yolo_dataset

    return run_split_yolo_dataset(payload)


def _run_annotate_visualize(payload: BaseModel) -> BaseModel:
    from app.services.annotation_visualize import run_annotate_visualize

    return run_annotate_visualize(payload)


def _run_clean_nested_dataset(payload: BaseModel) -> BaseModel:
    from app.services.nested_dataset import run_clean_nested_dataset

    return run_clean_nested_dataset(payload)


def _run_discover_leaf_dirs(payload: BaseModel) -> BaseModel:
    from app.services.nested_dataset import run_discover_leaf_dirs

    return run_discover_leaf_dirs(payload)


def _run_aggregate_nested_dataset(payload: BaseModel) -> BaseModel:
    from app.services.nested_dataset import run_aggregate_nested_dataset

    return run_aggregate_nested_dataset(payload)


def _run_build_yolo_yaml(payload: BaseModel) -> BaseModel:
    from app.services.build_yolo_yaml import run_build_yolo_yaml

    return run_build_yolo_yaml(payload)


def _run_zip_folder(payload: BaseModel) -> BaseModel:
    from app.services.file_operations import run_zip_folder

    return run_zip_folder(payload)


def _run_unzip_archive(payload: BaseModel) -> BaseModel:
    from app.services.file_operations import run_unzip_archive

    return run_unzip_archive(payload)


def _run_move_path(payload: BaseModel) -> BaseModel:
    from app.services.file_operations import run_move_path

    return run_move_path(payload)


def _run_copy_path(payload: BaseModel) -> BaseModel:
    from app.services.file_operations import run_copy_path

    return run_copy_path(payload)


def _run_publish_incremental_yolo_dataset(payload: BaseModel) -> BaseModel:
    from app.services.publish_yolo_dataset import run_publish_incremental_yolo_dataset

    return run_publish_incremental_yolo_dataset(payload)


def _run_reset_yolo_label_index(payload: BaseModel) -> BaseModel:
    from app.services.reset_yolo_labels_index import run_reset_yolo_labels_index

    return run_reset_yolo_labels_index(payload)


def _run_scan_yolo_label_indices(payload: BaseModel) -> BaseModel:
    from app.services.yolo_label_indices import scan_yolo_label_indices

    return scan_yolo_label_indices(payload)


def _run_rewrite_yolo_label_indices(payload: BaseModel) -> BaseModel:
    from app.services.yolo_label_indices import rewrite_yolo_label_indices

    return rewrite_yolo_label_indices(payload)


def _run_voc_bar_crop(payload: BaseModel) -> BaseModel:
    from app.services.voc_bar_crop import run_voc_bar_crop

    return run_voc_bar_crop(payload)


def _run_restore_voc_crops_batch(payload: BaseModel) -> BaseModel:
    from app.services.restore_voc_crops_batch import run_restore_voc_crops_batch

    return run_restore_voc_crops_batch(payload)


_TOOL_DEFINITIONS = [
    AgentToolDefinition(
        name="discover-leaf-dirs",
        description="Discover lowest-level leaf data folders under a root directory.",
        request_model=DiscoverLeafDirsRequest,
        runner=lambda payload: _run_discover_leaf_dirs(payload),
        normalize_arguments=_normalize_discover_leaf_dirs,
        argument_hint="{input_dir, recursive?, extensions?}",
    ),
    AgentToolDefinition(
        name="xml-to-yolo",
        description="Convert Pascal VOC XML annotations to YOLO labels.",
        request_model=XmlToYoloRequest,
        runner=lambda payload: _run_xml_to_yolo(payload),
        async_task=True,
        task_type="xml_to_yolo",
        argument_hint="{input_dir}",
    ),
    AgentToolDefinition(
        name="yolo-sliding-window-crop",
        description="Crop large YOLO images with synchronized sliding-window labels.",
        request_model=YoloSlidingWindowCropRequest,
        runner=lambda payload: _run_yolo_sliding_window_crop(payload),
        async_task=True,
        task_type="yolo_sliding_window_crop",
        normalize_arguments=_normalize_yolo_sliding_window_crop,
        argument_hint="{input_dir, output_dir, window_width?, window_height?, stride_x?, stride_y?, min_vis_ratio?, stride_ratio?, ignore_vis_ratio?, only_wide?}",
    ),
    AgentToolDefinition(
        name="yolo-augment",
        description="Apply YOLO dataset augmentation.",
        request_model=YoloAugmentRequest,
        runner=lambda payload: _run_yolo_augment(payload),
        async_task=True,
        task_type="yolo_augment",
        normalize_arguments=_normalize_yolo_augment,
        argument_hint="{input_dir, output_dir, recursive?, overwrite?, horizontal_flip?, vertical_flip?, brightness_up?, brightness_down?, contrast_up?, contrast_down?, gaussian_blur?}",
    ),
    AgentToolDefinition(
        name="split-yolo-dataset",
        description="Split a YOLO dataset into train, val, and test subsets.",
        request_model=SplitYoloDatasetRequest,
        runner=lambda payload: _run_split_yolo_dataset(payload),
        async_task=True,
        task_type="split_yolo_dataset",
        normalize_arguments=_normalize_split_yolo_dataset,
        argument_hint="{input_dir, output_dir, mode?, train_ratio?, val_ratio?, test_ratio?}",
    ),
    AgentToolDefinition(
        name="annotate-visualize",
        description="Render annotation preview images for a dataset.",
        request_model=AnnotateVisualizeRequest,
        runner=lambda payload: _run_annotate_visualize(payload),
        async_task=True,
        task_type="annotate_visualize",
        normalize_arguments=_normalize_annotate_visualize,
        argument_hint="{input_dir, output_dir, recursive?, extensions?, include_difficult?, line_width?, overwrite?, classes?, classes_file?}",
    ),
    AgentToolDefinition(
        name="clean-nested-dataset-flat",
        description="Flatten nested image/XML datasets into images and xmls folders.",
        request_model=CleanNestedDatasetRequest,
        runner=lambda payload: _run_clean_nested_dataset(payload),
        async_task=True,
        task_type="clean_nested_dataset",
        normalize_arguments=_normalize_clean_nested_dataset_flat,
        argument_hint="{input_dir, output_dir, recursive?, pairing_mode?, flatten?, include_backgrounds?, copy_files?, overwrite?, images_dir_aliases?, xmls_dir_aliases?}",
    ),
    AgentToolDefinition(
        name="aggregate-nested-dataset",
        description="Aggregate cleaned dataset fragments into a unified dataset root.",
        request_model=AggregateNestedDatasetRequest,
        runner=lambda payload: _run_aggregate_nested_dataset(payload),
        async_task=True,
        task_type="aggregate_nested_dataset",
        normalize_arguments=_normalize_aggregate_nested_dataset,
        argument_hint="{input_dir, output_dir?, images_dir_name?, labels_dir_name?, backgrounds_dir_name?, classes_file_name?, recursive?, extensions?, require_non_empty_labels?, overwrite?}",
    ),
    AgentToolDefinition(
        name="build-yolo-yaml",
        description="Build a YOLO data.yaml from a dataset root.",
        request_model=BuildYoloYamlRequest,
        runner=lambda payload: _run_build_yolo_yaml(payload),
        async_task=True,
        task_type="build_yolo_yaml",
        normalize_arguments=_normalize_build_yolo_yaml,
        argument_hint="{input_dir, output_yaml_path?, classes_file?, split_names?, images_subdir_name?, last_yaml?, sftp_username?, sftp_private_key_path?, sftp_port?, project_root_dir?, detector_name?, dataset_version?}",
    ),
    AgentToolDefinition(
        name="zip-folder",
        description="Package a directory into a zip archive.",
        request_model=ZipFolderRequest,
        runner=lambda payload: _run_zip_folder(payload),
        normalize_arguments=_normalize_zip_folder,
        argument_hint="{input_dir, output_zip_path?, include_root_dir?, overwrite?}",
    ),
    AgentToolDefinition(
        name="unzip-archive",
        description="Extract a zip archive into a directory.",
        request_model=UnzipArchiveRequest,
        runner=lambda payload: _run_unzip_archive(payload),
        normalize_arguments=_normalize_unzip_archive,
        argument_hint="{archive_path, output_dir?, overwrite?}",
    ),
    AgentToolDefinition(
        name="move-path",
        description="Move a file or directory into a target directory.",
        request_model=MovePathRequest,
        runner=lambda payload: _run_move_path(payload),
        argument_hint="{source_path, target_dir, overwrite?}",
    ),
    AgentToolDefinition(
        name="copy-path",
        description="Copy a file or directory into a target directory.",
        request_model=CopyPathRequest,
        runner=lambda payload: _run_copy_path(payload),
        argument_hint="{source_path, target_dir, overwrite?}",
    ),
    AgentToolDefinition(
        name="publish-incremental-yolo-dataset",
        description="Publish incremental YOLO data to a remote dataset location.",
        request_model=PublishIncrementalYoloDatasetRequest,
        runner=lambda payload: _run_publish_incremental_yolo_dataset(payload),
        async_task=True,
        task_type="publish_incremental_yolo_dataset",
        normalize_arguments=_normalize_publish_incremental_yolo_dataset,
        argument_hint="{last_yaml, local_paths}",
    ),
    AgentToolDefinition(
        name="reset-yolo-label-index",
        description="Rewrite all YOLO label indices to zero across labels directories.",
        request_model=ResetYoloLabelIndexRequest,
        runner=lambda payload: _run_reset_yolo_label_index(payload),
        normalize_arguments=_normalize_reset_yolo_label_index,
        argument_hint="{input_dir, recursive?}",
    ),
    AgentToolDefinition(
        name="scan-yolo-label-indices",
        description="Scan YOLO label index usage and counts.",
        request_model=ScanYoloLabelIndicesRequest,
        runner=lambda payload: _run_scan_yolo_label_indices(payload),
        argument_hint="{input_dir}",
    ),
    AgentToolDefinition(
        name="rewrite-yolo-label-indices",
        description="Rewrite or remap YOLO label indices.",
        request_model=RewriteYoloLabelIndicesRequest,
        runner=lambda payload: _run_rewrite_yolo_label_indices(payload),
        argument_hint="{input_dir, mapping?, default_target_index?}",
    ),
    AgentToolDefinition(
        name="voc-bar-crop",
        description="Crop square VOC patches centered on horizontal bar-like objects.",
        request_model=VocBarCropRequest,
        runner=lambda payload: _run_voc_bar_crop(payload),
        async_task=True,
        task_type="voc_bar_crop",
        normalize_arguments=_normalize_voc_bar_crop,
        argument_hint="{input_dir, output_dir?, recursive?}",
    ),
    AgentToolDefinition(
        name="restore-voc-crops-batch",
        description="Paste edited voc-bar crops back onto originals and merge VOC XMLs.",
        request_model=RestoreVocCropsBatchRequest,
        runner=lambda payload: _run_restore_voc_crops_batch(payload),
        async_task=True,
        task_type="restore_voc_crops_batch",
        normalize_arguments=_normalize_restore_voc_crops_batch,
        argument_hint="{original_images_dir, original_xmls_dir, edited_crops_images_dir, edited_crops_xmls_dir, output_dir?, recursive?, skip_unparsed_names?}",
    ),
]

_TOOL_DEFINITION_BY_NAME = {tool.name: tool for tool in _TOOL_DEFINITIONS}


def get_tool_definitions() -> list[AgentToolDefinition]:
    return list(_TOOL_DEFINITIONS)


def get_tool_definition(name: str) -> AgentToolDefinition | None:
    return _TOOL_DEFINITION_BY_NAME.get(name)


def get_tool_specs() -> list[ToolSpec]:
    return [tool.to_spec() for tool in _TOOL_DEFINITIONS]
