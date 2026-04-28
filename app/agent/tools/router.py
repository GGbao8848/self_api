import re
from collections.abc import Callable


_PATH_RE = re.compile(r"(/[^\s，。；;,]+)")
_LAST_YAML_RE = re.compile(r"((?:sftp|ssh)://[^\s，。；;,]+|/[^\s，。；;,]+\.ya?ml)")


def route_message_to_tool(message: str) -> tuple[str, dict] | None:
    """Minimal fallback router.

    This router is intentionally conservative. The main selection path is the LLM.
    We only keep a few unambiguous aliases here so the fallback stays cheap to maintain.
    """
    text = message.strip()
    if not text:
        return None

    input_dir = _extract_first_path(text)
    lower = text.lower()

    for tool_name, matcher, builder in _FALLBACK_ROUTES:
        if matcher(text, lower):
            return tool_name, builder(input_dir, text)
    return None


def _extract_first_path(text: str) -> str | None:
    match = _PATH_RE.search(text)
    if match is None:
        return None
    return match.group(1).rstrip("'\"`)]}")


def _extract_last_yaml_like_path(text: str) -> str | None:
    match = _LAST_YAML_RE.search(text)
    if match is None:
        return None
    return match.group(1).rstrip("'\"`)]}")


def _build_input_only(input_dir: str | None, _text: str) -> dict:
    return {"input_dir": input_dir} if input_dir else {}


def _build_with_output_suffix(suffix: str) -> Callable[[str | None, str], dict]:
    def _builder(input_dir: str | None, _text: str) -> dict:
        if not input_dir:
            return {}
        normalized_input = input_dir.rstrip("/")
        return {
            "input_dir": normalized_input,
            "output_dir": f"{normalized_input}{suffix}",
        }

    return _builder


def _build_clean_nested_dataset_flat(input_dir: str | None, _text: str) -> dict:
    if not input_dir:
        return {}
    normalized_input = input_dir.rstrip("/")
    return {
        "input_dir": normalized_input,
        "output_dir": f"{normalized_input}_cleaned_flat",
        "recursive": True,
        "pairing_mode": "images_xmls_subfolders",
        "flatten": True,
        "include_backgrounds": False,
        "copy_files": True,
        "overwrite": True,
        "images_dir_aliases": ["images", "image"],
        "xmls_dir_aliases": ["xmls", "xml"],
    }


def _build_build_yolo_yaml(input_dir: str | None, _text: str) -> dict:
    if not input_dir:
        return {}
    normalized_input = input_dir.rstrip("/")
    return {
        "input_dir": normalized_input,
        "output_yaml_path": f"{normalized_input}/data.yaml",
    }


def _build_publish_incremental(input_dir: str | None, text: str) -> dict:
    arguments: dict = {}
    if input_dir:
        arguments["local_paths"] = [input_dir.rstrip("/")]
    last_yaml = _extract_last_yaml_like_path(text)
    if last_yaml:
        arguments["last_yaml"] = last_yaml
    return arguments


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _matches_xml_to_yolo(text: str, lower: str) -> bool:
    return "xml-to-yolo" in lower or _has_any(lower, ("xml转yolo", "voc转yolo", "xml to yolo"))


def _matches_yolo_sliding_window_crop(text: str, lower: str) -> bool:
    return "yolo-sliding-window-crop" in lower or _has_any(
        text, ("滑窗裁剪", "滑窗裁图")
    )


def _matches_yolo_augment(text: str, lower: str) -> bool:
    return "yolo-augment" in lower or _has_any(text, ("数据增强",))


def _matches_split_yolo_dataset(text: str, lower: str) -> bool:
    return "split-yolo-dataset" in lower or _has_any(text, ("划分数据集",))


def _matches_annotate_visualize(text: str, lower: str) -> bool:
    return "annotate-visualize" in lower or _has_any(
        text, ("标注可视化", "可视化框", "画框预览")
    )


def _matches_clean_nested_dataset_flat(text: str, lower: str) -> bool:
    return "clean-nested-dataset-flat" in lower or _has_any(
        text, ("扁平化输出", "数据扁平化")
    )


def _matches_build_yolo_yaml(text: str, lower: str) -> bool:
    return "build-yolo-yaml" in lower or _has_any(
        text, ("生成data.yaml", "生成 data.yaml", "构建yolo yaml")
    )


def _matches_publish_incremental(text: str, lower: str) -> bool:
    return "publish-incremental-yolo-dataset" in lower or _has_any(
        text, ("增量发布数据集", "增量发布")
    )


def _matches_scan_indices(text: str, lower: str) -> bool:
    return "scan-yolo-label-indices" in lower or (
        _has_any(text, ("查看标签索引", "统计标签索引", "检查标签索引")) and "索引" in text
    )


def _matches_rewrite_indices(text: str, lower: str) -> bool:
    return "rewrite-yolo-label-indices" in lower or (
        _has_any(text, ("修改标签索引", "重映射标签索引", "合并标签索引")) and "索引" in text
    )


_FALLBACK_ROUTES: list[tuple[str, Callable[[str, str], bool], Callable[[str | None, str], dict]]] = [
    ("xml-to-yolo", _matches_xml_to_yolo, _build_input_only),
    ("yolo-sliding-window-crop", _matches_yolo_sliding_window_crop, _build_with_output_suffix("_yolo-sliding-window-crop")),
    ("yolo-augment", _matches_yolo_augment, _build_with_output_suffix("_aug")),
    ("clean-nested-dataset-flat", _matches_clean_nested_dataset_flat, _build_clean_nested_dataset_flat),
    ("build-yolo-yaml", _matches_build_yolo_yaml, _build_build_yolo_yaml),
    ("split-yolo-dataset", _matches_split_yolo_dataset, _build_with_output_suffix("_split")),
    ("annotate-visualize", _matches_annotate_visualize, _build_with_output_suffix("_visualized")),
    ("publish-incremental-yolo-dataset", _matches_publish_incremental, _build_publish_incremental),
    ("scan-yolo-label-indices", _matches_scan_indices, _build_input_only),
    ("rewrite-yolo-label-indices", _matches_rewrite_indices, _build_input_only),
]
