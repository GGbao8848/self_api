import re


_PATH_RE = re.compile(r"(/[^\s，。；;,]+)")


def route_message_to_tool(message: str) -> tuple[str, dict] | None:
    text = message.strip()
    if not text:
        return None

    input_dir = _extract_first_path(text)
    lower = text.lower()

    if _looks_like_xml_to_yolo_request(text, lower):
        arguments: dict = {}
        if input_dir:
            arguments["input_dir"] = input_dir
        return "xml-to-yolo", arguments

    if _looks_like_split_request(text, lower):
        arguments = {}
        if input_dir:
            normalized_input = input_dir.rstrip("/")
            arguments["input_dir"] = normalized_input
            arguments["output_dir"] = f"{normalized_input}_split"
        mode_and_ratios = _extract_split_mode_and_ratios(text)
        arguments.update(mode_and_ratios)
        return "split-yolo-dataset", arguments

    if _looks_like_rewrite_request(text, lower):
        arguments: dict = {}
        if input_dir:
            arguments["input_dir"] = input_dir
        default_target_index = _extract_default_target_index(text)
        if default_target_index is not None:
            arguments["default_target_index"] = default_target_index
        mapping = _extract_simple_range_mapping(text)
        if mapping:
            arguments["mapping"] = mapping
        return "rewrite-yolo-label-indices", arguments

    if _looks_like_scan_request(text, lower):
        arguments = {}
        if input_dir:
            arguments["input_dir"] = input_dir
        return "scan-yolo-label-indices", arguments

    return None


def _extract_first_path(text: str) -> str | None:
    match = _PATH_RE.search(text)
    if match is None:
        return None
    return match.group(1).rstrip("'\"`)]}")


def _looks_like_scan_request(text: str, lower: str) -> bool:
    if "scan-yolo-label-indices" in lower:
        return True
    has_index_word = "标签索引" in text or "索引" in text or "label" in lower
    has_scan_verb = any(word in text for word in ("查看", "统计", "检查", "扫描"))
    return has_index_word and has_scan_verb


def _looks_like_xml_to_yolo_request(text: str, lower: str) -> bool:
    if "xml-to-yolo" in lower:
        return True
    return any(word in lower for word in ("xml转yolo", "voc转yolo", "xml to yolo"))


def _looks_like_split_request(text: str, lower: str) -> bool:
    if "split-yolo-dataset" in lower:
        return True
    return "划分" in text or "split" in lower


def _looks_like_rewrite_request(text: str, lower: str) -> bool:
    if "rewrite-yolo-label-indices" in lower:
        return True
    has_index_word = "标签索引" in text or "索引" in text or "label" in lower
    has_rewrite_verb = any(word in text for word in ("修改", "重映射", "合并", "改成", "全部改"))
    return has_index_word and has_rewrite_verb


def _extract_split_mode_and_ratios(text: str) -> dict:
    if "仅训练" in text or "只要训练" in text or "train_only" in text:
        return {
            "mode": "train_only",
            "train_ratio": 1.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
        }

    ratio_match = re.search(r"(\d+(?:\.\d+)?)\s*[:：]\s*(\d+(?:\.\d+)?)", text)
    if ratio_match:
        first = float(ratio_match.group(1))
        second = float(ratio_match.group(2))
        total = first + second
        if total > 0:
            return {
                "mode": "train_val",
                "train_ratio": first / total,
                "val_ratio": second / total,
                "test_ratio": 0.0,
            }

    return {}


def _extract_default_target_index(text: str) -> int | None:
    patterns = [
        r"default_target_index\s*[:=]\s*(\d+)",
        r"全部(?:标签|索引|类别)?(?:都)?改成\s*(\d+)",
        r"其余(?:都)?改成\s*(\d+)",
        r"改成\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_simple_range_mapping(text: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    range_match = re.search(r"(\d+)\s*(?:到|-|~)\s*(\d+)\s*改成\s*(\d+)", text)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        target = int(range_match.group(3))
        if start <= end:
            mapping.update({index: target for index in range(start, end + 1)})

    for source, target in re.findall(r"(\d+)\s*(?:->|=>|改成)\s*(\d+)", text):
        mapping[int(source)] = int(target)

    return mapping
