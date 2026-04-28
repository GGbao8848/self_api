from pydantic import ValidationError

from app.agent.tools.async_runner import submit_and_wait_for_task
from app.agent.types import ToolCallRecord
from app.schemas.preprocess import (
    RewriteYoloLabelIndicesRequest,
    ScanYoloLabelIndicesRequest,
    SplitYoloDatasetRequest,
    XmlToYoloRequest,
)
from app.services.split_yolo_dataset import run_split_yolo_dataset
from app.services.xml_to_yolo import run_xml_to_yolo
from app.services.yolo_label_indices import (
    rewrite_yolo_label_indices,
    scan_yolo_label_indices,
)


class AgentToolError(ValueError):
    pass


def execute_tool(name: str, arguments: dict) -> ToolCallRecord:
    arguments = _normalize_tool_arguments(name, arguments)
    if name == "xml-to-yolo":
        return _execute_xml_to_yolo(arguments)
    if name == "split-yolo-dataset":
        return _execute_split_yolo_dataset(arguments)
    if name == "scan-yolo-label-indices":
        return _execute_scan_yolo_label_indices(arguments)
    if name == "rewrite-yolo-label-indices":
        return _execute_rewrite_yolo_label_indices(arguments)
    raise AgentToolError(f"tool is not executable yet: {name}")


def _normalize_tool_arguments(name: str, arguments: dict) -> dict:
    normalized = dict(arguments)
    if "input_dir" not in normalized:
        for alias in ("path", "dir", "dataset_dir", "dataset_path", "labels_dir"):
            value = normalized.pop(alias, None)
            if value:
                normalized["input_dir"] = value
                break

    if name == "split-yolo-dataset" and "output_dir" not in normalized:
        input_dir = normalized.get("input_dir")
        if isinstance(input_dir, str) and input_dir.strip():
            normalized["output_dir"] = f"{input_dir.rstrip('/')}_split"

    return normalized


def _execute_xml_to_yolo(arguments: dict) -> ToolCallRecord:
    try:
        payload = XmlToYoloRequest(**arguments)
        result = submit_and_wait_for_task(
            task_type="xml_to_yolo",
            payload=payload,
            runner=lambda: run_xml_to_yolo(payload).model_dump(),
        )
    except (ValidationError, ValueError) as exc:
        return ToolCallRecord(
            name="xml-to-yolo",
            arguments=arguments,
            error=str(exc),
        )
    return ToolCallRecord(
        name="xml-to-yolo",
        arguments=payload.model_dump(),
        result=result,
        error=result.get("error"),
    )


def _execute_split_yolo_dataset(arguments: dict) -> ToolCallRecord:
    try:
        payload = SplitYoloDatasetRequest(**arguments)
        result = submit_and_wait_for_task(
            task_type="split_yolo_dataset",
            payload=payload,
            runner=lambda: run_split_yolo_dataset(payload).model_dump(),
        )
    except (ValidationError, ValueError) as exc:
        return ToolCallRecord(
            name="split-yolo-dataset",
            arguments=arguments,
            error=str(exc),
        )
    return ToolCallRecord(
        name="split-yolo-dataset",
        arguments=payload.model_dump(),
        result=result,
        error=result.get("error"),
    )


def _execute_scan_yolo_label_indices(arguments: dict) -> ToolCallRecord:
    try:
        payload = ScanYoloLabelIndicesRequest(**arguments)
        result = scan_yolo_label_indices(payload)
    except (ValidationError, ValueError) as exc:
        return ToolCallRecord(
            name="scan-yolo-label-indices",
            arguments=arguments,
            error=str(exc),
        )
    return ToolCallRecord(
        name="scan-yolo-label-indices",
        arguments=payload.model_dump(),
        result=result.model_dump(),
    )


def _execute_rewrite_yolo_label_indices(arguments: dict) -> ToolCallRecord:
    try:
        payload = RewriteYoloLabelIndicesRequest(**arguments)
        result = rewrite_yolo_label_indices(payload)
    except (ValidationError, ValueError) as exc:
        return ToolCallRecord(
            name="rewrite-yolo-label-indices",
            arguments=arguments,
            error=str(exc),
        )
    return ToolCallRecord(
        name="rewrite-yolo-label-indices",
        arguments=payload.model_dump(),
        result=result.model_dump(),
    )
