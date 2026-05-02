from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.tools.executor import AgentToolError, execute_tool
from app.agent.types import ToolCallRecord


class PipelineState(TypedDict, total=False):
    message: str
    dataset_root: str
    current_dataset_dir: str
    labels_dir: str | None
    current_output_dir: str | None
    enable_xml_to_yolo: bool
    enable_reset_to_zero: bool
    enable_train_only_split: bool
    enable_sliding_window_crop: bool
    enable_augment: bool
    tool_calls: list[ToolCallRecord]
    error: str | None


def matches_langgraph_pipeline_request(message: str) -> bool:
    text = message.strip().lower()
    return bool(
        _extract_dataset_root(message)
        and any(keyword in text for keyword in ("xml转yolo", "xml to yolo", "xml-to-yolo"))
        and any(keyword in text for keyword in ("train_only", "滑窗", "裁剪", "增强", "augment", "全转为0"))
    )


def build_pipeline_state(message: str) -> PipelineState:
    dataset_root = _extract_dataset_root(message)
    if dataset_root is None:
        raise AgentToolError("could not infer dataset root path from request")
    lowered = message.lower()
    return PipelineState(
        message=message,
        dataset_root=dataset_root,
        current_dataset_dir=dataset_root,
        labels_dir=None,
        current_output_dir=None,
        enable_xml_to_yolo=any(keyword in lowered for keyword in ("xml转yolo", "xml to yolo", "xml-to-yolo")),
        enable_reset_to_zero="全转为0" in lowered or "转为0" in lowered or "reset" in lowered,
        enable_train_only_split="train_only" in lowered,
        enable_sliding_window_crop="滑窗" in lowered or "裁剪" in lowered or "crop" in lowered,
        enable_augment="增强" in lowered or "augment" in lowered,
        tool_calls=[],
        error=None,
    )


def run_pipeline(message: str) -> PipelineState:
    graph = _build_graph().compile()
    state = build_pipeline_state(message)
    return graph.invoke(state)


def _build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("xml_to_yolo", _xml_to_yolo_node)
    graph.add_node("reset_yolo_zero", _reset_yolo_zero_node)
    graph.add_node("split_train_only", _split_train_only_node)
    graph.add_node("sliding_window_crop", _sliding_window_crop_node)
    graph.add_node("yolo_augment", _yolo_augment_node)

    graph.add_edge(START, "xml_to_yolo")
    graph.add_edge("xml_to_yolo", "reset_yolo_zero")
    graph.add_edge("reset_yolo_zero", "split_train_only")
    graph.add_edge("split_train_only", "sliding_window_crop")
    graph.add_edge("sliding_window_crop", "yolo_augment")
    graph.add_edge("yolo_augment", END)
    return graph


def _xml_to_yolo_node(state: PipelineState) -> PipelineState:
    if state.get("error") or not state.get("enable_xml_to_yolo"):
        return state
    tool_call = execute_tool("xml-to-yolo", {"input_dir": state["dataset_root"]})
    next_state = _append_tool_call(state, tool_call)
    if tool_call.error:
        next_state["error"] = tool_call.error
        return next_state
    labels_dir = _extract_value(tool_call, "labels_dir")
    if labels_dir:
        next_state["labels_dir"] = labels_dir
    next_state["current_dataset_dir"] = state["dataset_root"]
    next_state["current_output_dir"] = labels_dir
    return next_state


def _reset_yolo_zero_node(state: PipelineState) -> PipelineState:
    if state.get("error") or not state.get("enable_reset_to_zero"):
        return state
    labels_dir = state.get("labels_dir") or f"{state['dataset_root'].rstrip('/')}/labels"
    tool_call = execute_tool("reset-yolo-label-index", {"input_dir": labels_dir, "recursive": True})
    next_state = _append_tool_call(state, tool_call)
    if tool_call.error:
        next_state["error"] = tool_call.error
        return next_state
    next_state["labels_dir"] = _extract_value(tool_call, "labels_dir") or labels_dir
    next_state["current_output_dir"] = next_state["labels_dir"]
    return next_state


def _split_train_only_node(state: PipelineState) -> PipelineState:
    if state.get("error") or not state.get("enable_train_only_split"):
        return state
    tool_call = execute_tool(
        "split-yolo-dataset",
        {
            "input_dir": state["current_dataset_dir"],
            "mode": "train_only",
            "train_ratio": 1.0,
        },
    )
    next_state = _append_tool_call(state, tool_call)
    if tool_call.error:
        next_state["error"] = tool_call.error
        return next_state
    output_dir = _extract_value(tool_call, "output_dir")
    if output_dir:
        next_state["current_dataset_dir"] = output_dir
        next_state["current_output_dir"] = output_dir
    return next_state


def _sliding_window_crop_node(state: PipelineState) -> PipelineState:
    if state.get("error") or not state.get("enable_sliding_window_crop"):
        return state
    tool_call = execute_tool(
        "yolo-sliding-window-crop",
        {
            "input_dir": state["current_dataset_dir"],
        },
    )
    next_state = _append_tool_call(state, tool_call)
    if tool_call.error:
        next_state["error"] = tool_call.error
        return next_state
    output_dir = _extract_value(tool_call, "output_dir")
    if output_dir:
        next_state["current_dataset_dir"] = output_dir
        next_state["current_output_dir"] = output_dir
    return next_state


def _yolo_augment_node(state: PipelineState) -> PipelineState:
    if state.get("error") or not state.get("enable_augment"):
        return state
    tool_call = execute_tool(
        "yolo-augment",
        {
            "input_dir": state["current_dataset_dir"],
        },
    )
    next_state = _append_tool_call(state, tool_call)
    if tool_call.error:
        next_state["error"] = tool_call.error
        return next_state
    output_dir = _extract_value(tool_call, "output_dir")
    if output_dir:
        next_state["current_dataset_dir"] = output_dir
        next_state["current_output_dir"] = output_dir
    return next_state


def _append_tool_call(state: PipelineState, tool_call: ToolCallRecord) -> PipelineState:
    return {
        **state,
        "tool_calls": [*state.get("tool_calls", []), tool_call],
    }


def _extract_value(tool_call: ToolCallRecord, key: str) -> str | None:
    if isinstance(tool_call.result, dict):
        nested_result = tool_call.result.get("result")
        if isinstance(nested_result, dict):
            value = nested_result.get(key)
            if isinstance(value, str) and value.strip():
                return value
        value = tool_call.result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    value = tool_call.arguments.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _extract_dataset_root(message: str) -> str | None:
    match = re.search(r"(/[^\s]+)", message)
    if match is None:
        return None
    candidate = match.group(1).strip()
    if not candidate:
        return None
    return str(Path(candidate).expanduser())
