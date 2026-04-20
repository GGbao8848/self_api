"""LangGraph Pipeline 构建。

Graph 节点顺序：
  healthcheck → discover_classes → xml_to_yolo → review_labels
  → split_dataset → crop_augment → build_yaml
  → publish_transfer → train → poll_train → review_result

Gate 系统（interrupt）由各节点内部通过 _maybe_interrupt() 触发；
LangGraph 的 MemorySaver 以 thread_id=run_id 保存断点状态。

使用方式：
    config = {"configurable": {"thread_id": run_id}}
    graph.invoke(initial_state, config)      # 启动或从断点恢复
    graph.invoke(Command(resume=payload), config)  # 人工确认后继续
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command  # re-export for callers

from app.graph.nodes import (
    node_build_yaml,
    node_crop_augment,
    node_discover_classes,
    node_healthcheck,
    node_poll_train,
    node_publish_transfer,
    node_review_labels,
    node_review_result,
    node_split_dataset,
    node_train,
    node_xml_to_yolo,
)
from app.graph.state import PipelineState


def _should_abort(state: PipelineState) -> str:
    """若 completed=True 且有 error，路由到 END（提前终止）。"""
    if state.get("completed") or state.get("error"):
        return "end"
    return "continue"


_checkpointer = MemorySaver()


def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    # 注册节点
    g.add_node("healthcheck",       node_healthcheck)
    g.add_node("discover_classes",  node_discover_classes)
    g.add_node("xml_to_yolo",       node_xml_to_yolo)
    g.add_node("review_labels",     node_review_labels)
    g.add_node("split_dataset",     node_split_dataset)
    g.add_node("crop_augment",      node_crop_augment)
    g.add_node("build_yaml",        node_build_yaml)
    g.add_node("publish_transfer",  node_publish_transfer)
    g.add_node("train",             node_train)
    g.add_node("poll_train",        node_poll_train)
    g.add_node("review_result",     node_review_result)

    # 入口
    g.set_entry_point("healthcheck")

    # 每步之后检查是否需要提前退出
    for src, dst in [
        ("healthcheck",      "discover_classes"),
        ("discover_classes", "xml_to_yolo"),
        ("xml_to_yolo",      "review_labels"),
        ("review_labels",    "split_dataset"),
        ("split_dataset",    "crop_augment"),
        ("crop_augment",     "build_yaml"),
        ("build_yaml",       "publish_transfer"),
        ("publish_transfer", "train"),
        ("train",            "poll_train"),
        ("poll_train",       "review_result"),
    ]:
        g.add_conditional_edges(
            src,
            _should_abort,
            {"end": END, "continue": dst},
        )

    g.add_edge("review_result", END)

    return g


# 全局编译后的 graph（单例，供 endpoint 调用）
compiled_graph = build_graph().compile(checkpointer=_checkpointer)
