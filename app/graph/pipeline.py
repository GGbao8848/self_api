"""LangGraph Pipeline 构建。

Graph 节点顺序：
  healthcheck → discover_classes → xml_to_yolo → review_labels
  → split_dataset → crop_augment → build_yaml
  → publish_transfer → train → poll_train → review_result

Gate 系统（interrupt）由各节点内部通过 _maybe_interrupt() 触发；
checkpointer 以 thread_id=run_id 保存断点状态。支持两种：
  - MemorySaver（进程内，重启丢失）
  - SqliteSaver（默认路径 storage/pipeline_checkpoints.sqlite）
通过 settings.pipeline_checkpointer 选择。

使用方式：
    config = {"configurable": {"thread_id": run_id}}
    graph.invoke(initial_state, config)      # 启动或从断点恢复
    graph.invoke(Command(resume=payload), config)  # 人工确认后继续
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command  # re-export for callers

from app.core.config import get_settings
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

logger = logging.getLogger(__name__)


def _build_checkpointer() -> BaseCheckpointSaver:
    """按配置创建 checkpointer；SQLite 不可用时回退到内存实现。"""
    settings = get_settings()
    kind = (settings.pipeline_checkpointer or "memory").strip().lower()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            logger.warning(
                "pipeline_checkpointer=sqlite 但 langgraph-checkpoint-sqlite 未安装，回退到 MemorySaver"
            )
            return MemorySaver()

        db_path = Path(settings.pipeline_sqlite_path)
        if not db_path.is_absolute():
            db_path = (settings.project_root / db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        logger.info("LangGraph pipeline checkpointer: sqlite → %s", db_path)
        return saver

    logger.info("LangGraph pipeline checkpointer: memory")
    return MemorySaver()


def _should_abort(state: PipelineState) -> str:
    """若 completed=True 且有 error，路由到 END（提前终止）。"""
    if state.get("completed") or state.get("error"):
        return "end"
    return "continue"


def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

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

    g.set_entry_point("healthcheck")

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


_checkpointer = _build_checkpointer()

compiled_graph = build_graph().compile(checkpointer=_checkpointer)
