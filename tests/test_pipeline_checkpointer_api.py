"""Checkpointer 选择器测试。

验证：
- 默认 memory → 返回 MemorySaver 实例
- 配置 sqlite → 返回 SqliteSaver 实例，并在指定路径创建 sqlite 文件
- sqlite 持久化后，跨"重新编译"可以恢复已有 run 的 state
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from app.core.config import get_settings
from app.graph.state import PipelineState


def _build_tiny_graph(checkpointer):
    """一个仅有 healthcheck_stub 节点的最小 graph，避免触发 urlopen。"""
    g = StateGraph(PipelineState)

    def _stub(state: PipelineState) -> dict:
        return {
            "current_step": "tiny",
            "completed": True,
            "step_results": {"tiny": {"status": "ok", "summary": "done", "data": {}}},
        }

    g.add_node("tiny", _stub)
    g.set_entry_point("tiny")
    g.add_edge("tiny", END)
    return g.compile(checkpointer=checkpointer)


def test_build_checkpointer_memory_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELF_API_PIPELINE_CHECKPOINTER", "memory")
    get_settings.cache_clear()

    from app.graph.pipeline import _build_checkpointer

    saver = _build_checkpointer()
    assert isinstance(saver, MemorySaver)


def test_build_checkpointer_sqlite_creates_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "pipeline.sqlite"
    monkeypatch.setenv("SELF_API_PIPELINE_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("SELF_API_PIPELINE_SQLITE_PATH", str(db_path))
    get_settings.cache_clear()

    from app.graph.pipeline import _build_checkpointer

    saver = _build_checkpointer()
    assert isinstance(saver, SqliteSaver)
    assert db_path.exists(), "SqliteSaver 应该创建数据库文件"


def test_sqlite_checkpointer_persists_across_compilations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """在 SqliteSaver 上跑一次 run，用同一个文件重新加载后，state 仍可读出。"""
    db_path = tmp_path / "pipeline.sqlite"
    conn1 = sqlite3.connect(str(db_path), check_same_thread=False)
    saver1 = SqliteSaver(conn1)
    saver1.setup()

    graph1 = _build_tiny_graph(saver1)
    run_id = "persist-test-run"
    initial: PipelineState = {
        "run_id": run_id,
        "detector_name": "persist",
        "full_access": True,
    }
    graph1.invoke(initial, {"configurable": {"thread_id": run_id}})
    conn1.close()

    conn2 = sqlite3.connect(str(db_path), check_same_thread=False)
    saver2 = SqliteSaver(conn2)
    graph2 = _build_tiny_graph(saver2)

    snapshot = graph2.get_state({"configurable": {"thread_id": run_id}})
    assert snapshot.values.get("run_id") == run_id
    assert snapshot.values.get("completed") is True
    assert snapshot.values["step_results"]["tiny"]["status"] == "ok"
    conn2.close()


def test_build_checkpointer_sqlite_relative_path_resolves_under_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """相对路径应解析到 project_root 下。"""
    rel_path = "tmp_datasets/pipeline_ckpt_test.sqlite"
    monkeypatch.setenv("SELF_API_PIPELINE_CHECKPOINTER", "sqlite")
    monkeypatch.setenv("SELF_API_PIPELINE_SQLITE_PATH", rel_path)
    get_settings.cache_clear()

    from app.graph.pipeline import _build_checkpointer

    settings = get_settings()
    expected = (settings.project_root / rel_path).resolve()
    try:
        saver = _build_checkpointer()
        assert isinstance(saver, SqliteSaver)
        assert expected.exists()
    finally:
        if expected.exists():
            expected.unlink()
