import time
import json

from app.agent.sessions import agent_session_store
from app.agent.providers import ProviderCallError
from app.agent.types import AgentRunRecord, AgentStepRecord, LLMToolDecision
from app.core.config import get_settings
from app.services import task_manager
from tests.data_helpers import create_image, create_voc_xml


def _wait_for_run(client, run_id: str, timeout_seconds: float = 10.0) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/v1/agent/runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["final_state"] in {
            "completed",
            "failed",
            "cancelled",
            "clarification_required",
            "requires_provider",
        }:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"agent run not finished in time: {run_id}")


def test_agent_async_run_executes_multiple_steps(client, case_dir, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")

    dataset_dir = case_dir / "long_run_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )

    from app.agent import runtime as runtime_module

    calls = {"count": 0}

    def fake_request_tool_decision(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        if calls["count"] == 2:
            assert "Current run progress:" in (kwargs.get("conversation_context") or "")
            return LLMToolDecision(
                action="execute",
                tool_name="scan-yolo-label-indices",
                tool_arguments={},
            )
        return LLMToolDecision(action="respond", message="处理完成")

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)
    get_settings.cache_clear()

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"{dataset_dir} 转 yolo 然后检查索引", "async_run": True},
    )

    assert response.status_code == 200
    accepted = response.json()
    assert accepted["final_state"] == "accepted"
    assert accepted["steps"] == []

    completed = _wait_for_run(client, accepted["run_id"])
    assert completed["final_state"] == "completed"
    assert completed["message"] == "处理完成"
    assert completed["root_run_id"] == accepted["run_id"]
    assert completed["trigger_kind"] == "new"
    assert completed["plan_summary"]
    assert [item["name"] for item in completed["tool_calls"]] == [
        "xml-to-yolo",
        "scan-yolo-label-indices",
    ]
    assert len(completed["steps"]) >= 5
    assert completed["steps"][0]["kind"] == "plan"
    assert any(step["task_id"] for step in completed["steps"] if step["tool_name"] == "xml-to-yolo")
    assert completed["tool_calls"][1]["arguments"]["input_dir"] == str(dataset_dir / "labels")


def test_agent_run_events_endpoint_streams_snapshots(client, case_dir, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")

    dataset_dir = case_dir / "events_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(10, 100, 255), size=(64, 64))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(64, 64),
        objects=[("cat", (5, 5, 40, 40))],
    )

    from app.agent import runtime as runtime_module

    calls = {"count": 0}

    def fake_request_tool_decision(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        return LLMToolDecision(action="respond", message="完成")

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)
    get_settings.cache_clear()

    accepted = client.post(
        "/api/v1/agent/chat",
        json={"message": "转一下", "session_id": "events-session", "async_run": True},
    ).json()

    response = client.get(f"/api/v1/agent/runs/{accepted['run_id']}/events")
    assert response.status_code == 200
    assert "event: snapshot" in response.text
    assert "event: end" in response.text
    snapshots = []
    for chunk in response.text.split("\n\n"):
        if not chunk.strip():
            continue
        lines = chunk.splitlines()
        event_name = lines[0].split(": ", 1)[1]
        payload = json.loads(lines[1].split("data: ", 1)[1])
        snapshots.append((event_name, payload))
    assert snapshots[-1][0] == "end"
    assert snapshots[-1][1]["run"]["final_state"] == "completed"
    assert snapshots[-1][1]["run"]["steps"]
    assert snapshots[-1][1]["run"]["plan_summary"]


def test_agent_retry_run_creates_followup_run(client, case_dir, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")

    dataset_dir = case_dir / "retry_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(10, 100, 255), size=(64, 64))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(64, 64),
        objects=[("cat", (5, 5, 40, 40))],
    )

    from app.agent import runtime as runtime_module

    calls = {"count": 0}

    def fake_request_tool_decision(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return LLMToolDecision(action="clarify", message="请确认要不要继续")
        if calls["count"] == 2:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        return LLMToolDecision(action="respond", message="处理完成")

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)
    get_settings.cache_clear()

    first = client.post(
        "/api/v1/agent/chat",
        json={"message": "帮我处理", "session_id": "retry-session"},
    ).json()
    assert first["final_state"] == "clarification_required"

    retried = client.post(
        f"/api/v1/agent/runs/{first['run_id']}/retry",
        json={"message": "继续，直接执行", "async_run": True},
    ).json()
    assert retried["session_id"] == first["session_id"]
    assert retried["final_state"] == "accepted"
    assert retried["parent_run_id"] == first["run_id"]
    assert retried["trigger_kind"] == "retry"

    completed = _wait_for_run(client, retried["run_id"])
    assert completed["final_state"] == "completed"
    assert completed["tool_calls"][0]["name"] == "xml-to-yolo"
    assert completed["parent_run_id"] == first["run_id"]
    assert completed["root_run_id"] == first["run_id"]


def test_agent_continue_run_preserves_lineage(client, case_dir, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")

    dataset_dir = case_dir / "continue_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(10, 100, 255), size=(64, 64))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(64, 64),
        objects=[("cat", (5, 5, 40, 40))],
    )

    from app.agent import runtime as runtime_module

    calls = {"count": 0}

    def fake_request_tool_decision(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return LLMToolDecision(action="respond", message="先告诉我下一步")
        if calls["count"] == 2:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        return LLMToolDecision(action="respond", message="继续处理完成")

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)
    get_settings.cache_clear()

    first = client.post(
        "/api/v1/agent/chat",
        json={"message": "先分析一下", "session_id": "continue-session"},
    ).json()
    assert first["final_state"] == "completed"

    continued = client.post(
        f"/api/v1/agent/runs/{first['run_id']}/continue",
        json={"message": "继续，直接转 yolo", "async_run": True},
    ).json()
    assert continued["final_state"] == "accepted"
    assert continued["parent_run_id"] == first["run_id"]
    assert continued["root_run_id"] == first["run_id"]
    assert continued["trigger_kind"] == "continue"

    completed = _wait_for_run(client, continued["run_id"])
    assert completed["final_state"] == "completed"
    assert completed["parent_run_id"] == first["run_id"]
    assert completed["root_run_id"] == first["run_id"]
    assert completed["trigger_kind"] == "continue"
    assert completed["tool_calls"][0]["name"] == "xml-to-yolo"


def test_agent_async_run_retries_transient_provider_failure_between_steps(
    client,
    case_dir,
    monkeypatch,
    isolated_runtime,
) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")

    dataset_dir = case_dir / "retry_transient_provider_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(50, 180, 80), size=(96, 96))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(96, 96),
        objects=[("cat", (12, 12, 72, 72))],
    )

    from app.agent import runtime as runtime_module

    calls = {"count": 0, "empty_failures": 0}

    def fake_request_tool_decision(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        if calls["count"] == 2:
            return LLMToolDecision(
                action="execute",
                tool_name="reset-yolo-label-index",
                tool_arguments={"input_dir": str(dataset_dir / "labels")},
            )
        if calls["empty_failures"] == 0:
            calls["empty_failures"] += 1
            raise ProviderCallError("provider returned empty message content")
        if calls["count"] == 4:
            assert "Current run progress:" in (kwargs.get("conversation_context") or "")
            return LLMToolDecision(
                action="execute",
                tool_name="split-yolo-dataset",
                tool_arguments={
                    "input_dir": str(dataset_dir),
                    "mode": "train_only",
                },
            )
        return LLMToolDecision(action="respond", message="处理完成")

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)
    get_settings.cache_clear()

    accepted = client.post(
        "/api/v1/agent/chat",
        json={"message": f"{dataset_dir} 转 yolo 后全转 0 再 train_only 划分", "async_run": True},
    ).json()

    completed = _wait_for_run(client, accepted["run_id"])
    assert completed["final_state"] == "completed"
    assert [item["name"] for item in completed["tool_calls"]] == [
        "xml-to-yolo",
        "reset-yolo-label-index",
        "split-yolo-dataset",
    ]
    assert completed["steps"][-1]["kind"] == "decision"
    assert completed["steps"][-1]["message"] == "处理完成"


def test_agent_async_run_chains_followup_tools_from_previous_output_dir(
    client,
    case_dir,
    monkeypatch,
    isolated_runtime,
) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")

    dataset_dir = case_dir / "chain_output_dataset"
    split_dir = case_dir / "chain_output_dataset_split"
    crop_dir = case_dir / "chain_output_dataset_split_yolo-sliding-window-crop"
    aug_dir = case_dir / "chain_output_dataset_split_yolo-sliding-window-crop_aug"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(120, 30, 200), size=(96, 96))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(96, 96),
        objects=[("cat", (10, 10, 70, 70))],
    )

    from app.agent import runtime as runtime_module

    calls = {"count": 0}

    def fake_request_tool_decision(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        if calls["count"] == 2:
            return LLMToolDecision(
                action="execute",
                tool_name="split-yolo-dataset",
                tool_arguments={"mode": "train_only", "output_dir": str(split_dir)},
            )
        if calls["count"] == 3:
            return LLMToolDecision(
                action="execute",
                tool_name="yolo-sliding-window-crop",
                tool_arguments={},
            )
        if calls["count"] == 4:
            return LLMToolDecision(
                action="execute",
                tool_name="yolo-augment",
                tool_arguments={},
            )
        if calls["count"] == 5:
            return LLMToolDecision(
                action="respond",
                message="处理完成",
            )
        raise AssertionError(f"unexpected decision call #{calls['count']}")

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)
    get_settings.cache_clear()

    accepted = client.post(
        "/api/v1/agent/chat",
        json={"message": f"{dataset_dir} 转 yolo 后 train_only 划分，再滑窗裁剪", "async_run": True},
    ).json()

    completed = _wait_for_run(client, accepted["run_id"], timeout_seconds=30.0)
    assert completed["final_state"] == "completed"
    assert [item["name"] for item in completed["tool_calls"]] == [
        "xml-to-yolo",
        "split-yolo-dataset",
        "yolo-sliding-window-crop",
        "yolo-augment",
    ]
    assert completed["tool_calls"][1]["arguments"]["input_dir"] == str(dataset_dir)
    assert completed["tool_calls"][2]["arguments"]["input_dir"] == str(split_dir)
    assert completed["tool_calls"][2]["arguments"]["output_dir"] == str(crop_dir)
    assert completed["tool_calls"][3]["arguments"]["input_dir"] == str(crop_dir)
    assert completed["tool_calls"][3]["arguments"]["output_dir"] == str(aug_dir)


def test_agent_long_run_falls_back_to_augment_when_model_repeats_after_crop(
    client,
    case_dir,
    monkeypatch,
    isolated_runtime,
) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "gemma4:e4b")

    dataset_dir = case_dir / "repeat_after_crop_dataset"
    split_dir = case_dir / "repeat_after_crop_dataset_split"
    crop_dir = case_dir / "repeat_after_crop_dataset_split_yolo-sliding-window-crop"
    aug_dir = case_dir / "repeat_after_crop_dataset_split_yolo-sliding-window-crop_aug"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(40, 200, 180), size=(128, 96))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(128, 96),
        objects=[("cat", (16, 16, 96, 72))],
    )

    from app.agent import runtime as runtime_module

    calls = {"count": 0}

    def fake_request_tool_decision(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        if calls["count"] == 2:
            return LLMToolDecision(
                action="execute",
                tool_name="split-yolo-dataset",
                tool_arguments={"input_dir": str(dataset_dir), "output_dir": str(split_dir), "mode": "train_only"},
            )
        if calls["count"] == 3:
            return LLMToolDecision(
                action="execute",
                tool_name="yolo-sliding-window-crop",
                tool_arguments={"input_dir": str(split_dir)},
            )
        if calls["count"] == 4:
            return LLMToolDecision(
                action="execute",
                tool_name="xml-to-yolo",
                tool_arguments={"input_dir": str(dataset_dir)},
            )
        return LLMToolDecision(action="respond", message="处理完成")

    monkeypatch.setattr(runtime_module, "request_tool_decision", fake_request_tool_decision)
    get_settings.cache_clear()

    accepted = client.post(
        "/api/v1/agent/chat",
        json={
            "message": f"{dataset_dir} xml转yolo,yolo全转为0，之后划分数据集train_only ,之后滑窗裁剪，对滑窗裁剪的数据进行数据增强",
            "async_run": True,
        },
    ).json()

    completed = _wait_for_run(client, accepted["run_id"], timeout_seconds=30.0)
    assert completed["final_state"] == "completed"
    assert [item["name"] for item in completed["tool_calls"]] == [
        "xml-to-yolo",
        "split-yolo-dataset",
        "yolo-sliding-window-crop",
        "yolo-augment",
    ]
    assert completed["tool_calls"][2]["arguments"]["output_dir"] == str(crop_dir)
    assert completed["tool_calls"][3]["arguments"]["input_dir"] == str(crop_dir)
    assert completed["tool_calls"][3]["arguments"]["output_dir"] == str(aug_dir)


def test_agent_store_marks_active_run_interrupted_on_restart(case_dir, isolated_runtime) -> None:
    agent_session_store.clear()
    dataset_dir = case_dir / "restart_mark_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(20, 20, 20), size=(64, 64))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(64, 64),
        objects=[("cat", (5, 5, 40, 40))],
    )

    run = AgentRunRecord(
        session_id="restart-session",
        run_id="run-restart-mark",
        user_message="转一下",
        message="waiting for background task",
        final_state="waiting_task",
        root_run_id="run-restart-mark",
        trigger_kind="new",
        plan_summary="Goal: 转一下",
        created_at="2026-05-01T00:00:00+00:00",
        updated_at="2026-05-01T00:00:01+00:00",
        request_payload={
            "session_id": "restart-session",
            "message": "转一下",
            "async_run": True,
            "tool_name": "xml-to-yolo",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
        checkpoint={
            "phase": "waiting_task",
            "current_tool": {
                "tool_name": "xml-to-yolo",
                "tool_arguments": {"input_dir": str(dataset_dir)},
                "task_id": "task-before-restart",
                "task_type": "xml_to_yolo",
                "step_id": "step-tool",
            },
        },
        steps=[
            AgentStepRecord(
                step_id="step-plan",
                step_index=1,
                kind="plan",
                status="completed",
                title="Frame objective and working plan",
                message="Goal: 转一下",
                started_at="2026-05-01T00:00:00+00:00",
                finished_at="2026-05-01T00:00:00+00:00",
            ),
            AgentStepRecord(
                step_id="step-tool",
                step_index=2,
                kind="tool",
                status="running",
                title="Execute xml-to-yolo",
                tool_name="xml-to-yolo",
                task_id="task-before-restart",
                task_type="xml_to_yolo",
                details={"request": {"input_dir": str(dataset_dir)}},
                started_at="2026-05-01T00:00:01+00:00",
            ),
        ],
    )
    agent_session_store.save_run(run)
    agent_session_store._initialized = False

    reloaded = agent_session_store.get_run(run.run_id)
    assert reloaded is not None
    assert reloaded.final_state == "interrupted"
    assert reloaded.checkpoint["resume_required"] is True
    assert reloaded.steps[-1].status == "interrupted"
    assert reloaded.steps[-1].details["resume_required"] is True


def test_agent_resume_run_replays_interrupted_async_tool(client, case_dir, isolated_runtime) -> None:
    agent_session_store.clear()
    dataset_dir = case_dir / "resume_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(200, 40, 40), size=(80, 80))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(80, 80),
        objects=[("cat", (8, 8, 60, 60))],
    )

    task_manager._ensure_initialized()
    with task_manager._connect() as connection:
        connection.execute(
            """
            INSERT INTO tasks (
                task_id,
                task_type,
                state,
                created_at,
                updated_at,
                finished_at,
                result_json,
                error,
                cancellation_requested,
                callback_url,
                callback_state,
                callback_sent_at,
                callback_status_code,
                callback_error,
                callback_events_json,
                artifacts_json,
                progress_current,
                progress_total,
                progress_message,
                events_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "task-interrupted-1",
                "xml_to_yolo",
                "interrupted",
                "2026-05-01T00:00:00+00:00",
                "2026-05-01T00:00:01+00:00",
                None,
                None,
                "task interrupted by service restart before completion",
                0,
                None,
                "succeeded",
                None,
                None,
                None,
                "[]",
                "[]",
                None,
                None,
                None,
                json.dumps([], ensure_ascii=True),
            ),
        )

    run = AgentRunRecord(
        session_id="resume-session",
        run_id="run-resume-1",
        user_message="转一下",
        message="agent run interrupted by service restart; resume is available",
        final_state="interrupted",
        root_run_id="run-resume-1",
        trigger_kind="new",
        plan_summary="Goal: 转一下",
        created_at="2026-05-01T00:00:00+00:00",
        updated_at="2026-05-01T00:00:01+00:00",
        request_payload={
            "session_id": "resume-session",
            "message": "转一下",
            "async_run": True,
            "tool_name": "xml-to-yolo",
            "tool_arguments": {"input_dir": str(dataset_dir)},
            "trigger_kind": "new",
        },
        checkpoint={
            "mode": "long_run",
            "phase": "waiting_task",
            "max_steps": 8,
            "resume_required": True,
            "current_tool": {
                "tool_name": "xml-to-yolo",
                "tool_arguments": {"input_dir": str(dataset_dir)},
                "task_id": "task-interrupted-1",
                "task_type": "xml_to_yolo",
                "step_id": "step-tool",
            },
        },
        steps=[
            AgentStepRecord(
                step_id="step-plan",
                step_index=1,
                kind="plan",
                status="completed",
                title="Frame objective and working plan",
                message="Goal: 转一下",
                started_at="2026-05-01T00:00:00+00:00",
                finished_at="2026-05-01T00:00:00+00:00",
            ),
            AgentStepRecord(
                step_id="step-tool",
                step_index=2,
                kind="tool",
                status="interrupted",
                title="Execute xml-to-yolo",
                tool_name="xml-to-yolo",
                task_id="task-interrupted-1",
                task_type="xml_to_yolo",
                details={"request": {"input_dir": str(dataset_dir)}, "resume_required": True},
                started_at="2026-05-01T00:00:01+00:00",
                finished_at="2026-05-01T00:00:01+00:00",
            ),
        ],
    )
    agent_session_store.save_run(run)

    response = client.post("/api/v1/agent/runs/run-resume-1/resume", json={})
    assert response.status_code == 200
    accepted = response.json()
    assert accepted["run_id"] == "run-resume-1"
    assert accepted["final_state"] == "accepted"

    completed = _wait_for_run(client, "run-resume-1")
    assert completed["final_state"] == "completed"
    assert completed["tool_calls"][0]["name"] == "xml-to-yolo"
    assert completed["tool_calls"][0]["result"]["state"] == "succeeded"
    assert completed["steps"][-1]["details"]["resumed_from_task_id"] == "task-interrupted-1"
