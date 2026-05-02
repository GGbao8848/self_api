import time

from app.agent.sessions import agent_session_store
from app.core.config import get_settings
from tests.data_helpers import create_image, create_voc_xml


def _wait_for_run(client, run_id: str, timeout_seconds: float = 30.0) -> dict:
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


def test_agent_langgraph_pipeline_runs_without_llm_provider(client, case_dir, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "")
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_MODEL", "")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "")
    monkeypatch.setenv("SELF_API_OPENROUTER_API_KEY", "")
    monkeypatch.setenv("SELF_API_OPENAI_API_KEY", "")

    dataset_dir = case_dir / "langgraph_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(200, 80, 40), size=(128, 96))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(128, 96),
        objects=[("cat", (12, 12, 100, 72))],
    )
    get_settings.cache_clear()

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": f"{dataset_dir} xml转yolo,yolo全转为0，之后划分数据集train_only ,之后滑窗裁剪，对滑窗裁剪的数据进行数据增强",
            "async_run": True,
        },
    )

    assert response.status_code == 200
    accepted = response.json()
    assert accepted["provider"] == "langgraph"

    completed = _wait_for_run(client, accepted["run_id"])
    assert completed["final_state"] == "completed"
    assert completed["provider"] == "langgraph"
    assert [item["name"] for item in completed["tool_calls"]] == [
        "xml-to-yolo",
        "reset-yolo-label-index",
        "split-yolo-dataset",
        "yolo-sliding-window-crop",
        "yolo-augment",
    ]
    assert completed["tool_calls"][-1]["arguments"]["output_dir"].endswith("_aug")
    assert completed["message"].startswith("LangGraph pipeline completed; output=")
