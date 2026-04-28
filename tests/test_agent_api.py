from app.agent.sessions import agent_session_store
from app.agent.types import LLMToolDecision
from app.core.config import get_settings
from tests.data_helpers import create_image, create_voc_xml, create_yolo_dataset


def test_agent_chat_requires_provider(client, isolated_runtime) -> None:
    agent_session_store.clear()
    get_settings.cache_clear()

    response = client.post("/api/v1/agent/chat", json={"message": "你好"})

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "requires_provider"
    assert data["session_id"]
    assert data["run_id"]
    assert data["tool_calls"] == []

    run_response = client.get(f"/api/v1/agent/runs/{data['run_id']}")
    assert run_response.status_code == 200
    assert run_response.json()["run_id"] == data["run_id"]

    session_response = client.get(f"/api/v1/agent/sessions/{data['session_id']}")
    assert session_response.status_code == 200
    assert session_response.json()["runs"][0]["run_id"] == data["run_id"]


def test_agent_chat_accepts_configured_ollama(client, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "ollama")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "qwen2.5:7b")
    from app.agent import runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "request_tool_decision",
        lambda **_: LLMToolDecision(action="respond", message="可执行 4 个工具"),
    )
    get_settings.cache_clear()

    response = client.post("/api/v1/agent/chat", json={"message": "列出能力"})

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["provider"] == "ollama"
    assert data["model"] == "qwen2.5:7b"


def test_agent_tools_list(client, isolated_runtime) -> None:
    response = client.get("/api/v1/agent/tools")

    assert response.status_code == 200
    data = response.json()
    names = {item["name"] for item in data["items"]}
    assert data["total"] == len(data["items"])
    assert "scan-yolo-label-indices" in names
    assert "rewrite-yolo-label-indices" in names
    assert "xml-to-yolo" in names


def test_agent_executes_structured_scan_tool(client, case_dir, isolated_runtime) -> None:
    labels_dir = case_dir / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text(
        "0 0.1 0.2 0.3 0.4\n2 0.2 0.3 0.4 0.5\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "scan labels",
            "tool_name": "scan-yolo-label-indices",
            "tool_arguments": {"input_dir": str(case_dir / "dataset")},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "scan-yolo-label-indices"
    assert data["tool_calls"][0]["error"] is None
    assert data["tool_calls"][0]["result"]["indices"] == [
        {"index": 0, "count": 1},
        {"index": 2, "count": 1},
    ]


def test_agent_routes_chinese_scan_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    labels_dir = case_dir / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "a.txt").write_text("3 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "request_tool_decision", lambda **_: LLMToolDecision(
        action="execute",
        tool_name="scan-yolo-label-indices",
        tool_arguments={"input_dir": str(case_dir / "dataset")},
    ))

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"查看标签索引 {case_dir / 'dataset'}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "scan-yolo-label-indices"
    assert data["tool_calls"][0]["result"]["indices"] == [{"index": 3, "count": 1}]


def test_agent_routes_chinese_rewrite_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    labels_dir = case_dir / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    label_file = labels_dir / "a.txt"
    label_file.write_text(
        "0 0.1 0.2 0.3 0.4\n2 0.2 0.3 0.4 0.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_module, "request_tool_decision", lambda **_: LLMToolDecision(
        action="execute",
        tool_name="rewrite-yolo-label-indices",
        tool_arguments={"input_dir": str(case_dir / "dataset"), "default_target_index": 1},
    ))

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"把标签索引全部改成1 {case_dir / 'dataset'}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "rewrite-yolo-label-indices"
    assert data["tool_calls"][0]["result"]["changed_lines"] == 2
    assert label_file.read_text(encoding="utf-8").splitlines() == [
        "1 0.1 0.2 0.3 0.4",
        "1 0.2 0.3 0.4 0.5",
    ]


def test_agent_tool_missing_input_dir_returns_failed_state(client, isolated_runtime) -> None:
    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "scan labels",
            "tool_name": "scan-yolo-label-indices",
            "tool_arguments": {},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "failed"
    assert data["tool_calls"][0]["error"]


def test_agent_executes_structured_xml_to_yolo_tool(client, case_dir, isolated_runtime) -> None:
    dataset_dir = case_dir / "voc_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "xml to yolo",
            "tool_name": "xml-to-yolo",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_result = data["tool_calls"][0]["result"]
    assert data["tool_calls"][0]["name"] == "xml-to-yolo"
    assert tool_result["state"] == "succeeded"
    assert tool_result["result"]["converted_files"] == 1
    assert tool_result["result"]["details_count"] == 1
    assert "details" not in tool_result["result"]
    assert (dataset_dir / "labels" / "img_1.txt").exists()


def test_agent_routes_chinese_xml_to_yolo_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    dataset_dir = case_dir / "voc_dataset"
    create_image(dataset_dir / "images" / "img_1.jpg", color=(255, 10, 10), size=(100, 100))
    create_voc_xml(
        dataset_dir / "xmls" / "img_1.xml",
        filename="img_1.jpg",
        size=(100, 100),
        objects=[("cat", (10, 20, 60, 80))],
    )
    monkeypatch.setattr(runtime_module, "request_tool_decision", lambda **_: LLMToolDecision(
        action="execute",
        tool_name="xml-to-yolo",
        tool_arguments={"input_dir": str(dataset_dir)},
    ))

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"xml转yolo {dataset_dir}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "xml-to-yolo"
    assert data["tool_calls"][0]["result"]["state"] == "succeeded"


def test_agent_routes_chinese_split_request(client, case_dir, monkeypatch, isolated_runtime) -> None:
    from app.agent import runtime as runtime_module

    dataset_dir = case_dir / "yolo_dataset"
    output_dir = case_dir / "yolo_dataset_split"
    create_yolo_dataset(dataset_dir, sample_count=5)
    monkeypatch.setattr(runtime_module, "request_tool_decision", lambda **_: LLMToolDecision(
        action="execute",
        tool_name="split-yolo-dataset",
        tool_arguments={"input_dir": str(dataset_dir), "output_dir": str(output_dir), "mode": "train_val", "train_ratio": 0.8, "val_ratio": 0.2, "test_ratio": 0.0},
    ))

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": f"划分数据集 8:2 {dataset_dir}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    assert data["tool_calls"][0]["name"] == "split-yolo-dataset"
    tool_result = data["tool_calls"][0]["result"]
    assert tool_result["state"] == "succeeded"
    assert tool_result["result"]["mode"] == "train_val"
    assert tool_result["result"]["train_images"] == 4
    assert tool_result["result"]["val_images"] == 1
    assert tool_result["result"]["test_images"] == 0
    assert tool_result["result"]["details_count"] == 5
    assert (output_dir / "train" / "images").exists()


def test_agent_api_requires_auth_when_enabled(client, monkeypatch, isolated_runtime) -> None:
    monkeypatch.setenv("SELF_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("SELF_API_AUTH_ADMIN_PASSWORD", "secret-pass")
    monkeypatch.setenv("SELF_API_AUTH_SECRET_KEY", "unit-test-secret")
    get_settings.cache_clear()

    response = client.get("/api/v1/agent/tools")

    assert response.status_code == 401
