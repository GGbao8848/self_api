from app.agent.sessions import agent_session_store
from app.agent.types import LLMToolDecision
from app.core.config import get_settings
from tests.data_helpers import create_image, create_voc_xml, create_yolo_dataset


def test_agent_chat_requires_provider(client, monkeypatch, isolated_runtime) -> None:
    agent_session_store.clear()
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_PROVIDER", "")
    monkeypatch.setenv("SELF_API_LLM_DEFAULT_MODEL", "")
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "")
    monkeypatch.setenv("SELF_API_OPENAI_API_KEY", "")
    monkeypatch.setenv("SELF_API_OPENROUTER_API_KEY", "")
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
    monkeypatch.setenv("SELF_API_OLLAMA_MODEL", "qwen3.5:9b")
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
    assert data["model"] == "qwen3.5:9b"


def test_agent_tools_list(client, isolated_runtime) -> None:
    response = client.get("/api/v1/agent/tools")

    assert response.status_code == 200
    data = response.json()
    names = {item["name"] for item in data["items"]}
    assert data["total"] == len(data["items"])
    assert "scan-yolo-label-indices" in names
    assert "rewrite-yolo-label-indices" in names
    assert "xml-to-yolo" in names
    assert "yolo-sliding-window-crop" in names
    assert "yolo-augment" in names
    assert "annotate-visualize" in names
    assert "clean-nested-dataset-flat" in names
    assert "build-yolo-yaml" in names
    assert "publish-incremental-yolo-dataset" in names
    hints = {item["name"]: item["argument_hint"] for item in data["items"]}
    assert hints["build-yolo-yaml"]
    assert "output_yaml_path" in hints["build-yolo-yaml"]
    assert hints["publish-incremental-yolo-dataset"] == "{last_yaml, local_paths}"
    assert "yolo-train" not in names
    assert "remote-sbatch-yolo-train" not in names


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


def test_agent_executes_structured_yolo_sliding_window_crop_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "crop_dataset"
    create_yolo_dataset(dataset_dir, sample_count=1)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "crop dataset",
            "tool_name": "yolo-sliding-window-crop",
            "tool_arguments": {"input_dir": str(dataset_dir), "only_wide": False},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "yolo-sliding-window-crop"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_yolo-sliding-window-crop"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["generated_crops"] == 1


def test_agent_executes_structured_yolo_augment_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "augment_dataset"
    create_yolo_dataset(dataset_dir, sample_count=1)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "augment dataset",
            "tool_name": "yolo-augment",
            "tool_arguments": {
                "input_dir": str(dataset_dir),
                "vertical_flip": False,
                "brightness_up": False,
                "brightness_down": False,
                "contrast_up": False,
                "contrast_down": False,
                "gaussian_blur": False,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "yolo-augment"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_aug"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["generated_images"] == 1


def test_agent_executes_structured_annotate_visualize_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "visualize_dataset"
    create_yolo_dataset(dataset_dir, sample_count=1)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "visualize dataset",
            "tool_name": "annotate-visualize",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "annotate-visualize"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_visualized"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["written_images"] == 1


def test_agent_executes_structured_clean_nested_dataset_flat_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "nested_raw"
    leaf = dataset_dir / "session_a"
    create_image(leaf / "images" / "a.jpg", color=(255, 0, 0), size=(16, 16))
    create_voc_xml(
        leaf / "xmls" / "a.xml",
        filename="a.jpg",
        size=(16, 16),
        objects=[("cat", (2, 2, 8, 8))],
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "flatten dataset",
            "tool_name": "clean-nested-dataset-flat",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "clean-nested-dataset-flat"
    assert tool_call["arguments"]["output_dir"] == f"{dataset_dir}_cleaned_flat"
    assert tool_call["arguments"]["flatten"] is True
    assert tool_call["arguments"]["include_backgrounds"] is False
    assert tool_call["arguments"]["pairing_mode"] == "images_xmls_subfolders"
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["labeled_images"] == 1


def test_agent_executes_structured_build_yolo_yaml_tool(
    client,
    case_dir,
    isolated_runtime,
) -> None:
    dataset_dir = case_dir / "yaml_dataset"
    create_image(dataset_dir / "train" / "images" / "a.jpg", color=(10, 20, 30), size=(16, 16))
    create_image(dataset_dir / "val" / "images" / "b.jpg", color=(20, 30, 40), size=(16, 16))
    (dataset_dir / "classes.txt").write_text("object\n", encoding="utf-8")

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "build yaml",
            "tool_name": "build-yolo-yaml",
            "tool_arguments": {"input_dir": str(dataset_dir)},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "build-yolo-yaml"
    assert tool_call["arguments"]["output_yaml_path"] == str(dataset_dir / "data.yaml")
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["splits_included"] == ["train", "val"]
    assert (dataset_dir / "data.yaml").exists()


def test_agent_executes_structured_publish_incremental_yolo_dataset_tool(
    client,
    case_dir,
    monkeypatch,
    isolated_runtime,
) -> None:
    from app.schemas.preprocess import PublishYoloDatasetResponse

    dataset_dir = case_dir / "dataset_incremental"
    dataset_aug_dir = case_dir / "dataset_incremental_aug"
    create_yolo_dataset(dataset_dir, sample_count=1)
    create_yolo_dataset(dataset_aug_dir, sample_count=1)

    def fake_run_publish_incremental_yolo_dataset(_payload):
        return PublishYoloDatasetResponse(
            publish_mode="remote_sftp",
            output_yaml_path="/remote/datasets/demo_20260428/demo_20260428.yaml",
            dataset_root=str(dataset_dir),
            source_dataset_roots=[str(dataset_dir), str(dataset_aug_dir)],
            splits_included=["train"],
            classes_count=1,
            dataset_version="demo_20260428",
            published_dataset_dir="/remote/datasets/demo_20260428",
            staging_published_dataset_dir=str(case_dir / "staging"),
            staging_output_yaml_path=str(case_dir / "staging" / "demo_20260428.yaml"),
            local_archive_path=str(case_dir / "staging" / "demo_20260428.zip"),
            remote_target_host="172.31.1.42",
            remote_target_port=22,
            remote_archive_path="/remote/datasets/demo_20260428.zip",
            recommended_train_project="/remote/runs/detect",
            recommended_train_name="demo_20260428",
            last_yaml_merged=True,
            last_yaml_source="sftp",
        )

    monkeypatch.setattr(
        "app.agent.tools.registry._run_publish_incremental_yolo_dataset",
        fake_run_publish_incremental_yolo_dataset,
    )

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": "publish incremental dataset",
            "tool_name": "publish-incremental-yolo-dataset",
            "tool_arguments": {
                "input_dir": str(dataset_dir),
                "last_yaml": "sftp://172.31.1.42/remote/datasets/demo_prev/demo_prev.yaml",
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["final_state"] == "completed"
    tool_call = data["tool_calls"][0]
    assert tool_call["name"] == "publish-incremental-yolo-dataset"
    assert tool_call["arguments"]["local_paths"] == [str(dataset_dir)]
    assert tool_call["result"]["state"] == "succeeded"
    assert tool_call["result"]["result"]["published_dataset_dir"] == "/remote/datasets/demo_20260428"


def test_agent_api_requires_auth_when_enabled(client, monkeypatch, isolated_runtime) -> None:
    monkeypatch.setenv("SELF_API_AUTH_ENABLED", "true")
    monkeypatch.setenv("SELF_API_AUTH_ADMIN_PASSWORD", "secret-pass")
    monkeypatch.setenv("SELF_API_AUTH_SECRET_KEY", "unit-test-secret")
    get_settings.cache_clear()

    response = client.get("/api/v1/agent/tools")

    assert response.status_code == 401
